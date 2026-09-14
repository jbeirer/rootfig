"""Reading branches and filling histograms without drawing: :func:`load`, :func:`histograms`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import awkward as ak

from rootfig._typing import Hist
from rootfig.api._common import single_sample
from rootfig.errors import SelectionError, SourceError
from rootfig.expressions import parse
from rootfig.histograms import (
    Histogram,
    NormalizeSpec,
    build_histograms,
    combined_selection,
    read_arrays,
)
from rootfig.histograms import normalize as normalize_histogram
from rootfig.model import (
    Bins,
    CutLike,
    RangeSpec,
    SystematicLike,
    Variable,
    as_samples,
    as_variable,
)
from rootfig.selection import NonFinitePolicy, boolean_mask, depth_of

__all__ = ["histogram", "histograms", "load"]


def load(
    data: Any,
    expressions: str | Sequence[str] | Mapping[str, str] | None = None,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> ak.Array:
    """Read branches (or evaluate expressions) into an Awkward record array.

    Parameters
    ----------
    data
        File path(s), glob, ``"path:tree"``, a :class:`~rootfig.model.Sample`,
        or in-memory arrays.
    expressions
        Branch names or expressions to evaluate. A mapping gives the output
        field names explicitly (``{"pt": "Muon_pt / 1000"}``). ``None`` reads
        every branch.
    tree
        Tree name when ``data`` is a file specification.
    selection
        An event-level boolean expression; events failing it are dropped. A
        per-object selection raises :class:`~rootfig.errors.SelectionError`
        (apply object cuts inside the expressions instead, e.g.
        ``"Muon_pt[Muon_pt > 20]"``), and so does a numeric one (an integer flag
        would otherwise be taken as an index array; write ``"flag != 0"``).
    entry_start, entry_stop
        Entry range to read (ignored for a ``Sample``, which carries its own).

    Returns
    -------
    awkward.Array
        A record array with one field per expression.
    """
    sample = single_sample(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
    available = sample.source.branches()
    if expressions is None:
        fields: dict[str, str] = {name: f"`{name}`" for name in available}
    elif isinstance(expressions, str):
        fields = {expressions: expressions}
    elif isinstance(expressions, Mapping):
        fields = dict(expressions)
    else:
        fields = {e: e for e in expressions}
    if not fields:
        msg = "no expressions to load"
        raise SourceError(msg)

    parsed = {name: parse(text) for name, text in fields.items()}
    cut = combined_selection(sample, selection)
    arrays, n_events = read_arrays(sample, [*parsed.values(), *([cut.parsed()] if cut else [])])
    result = {
        name: expression.evaluate(arrays, length=n_events) for name, expression in parsed.items()
    }
    if cut is not None:
        mask = boolean_mask(cut.parsed(), arrays, length=n_events)
        if depth_of(mask) != 1:
            msg = (
                f"selection {cut.expression!r} is per-object; load() only supports per-event "
                "selections. Reduce it with any()/all()/count() or apply it inside the "
                "expressions, e.g. 'Muon_pt[Muon_pt > 20]'"
            )
            raise SelectionError(msg)
        result = {name: array[mask] for name, array in result.items()}
    return ak.Array(result)


def histograms(
    data: Any,
    variable: str | Variable,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | None = None,
    range: RangeSpec = None,
    label: str | Sequence[str] | None = None,
    normalize: NormalizeSpec = None,
    nonfinite: NonFinitePolicy = "drop",
    systematics: Mapping[str, SystematicLike] | None = None,
) -> list[Histogram]:
    """Fill one :class:`~rootfig.histograms.Histogram` per sample with shared binning.

    See :func:`plot` for the meaning of the arguments; this function stops
    before drawing. An integer ``bins`` without a ``range`` infers one robustly
    (``range="auto"`` for the full finite minimum and maximum). Systematic
    variations are in :attr:`~rootfig.histograms.Histogram.variations` and
    summarised by :func:`~rootfig.histograms.uncertainty`.
    """
    samples = as_samples(data, tree=tree, labels=label)
    var = as_variable(variable, bins=bins, range=range)
    hists = build_histograms(
        samples,
        var,
        selection=selection,
        weight=weight,
        lumi=lumi,
        nonfinite=nonfinite,
        systematics=systematics,
    )
    if normalize is None or normalize is False:
        return hists
    return [normalize_histogram(h, normalize) for h in hists]


def histogram(
    data: Any,
    variable: str | Variable,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | None = None,
    range: RangeSpec = None,
    normalize: NormalizeSpec = None,
    nonfinite: NonFinitePolicy = "drop",
) -> Hist:
    """Fill a single histogram and return it as a plain ``hist.Hist``.

    See :func:`plot` for the arguments. An integer ``bins`` without a ``range``
    infers one robustly (``range="auto"`` for the full finite minimum and
    maximum).

    Examples
    --------
    >>> h = rf.histogram(
    ...     "events.root", "Muon_pt", tree="events", selection="Muon_pt > 20", bins=(50, 0, 200)
    ... )  # doctest: +SKIP
    >>> h.values().sum()  # doctest: +SKIP
    """
    samples = [sample.replace(systematics={}) for sample in as_samples(data, tree=tree)]
    results = histograms(
        samples,
        variable,
        tree=tree,
        selection=selection,
        weight=weight,
        lumi=lumi,
        bins=bins,
        range=range,
        normalize=normalize,
        nonfinite=nonfinite,
    )
    if len(results) != 1:
        msg = f"histogram() takes a single sample, got {len(results)}; use histograms() instead"
        raise SourceError(msg)
    return results[0].hist
