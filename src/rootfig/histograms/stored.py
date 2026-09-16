"""Histograms already stored in ROOT files, read instead of filled.

A variable that is a bare name (``"mz"``) may address a ``TH1``/``TH2`` object
stored under that name rather than a branch. :func:`stored_mode` decides,
deterministically, whether a set of samples is read that way, and
:func:`read_stored` turns the stored histograms into
:class:`~rootfig.histograms.Histogram` objects with the sample's label, colour,
scale and luminosity scaling applied. Everything that needs event data
(selections, weights, range inference, weight systematics) is refused with an
error that says so, for the nominal samples and for the samples a systematic
variation reads alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from rootfig._typing import Hist
from rootfig.errors import BinningError, SelectionError, SourceError, SystematicError, annotate
from rootfig.histograms.build import Histogram, from_sample
from rootfig.io import FileSource
from rootfig.model.binning import DEFAULT_RANGE
from rootfig.model.cuts import CutLike
from rootfig.model.samples import Sample
from rootfig.model.systematics import Systematic, SystematicLike, as_systematics
from rootfig.model.variables import Variable
from rootfig.selection import NonFinitePolicy

__all__ = ["read_stored", "stored_mode"]

_ROOT_DEFAULT_AXIS_TITLES = frozenset({"", "xaxis", "yaxis", "zaxis"})


def stored_mode(samples: Sequence[Sample], variables: Sequence[Variable]) -> bool:
    """Decide whether ``variables`` name histograms stored in the files of ``samples``.

    Stored mode applies when every sample reads files without an explicit tree
    or entry range, the variable is a bare name and the first file of every
    sample holds a 1D/2D histogram of that name at top level, and the file has
    no tree or its only tree has no branch of that name. Explicit intent wins:
    a ``tree=`` or an entry range always means a branch. For a 2D plot both
    variables must be stored histograms of the same name.

    Raises
    ------
    SourceError
        If some samples resolve to a stored histogram and others to a branch,
        or a file holds several trees besides the histogram (pass ``tree=`` to
        read a branch).
    """
    name = _stored_name(variables)
    if name is None:
        return False
    verdicts = [(sample, _sample_reads_stored(sample, name)) for sample in samples]
    stored = [sample.label for sample, verdict in verdicts if verdict]
    if stored and len(stored) != len(verdicts):
        branch = [sample.label for sample, verdict in verdicts if not verdict]
        msg = (
            f"{name!r} is a histogram stored in the files of {stored} but a branch for "
            f"{branch}; every sample of one plot must provide it the same way"
        )
        raise SourceError(msg)
    return bool(stored)


def _stored_name(variables: Sequence[Variable]) -> str | None:
    """Return the bare name shared by ``variables``, or ``None`` when they are expressions."""
    names = set()
    for variable in variables:
        parsed = variable.parsed()
        if not parsed.is_trivial:
            return None
        names.add(parsed.names[0])
    return names.pop() if len(names) == 1 else None


def _sample_reads_stored(sample: Sample, name: str) -> bool:
    source = sample.source
    if not isinstance(source, FileSource):
        return False
    if source.tree is not None or source.entry_start is not None or source.entry_stop is not None:
        return False
    if name not in source.histograms():
        return False
    trees = source.trees()
    if not trees:
        return True
    if len(trees) > 1:
        msg = (
            f"{source.files[0]!r} holds a histogram {name!r} and several trees {trees}; "
            "pass tree=... to read a branch, or FileSource.read_histogram() for the histogram"
        )
        raise SourceError(msg)
    return name not in source.branches()


def read_stored(
    samples: Sequence[Sample],
    variables: Sequence[Variable],
    *,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
    systematics: Mapping[str, SystematicLike] | None = None,
    assume_poisson: bool = False,
) -> list[Histogram]:
    """Read the histogram named by ``variables`` from every sample's files.

    One :class:`~rootfig.histograms.Histogram` per sample, in ``Weight``
    storage, scaled by the sample's ``scale`` and luminosity factor. An integer
    ``bins`` on a variable rebins the stored histogram to that many bins;
    ``label`` and ``unit`` replace the stored axis title. Systematics of kind
    ``"norm"`` scale the histogram and ``Systematic.samples`` reads the same
    name from other files, which are checked like the nominal ones; the other
    kinds need event data.

    Raises
    ------
    SelectionError
        For a selection, weight, ``nonfinite="error"`` or range request: those
        act on event data, which a stored histogram no longer has.
    BinningError
        If ``bins`` is not an integer dividing the stored bin count.
    SystematicError
        For weight or branch-replacement systematics, or a variation whose
        sample cannot read the stored histogram.
    SourceError
        If the stored histogram has another dimensionality than ``variables``,
        or its variances are unusable (see ``assume_poisson``).
    """
    name = _stored_name(variables)
    if name is None:
        msg = (
            "read_stored() needs the bare name of a stored histogram, not "
            f"{[variable.expression for variable in variables]}; stored_mode() tells whether "
            "samples provide one"
        )
        raise SourceError(msg)
    _reject_event_options(name, variables, selection=selection, weight=weight, nonfinite=nonfinite)
    plot_level = as_systematics(systematics, "plot") or {}
    result = []
    for sample in samples:
        source = _stored_source(sample, name)
        nominal = _read_scaled(
            sample, source, name, variables, lumi=lumi, assume_poisson=assume_poisson
        )
        sources = {} if sample.is_data else {**plot_level, **sample.systematics}
        variations = {
            syst_name: _variation(
                sample,
                syst_name,
                syst,
                name=name,
                nominal=nominal,
                variables=variables,
                lumi=lumi,
                assume_poisson=assume_poisson,
            )
            for syst_name, syst in sources.items()
        }
        histogram = from_sample(sample, nominal, variations=variations)
        # integer bin counts were checked above; None keeps the stored binning of that axis
        counts = [
            variable.bins if isinstance(variable.bins, int) else None for variable in variables
        ]
        result.append(histogram.rebinned_to(counts))
    return result


def _reject_event_options(
    name: str,
    variables: Sequence[Variable],
    *,
    selection: CutLike | None,
    weight: str | None,
    nonfinite: NonFinitePolicy,
) -> None:
    hint = f"{name!r} is a histogram stored in the file, not a branch"
    if selection is not None:
        msg = f"{hint}; a selection cannot be applied to it. Plot the branch from the tree instead"
        raise SelectionError(msg)
    if weight is not None:
        msg = f"{hint}; a weight cannot be applied to it. Use Sample(scale=...) for a constant"
        raise SelectionError(msg)
    if nonfinite != "drop":
        msg = f"{hint}; nonfinite= applies only when filling from event data"
        raise SelectionError(msg)
    for variable in variables:
        if variable.range not in (None, DEFAULT_RANGE):
            msg = (
                f"{hint}; its axis range is fixed. Use xlim= to zoom, or rebin with an "
                "integer bins="
            )
            raise BinningError(msg)
        if variable.bins is not None and not isinstance(variable.bins, int):
            msg = (
                f"{hint}; it can only be rebinned by merging adjacent bins, so bins= must be "
                f"an integer dividing its bin count, not {variable.bins!r}"
            )
            raise BinningError(msg)


def _stored_source(sample: Sample, name: str, *, variation: bool = False) -> FileSource:
    """Return the files ``sample`` reads the stored ``name`` from.

    Refuses everything that would need event data: an in-memory source, an
    explicit tree or entry range (they address a branch), and the sample's own
    selection or weight. The sample of a systematic ``variation`` gets a
    :class:`SystematicError` for all of these, the nominal one a
    :class:`SourceError` or :class:`SelectionError`.
    """
    hint = f"{name!r} is a histogram stored in the files of sample {sample.label!r}"
    source_error = SystematicError if variation else SourceError
    option_error = SystematicError if variation else SelectionError
    source = sample.source
    if not isinstance(source, FileSource):
        msg = f"{hint}; it cannot be read from {source.describe()}"
        raise source_error(msg)
    if source.tree is not None:
        msg = f"{hint}; tree={source.tree!r} addresses a branch instead. Drop tree= to read it"
        raise source_error(msg)
    if source.entry_start is not None or source.entry_stop is not None:
        msg = f"{hint}; an entry range applies to a tree and cannot be applied to it"
        raise source_error(msg)
    if sample.selection is not None:
        msg = f"{hint}; the sample's selection cannot be applied to it"
        raise option_error(msg)
    if sample.weight is not None:
        msg = f"{hint}; the sample's weight cannot be applied to it (use scale= for a constant)"
        raise option_error(msg)
    return source


def _read_scaled(
    sample: Sample,
    source: FileSource,
    name: str,
    variables: Sequence[Variable],
    *,
    lumi: float | str | None,
    assume_poisson: bool,
) -> Hist:
    stored = source.read_histogram(name, assume_poisson=assume_poisson)
    if stored.ndim != len(variables):
        other = "plot2d" if stored.ndim == 2 else "plot"
        msg = (
            f"{name!r} in {source.files[0]!r} is a {stored.ndim}D histogram; draw it with {other}()"
        )
        raise SourceError(msg)
    named = _named(stored, variables)
    factor = sample.scale * sample.lumi_scale(lumi)
    return named if factor == 1.0 else named * factor


def _named(stored: Hist, variables: Sequence[Variable]) -> Hist:
    """Copy ``stored`` with its axes named after the variables, keeping their kind.

    Category axes (labelled bins) survive as such. The stored axis titles are
    kept unless the variable has a label of its own or the title is ROOT's
    placeholder (``"xaxis"``); a unit on the variable is appended once.
    """
    result = stored.copy()
    for index, (axis, variable) in enumerate(zip(result.axes, variables, strict=True)):
        title = str(axis.label or "")
        if variable.label is not None or title in _ROOT_DEFAULT_AXIS_TITLES:
            label = variable.axis_label
        elif variable.unit and not title.endswith(f"[{variable.unit}]"):
            label = f"{title} [{variable.unit}]"
        else:
            label = title
        _rename_axis(axis, variable.safe_name if index == 0 else f"{variable.safe_name}_y")
        axis.label = label
    return result


def _rename_axis(axis: Any, name: str) -> None:
    """Set the name of ``axis`` in place.

    hist exposes an axis name read-only, because it identifies the axis inside
    a histogram, and keeps it in the axis' metadata mapping; this is the one
    place that writes there, and only on a copy nothing else refers to.
    """
    axis._raw_metadata["name"] = name


def _variation(
    sample: Sample,
    syst_name: str,
    syst: Systematic,
    *,
    name: str,
    nominal: Hist,
    variables: Sequence[Variable],
    lumi: float | str | None,
    assume_poisson: bool,
) -> tuple[Hist, Hist | None]:
    if syst.kind not in ("norm", "samples"):
        msg = (
            f"systematic {syst_name!r} of sample {sample.label!r} varies the "
            f"{'weight' if syst.kind == 'weight' else 'branches'}, which needs event data; a "
            "stored histogram takes normalisation factors or Systematic.samples() only"
        )
        raise SystematicError(msg)
    shifts: list[Hist | None] = []
    for direction, spec in (("up", syst.up), ("down", syst.down)):
        if spec is None:
            shifts.append(None)
        elif syst.kind == "norm":
            shifts.append(nominal * float(spec))
        else:
            context = f"{sample.label} [{syst_name} {direction}]"
            with annotate(f"while evaluating the systematic variation {context}"):
                variant = _variant_sample(sample, spec, context)
                source = _stored_source(variant, name, variation=True)
                shifts.append(
                    _read_scaled(
                        variant, source, name, variables, lumi=lumi, assume_poisson=assume_poisson
                    )
                )
    up = shifts[0]
    assert up is not None  # an up variation always exists
    return up, shifts[1]


def _variant_sample(sample: Sample, spec: Any, context: str) -> Sample:
    """Return the sample a ``Systematic.samples`` variation reads: other files, nominal settings."""
    if isinstance(spec, Sample):
        return spec.replace(label=context, systematics={})
    if isinstance(spec, str | PathLike) or (
        isinstance(spec, list | tuple) and all(isinstance(f, str | PathLike) for f in spec)
    ):
        try:
            source = FileSource(spec)
        except SourceError as exc:
            msg = f"{context}: cannot use {spec!r} as varied data: {exc}"
            raise SystematicError(msg) from exc
        return sample.replace(source=source, systematics={}, label=context)
    msg = f"{context}: a stored histogram takes its variations from other ROOT files, not {spec!r}"
    raise SystematicError(msg)
