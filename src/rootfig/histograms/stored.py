"""Histograms already stored in ROOT files, read instead of filled.

A variable that is a bare name (``"mz"``, or ``"`sel/mz`"`` for a histogram
inside a directory) may address a ``TH1``/``TH2`` object stored under that name
rather than a branch. :func:`stored_mode` decides,
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
from rootfig.errors import SelectionError, SourceError, SystematicError, annotate
from rootfig.histograms.build import Histogram, from_sample
from rootfig.io import FileSource
from rootfig.model.binning import merge_target
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
    sample holds a 1D/2D histogram of that name, and the file has no tree or its
    only tree has no branch of that name. A histogram inside a directory is
    named by its path in backticks (``"`sel/mz`"``), like a branch with odd
    characters. Explicit intent wins: a ``tree=`` or an entry range always means
    a branch. For a 2D plot both variables must be stored histograms of the same
    name.

    Raises
    ------
    SourceError
        If some samples resolve to a stored histogram and others do not (the
        message says why for each: a branch of the same name, a tree or entry
        range given explicitly, in-memory data, no such histogram), or a file
        holds several trees besides the histogram (pass ``tree=`` to read a
        branch).
    """
    name = _stored_name(variables)
    if name is None:
        return False
    verdicts = [(sample, _why_not_stored(sample, name)) for sample in samples]
    stored = [sample.label for sample, reason in verdicts if reason is None]
    if stored and len(stored) != len(verdicts):
        others = "; ".join(
            f"{sample.label!r}: {reason}" for sample, reason in verdicts if reason is not None
        )
        msg = (
            f"{name!r} is a histogram stored in the files of {stored}, but not for every sample "
            f"({others}); every sample of one plot must provide it the same way"
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


def _why_not_stored(sample: Sample, name: str) -> str | None:
    """Why ``sample`` does not read ``name`` as a stored histogram; ``None`` when it does."""
    source = sample.source
    if not isinstance(source, FileSource):
        return f"{source.describe()} hold no stored histograms"
    explicit = _addresses_a_tree(source)
    if explicit is not None:
        return explicit
    if name not in source.histograms():
        return f"{source.files[0]!r} holds no histogram of that name"
    trees = source.trees()
    if len(trees) > 1:
        msg = (
            f"{source.files[0]!r} holds a histogram {name!r} and several trees {trees}; "
            "pass tree=... to read a branch, or FileSource.read_histogram() for the histogram"
        )
        raise SourceError(msg)
    if trees and name in source.branches():
        return f"tree {trees[0]!r} has a branch of that name, which wins"
    return None


def _addresses_a_tree(source: FileSource) -> str | None:
    """Why ``source`` addresses a tree rather than stored histograms, else ``None``."""
    if source.tree is not None:
        return f"tree={source.tree!r} addresses a branch"
    if source.entry_start is not None or source.entry_stop is not None:
        return "an entry range addresses a tree"
    return None


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
    storage, scaled by the sample's ``scale`` and luminosity factor. A
    variable's ``bins`` merges the stored bins: an integer count, or edges that
    coincide with the stored ones (see
    :func:`~rootfig.model.binning.merge_target`); ``label`` and ``unit``
    replace the stored axis title. Systematics of kind ``"norm"`` scale the
    histogram and ``Systematic.samples`` reads the same name from other files,
    which are checked like the nominal ones; the other kinds need event data.

    Raises
    ------
    SelectionError
        For a selection, weight or ``nonfinite="error"`` request: those act on
        event data, which a stored histogram no longer has.
    BinningError
        If ``bins`` asks for anything but a merge of the stored bins, or a
        ``range`` comes without bins (the stored range is fixed; ``xlim=`` zooms).
    SystematicError
        For weight or branch-replacement systematics, or a variation whose
        sample addresses a tree or in-memory data instead of stored histograms.
    SourceError
        If the stored histogram has another dimensionality than ``variables``,
        its variances are unusable (see ``assume_poisson``), or a file (of a
        sample or of a variation) lacks it.
    """
    name = _stored_name(variables)
    if name is None:
        msg = (
            "read_stored() needs the bare name of a stored histogram, not "
            f"{[variable.expression for variable in variables]}; stored_mode() tells whether "
            "samples provide one"
        )
        raise SourceError(msg)
    _reject_event_options(name, selection=selection, weight=weight, nonfinite=nonfinite)
    # what each axis is merged to; checked before any file is read
    targets = [merge_target(variable.bins, variable.range) for variable in variables]
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
        result.append(histogram.rebinned_to(targets))
    return result


def _reject_event_options(
    name: str, *, selection: CutLike | None, weight: str | None, nonfinite: NonFinitePolicy
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
