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

import re
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from rootfig._typing import Hist
from rootfig.errors import SelectionError, SourceError, SystematicError, annotate
from rootfig.histograms.build import Histogram, from_sample
from rootfig.histograms.sources import shared_source
from rootfig.io import FileSource, ReadCache
from rootfig.io.objects import is_tree_class
from rootfig.model.binning import merge_target
from rootfig.model.cuts import CutLike
from rootfig.model.samples import Sample
from rootfig.model.systematics import Systematic, SystematicLike, as_systematics
from rootfig.model.variables import Variable
from rootfig.selection import NonFinitePolicy

__all__ = ["describe_axes", "read_stored", "stored_mode"]

_ROOT_DEFAULT_AXIS_TITLES = frozenset({"", "xaxis", "yaxis", "zaxis"})
_UPROOT_DEFAULT_AXIS_TITLE = re.compile(r"Axis \d+")  # what uproot writes for a label-less axis


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
        return _no_histogram(source, name)
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


def _no_histogram(source: FileSource, name: str) -> str:
    """Why ``name`` is not a stored histogram of ``source`` although no branch has been checked yet.

    An object of that name that is neither a ``TH1``/``TH2`` nor a tree (a
    ``TProfile``, ``TH3``, ``TParameter``, ...) cannot be plotted at all, so it
    is reported as such instead of leaving the tree lookup to fail on a missing
    branch, unless a branch of that name takes the name (a branch wins over any
    stored object). With several trees the tree lookup reports the ambiguity.

    Raises
    ------
    SourceError
        For an object rootfig cannot plot, naming its class.
    """
    path = source.files[0]
    classname = source.objects().get(name)
    if classname is None or is_tree_class(classname):
        return f"{path!r} holds no histogram of that name"
    trees = source.trees()
    if len(trees) > 1 or (trees and name in source.branches()):
        return f"{path!r} holds a {classname} of that name, which is not a histogram to read"
    msg = (
        f"object {name!r} in {path!r} is a {classname}, which rootfig cannot plot; stored ROOT "
        "histograms are read as TH1 and TH2 only"
    )
    raise SourceError(msg)


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
    include_systematics: bool = True,
    cache: ReadCache | None = None,
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
    ``include_systematics=False`` reads the nominal histograms only and leaves
    the samples' systematics unexamined, for callers that draw no variations
    (2D plots); the samples themselves are kept on the result as given. A
    ``cache`` (:class:`~rootfig.io.ReadCache`) serves the stored histograms it
    holds and reads the rest.

    Raises
    ------
    SelectionError
        For a selection, weight or ``nonfinite="error"`` request: those act on
        event data, which a stored histogram no longer has.
    BinningError
        If ``bins`` asks for anything but a merge of the stored bins, or a
        ``(low, high)`` range comes without bins (the stored range is fixed;
        ``xlim=`` zooms).
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
            sample, source, name, variables, lumi=lumi, assume_poisson=assume_poisson, cache=cache
        )
        sources = (
            {}
            if sample.is_data or not include_systematics
            else {**plot_level, **sample.systematics}
        )
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
                cache=cache,
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
    cache: ReadCache | None = None,
) -> Hist:
    if cache is not None:
        stored = cache.histogram(source, name, assume_poisson=assume_poisson)
    else:
        stored = source.read_histogram(name, assume_poisson=assume_poisson)
    if stored.ndim != len(variables):
        other = "plot2d" if stored.ndim == 2 else "plot"
        msg = (
            f"{name!r} in {source.files[0]!r} is a {stored.ndim}D histogram; draw it with {other}()"
        )
        raise SourceError(msg)
    named = describe_axes(stored, variables)
    factor = sample.scale * sample.lumi_scale(lumi)
    return named if factor == 1.0 else named * factor


def describe_axes(h: Hist, variables: Sequence[Variable | None]) -> Hist:
    """Copy ``h`` with its axes named and labelled after ``variables``, keeping their kind.

    One entry per axis; ``None`` leaves an axis as it is. Axis names follow the
    rule for histograms filled from trees (see :func:`_axis_names`). Category
    axes (labelled bins) survive as such. The existing axis titles are kept
    unless the variable has a label of its own or the title is a placeholder
    (ROOT's ``"xaxis"`` or empty, uproot's ``"Axis 1"``), which gives way to the
    variable; a unit on the variable is appended once. A ``TH2`` addressed by
    one variable lends it to the y axis too, where the object name would
    describe the histogram rather than that axis: a placeholder title there
    leaves the axis without a label of its own (hist then presents the axis
    name, ``<name>_y``). The histogram given is not modified.
    """
    result = h.copy()
    names = _axis_names(result.axes, variables)
    first = variables[0]
    for index, (axis, variable, name) in enumerate(zip(result.axes, variables, names, strict=True)):
        if variable is None:
            _rename_axis(axis, name)
            continue
        title = str(
            axis.label or ""
        )  # read before renaming: hist presents an unlabelled axis by name
        borrowed = index > 0 and first is not None and variable.expression == first.expression
        if variable.label is not None:
            label = variable.axis_label
        elif _is_placeholder_title(title):
            label = "" if borrowed else variable.axis_label
        elif variable.unit and not title.endswith(f"[{variable.unit}]"):
            label = f"{title} [{variable.unit}]"
        else:
            label = title
        _rename_axis(axis, name)
        axis.label = label
    return result


def _is_placeholder_title(title: str) -> bool:
    """Return True for an axis title that names no quantity, only the axis itself."""
    return (
        title in _ROOT_DEFAULT_AXIS_TITLES
        or _UPROOT_DEFAULT_AXIS_TITLE.fullmatch(title) is not None
    )


def _axis_names(axes: Sequence[Any], variables: Sequence[Variable | None]) -> list[str]:
    """Return one axis name per variable, as a histogram filled from a tree would carry.

    An axis without a variable keeps its name. The y axis gets a ``_y`` suffix
    only when both axes would share a name (the same stored ``TH2`` read for
    both, or a variable named like the axis left alone), since hist requires
    distinct names.
    """
    names = [
        axis.name if variable is None else variable.safe_name
        for axis, variable in zip(axes, variables, strict=True)
    ]
    if len(names) > 1 and names[1] == names[0]:
        names[1] = f"{names[1]}_y"
    return names


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
    cache: ReadCache | None = None,
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
                variant = _variant_sample(sample, spec, context, cache=cache)
                source = _stored_source(variant, name, variation=True)
                shifts.append(
                    _read_scaled(
                        variant,
                        source,
                        name,
                        variables,
                        lumi=lumi,
                        assume_poisson=assume_poisson,
                        cache=cache,
                    )
                )
    up = shifts[0]
    assert up is not None  # an up variation always exists
    return up, shifts[1]


def _variant_sample(
    sample: Sample, spec: Any, context: str, *, cache: ReadCache | None = None
) -> Sample:
    """Return the sample a ``Systematic.samples`` variation reads: other files, nominal settings.

    Its files are read through the instance ``cache`` holds for them
    (:func:`~rootfig.histograms.sources.shared_source`).
    """
    if isinstance(spec, Sample):
        return shared_source(spec.replace(label=context, systematics={}), cache)
    if isinstance(spec, str | PathLike) or (
        isinstance(spec, list | tuple) and all(isinstance(f, str | PathLike) for f in spec)
    ):
        try:
            source = FileSource(spec)
        except SourceError as exc:
            msg = f"{context}: cannot use {spec!r} as varied data: {exc}"
            raise SystematicError(msg) from exc
        return shared_source(sample.replace(source=source, systematics={}, label=context), cache)
    msg = f"{context}: a stored histogram takes its variations from other ROOT files, not {spec!r}"
    raise SystematicError(msg)
