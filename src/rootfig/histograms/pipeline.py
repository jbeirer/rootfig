"""From samples and variables to filled histograms.

This is the orchestration layer: for each sample it determines the union of
branches needed by the variable, selection and weight expressions, reads
them once, applies the selection semantics, chooses a common binning across
all samples, and fills one :class:`~rootfig.histograms.Histogram` per sample.
A :class:`~rootfig.model.Group` is filled through its samples and gets the sum
of their histograms.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import pairwise
from os import PathLike
from typing import Any

import awkward as ak
import numpy as np

from rootfig._typing import Hist
from rootfig.errors import (
    MissingBranchError,
    RootfigError,
    SourceError,
    SystematicError,
    annotate,
)
from rootfig.expressions import Expression, parse
from rootfig.histograms.build import Histogram, fill, from_sample, mirror
from rootfig.histograms.groups import regroup_histograms
from rootfig.histograms.sources import shared_source
from rootfig.histograms.stats import summarize
from rootfig.histograms.stored import read_stored, stored_mode
from rootfig.io import CHUNK_BYTES, ArraySource, FileSource, ReadCache, Source, as_source
from rootfig.io.sources import resolve_files
from rootfig.model.binning import Axis, resolve_axis
from rootfig.model.cuts import Cut, CutLike, as_cut
from rootfig.model.groups import Group
from rootfig.model.inputs import leaf_samples
from rootfig.model.samples import Sample
from rootfig.model.systematics import Systematic, SystematicLike, as_systematics
from rootfig.model.variables import Variable, as_variable
from rootfig.selection import Columns, NonFinitePolicy, Request, prepare_chunks

__all__ = [
    "branch_names",
    "build_histograms",
    "build_histograms_2d",
    "combined_selection",
    "combined_weight",
    "load_columns",
    "load_columns_each",
    "read_arrays",
    "read_chunks",
]


def combined_selection(sample: Sample, selection: CutLike | None) -> Cut | None:
    """Combine a sample's own selection with a plot-level selection using ``&``."""
    plot_cut = as_cut(selection)
    if sample.selection is None:
        return plot_cut
    if plot_cut is None:
        return sample.selection
    return sample.selection & plot_cut


def combined_weight(sample: Sample, weight: str | None) -> str | None:
    """Combine a sample's own weight with a plot-level weight multiplicatively."""
    return _product(sample.weight, weight)


def _product(first: str | None, second: str | None) -> str | None:
    if second is not None and not second.strip():
        second = None
    if first is None:
        return second
    if second is None:
        return first
    return f"({first}) * ({second})"


def read_arrays(
    sample: Sample, expressions: Sequence[Any], *, cache: ReadCache | None = None
) -> tuple[dict[str, Any], int]:
    """Read the branches ``expressions`` need from ``sample`` and return them with the event count.

    Only the union of the required branches is read (:func:`branch_names`). A
    ``cache`` serves the branches it holds and reads the rest, through the
    source instance it holds for these files, so a source rebuilt from the same
    paths reads neither the branches nor what the file says about itself again;
    in-memory sources are read directly. The number of events is taken from the
    arrays, or from the source when nothing had to be read (all expressions
    constant), so ``"1"`` or ``"True"`` still know how many events there are.
    """
    source = sample.source
    if cache is not None and isinstance(source, FileSource):
        # the instance the cache holds for these files, which has already learnt about them
        source = cache.source(source)
        needed = branch_names(source, expressions)
        arrays = cache.arrays(source, needed)
    else:
        needed = branch_names(source, expressions)
        arrays = source.arrays(needed)
    if needed:
        return arrays, len(next(iter(arrays.values())))
    return arrays, source_length(source)


def read_chunks(
    sample: Sample, expressions: Sequence[Any], *, cache: ReadCache | None = None
) -> Iterator[tuple[dict[str, Any], int]]:
    """Return the branches ``expressions`` need from ``sample``, a chunk of events at a time.

    Each chunk comes with its number of events, and the chunks, joined in order,
    are what :func:`read_arrays` returns. Files are read chunk by chunk
    (:meth:`~rootfig.io.FileSource.iterate`), so only the chunks being prepared
    are held, never all branches of all events; arrays already in memory (in-memory
    data, a ``cache``) are cut into slices of about as many bytes, views that copy
    nothing, so that they are prepared in parallel alike. The branch names are
    checked before this returns, and nothing is read until the first chunk is
    asked for. A chunk holds each branch in the type its file does, which the
    whole read promotes to one type for all files; a chunk whose types differ from
    the first chunk's raises :class:`_MixedTypesError` (see :func:`_prepared`).
    """
    source = sample.source
    if cache is None and isinstance(source, FileSource):
        needed = branch_names(source, expressions)
        if needed:
            return _same_types(source.iterate(needed))
    arrays, n_events = read_arrays(sample, expressions, cache=cache)
    return _slices(arrays, n_events)


class _MixedTypesError(Exception):
    """A chunk holds a branch in another type than the first chunk did."""


def _same_types(chunks: Iterator[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield ``chunks`` with their numbers of events, as long as their types stay the first's."""
    try:
        first = None
        for arrays in chunks:
            types = [array.type.content for array in arrays.values()]
            if first is None:
                first = types
            elif types != first:
                raise _MixedTypesError
            yield arrays, len(next(iter(arrays.values())))
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()


def _prepared(
    sample: Sample,
    expressions: Sequence[Any],
    requests: Sequence[Request],
    *,
    cache: ReadCache | None = None,
) -> list[Columns]:
    """Run ``requests`` on the chunks of ``sample`` that :func:`read_chunks` reads.

    Where the files hold a branch in different types (``int32`` in one, ``int64``
    in another), the sample is read whole instead (:func:`read_arrays`), which
    promotes them to one type before anything is evaluated: ``x * x`` would
    otherwise overflow in the ``int32`` file's chunks, and the result would depend
    on where the chunks start and on whether a cache was used. The difference shows
    at the first chunk of another type, so the chunks before it have been prepared
    in their own type. Their columns are discarded, and an error they raise that
    rootfig did not raise itself (NumPy's under ``np.errstate``, a warning turned
    into an error, a NumPy error handler's, Awkward's) sends the sample to the whole
    read as well if its files' types differ, which is only then looked up: the
    sample fails or succeeds as the whole read does. Running out of memory does not,
    since the whole read takes more. Their warnings are not taken back. Knowing
    every file's types beforehand would cost opening each file's metadata once
    more, a quarter of a second for a tree of 1 000 branches.
    """
    try:
        return prepare_chunks(read_chunks(sample, expressions, cache=cache), requests)
    except _MixedTypesError:
        pass
    except Exception as exc:
        chain = list(_chain(exc))
        if (
            cache is not None
            or isinstance(exc, SourceError)
            or all(isinstance(each, RootfigError) for each in chain)
            or any(isinstance(each, MemoryError) for each in chain)  # a whole read needs more
        ):
            raise
        try:
            differ = _types_differ(sample, expressions)
        except Exception:  # a file that cannot tell: the error stands as raised
            differ = False
        if not differ:
            raise
    return prepare_chunks(_slices(*read_arrays(sample, expressions, cache=cache)), requests)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and the exceptions it was raised from or while handling, once each."""
    seen: set[int] = set()
    cause: BaseException | None = exc
    while cause is not None and id(cause) not in seen:
        yield cause
        seen.add(id(cause))
        cause = cause.__cause__ or cause.__context__


def _types_differ(sample: Sample, expressions: Sequence[Any]) -> bool:
    """Return whether the files of ``sample`` hold the branches read in different types.

    The types come from each file's metadata (:meth:`~rootfig.io.FileSource.branch_forms`),
    so an empty tree has them too.
    """
    source = sample.source
    if not isinstance(source, FileSource) or len(source.files) < 2:
        return False
    needed = branch_names(source, expressions)
    tree = source.resolved_tree()

    def types(path: str) -> list[Any]:
        forms = FileSource(path, tree).branch_forms()
        return [forms[name].type if name in forms else None for name in needed]

    first = types(source.files[0])
    return any(types(path) != first for path in source.files[1:])


def _slices(arrays: dict[str, Any], n_events: int) -> Iterator[tuple[dict[str, Any], int]]:
    """Cut ``arrays`` into consecutive slices of at most about :data:`~rootfig.io.CHUNK_BYTES`.

    With nothing read (every expression a constant), each event still becomes a
    ``float64`` once a constant is broadcast to it, and those are what is counted.
    """
    size = sum(getattr(array, "nbytes", 0) for array in arrays.values()) or 8 * n_events
    count = max(1, min(n_events, -(-size // CHUNK_BYTES)))
    if count == 1:
        yield arrays, n_events
        return
    for start, stop in pairwise(n_events * part // count for part in range(count + 1)):
        yield {name: array[start:stop] for name, array in arrays.items()}, stop - start


def branch_names(source: Source, expressions: Sequence[Any]) -> list[str]:
    """Return the branches of ``source`` that ``expressions`` need, once each, in order of use.

    Raises
    ------
    MissingBranchError
        For a name that is neither a branch nor a constant.
    """
    available = source.branches()
    needed: list[str] = []
    for expression in expressions:
        for name in expression.required_branches(available):
            if name not in needed:
                needed.append(name)
    return needed


def source_length(source: Any) -> int:
    """Return the number of events in ``source`` (``num_entries()``, else a branch length)."""
    counter = getattr(source, "num_entries", None)
    if callable(counter):
        return int(counter())
    first = source.branches()[:1]
    if not first:
        return 0
    return len(next(iter(source.arrays(first).values())))


def load_columns(
    sample: Sample,
    variables: Sequence[Variable | str],
    *,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
    cache: ReadCache | None = None,
    scaled: bool = True,
) -> Columns:
    """Read the required branches of ``sample`` and prepare flat columns.

    The selection and weight given here are combined with those defined on the
    sample itself (see :func:`combined_selection` and :func:`combined_weight`).
    ``lumi`` scales samples that carry a cross section (see
    :meth:`~rootfig.model.Sample.lumi_scale`). ``scaled=False`` leaves the
    sample's scale and luminosity factor out, for an efficiency, in which they
    cancel: the weights are the event weights alone. With a ``cache`` the sample reads
    through the source instance it holds for its files
    (:func:`~rootfig.histograms.sources.shared_source`), which is then also
    where the branches and the cross-section numbers are read.
    """
    sample = shared_source(sample, cache)
    var_exprs = [as_variable(v).expression for v in variables]
    cut = combined_selection(sample, selection)
    weight_expr = combined_weight(sample, weight)
    request = Request(
        tuple(var_exprs),
        selection=None if cut is None else cut.expression,
        weight=weight_expr,
        scale=sample.scale * sample.lumi_scale(lumi) if scaled else 1.0,
        nonfinite=nonfinite,
        context=sample.label,
    )
    expressions = _expressions(var_exprs, cut, weight_expr)
    return _prepared(sample, expressions, [request], cache=cache)[0]


def load_columns_each(
    sample: Sample,
    variables: Sequence[Variable | str],
    *,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
) -> list[Columns]:
    """Like :func:`load_columns` once per variable, reading the source only once.

    The union of the branches needed by all variables, the selection and the
    weight is read in a single pass; each variable is then prepared on its own,
    so variables of different structure (per-event and per-object) may be mixed
    and each keeps the semantics it would have alone.
    """
    var_exprs = [as_variable(v).expression for v in variables]
    cut = combined_selection(sample, selection)
    weight_expr = combined_weight(sample, weight)
    scale = sample.scale * sample.lumi_scale(lumi)
    requests = [
        Request(
            (expression,),
            selection=None if cut is None else cut.expression,
            weight=weight_expr,
            scale=scale,
            nonfinite=nonfinite,
            context=sample.label,
        )
        for expression in var_exprs
    ]
    return _prepared(sample, _expressions(var_exprs, cut, weight_expr), requests)


def _expressions(
    var_exprs: Sequence[str], cut: Cut | None, weight_expr: str | None
) -> list[Expression]:
    """Return the parsed variables, selection and weight, whose branches are read."""
    expressions = [parse(v) for v in var_exprs]
    if cut is not None:
        expressions.append(cut.parsed())
    if weight_expr is not None:
        expressions.append(parse(weight_expr))
    return expressions


def build_histograms(
    items: Sequence[Sample | Group],
    variable: Variable | str,
    *,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
    systematics: Mapping[str, SystematicLike] | None = None,
    assume_poisson: bool = False,
    cache: ReadCache | None = None,
) -> list[Histogram]:
    """Fill one 1D histogram per sample or group, with a binning shared by all of them.

    Systematic variations (the sample's own and ``systematics``, which apply to
    every non-data sample; a sample's own source of the same name wins) are filled into
    :attr:`~rootfig.histograms.Histogram.variations` with the binning chosen
    from the nominal values. A :class:`~rootfig.model.Group` is filled through
    its samples, which share the binning like any other, and gets the sum of
    their histograms (:func:`~rootfig.histograms.groups.group_histogram`).

    A bare variable name that addresses a histogram stored in the samples'
    files (see :func:`~rootfig.histograms.stored_mode`) is read instead of
    filled; ``assume_poisson`` then accepts stored histograms without a sum of
    squared weights.

    A ``cache`` (:class:`~rootfig.io.ReadCache`) serves the branch arrays and
    stored histograms it holds and reads the rest, so several histograms built
    from the same files read them once; the result does not depend on it.
    """
    var = as_variable(variable)
    samples = [shared_source(s, cache) for s in leaf_samples(items)]
    if stored_mode(samples, [var]):
        stored = read_stored(
            samples,
            [var],
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
            systematics=systematics,
            assume_poisson=assume_poisson,
            cache=cache,
        )
        return regroup_histograms(items, stored)
    plot_level = as_systematics(systematics, "plot")
    loaded = [
        _load_with_variations(
            s,
            var,
            _sample_systematics(s, plot_level),
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
            cache=cache,
        )
        for s in samples
    ]
    axis = resolve_axis(
        var,
        [item.nominal.values for item in loaded],
        name=var.safe_name,
        weights=[item.nominal.weights for item in loaded],
    )
    # unbinned statistics feed the stats box and Histogram.entries, which a group's histogram
    # does not carry, so only the leaves that are top-level samples are summarised
    with_stats = [
        not isinstance(item, Group)
        for item in items
        for _ in (item.samples if isinstance(item, Group) else (item,))
    ]
    histograms = []
    for sample, item, keep in zip(samples, loaded, with_stats, strict=True):
        nominal = fill([axis], item.nominal)
        histograms.append(
            from_sample(
                sample,
                nominal,
                stats=summarize(item.nominal) if keep else None,
                per_object=item.nominal.per_object,
                weighted=item.nominal.weights is not None,
                variations={
                    name: _fill_variation(axis, nominal, up, down)
                    for name, (up, down) in item.variations.items()
                },
            )
        )
    return regroup_histograms(items, histograms)


def _sample_systematics(
    sample: Sample, plot_level: Mapping[str, Systematic] | None = None
) -> dict[str, Systematic]:
    """Return the systematics of ``sample``: ``plot_level`` ones, overridden by its own.

    Data carries no systematics (``Sample`` refuses them); plot-level ones skip it.
    """
    if sample.is_data:
        return {}
    return {**(plot_level or {}), **sample.systematics}


@dataclass(frozen=True)
class _Loaded:
    """Nominal columns of a sample and, per source, what its up and down variations are."""

    nominal: Columns
    variations: dict[str, tuple[Columns | float | None, Columns | float | None]]


def _load_with_variations(
    sample: Sample,
    var: Variable,
    systematics: Mapping[str, Systematic],
    *,
    selection: CutLike | None,
    weight: str | None,
    lumi: float | str | None,
    nonfinite: NonFinitePolicy,
    cache: ReadCache | None = None,
) -> _Loaded:
    """Prepare the nominal columns and every variation, reading the sample's branches once.

    Weight and branch variations are prepared from the chunks read for the
    nominal (:func:`read_chunks`), chunk by chunk alongside it; variations from
    other data read those. A variation is a :class:`Columns` to fill, a factor
    for the nominal histogram, or ``None`` for a down direction mirrored from the
    up one.
    """
    cut = combined_selection(sample, selection)
    cut_text = None if cut is None else cut.expression
    weight_expr = combined_weight(sample, weight)
    scale = sample.scale * sample.lumi_scale(lumi)
    expressions, used = _read_plan(sample, var, systematics, selection=selection, weight=weight)

    def request(context: str, weight: str | None, **extra: Any) -> Request:
        return Request(
            (var.expression,),
            selection=cut_text,
            weight=weight,
            scale=scale,
            nonfinite=nonfinite,
            context=context,
            **extra,
        )

    requests = [request(sample.label, weight_expr)]
    shifted: dict[tuple[str, str], int] = {}  # the request of each weight or branch variation
    for name, syst in systematics.items():
        for direction, spec in (("up", syst.up), ("down", syst.down)):
            if spec is None or syst.kind not in ("weight", "replace"):
                continue
            context = f"{sample.label} [{name} {direction}]"
            shifted[name, direction] = len(requests)
            note = _variation_note(context)
            if syst.kind == "weight":
                requests.append(request(context, _product(spec, weight), note=note))
            else:
                # the replacing branches take the place of the replaced ones, so the
                # variable, selection and weight all see the shifted values
                replaced = {old: new for old, new in spec.items() if old in used}
                requests.append(request(context, weight_expr, substitutes=replaced, note=note))
    prepared = _prepared(sample, expressions, requests, cache=cache)

    variations: dict[str, tuple[Columns | float | None, Columns | float | None]] = {}
    for name, syst in systematics.items():
        shifts: list[Columns | float | None] = []
        for direction, spec in (("up", syst.up), ("down", syst.down)):
            context = f"{sample.label} [{name} {direction}]"
            with _in_variation(context):
                if spec is None:
                    shifts.append(None)
                elif syst.kind == "norm":
                    shifts.append(float(spec))
                elif syst.kind == "samples":
                    variant = _variant_sample(sample, spec, context, cache=cache)
                    shifts.append(
                        load_columns(
                            variant,
                            [var],
                            selection=selection,
                            weight=weight,
                            lumi=lumi,
                            nonfinite=nonfinite,
                            cache=cache,
                        )
                    )
                else:
                    shifts.append(prepared[shifted[name, direction]])
        variations[name] = (shifts[0], shifts[1])
    return _Loaded(prepared[0], variations)


def _read_plan(
    sample: Sample,
    var: Variable,
    systematics: Mapping[str, Systematic],
    *,
    selection: CutLike | None,
    weight: str | None,
) -> tuple[list[Expression], set[str]]:
    """Return what :func:`_load_with_variations` reads for ``var``, and the branches it uses.

    The variable, the sample's selection and weight combined with the given
    ones, the weight expressions of ``weight`` systematics and the branches that
    replace used ones under ``replace`` systematics; the set holds the branches
    of the nominal expressions, which decides what a replacement applies to.
    :func:`~rootfig.histograms.prefetch` reads the same plan ahead, so both read
    the same branches.

    Raises
    ------
    MissingBranchError
        For a name that is not a branch, also in a variation (the message names
        the sample, source and direction).
    """
    cut = combined_selection(sample, selection)
    weight_expr = combined_weight(sample, weight)
    parsed = [parse(var.expression)]
    if cut is not None:
        parsed.append(cut.parsed())
    if weight_expr is not None:
        parsed.append(parse(weight_expr))
    available = sample.source.branches()
    used = {name for expression in parsed for name in expression.required_branches(available)}
    # weight variations and the branches that replace used ones join the single read
    extra: list[Expression] = []
    for name, syst in systematics.items():
        for direction, spec in (("up", syst.up), ("down", syst.down)):
            if spec is None or syst.kind not in ("weight", "replace"):
                continue
            if syst.kind == "weight":
                with _in_variation(f"{sample.label} [{name} {direction}]"):
                    spec_expression = parse(spec)
                    spec_expression.required_branches(available)
                extra.append(spec_expression)
            else:
                for old, new in spec.items():
                    if old not in used:
                        continue
                    if new not in available:
                        raise MissingBranchError(
                            new,
                            available=available,
                            context=f"{sample.label} [{name} {direction}]: replacing {old!r}",
                        )
                    extra.append(parse(f"`{new}`"))
    return [*parsed, *extra], used


def _in_variation(context: str) -> AbstractContextManager[None]:
    """Name the variation (sample, source, direction) in any rootfig error raised inside."""
    return annotate(_variation_note(context))


def _variation_note(context: str) -> str:
    """Return the note naming the variation ``context`` (sample, source, direction)."""
    return f"while evaluating the systematic variation {context}"


def _variant_sample(
    sample: Sample, spec: Any, context: str, *, cache: ReadCache | None = None
) -> Sample:
    """Return the sample a ``Systematic.samples`` variation reads: ``spec``, or its data.

    It is labelled ``context`` so its warnings and errors name the variation. Data
    that cannot look up a string ``ngen`` itself (in-memory arrays) takes the
    nominal sample's resolved number of generated events; files read their own,
    through the instance ``cache`` holds for them
    (:func:`~rootfig.histograms.sources.shared_source`).
    """
    if isinstance(spec, Sample):
        return shared_source(spec.replace(label=context), cache)
    nominal = sample.source
    try:
        if isinstance(nominal, FileSource | ArraySource) and (
            _is_file_spec(spec) or isinstance(spec, Mapping | ak.Array | np.ndarray)
        ):
            # the nominal tree name and entry range apply unless the spec names its own tree
            tree = (
                nominal.resolved_tree()
                if isinstance(nominal, FileSource)
                and _is_file_spec(spec)
                and resolve_files(spec)[1] is None
                else None
            )
            source = as_source(
                spec, tree=tree, entry_start=nominal.entry_start, entry_stop=nominal.entry_stop
            )
        else:
            source = as_source(spec)
    except (SourceError, OSError, TypeError, ValueError) as exc:
        msg = f"{context}: cannot use {spec!r} as varied data: {exc}"
        raise SystematicError(msg) from exc
    changes: dict[str, Any] = {"source": source, "systematics": {}, "label": context}
    if (
        sample.xsec is not None
        and isinstance(sample.ngen, str)
        and not callable(getattr(source, "read_scalar", None))
    ):
        changes["ngen"] = sample.generated_events()
    return shared_source(sample.replace(**changes), cache)


def _is_file_spec(spec: Any) -> bool:
    if isinstance(spec, str | PathLike):
        return True
    return isinstance(spec, list | tuple) and all(isinstance(f, str | PathLike) for f in spec)


def _fill_variation(
    axis: Axis, nominal: Hist, up: Columns | float | None, down: Columns | float | None
) -> tuple[Hist, Hist]:
    up_hist = _fill_shift(axis, nominal, up)
    return up_hist, mirror(nominal, up_hist) if down is None else _fill_shift(axis, nominal, down)


def _fill_shift(axis: Axis, nominal: Hist, shift: Columns | float | None) -> Hist:
    if isinstance(shift, float):
        return nominal * shift
    if shift is None:  # pragma: no cover - an up variation always exists
        msg = "internal error: missing up variation"
        raise SystematicError(msg)
    return fill([axis], shift)


def build_histograms_2d(
    samples: Sequence[Sample],
    x: Variable | str,
    y: Variable | str | None = None,
    *,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
    assume_poisson: bool = False,
) -> list[Histogram]:
    """Fill one 2D histogram per sample; ``x`` and ``y`` must share their structure.

    With ``y`` omitted, ``x`` must name a 2D histogram stored in the samples'
    files, which is read instead (see :func:`~rootfig.histograms.stored_mode`).
    Systematics are ignored either way: a 2D plot draws no variations, so the
    samples' sources are neither filled nor checked.
    """
    var_x = as_variable(x)
    # the y axis of a stored 2D histogram named by x alone: the same name (so the axes are
    # told apart by a suffix), none of x's label, unit or bins, which describe the x axis only
    var_y = Variable(var_x.expression, name=var_x.name) if y is None else as_variable(y)
    if stored_mode(samples, [var_x, var_y]):
        return read_stored(
            samples,
            [var_x, var_y],
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
            assume_poisson=assume_poisson,
            include_systematics=False,
        )
    if y is None:
        msg = (
            f"plot2d needs two variables, or the name of a 2D histogram stored in the "
            f"file; {var_x.expression!r} is neither"
        )
        raise SourceError(msg)
    columns = [
        load_columns(
            s, [var_x, var_y], selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
        )
        for s in samples
    ]
    weights = [c.weights for c in columns]
    axis_x = resolve_axis(
        var_x, [c.arrays[0] for c in columns], name=var_x.safe_name, weights=weights
    )
    name_y = var_y.safe_name if var_y.safe_name != var_x.safe_name else f"{var_y.safe_name}_y"
    axis_y = resolve_axis(var_y, [c.arrays[1] for c in columns], name=name_y, weights=weights)
    return [
        from_sample(
            sample,
            fill([axis_x, axis_y], cols),
            stats=summarize(cols),
            weighted=cols.weights is not None,
        )
        for sample, cols in zip(samples, columns, strict=True)
    ]
