"""From samples and variables to filled histograms.

This is the orchestration layer: for each sample it determines the union of
branches needed by the variable, selection and weight expressions, reads
them once, applies the selection semantics, chooses a common binning across
all samples, and fills one :class:`~rootfig.histograms.Histogram` per sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rootfig.expressions import parse
from rootfig.histograms.build import Histogram, fill
from rootfig.histograms.stats import summarize
from rootfig.model.binning import resolve_axis
from rootfig.model.cuts import Cut, CutLike, as_cut
from rootfig.model.samples import Sample
from rootfig.model.variables import Variable, as_variable
from rootfig.selection import Columns, NonFinitePolicy, prepare

__all__ = [
    "build_histograms",
    "build_histograms_2d",
    "combined_selection",
    "combined_weight",
    "load_columns",
    "load_columns_each",
    "read_arrays",
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
    if weight is not None and not weight.strip():
        weight = None
    if sample.weight is None:
        return weight
    if weight is None:
        return sample.weight
    return f"({sample.weight}) * ({weight})"


def read_arrays(sample: Sample, expressions: Sequence[Any]) -> tuple[dict[str, Any], int]:
    """Read the branches ``expressions`` need from ``sample`` and return them with the event count.

    Only the union of the required branches is read. The number of events is
    taken from the arrays, or from the source when nothing had to be read
    (all expressions constant), so ``"1"`` or ``"True"`` still know how many
    events there are.
    """
    available = sample.source.branches()
    needed: list[str] = []
    for expression in expressions:
        for name in expression.required_branches(available):
            if name not in needed:
                needed.append(name)
    arrays = sample.source.arrays(needed)
    if needed:
        return arrays, len(next(iter(arrays.values())))
    return arrays, source_length(sample.source)


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
) -> Columns:
    """Read the required branches of ``sample`` and prepare flat columns.

    The selection and weight given here are combined with those defined on the
    sample itself (see :func:`combined_selection` and :func:`combined_weight`).
    ``lumi`` scales samples that carry a cross section (see
    :meth:`~rootfig.model.Sample.lumi_scale`).
    """
    var_exprs = [as_variable(v).expression for v in variables]
    cut = combined_selection(sample, selection)
    weight_expr = combined_weight(sample, weight)
    arrays, n_events = _read_for(sample, var_exprs, cut, weight_expr)
    return prepare(
        arrays,
        var_exprs,
        selection=None if cut is None else cut.expression,
        weight=weight_expr,
        scale=sample.scale * sample.lumi_scale(lumi),
        nonfinite=nonfinite,
        context=sample.label,
        n_events=n_events,
    )


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
    arrays, n_events = _read_for(sample, var_exprs, cut, weight_expr)
    scale = sample.scale * sample.lumi_scale(lumi)
    return [
        prepare(
            arrays,
            [expression],
            selection=None if cut is None else cut.expression,
            weight=weight_expr,
            scale=scale,
            nonfinite=nonfinite,
            context=sample.label,
            n_events=n_events,
        )
        for expression in var_exprs
    ]


def _read_for(
    sample: Sample, var_exprs: Sequence[str], cut: Cut | None, weight_expr: str | None
) -> tuple[dict[str, Any], int]:
    """Read the branches needed by the variables, the selection and the weight."""
    expressions = [parse(v) for v in var_exprs]
    if cut is not None:
        expressions.append(cut.parsed())
    if weight_expr is not None:
        expressions.append(parse(weight_expr))
    return read_arrays(sample, expressions)


def build_histograms(
    samples: Sequence[Sample],
    variable: Variable | str,
    *,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
) -> list[Histogram]:
    """Fill one 1D histogram per sample with a binning shared by all of them."""
    var = as_variable(variable)
    columns = [
        load_columns(s, [var], selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite)
        for s in samples
    ]
    axis = resolve_axis(
        var,
        [c.values for c in columns],
        name=var.safe_name,
        weights=[c.weights for c in columns],
    )
    return [
        Histogram(
            hist=fill([axis], cols),
            label=sample.label,
            sample=sample,
            stats=summarize(cols),
            is_data=sample.is_data,
            color=sample.color,
            histtype=sample.histtype,
        )
        for sample, cols in zip(samples, columns, strict=True)
    ]


def build_histograms_2d(
    samples: Sequence[Sample],
    x: Variable | str,
    y: Variable | str,
    *,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
) -> list[Histogram]:
    """Fill one 2D histogram per sample; ``x`` and ``y`` must share their structure."""
    var_x, var_y = as_variable(x), as_variable(y)
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
        Histogram(
            hist=fill([axis_x, axis_y], cols),
            label=sample.label,
            sample=sample,
            stats=summarize(cols),
            is_data=sample.is_data,
            color=sample.color,
            histtype=sample.histtype,
        )
        for sample, cols in zip(samples, columns, strict=True)
    ]
