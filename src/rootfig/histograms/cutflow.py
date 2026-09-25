"""Cut flows: event counts and weighted yields after successive selections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rootfig.expressions import parse
from rootfig.histograms.intervals import wilson_interval
from rootfig.histograms.pipeline import combined_weight, read_arrays
from rootfig.model.cuts import Cut, CutLike, as_cut
from rootfig.model.samples import Sample
from rootfig.selection import NonFinitePolicy, event_mask, event_weights

__all__ = ["Cutflow", "CutflowStep", "CutflowTable", "cutflow"]


@dataclass(frozen=True)
class CutflowStep:
    """Yields after one more cut has been applied.

    Attributes
    ----------
    label
        The cut's label (or expression). For the first step: the label (or
        expression) of the sample's own selection, ``"All"`` if it has none.
    expression
        The cut expression, empty for the first step.
    events
        Raw number of events passing all cuts so far.
    yield_
        Weighted yield (sum of weights, including the sample scale and luminosity).
    error
        Statistical uncertainty on ``yield_``, ``sqrt(sum w^2)``.
    negative_weights
        Whether an event passing all cuts so far has a negative weight; the
        efficiencies measured against this step then have no binomial interval.
    """

    label: str
    expression: str
    events: int
    yield_: float
    error: float
    negative_weights: bool = False


@dataclass(frozen=True)
class Cutflow:
    """The cut flow of one sample: a sequence of :class:`CutflowStep`."""

    sample: str
    steps: tuple[CutflowStep, ...]

    @property
    def labels(self) -> list[str]:
        """Step labels."""
        return [step.label for step in self.steps]

    @property
    def yields(self) -> np.ndarray:
        """Weighted yield per step."""
        return np.array([step.yield_ for step in self.steps], dtype=float)

    @property
    def events(self) -> np.ndarray:
        """Raw event count per step."""
        return np.array([step.events for step in self.steps], dtype=int)

    @property
    def efficiencies(self) -> np.ndarray:
        """Weighted efficiency of each step relative to the previous one (1 for the first).

        The plain ratio of yields: ``nan`` where the previous yield is zero. With
        signed (NLO) weights a yield can be negative, and the ratio may then lie
        outside ``[0, 1]``; it is still reported.
        """
        y = self.yields
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(y[:-1] != 0, y[1:] / y[:-1], np.nan)
        return np.concatenate([[1.0], rel])

    @property
    def absolute_efficiencies(self) -> np.ndarray:
        """Weighted efficiency of each step relative to the first (see :attr:`efficiencies`)."""
        y = self.yields
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y[0] != 0, y / y[0], np.nan)

    @property
    def efficiency_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """``(down, up)`` statistical errors of :attr:`efficiencies`, 0 for the first step.

        A step keeps a subset of the previous step's events, so its efficiency
        has a binomial uncertainty: the Wilson score interval (one standard
        deviation) of the two yields, with the effective entries of the previous
        step for weighted events (see :func:`~rootfig.histograms.intervals.wilson_interval`).
        ``nan`` where
        the efficiency is undefined or lies outside ``[0, 1]``, and after a step
        holding a negative weight (:attr:`CutflowStep.negative_weights`): no
        binomial interval describes signed weights.
        """
        return self._errors([self.steps[0], *self.steps[:-1]], self.efficiencies)

    @property
    def absolute_efficiency_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """``(down, up)`` errors of :attr:`absolute_efficiencies`, as :attr:`efficiency_errors`.

        Every step keeps a subset of the first step's events, the denominator.
        """
        return self._errors([self.steps[0]] * len(self.steps), self.absolute_efficiencies)

    def _errors(
        self, before: Sequence[CutflowStep], values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Wilson errors of ``values``, each step measured against the one ``before`` it."""
        lower, upper = wilson_interval(
            self.yields, [b.yield_ for b in before], [b.error**2 for b in before]
        )
        signed = np.array([b.negative_weights for b in before])
        down = np.where(signed, np.nan, values - lower)
        up = np.where(signed, np.nan, upper - values)
        # the first step is the reference itself: its efficiency is exact where defined
        exact = 0.0 if np.isfinite(values[0]) else np.nan
        down[0] = up[0] = exact
        return down, up


@dataclass(frozen=True)
class CutflowTable:
    """Cut flows of several samples with the same cuts; ``str(table)`` is an aligned text table."""

    rows: tuple[Cutflow, ...]

    def get(self, sample: str) -> Cutflow:
        """Return the cut flow of the sample labelled ``sample``."""
        for flow in self.rows:
            if flow.sample == sample:
                return flow
        msg = f"no cut flow for sample {sample!r}; have {[f.sample for f in self.rows]}"
        raise KeyError(msg)

    @property
    def samples(self) -> list[str]:
        """Sample labels in order."""
        return [flow.sample for flow in self.rows]

    @property
    def labels(self) -> list[str]:
        """Step labels (identical for all samples)."""
        return self.rows[0].labels if self.rows else []

    def __str__(self) -> str:
        if not self.rows:
            return ""
        header = ["", *(f"{flow.sample}" for flow in self.rows)]
        lines = [header]
        for index, label in enumerate(self.labels):
            cells = [label]
            for flow in self.rows:
                step = flow.steps[index]
                eff = flow.efficiencies[index]
                if step.events == step.yield_ == int(step.yield_):
                    cell = f"{step.events}"
                else:
                    cell = f"{_fmt(step.yield_)} ± {_fmt(step.error)} ({step.events})"
                if index > 0:
                    cell += f"  {100 * eff:5.1f}%"
                cells.append(cell)
            lines.append(cells)
        widths = [max(len(row[i]) for row in lines) for i in range(len(header))]
        return "\n".join(
            "  ".join(
                cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                for i, cell in enumerate(row)
            )
            for row in lines
        )


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def cutflow(
    sample: Sample,
    cuts: Sequence[CutLike],
    *,
    weight: str | None = None,
    lumi: float | str | None = None,
    nonfinite: NonFinitePolicy = "drop",
) -> Cutflow:
    """Apply ``cuts`` one after another to ``sample`` and count events and yields.

    The sample's own selection (if any) defines the first step; the plot-level
    ``weight`` is multiplied with the sample's weight, scale and luminosity
    factor. Per-object cuts pass an event when any object passes. Events with a
    missing or non-finite weight are excluded from every step (counts and
    yields); non-finite ones are reported following ``nonfinite`` (a
    :class:`~rootfig.errors.RootfigWarning`, or a
    :class:`~rootfig.errors.SelectionError` for ``"error"``).
    """
    steps: list[Cut] = []
    for item in cuts:
        cut = as_cut(item)
        if cut is not None:
            steps.append(cut)
    weight_expr = combined_weight(sample, weight)
    expressions = [parse(c.expression) for c in steps]
    if sample.selection is not None:
        expressions.append(sample.selection.parsed())
    if weight_expr is not None:
        expressions.append(parse(weight_expr))
    arrays, n_events = read_arrays(sample, expressions)
    weights = event_weights(
        weight_expr, arrays, n_events, nonfinite=nonfinite, context=sample.label
    )
    passing = np.isfinite(weights)
    weights = np.where(passing, weights, 0.0) * sample.scale * sample.lumi_scale(lumi)
    if sample.selection is not None:
        passing &= event_mask(sample.selection.expression, arrays, length=n_events)

    def step(label: str, expression: str) -> CutflowStep:
        selected = weights[passing]
        return CutflowStep(
            label=label,
            expression=expression,
            events=int(np.count_nonzero(passing)),
            yield_=float(selected.sum()),
            error=float(np.sqrt(np.sum(selected**2))),
            negative_weights=bool(np.any(selected < 0)),
        )

    base = sample.selection
    first_label = (base.label or base.expression) if base is not None else "All"
    result = [step(first_label, base.expression if base is not None else "")]
    for cut in steps:
        passing &= event_mask(cut.expression, arrays, length=n_events)
        result.append(step(cut.label or cut.expression, cut.expression))
    return Cutflow(sample=sample.label, steps=tuple(result))
