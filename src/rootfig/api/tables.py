"""Unbinned statistics and cut flows: :func:`summarize`, :func:`cutflow`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rootfig.histograms import (
    CutflowTable,
    EfficiencyInterval,
    Summary,
    describe_table,
    load_columns_each,
)
from rootfig.histograms import cutflow as cutflow_of
from rootfig.histograms import summarize as summarize_columns
from rootfig.model import (
    CutLike,
    Variable,
    as_samples,
    as_variable,
)
from rootfig.selection import NonFinitePolicy

__all__ = ["SummaryTable", "cutflow", "summarize"]


@dataclass(frozen=True)
class SummaryTable:
    """Summary statistics for several variables and samples.

    ``str(table)`` gives an aligned text table; :meth:`get` returns a single
    :class:`~rootfig.histograms.Summary`.
    """

    rows: tuple[tuple[str, str, Summary], ...]
    """``(sample label, variable expression, summary)`` triples."""

    def get(self, variable: str, sample: str | None = None) -> Summary:
        """Return the summary for ``variable`` (and ``sample``, if several)."""
        matches = [
            s
            for label, var, s in self.rows
            if var == variable and (sample is None or label == sample)
        ]
        if not matches:
            msg = f"no summary for variable {variable!r}" + (
                f" and sample {sample!r}" if sample else ""
            )
            raise KeyError(msg)
        if len(matches) > 1:
            msg = f"several samples have variable {variable!r}; pass sample=..."
            raise KeyError(msg)
        return matches[0]

    @property
    def samples(self) -> list[str]:
        """Distinct sample labels in order of appearance."""
        return list(dict.fromkeys(label for label, _, _ in self.rows))

    @property
    def variables(self) -> list[str]:
        """Distinct variable expressions in order of appearance."""
        return list(dict.fromkeys(var for _, var, _ in self.rows))

    def __str__(self) -> str:
        multi = len(self.samples) > 1
        entries = [(f"{label}: {var}" if multi else var, s) for label, var, s in self.rows]
        return describe_table(entries)


def summarize(
    data: Any,
    variables: str | Variable | Sequence[str | Variable],
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    label: str | Sequence[str] | None = None,
    nonfinite: NonFinitePolicy = "drop",
) -> SummaryTable:
    """Compute entries, mean, standard deviation, skewness, ... for variables and samples.

    Examples
    --------
    >>> table = rf.summarize(
    ...     "events.root", ["MET", "Muon_pt"], tree="events", selection="nMuon > 0"
    ... )  # doctest: +SKIP
    >>> print(table)  # doctest: +SKIP
    >>> table.get("MET").mean  # doctest: +SKIP
    """
    samples = as_samples(data, tree=tree, labels=label)
    var_list = [variables] if isinstance(variables, str | Variable) else list(variables)
    rows: list[tuple[str, str, Summary]] = []
    for sample in samples:
        # One read per sample: the branches of all variables are fetched together.
        per_variable = load_columns_each(
            sample, var_list, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
        )
        for var, columns in zip(var_list, per_variable, strict=True):
            rows.append((sample.label, as_variable(var).expression, summarize_columns(columns)))
    return SummaryTable(tuple(rows))


def cutflow(
    data: Any,
    cuts: Sequence[CutLike],
    *,
    tree: str | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    label: str | Sequence[str] | None = None,
    nonfinite: NonFinitePolicy = "drop",
    interval: EfficiencyInterval = "auto",
) -> CutflowTable:
    """Count events and weighted yields after each successive cut, per sample.

    The first row holds all events (after the sample's own selection, if any);
    every further row applies one more cut. Per-object cuts pass an event when
    any object passes. ``weight``, ``lumi`` and ``nonfinite`` work as in
    :func:`plot`: events with a ``nan``/``inf`` weight are excluded from all
    steps with a warning, or raise for ``nonfinite="error"``. Yields carry
    ``sqrt(sum w^2)``; efficiencies carry the confidence interval ``interval``
    names (:attr:`~rootfig.Cutflow.efficiency_errors`): ``"auto"`` is ROOT's
    ``TEfficiency`` default, Clopper-Pearson when the event weights (before the
    sample's scale and luminosity factor) are unweighted as ROOT decides (their
    sum equals the sum of their squares: weights of 1, or 0 and 1) and the
    normal approximation otherwise. Systematics are not propagated.

    Examples
    --------
    >>> table = rf.cutflow(
    ...     [sig, bkg], ["nMuon >= 2", rf.Cut("MET > 50", label="MET"), "any(Jet_btag > 0.8)"]
    ... )  # doctest: +SKIP
    >>> print(table)  # doctest: +SKIP
    >>> table.get("Signal").efficiencies  # doctest: +SKIP
    >>> table.get("Signal").efficiency_errors  # (down, up)  # doctest: +SKIP
    """
    samples = as_samples(data, tree=tree, labels=label)
    return CutflowTable(
        tuple(
            cutflow_of(s, cuts, weight=weight, lumi=lumi, nonfinite=nonfinite, interval=interval)
            for s in samples
        )
    )
