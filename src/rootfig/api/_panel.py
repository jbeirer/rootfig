"""The lower panel of :func:`~rootfig.plot`: what it compares with what, and how.

:func:`resolve` decides the roles before the figure exists, so a bad ``panel=``
or ``reference=`` leaves nothing open; :meth:`PanelPlan.comparisons` computes
them once the stack is drawn, against the very total :attr:`Plot.stack
<rootfig.Plot.stack>` holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rootfig.histograms import (
    COMPARISON_KINDS,
    Comparison,
    ComparisonKind,
    Histogram,
    UncertaintyMode,
    compare,
    sum_histograms,
)
from rootfig.histograms.comparison import BAND_KINDS, SIGNIFICANCE_KINDS
from rootfig.plotting.panel import comparison_label

__all__ = ["PanelPlan", "resolve"]


@dataclass(frozen=True)
class PanelPlan:
    """The roles of a lower panel, decided before anything is drawn.

    Attributes
    ----------
    kind
        What the panel shows.
    numerators
        The histograms compared with the reference, in the order drawn.
    reference
        The reference (the background of a significance), or ``None`` for the
        stack total, which exists only once the stack is drawn.
    modes
        The :data:`~rootfig.histograms.UncertaintyMode` of each numerator.
    label
        The y label, given or built from the roles.
    """

    kind: ComparisonKind
    numerators: tuple[Histogram, ...]
    reference: Histogram | None
    modes: tuple[UncertaintyMode, ...]
    label: str

    @property
    def observed(self) -> list[bool]:
        """Which numerators are observed data."""
        return [h.is_data for h in self.numerators]

    def comparisons(self, total: Histogram | None) -> list[Comparison]:
        """Compare every numerator with the reference, the stack ``total`` if it is ``None``."""
        reference = self.reference if self.reference is not None else total
        assert reference is not None  # the stack total is the reference only with a stack
        return [
            compare(h, reference, kind=self.kind, uncertainty=mode)
            for h, mode in zip(self.numerators, self.modes, strict=True)
        ]


def resolve(
    histograms: Sequence[Histogram],
    *,
    panel: object,
    reference: str | None,
    uncertainty: UncertaintyMode | None,
    label: str | None,
    stacked: Sequence[Histogram],
    overlaid: Sequence[Histogram],
    data: Sequence[Histogram],
) -> PanelPlan | None:
    """Return the roles of the lower panel ``panel``, or ``None`` without one.

    ``stacked``, ``overlaid`` and ``data`` split ``histograms`` as drawn. Without
    ``reference``, a ratio-like kind compares data with the stack total, or the
    overlays with it when there is no data; without a stack, data with the first
    non-data histogram, and without data every further histogram with the first.
    With observed data alone (nothing to stack), every further data histogram is
    compared with the first.
    A significance takes the overlays as signals over the stack total, or else
    the last non-data histogram over the sum of the others. ``reference`` names
    the reference (the background) for every other histogram (non-data for a
    significance).

    Raises
    ------
    ValueError
        For an unknown ``panel``, ``reference`` without ``panel``, a
        ``reference`` naming no drawn histogram or several, data as the
        background of a significance, ``uncertainty`` for a kind without a band,
        and roles the drawn histograms cannot fill.
    """
    if panel is None:
        if reference is not None:
            msg = (
                f"reference={reference!r} names what the lower panel compares with; "
                "choose the panel too, e.g. panel='ratio'"
            )
            raise ValueError(msg)
        return None
    if not isinstance(panel, str) or panel not in COMPARISON_KINDS:
        msg = (
            f"panel={panel!r} is not one of {COMPARISON_KINDS}; for a data/MC ratio write "
            "panel='ratio', for a significance panel='s/sqrt(b)'"
        )
        raise ValueError(msg)
    kind: ComparisonKind = panel
    if uncertainty is not None:
        if kind not in BAND_KINDS:
            msg = (
                f"panel_uncertainty chooses how a reference band is drawn, which "
                f"panel={kind!r} does not have; it applies to {BAND_KINDS}"
            )
            raise ValueError(msg)
        if uncertainty not in ("propagate", "numerator"):
            msg = f"panel_uncertainty must be 'propagate' or 'numerator', got {uncertainty!r}"
            raise ValueError(msg)
    named = None if reference is None else _named(histograms, reference)
    if kind in SIGNIFICANCE_KINDS:
        numerators, background = _significance_roles(
            histograms, named, stacked=stacked, overlaid=overlaid
        )
        modes: list[UncertaintyMode] = ["propagate"] * len(numerators)
        return PanelPlan(
            kind,
            tuple(numerators),
            background,
            tuple(modes),
            label if label is not None else comparison_label(kind, ""),
        )
    numerators, chosen = _ratio_roles(
        histograms, named, kind=kind, stacked=stacked, overlaid=overlaid, data=data
    )
    reference_is_data = chosen is not None and chosen.is_data
    if kind == "pull":
        modes = ["propagate"] * len(numerators)
    elif uncertainty is not None:
        modes = [uncertainty] * len(numerators)
    else:
        # Data over simulation keeps the reference as a band; every other comparison
        # propagates both uncertainties, so shared systematic sources cancel.
        modes = [
            "numerator" if h.is_data and not reference_is_data else "propagate" for h in numerators
        ]
    if label is None:
        # a named reference can mix data and simulation in the panel: only a panel of
        # data alone is labelled as data
        over_simulation = all(h.is_data for h in numerators) and not reference_is_data
        label = comparison_label(
            kind, "MC" if chosen is None else chosen.label, data=over_simulation
        )
    return PanelPlan(kind, tuple(numerators), chosen, tuple(modes), label)


def _named(histograms: Sequence[Histogram], reference: str) -> Histogram:
    """Return the one drawn histogram labelled ``reference``."""
    labels = [h.label for h in histograms]
    matches = [h for h in histograms if h.label == reference]
    if not matches:
        msg = f"reference={reference!r} is not the label of a drawn histogram (labels: {labels})"
        raise ValueError(msg)
    if len(matches) > 1:
        msg = (
            f"reference={reference!r} is the label of {len(matches)} drawn histograms; "
            "a reference is one histogram, so give them distinct labels"
        )
        raise ValueError(msg)
    return matches[0]


def _significance_roles(
    histograms: Sequence[Histogram],
    background: Histogram | None,
    *,
    stacked: Sequence[Histogram],
    overlaid: Sequence[Histogram],
) -> tuple[list[Histogram], Histogram | None]:
    """Return the signals and their background (``None``: the stack total).

    A named background is compared with every other non-data histogram. Without
    one, histograms overlaid on a stack are the signals and the stack is their
    background; otherwise (no stack, or everything stacked) the last non-data
    histogram is the signal and the others are summed into the background.
    """
    simulated = [h for h in histograms if not h.is_data]
    if background is not None:
        if background.is_data:
            msg = (
                f"reference={background.label!r} is observed data, which cannot be the "
                "background of a significance; name a simulated histogram"
            )
            raise ValueError(msg)
        signals = [h for h in simulated if h is not background]
        if not signals:
            msg = "a significance panel needs a non-data histogram besides the background"
            raise ValueError(msg)
        return signals, background
    if stacked and overlaid:
        return list(overlaid), None
    if len(simulated) < 2:
        msg = "a significance panel needs at least two non-data histograms (signal and background)"
        raise ValueError(msg)
    # summed like a stack total: bin by bin, whatever the axis names and labels, checked
    return [simulated[-1]], sum_histograms(simulated[:-1], label="Background")


def _ratio_roles(
    histograms: Sequence[Histogram],
    reference: Histogram | None,
    *,
    kind: ComparisonKind,
    stacked: Sequence[Histogram],
    overlaid: Sequence[Histogram],
    data: Sequence[Histogram],
) -> tuple[list[Histogram], Histogram | None]:
    """Return the numerators and the reference (``None``: the stack total) of a ratio-like kind."""
    if reference is not None:
        numerators = [h for h in histograms if h is not reference]
    elif stacked:
        numerators = list(data or overlaid)
        if not numerators:
            msg = (
                f"panel={kind!r} with every non-data histogram stacked needs observed data "
                "(observed=...) or a histogram outside the stack (stack=[...])"
            )
            raise ValueError(msg)
        return numerators, None
    elif data and overlaid:
        numerators, reference = list(data), overlaid[0]
    else:
        # all simulation, or all observed data (two run periods): later ones / the first
        numerators, reference = list(histograms[1:]), histograms[0]
    if not numerators:
        msg = f"a {kind} panel needs at least two histograms"
        raise ValueError(msg)
    return numerators, reference
