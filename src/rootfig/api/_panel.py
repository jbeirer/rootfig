"""The lower panel of :func:`~rootfig.plot`: what it compares with what, and how.

:func:`resolve` decides the roles before the figure exists, so a bad ``panel=``
or ``reference=`` leaves nothing open; :meth:`PanelPlan.comparisons` computes
them once the stack is drawn, against the very total :attr:`Plot.stack
<rootfig.Plot.stack>` holds. :func:`resolve_points` does the same for the
efficiencies and profiles of :func:`~rootfig.efficiency` and :func:`~rootfig.profile`.
:func:`fit_results` tests the pairs a ratio panel would compare (``goodness_of_fit=``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from rootfig.errors import BinningError
from rootfig.histograms import (
    COMPARISON_KINDS,
    GOODNESS_OF_FIT_TESTS,
    Comparison,
    ComparisonKind,
    Efficiency,
    GoodnessOfFit,
    GoodnessOfFitTest,
    Histogram,
    Profile,
    UncertaintyMode,
    compare,
    compatible_binning,
    goodness_of_fit,
    sum_histograms,
)
from rootfig.histograms.comparison import BAND_KINDS, SIGNIFICANCE_KINDS
from rootfig.plotting import FlowSpec, StackSpec, fold_flow_bins, show_flow_bins, split_stack
from rootfig.plotting.panel import comparison_label

__all__ = ["PanelPlan", "fit_results", "resolve", "resolve_points"]

Compared: TypeAlias = Histogram | Efficiency | Profile
"""What a lower panel compares: histograms, or the points of an efficiency or a profile."""


class _Labelled(Protocol):
    @property
    def label(self) -> str: ...


@dataclass(frozen=True)
class PanelPlan:
    """The roles of a lower panel, decided before anything is drawn.

    Attributes
    ----------
    kind
        What the panel shows.
    numerators
        What is compared with the reference, in the order drawn.
    reference
        The reference (the background of a significance), or ``None`` for the
        stack total, which exists only once the stack is drawn.
    modes
        The :data:`~rootfig.histograms.UncertaintyMode` of each numerator.
    label
        The y label, given or built from the roles.
    observed
        Which numerators are observed data.
    """

    kind: ComparisonKind
    numerators: tuple[Compared, ...]
    reference: Compared | None
    modes: tuple[UncertaintyMode, ...]
    label: str
    observed: tuple[bool, ...]

    def comparisons(self, total: Histogram | None = None) -> list[Comparison]:
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
    tested: bool = False,
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
    significance). It needs no ``panel`` when ``tested``: it then names the
    reference of the goodness of fit alone.

    Raises
    ------
    ValueError
        For an unknown ``panel``, ``reference`` without ``panel`` (unless
        ``tested``), a
        ``reference`` naming no drawn histogram or several, data as the
        background of a significance, ``uncertainty`` for a kind without a band,
        and roles the drawn histograms cannot fill.
    BinningError
        If a numerator does not share the reference's binning.
    """
    if panel is None:
        if not tested:
            _no_panel(
                reference,
                "the lower panel or the goodness of fit",
                "choose the panel too, e.g. panel='ratio', or goodness_of_fit=True",
            )
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
        if uncertainty not in ("propagate", "numerator", "poisson-ratio"):
            msg = (
                "panel_uncertainty must be 'propagate', 'numerator' or 'poisson-ratio', got "
                f"{uncertainty!r}"
            )
            raise ValueError(msg)
        if uncertainty == "poisson-ratio" and kind not in ("ratio", "relative_difference"):
            msg = f"panel_uncertainty='poisson-ratio' is the interval of a ratio, not of {kind!r}"
            raise ValueError(msg)
    named = None if reference is None else _named(histograms, reference)
    if kind in SIGNIFICANCE_KINDS:
        numerators, background = _significance_roles(
            histograms, named, stacked=stacked, overlaid=overlaid
        )
        _check_binning(numerators, background, what=f"panel={kind!r}", stacked=stacked)
        modes: list[UncertaintyMode] = ["propagate"] * len(numerators)
        return PanelPlan(
            kind,
            tuple(numerators),
            background,
            tuple(modes),
            label if label is not None else comparison_label(kind, ""),
            tuple(h.is_data for h in numerators),
        )
    numerators, chosen = _ratio_roles(
        histograms, named, what=f"panel={kind!r}", stacked=stacked, overlaid=overlaid, data=data
    )
    _check_binning(numerators, chosen, what=f"panel={kind!r}", stacked=stacked)
    reference_is_data = chosen is not None and chosen.is_data
    if kind not in BAND_KINDS:
        modes = ["propagate"] * len(numerators)
    elif uncertainty is not None:
        modes = [uncertainty] * len(numerators)
        if uncertainty == "poisson-ratio":  # both sides must be counts: checked before drawing
            counted = [*numerators, chosen if chosen is not None else sum_histograms(stacked)]
            for histogram_ in counted:
                histogram_.counts()
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
    observed = tuple(h.is_data for h in numerators)
    return PanelPlan(kind, tuple(numerators), chosen, tuple(modes), label, observed)


def resolve_points(
    points: Sequence[Efficiency | Profile],
    is_data: Sequence[bool],
    *,
    panel: object,
    reference: str | None,
    label: str | None,
) -> PanelPlan | None:
    """Return the roles of the lower panel of an efficiency or profile plot, or ``None``.

    ``is_data`` says which of ``points`` belong to observed data. The roles are
    those of :func:`~rootfig.plot` without a stack: data over the first
    simulated sample when there are both, every further sample over the first
    otherwise, or every other one over the one ``reference`` names. Both sides
    are propagated; significances, which count events, are refused.

    Raises
    ------
    ValueError
        For an unknown ``panel`` or a significance, ``reference`` without
        ``panel``, a ``reference`` naming no sample or several, or a single sample.
    """
    if panel is None:
        _no_panel(reference, "the lower panel", "choose the panel too, e.g. panel='ratio'")
        return None
    if not isinstance(panel, str) or panel not in COMPARISON_KINDS or panel in SIGNIFICANCE_KINDS:
        kinds = tuple(k for k in COMPARISON_KINDS if k not in SIGNIFICANCE_KINDS)
        msg = (
            f"panel={panel!r} is not one of {kinds}; significances count events and apply "
            "to rf.plot only. For a scale factor write panel='ratio'"
        )
        raise ValueError(msg)
    kind: ComparisonKind = panel
    items: list[Efficiency | Profile] = list(points)
    flags = dict(zip(map(id, items), is_data, strict=True))
    named = None if reference is None else _named(items, reference)
    numerators, chosen = _ratio_roles(
        items,
        named,
        what=f"panel={kind!r}",
        stacked=[],
        overlaid=[p for p in items if not flags[id(p)]],
        data=[p for p in items if flags[id(p)]],
    )
    assert chosen is not None  # without a stack there is always a named or first reference
    observed = tuple(flags[id(p)] for p in numerators)
    if label is None:
        label = comparison_label(kind, chosen.label, data=all(observed) and not flags[id(chosen)])
    modes: list[UncertaintyMode] = ["propagate"] * len(numerators)
    return PanelPlan(kind, tuple(numerators), chosen, tuple(modes), label, observed)


def fit_results(
    histograms: Sequence[Histogram],
    spec: bool | GoodnessOfFitTest,
    *,
    flow: FlowSpec,
    stack: StackSpec,
    reference: str | None,
) -> list[GoodnessOfFit]:
    """Test what a ratio panel would compare with its reference, whatever ``panel`` is.

    ``spec`` is ``plot(goodness_of_fit=)``: ``False`` for nothing, ``True`` for
    ``"chi2"``. ``histograms`` are taken before normalisation, which changes
    only the drawing, with the flow bins shown or folded as ``flow`` draws them.

    Raises
    ------
    ValueError
        For an unknown test and roles the histograms cannot fill (see :func:`resolve`).
    BinningError
        If a tested histogram does not share its reference's binning.
    """
    if spec is False:
        return []
    test = "chi2" if spec is True else spec
    if test not in GOODNESS_OF_FIT_TESTS:
        msg = (
            f"goodness_of_fit={spec!r} is not True, False or one of {GOODNESS_OF_FIT_TESTS}; "
            "for ROOT's chi-square test of the shapes write goodness_of_fit=True"
        )
        raise ValueError(msg)
    if flow == "show":
        histograms = show_flow_bins(histograms)[0]
    elif flow == "sum":
        histograms = fold_flow_bins(histograms)
    stacked, overlaid, data = split_stack(histograms, stack)
    named = None if reference is None else _named(histograms, reference)
    what = f"goodness_of_fit={spec!r}"
    numerators, chosen = _ratio_roles(
        histograms, named, what=what, stacked=stacked, overlaid=overlaid, data=data
    )
    _check_binning(numerators, chosen, what=what, stacked=stacked)
    against = chosen if chosen is not None else sum_histograms(stacked)
    return [goodness_of_fit(h, against, test=test) for h in numerators]


def _no_panel(reference: str | None, compares: str, remedy: str) -> None:
    """Refuse ``reference`` without anything it could name the reference of."""
    if reference is not None:
        msg = f"reference={reference!r} names what {compares} compares with; {remedy}"
        raise ValueError(msg)


def _check_binning(
    numerators: Sequence[Histogram],
    reference: Histogram | None,
    *,
    what: str,
    stacked: Sequence[Histogram],
) -> None:
    """Refuse a numerator that does not bin like the reference, before a figure is made.

    The stack total (``reference`` is ``None``) bins like the histograms it
    sums, so the first of those stands for it.
    """
    against = reference if reference is not None else stacked[0]
    for numerator in numerators:
        if not compatible_binning(numerator.hist, against.hist):
            msg = (
                f"{what} compares {numerator.label!r} with {against.label!r}, "
                "which do not share one binning; give them the same bins= and range= "
                "(histograms that already exist must be rebinned to match)"
            )
            raise BinningError(msg)


def _named[T: _Labelled](histograms: Sequence[T], reference: str) -> T:
    """Return the one drawn histogram (or efficiency, profile) labelled ``reference``."""
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


def _ratio_roles[T: _Labelled](
    histograms: Sequence[T],
    reference: T | None,
    *,
    what: str,
    stacked: Sequence[T],
    overlaid: Sequence[T],
    data: Sequence[T],
) -> tuple[list[T], T | None]:
    """Return the numerators and the reference (``None``: the stack total) of a ratio-like kind.

    ``what`` names the option asking, for the errors.
    """
    if reference is not None:
        numerators = [h for h in histograms if h is not reference]
    elif stacked:
        numerators = list(data or overlaid)
        if not numerators:
            msg = (
                f"{what} with every non-data histogram stacked needs observed data "
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
        msg = f"{what} needs at least two histograms"
        raise ValueError(msg)
    return numerators, reference
