"""Drawing the lower panel: the comparisons of histograms with a reference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from rootfig.histograms.comparison import SIGNIFICANCE_KINDS, Comparison, ComparisonKind
from rootfig.plotting.hist1d import band_label, in_view
from rootfig.plotting.style import color_cycle, foreground

__all__ = ["comparison_label", "draw_panel", "panel_ylim"]

DEFAULT_RATIO_YLIM = (0.5, 1.5)
"""The smallest automatic range of a ratio panel; a relative difference's is this minus one."""

PULL_YLIM = (3.0, 5.0)
"""The smallest and largest automatic half-range of a pull panel."""

_BASELINES: dict[str, float] = {"ratio": 1.0, "relative_difference": 0.0, "difference": 0.0}

_LABELS: dict[str, str] = {
    "ratio": "Ratio to {}",
    "relative_difference": "Rel. difference to {}",
    "difference": "Difference to {}",
    "pull": "Pull",
    "s/sqrt(b)": r"$S/\sqrt{{B}}$",
    "s/sqrt(s+b)": r"$S/\sqrt{{S+B}}$",
}


def comparison_label(kind: ComparisonKind, reference: str, *, data: bool = False) -> str:
    """Return the default y label of a ``kind`` panel comparing with ``reference``.

    ``data`` words it for observed data over a simulated reference
    (``"Data / MC"``); a pull and a significance are labelled by their kind alone.
    """
    minus = "\N{MINUS SIGN}"
    if data:
        match kind:
            case "ratio":
                return f"Data / {reference}"
            case "relative_difference":
                return f"(Data {minus} {reference}) / {reference}"
            case "difference":
                return f"Data {minus} {reference}"
    return _LABELS[kind].format(reference)


def draw_panel(
    comparisons: Sequence[Comparison],
    ax: Axes,
    *,
    colors: Sequence[str] | None = None,
    observed: Sequence[bool] | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: str | None = None,
    band: bool | None = None,
    view: Sequence[tuple[float, float]] | None = None,
) -> None:
    """Draw ``comparisons`` of one kind into the lower panel ``ax``.

    Ratios, relative differences and differences are points with error bars
    around their baseline (1 or 0, a dashed line), over the reference's
    uncertainty band; pulls are filled bars from 0, without error bars; a
    significance is points with error bars and no baseline.

    Parameters
    ----------
    comparisons
        What :func:`~rootfig.histograms.compare` returned, one per numerator,
        all of one kind.
    ax
        The panel's axes.
    colors
        One colour per comparison; defaults to the foreground colour for
        observed data and the style's cycle for the rest.
    observed
        Which comparisons have observed data as numerator; their markers are
        drawn larger. Defaults to none.
    ylim
        Vertical range; defaults to :func:`panel_ylim`.
    ylabel
        Label; defaults to :func:`comparison_label` for the first comparison's reference.
    band
        Draw the reference's uncertainty band of the first comparison, which
        belongs in the panel when the error bars leave it out (the ``"numerator"``
        mode). Defaults to drawing it whenever the comparisons carry one.
    view
        The x windows whose bins set the automatic y range.

    Raises
    ------
    ValueError
        Without comparisons, or with comparisons of different kinds.
    """
    if not comparisons:
        msg = "draw_panel needs at least one comparison"
        raise ValueError(msg)
    kind = comparisons[0].kind
    if any(c.kind != kind for c in comparisons):
        msg = f"draw_panel draws one kind; got {sorted({c.kind for c in comparisons})}"
        raise ValueError(msg)
    flags = [False] * len(comparisons) if observed is None else list(observed)
    if colors is None:
        cycle = iter(color_cycle(len(comparisons)))
        colors = [foreground() if is_data else next(cycle) for is_data in flags]

    band_edges: tuple[np.ndarray, np.ndarray] | None = None
    if kind in SIGNIFICANCE_KINDS:
        _draw_points(comparisons, ax, colors, [False] * len(comparisons), clip_errors=True)
    elif kind == "pull":
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
        for comparison, color in zip(comparisons, colors, strict=True):
            values = np.where(np.isfinite(comparison.values), comparison.values, 0.0)
            ax.stairs(values, comparison.edges, baseline=0.0, fill=True, color=color, alpha=0.6)
    else:
        baseline = _BASELINES[kind]
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.0, zorder=1)
        first = next((c for c in comparisons if c.band is not None), None)
        total = first.total_band() if first is not None else None
        if (band is None or band) and first is not None and total is not None:
            band_down, band_up = total
            has_systematics = first.syst_band is not None
            lower = np.where(np.isfinite(band_down), baseline - band_down, baseline)
            upper = np.where(np.isfinite(band_up), baseline + band_up, baseline)
            band_edges = (lower, upper)
            ax.fill_between(
                first.edges,
                np.append(lower, lower[-1]),
                np.append(upper, upper[-1]),
                step="post",
                facecolor="gray",
                alpha=0.3,
                linewidth=0,
                zorder=0,
                label=f"{first.reference} {band_label(systematics=has_systematics).lower()}",
            )
        _draw_points(comparisons, ax, colors, flags, clip_errors=False)

    ax.set_ylim(
        *(ylim if ylim is not None else panel_ylim(comparisons, band=band_edges, view=view))
    )
    edges = comparisons[0].edges
    ax.set_xlim(edges[0], edges[-1])
    if ylabel is None:
        ylabel = comparison_label(kind, comparisons[0].reference)
    ax.set_ylabel(ylabel, loc="center")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="upper"))


def _draw_points(
    comparisons: Sequence[Comparison],
    ax: Axes,
    colors: Sequence[str],
    observed: Sequence[bool],
    *,
    clip_errors: bool,
) -> None:
    """Draw every comparison as markers with error bars, skipping undefined bins.

    ``clip_errors`` draws undefined uncertainties of defined values as zero (a
    significance); otherwise the statistical and systematic errors are combined.
    """
    for comparison, color, is_data in zip(comparisons, colors, observed, strict=True):
        ok = np.isfinite(comparison.values)
        yerr: Any
        if clip_errors:
            yerr = np.where(np.isfinite(comparison.errors), comparison.errors, 0.0)[ok]
        elif comparison.syst_errors is None:
            yerr = comparison.errors[ok]
        else:
            errors_down, errors_up = comparison.total_errors()
            yerr = [errors_down[ok], errors_up[ok]]
        ax.errorbar(
            comparison.centers[ok],
            comparison.values[ok],
            yerr=yerr,
            xerr=comparison.half_widths[ok],
            fmt="o",
            markersize=5 if is_data else 4,
            color=color,
            elinewidth=1.0,
            capsize=0,
        )


def panel_ylim(
    comparisons: Sequence[Comparison],
    *,
    band: tuple[np.ndarray, np.ndarray] | None = None,
    view: Sequence[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Choose the automatic y range of a panel from the bins in the x windows ``view``.

    ``band`` is the lower and upper edge of the drawn reference band, if any.
    Statistical error bars never count: a few low-statistics bins with huge
    uncertainties would otherwise squash the panel. Per kind:

    * ratio: at least (0.5, 1.5), widened to the 5th to 95th percentile, padded
      by 10 percent, of the values, their systematic extent and the band edges,
      each judged on its own so it can widen the range but never narrow it;
      clipped to ``[0, 3]``, or to ``[-3, 3]`` with negative ratios (signed
      weights), whose lowest value then stays in view;
    * relative difference: the ratio's range of the values plus one, shifted back;
    * difference: symmetric, 1.1 times the largest magnitude of those
      percentiles, or ``(-1, 1)`` when everything is zero;
    * pull: symmetric, 1.1 times the 95th percentile of the magnitudes, at least
      3 and at most 5;
    * significance: from zero to 1.25 times the highest value plus its error.
    """
    kind = comparisons[0].kind if comparisons else "ratio"
    visible = [in_view(c.edges, view) for c in comparisons]
    if kind in SIGNIFICANCE_KINDS:
        tops = []
        for comparison, shown in zip(comparisons, visible, strict=True):
            errors = np.where(np.isfinite(comparison.errors), comparison.errors, 0.0)
            ok = np.isfinite(comparison.values) & shown
            if ok.any():
                tops.append(float(np.max(comparison.values[ok] + errors[ok])))
        top = max(tops, default=1.0)
        return (0.0, 1.25 * top if top > 0 else 1.0)
    if kind == "pull":
        magnitudes = _finite(
            [np.abs(c.values[shown]) for c, shown in zip(comparisons, visible, strict=True)]
        )
        smallest, largest = PULL_YLIM
        if not magnitudes.size:
            return (-smallest, smallest)
        half = min(max(smallest, 1.1 * float(np.percentile(magnitudes, 95))), largest)
        return (-half, half)
    shift = 1.0 if kind == "relative_difference" else 0.0
    ranges = _extents(comparisons, visible, band, view, shift=shift)
    if kind == "difference":
        bound = 0.0
        for lower, upper in ranges:
            if lower.size and upper.size:
                q_low, q_high = np.percentile(lower, 5), np.percentile(upper, 95)
                bound = max(bound, abs(float(q_low)), abs(float(q_high)))
        half = 1.1 * bound if bound > 0 else 1.0
        return (-half, half)
    low, high = _ratio_range(ranges)
    return (low - shift, high - shift)


def _extents(
    comparisons: Sequence[Comparison],
    visible: Sequence[np.ndarray],
    band: tuple[np.ndarray, np.ndarray] | None,
    view: Sequence[tuple[float, float]] | None,
    *,
    shift: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the ``(lower, upper)`` finite values of what an automatic range must cover.

    The values, their systematic extent and the band edges, each a pair, in the
    bins shown and moved by ``shift``.
    """
    values = _finite(
        [c.values[shown] + shift for c, shown in zip(comparisons, visible, strict=True)]
    )
    ranges = [(values, values)]
    with_syst = [
        (c, shown, c.syst_errors)
        for c, shown in zip(comparisons, visible, strict=True)
        if c.syst_errors is not None
    ]
    if with_syst:
        ranges.append(
            (
                _finite([(c.values - syst[0])[shown] + shift for c, shown, syst in with_syst]),
                _finite([(c.values + syst[1])[shown] + shift for c, shown, syst in with_syst]),
            )
        )
    if band is not None:
        shown = in_view(comparisons[0].edges, view) if comparisons else np.ones_like(band[0], bool)
        ranges.append((_finite([band[0][shown] + shift]), _finite([band[1][shown] + shift])))
    return ranges


def _ratio_range(ranges: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Return the range of a ratio panel covering ``ranges``; the first holds the values."""
    low, high = DEFAULT_RATIO_YLIM
    for lower, upper in ranges:
        if lower.size and upper.size:
            q_low = float(np.percentile(lower, 5))
            q_high = float(np.percentile(upper, 95))
            pad = 0.1 * max(q_high - q_low, 0.2)
            low = min(low, q_low - pad)
            high = max(high, q_high + pad)
    values = ranges[0][0]
    floor = 0.0
    if values.size and values.min() < 0:
        floor = -3.0
        low = min(low, float(values.min()) - 0.1 * max(high - float(values.min()), 0.2))
    return (max(low, floor), min(high, 3.0))


def _finite(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0)
    combined = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    return combined[np.isfinite(combined)]
