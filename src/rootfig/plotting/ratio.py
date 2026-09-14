"""Drawing the ratio panel."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from rootfig.histograms.build import Histogram
from rootfig.histograms.ratio import Ratio, RatioUncertainty, SignificanceKind, ratio
from rootfig.model.style import Style
from rootfig.plotting.hist1d import band_label
from rootfig.plotting.style import color_cycle, foreground

__all__ = ["draw_ratio_panel", "draw_significance_panel", "ratio_ylim"]

DEFAULT_RATIO_YLIM = (0.5, 1.5)


def draw_ratio_panel(
    numerators: Sequence[Histogram],
    reference: Histogram,
    ax: Axes,
    *,
    style: Style,
    uncertainty: RatioUncertainty | Sequence[RatioUncertainty],
    colors: Sequence[str] | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: str | None = None,
    band: bool | None = None,
) -> list[Ratio]:
    """Draw ``numerator / reference`` for every numerator and return the ratios.

    Parameters
    ----------
    numerators
        Histograms to divide by ``reference``.
    reference
        The denominator.
    ax
        The ratio axes.
    style
        Style providing colours.
    uncertainty
        ``"propagate"`` (error bars carry both uncertainties) or ``"numerator"``
        (error bars carry the numerator's; the reference uncertainty is a band),
        for all numerators or one per numerator. Systematic variations of the
        histograms are included in error bars and band alike (see
        :func:`~rootfig.histograms.ratio`).
    colors
        One colour per numerator; defaults to the numerator's own colour or the
        style cycle (``text.color`` for data).
    ylim
        Vertical range; defaults to :data:`DEFAULT_RATIO_YLIM` expanded to cover
        the points.
    ylabel
        Label; defaults to ``"Ratio to <reference>"`` or ``"Data / MC"``.
    band
        Draw the reference uncertainty band. Defaults to ``True`` when any
        numerator uses ``uncertainty="numerator"``.
    """
    modes = [uncertainty] * len(numerators) if isinstance(uncertainty, str) else list(uncertainty)
    if len(modes) != len(numerators):
        msg = f"got {len(modes)} uncertainty modes for {len(numerators)} numerators"
        raise ValueError(msg)
    ratios = [
        ratio(h, reference, uncertainty=mode) for h, mode in zip(numerators, modes, strict=True)
    ]
    if colors is None:
        cycle = iter(color_cycle(max(len(numerators), 1), style))
        colors = [
            (foreground() if h.is_data and not h.color else (h.color or next(cycle)))
            for h in numerators
        ]

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    show_band = ("numerator" in modes) if band is None else band
    band_edges: tuple[np.ndarray, np.ndarray] | None = None
    if show_band and ratios:
        first = ratios[0]
        band_down, band_up = first.total_band()
        has_systematics = first.syst_band is not None
        lower = np.where(np.isfinite(band_down), 1.0 - band_down, 1.0)
        upper = np.where(np.isfinite(band_up), 1.0 + band_up, 1.0)
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
            label=f"{reference.label} {band_label(systematics=has_systematics).lower()}",
        )

    for r, color, numerator in zip(ratios, colors, numerators, strict=True):
        ok = np.isfinite(r.values)
        marker: dict[str, Any] = {"fmt": "o", "markersize": 4 if not numerator.is_data else 5}
        errors_down, errors_up = r.total_errors()
        ax.errorbar(
            r.centers[ok],
            r.values[ok],
            yerr=r.errors[ok] if r.syst_errors is None else [errors_down[ok], errors_up[ok]],
            xerr=r.half_widths[ok],
            color=color,
            elinewidth=1.0,
            capsize=0,
            **marker,
        )

    ax.set_ylim(*(ylim if ylim is not None else ratio_ylim(ratios, band=band_edges)))
    ax.set_xlim(reference.edges[0], reference.edges[-1])
    if ylabel is None:
        if any(h.is_data for h in numerators) and not reference.is_data:
            ylabel = f"Data / {'MC' if reference.label in ('Total', 'MC') else reference.label}"
        else:
            ylabel = f"Ratio to {reference.label}"
    ax.set_ylabel(ylabel, loc="center")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="upper"))
    return ratios


def ratio_ylim(
    ratios: Sequence[Ratio], *, band: tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[float, float]:
    """Choose a ratio range: at least (0.5, 1.5), widened to cover the bulk of what is drawn.

    The bulk is the 5th to 95th percentile of the finite ratio values, padded
    by 10 percent, and likewise of the lower and upper edges of the reference
    ``band`` when one is drawn, so a large systematic band is not clipped. Error
    bars do not count: a few low-statistics bins with huge uncertainties would
    otherwise squash the panel. The result is clipped to ``[0, 3]``.
    """
    low, high = DEFAULT_RATIO_YLIM
    ranges = [(_finite([r.values for r in ratios]),) * 2]
    if band is not None:  # judged on its own, so the band can widen the range but never narrow it
        ranges.append((_finite([band[0]]), _finite([band[1]])))
    for lower, upper in ranges:
        if lower.size and upper.size:
            q_low = float(np.percentile(lower, 5))
            q_high = float(np.percentile(upper, 95))
            pad = 0.1 * max(q_high - q_low, 0.2)
            low = min(low, q_low - pad)
            high = max(high, q_high + pad)
    return (max(low, 0.0), min(high, 3.0))


def _finite(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0)
    combined = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    return combined[np.isfinite(combined)]


def draw_significance_panel(
    result: Ratio,
    ax: Axes,
    *,
    kind: SignificanceKind = "s/sqrt(b)",
    color: str | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: str | None = None,
) -> None:
    """Draw a per-bin significance (from :func:`~rootfig.histograms.significance`) as points."""
    ok = np.isfinite(result.values)
    errors = np.where(np.isfinite(result.errors), result.errors, 0.0)
    ax.errorbar(
        result.centers[ok],
        result.values[ok],
        yerr=errors[ok],
        xerr=result.half_widths[ok],
        fmt="o",
        markersize=4,
        capsize=0,
        elinewidth=1.0,
        color=color or foreground(),
    )
    if ylim is None:
        top = float(np.max(result.values[ok] + errors[ok])) if ok.any() else 1.0
        ylim = (0.0, 1.25 * top if top > 0 else 1.0)
    ax.set_ylim(*ylim)
    ax.set_xlim(result.edges[0], result.edges[-1])
    if ylabel is None:
        ylabel = r"$S/\sqrt{B}$" if kind == "s/sqrt(b)" else r"$S/\sqrt{S+B}$"
    ax.set_ylabel(ylabel, loc="center")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="upper"))
