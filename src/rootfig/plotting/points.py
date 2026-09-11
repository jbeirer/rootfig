"""Drawing efficiencies and profiles as points with error bars."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes

from rootfig.histograms.efficiency import Efficiency, Profile
from rootfig.model.style import Style
from rootfig.plotting.style import color_cycle

__all__ = ["draw_efficiencies", "draw_profiles"]


def draw_efficiencies(
    efficiencies: Sequence[Efficiency],
    ax: Axes,
    *,
    style: Style,
    colors: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Draw each efficiency as points with asymmetric error bars; return the y data range."""
    if colors is None:
        colors = color_cycle(max(len(efficiencies), 1), style)
    low, high = 1.0, 0.0
    for eff, color in zip(efficiencies, colors, strict=True):
        ok = np.isfinite(eff.values)
        lower_err, upper_err = eff.errors
        ax.errorbar(
            eff.centers[ok],
            eff.values[ok],
            yerr=[lower_err[ok], upper_err[ok]],
            xerr=eff.half_widths[ok],
            fmt="o",
            markersize=4,
            capsize=0,
            color=color,
            label=eff.label,
        )
        if ok.any():
            low = min(low, float(np.nanmin(eff.lower[ok])))
            high = max(high, float(np.nanmax(eff.upper[ok])))
    return (min(low, 0.0) if low < 0 else 0.0, max(high, 1.0))


def draw_profiles(
    profiles: Sequence[Profile],
    ax: Axes,
    *,
    style: Style,
    colors: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Draw each profile as points with error bars; return the y data range."""
    if colors is None:
        colors = color_cycle(max(len(profiles), 1), style)
    low, high = np.inf, -np.inf
    for prof, color in zip(profiles, colors, strict=True):
        ok = np.isfinite(prof.values)
        errors = np.where(np.isfinite(prof.errors), prof.errors, 0.0)
        ax.errorbar(
            prof.centers[ok],
            prof.values[ok],
            yerr=errors[ok],
            xerr=prof.half_widths[ok],
            fmt="o",
            markersize=4,
            capsize=0,
            color=color,
            label=prof.label,
        )
        if ok.any():
            low = min(low, float(np.min(prof.values[ok] - errors[ok])))
            high = max(high, float(np.max(prof.values[ok] + errors[ok])))
    if not np.isfinite(low):
        return (0.0, 1.0)
    return (low, high)
