"""Drawing two-dimensional histograms."""

from __future__ import annotations

import warnings
from typing import Any

import mplhep as hep
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize

from rootfig.errors import RootfigWarning
from rootfig.histograms.build import Histogram

__all__ = ["draw_hist2d"]


def draw_hist2d(
    histogram: Histogram,
    ax: Axes,
    *,
    logz: bool = False,
    cmap: str | Any = "viridis",
    colorbar: bool = True,
    zlabel: str | None = None,
    flow: str = "none",
    **kwargs: Any,
) -> Any:
    """Draw a 2D histogram as a colour mesh with mplhep.

    Empty bins (no content and no variance, i.e. nothing was ever filled there)
    are left blank so the colour scale is driven by the populated region; bins
    with negative content stay visible on a linear scale. With ``logz`` only
    positive bins can be shown. Returns the mplhep artists.
    """
    if histogram.ndim != 2:
        msg = f"draw_hist2d needs a two-dimensional histogram, got {histogram.ndim}D"
        raise ValueError(msg)
    values = histogram.values()
    positive = values[values > 0]
    norm: Normalize | None = None
    if logz:
        if positive.size == 0:
            warnings.warn(
                "logz=True but the histogram has no positive bins; drawing a linear colour "
                "scale instead",
                RootfigWarning,
                stacklevel=2,
            )
            logz = False
        else:
            norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    if norm is not None:
        # LogNorm masks non-positive values itself.
        kwargs["norm"] = norm
    else:
        populated = (values != 0) | (histogram.variances() > 0)
        if populated.any() and not populated.all():
            kwargs["mask"] = populated
    # mplhep's own colour bar (an axes divider) widens the figure and is ignored by
    # constrained layout; matplotlib's takes its space from the axes instead.
    artists = hep.hist2dplot(histogram.hist, ax=ax, cmap=cmap, cbar=False, flow=flow, **kwargs)
    if colorbar:
        cbar = ax.figure.colorbar(artists.pcolormesh, ax=ax, pad=0.02, fraction=0.05)
        if zlabel is not None:
            cbar.set_label(zlabel, loc="top")
        artists = artists._replace(cbar=cbar)
    ax.set_xlabel(histogram.hist.axes[0].label or "", loc="right")
    ax.set_ylabel(histogram.hist.axes[1].label or "", loc="top")
    return artists
