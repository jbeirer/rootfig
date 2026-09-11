"""Drawing two-dimensional histograms."""

from __future__ import annotations

from typing import Any

import mplhep as hep
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize

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
    # mplhep would otherwise widen a single-axes figure to fit the colour bar, so a 2D
    # plot would not have the same canvas as a 1D one; take the space from the axes.
    kwargs.setdefault("cbarextend", False)
    artists = hep.hist2dplot(histogram.hist, ax=ax, cmap=cmap, cbar=colorbar, flow=flow, **kwargs)
    if colorbar and zlabel is not None:
        cbar = getattr(artists, "cbar", None)
        if cbar is not None:
            cbar.set_label(zlabel, loc="top")
    ax.set_xlabel(histogram.hist.axes[0].label or "", loc="right")
    ax.set_ylabel(histogram.hist.axes[1].label or "", loc="top")
    return artists
