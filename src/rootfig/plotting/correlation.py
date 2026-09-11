"""Drawing correlation matrices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from rootfig._typing import FloatArray

__all__ = ["draw_correlation"]


def draw_correlation(
    matrix: FloatArray,
    labels: Sequence[str],
    ax: Axes,
    *,
    cmap: str | Any = "RdBu_r",
    annotate: bool = True,
    fmt: str | None = None,
    colorbar: bool = True,
    percent: bool = False,
) -> Any:
    """Draw ``matrix`` (values in [-1, 1]) as an annotated heat map.

    Parameters
    ----------
    matrix
        Square correlation matrix.
    labels
        Tick labels, one per row/column.
    ax
        Target axes.
    cmap
        Diverging colour map.
    annotate
        Write the coefficient into each cell.
    fmt
        Format for the annotations.
    colorbar
        Add a colour bar.
    percent
        Show coefficients in percent instead of fractions.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = f"correlation matrix must be square, got shape {matrix.shape}"
        raise ValueError(msg)
    if len(labels) != matrix.shape[0]:
        msg = f"got {len(labels)} labels for a {matrix.shape[0]}x{matrix.shape[0]} matrix"
        raise ValueError(msg)
    shown = matrix * 100.0 if percent else matrix
    fmt = fmt if fmt is not None else (".0f" if percent else ".2f")
    limit = 100.0 if percent else 1.0
    image = ax.imshow(shown, cmap=cmap, vmin=-limit, vmax=limit, origin="upper", aspect="equal")
    n = matrix.shape[0]
    ax.set_xticks(range(n), labels=list(labels), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), labels=list(labels))
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
    ax.tick_params(which="major", top=False, right=False, direction="out", length=3)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    if annotate:
        for i in range(n):
            for j in range(n):
                value = shown[i, j]
                if not np.isfinite(value):
                    continue
                text_color = "white" if abs(matrix[i, j]) > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    _format(value, fmt) + ("%" if percent else ""),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=plt.rcParams["legend.fontsize"],
                )
    if colorbar:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation" + (" [%]" if percent else ""), loc="top")
    return image


def _format(value: float, fmt: str) -> str:
    """Format ``value`` without a negative zero (``-0`` reads as a bug)."""
    text = format(value, fmt)
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text
