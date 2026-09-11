"""The :class:`Plot` object returned by every rootfig plotting function."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from rootfig._typing import FloatArray
    from rootfig.histograms import Efficiency, Histogram, Profile, Ratio
    from rootfig.model import Variable

__all__ = ["Plot"]


@dataclass
class Plot:
    """A finished figure plus the objects it was built from.

    Everything is a standard matplotlib or ``hist`` object, so further
    customisation is ordinary matplotlib code::

        p = rf.plot(...)
        p.ax.set_ylim(top=500)
        p.ax.axvline(91.2, color="gray", ls="--")
        p.save("z_mass.pdf")

    Attributes
    ----------
    fig
        The :class:`matplotlib.figure.Figure`.
    ax
        The main :class:`matplotlib.axes.Axes`.
    ratio_ax
        The ratio panel axes, or ``None``.
    ax_right, ratio_ax_right
        The right-hand segments when the x axis is broken (``xbreak``), else ``None``.
    histograms
        The :class:`~rootfig.histograms.Histogram` objects drawn (each wraps a
        ``hist.Hist``).
    ratios
        The :class:`~rootfig.histograms.Ratio` objects drawn in the ratio panel.
    variable
        The :class:`~rootfig.model.Variable` (x axis) if known; used for default
        file names.
    matrix
        For correlation plots, the correlation matrix.
    efficiencies, profiles
        For :func:`rootfig.efficiency` and :func:`rootfig.profile`, the computed
        :class:`~rootfig.histograms.Efficiency` / :class:`~rootfig.histograms.Profile` objects.
    """

    fig: Figure
    ax: Axes
    ratio_ax: Axes | None = None
    ax_right: Axes | None = None
    ratio_ax_right: Axes | None = None
    histograms: list[Histogram] = field(default_factory=list)
    ratios: list[Ratio] = field(default_factory=list)
    variable: Variable | None = None
    matrix: FloatArray | None = None
    efficiencies: list[Efficiency] = field(default_factory=list)
    profiles: list[Profile] = field(default_factory=list)

    @property
    def axes(self) -> tuple[Axes, ...]:
        """All axes in reading order: main (left, right), then ratio (left, right)."""
        candidates = (self.ax, self.ax_right, self.ratio_ax, self.ratio_ax_right)
        return tuple(a for a in candidates if a is not None)

    @property
    def hists(self) -> list[Any]:
        """The underlying ``hist.Hist`` objects, in drawing order."""
        return [h.hist for h in self.histograms]

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        formats: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[Path]:
        """Save the figure and return the written paths.

        Parameters
        ----------
        path
            Output file. If it is an existing directory (or ends with a
            separator), the file name is derived from the variable
            (``<dir>/<variable>.pdf``).
        formats
            Optional list of formats (``["pdf", "png"]``) to write instead of
            (or in addition to) the suffix of ``path``. Each replaces the suffix.
        **kwargs
            Forwarded to :meth:`matplotlib.figure.Figure.savefig`. Figures made
            by rootfig use constrained layout and are saved at exactly their
            ``figsize``. For figures drawn into user axes without a layout
            engine, ``bbox_inches="tight"`` is used unless given.
        """
        if self.fig.get_layout_engine() is None:
            kwargs.setdefault("bbox_inches", "tight")
            kwargs.setdefault("pad_inches", 0.04)
        target = Path(path)
        if str(path).endswith(("/", os.sep)) or target.is_dir():
            stem = self.variable.safe_name if self.variable is not None else "plot"
            target = target / f"{stem}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        outputs = (
            [target] if not formats else [target.with_suffix(f".{f.lstrip('.')}") for f in formats]
        )
        for output in outputs:
            self.fig.savefig(output, **kwargs)
        return outputs

    def show(self) -> None:
        """Display the figure (``plt.show()``)."""
        plt.show()

    def close(self) -> None:
        """Close the figure to free memory."""
        plt.close(self.fig)

    def __repr__(self) -> str:
        labels = ", ".join(repr(h.label) for h in self.histograms)
        panels = "main+ratio" if self.ratio_ax is not None else "main"
        return f"Plot(histograms=[{labels}], panels={panels})"
