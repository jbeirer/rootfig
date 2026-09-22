"""The :class:`Plot` object returned by every rootfig plotting function."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure

from rootfig.histograms import uncertainty

if TYPE_CHECKING:
    from rootfig._typing import FloatArray
    from rootfig.histograms import Comparison, Efficiency, Histogram, Profile, Uncertainty
    from rootfig.model import Variable

__all__ = ["Plot", "normalize_formats"]


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
    panel_ax
        The lower panel's axes (``panel=``), or ``None``.
    ax_right, panel_ax_right
        The right-hand segments when the x axis is broken (``xbreak``), else ``None``.
    histograms
        The :class:`~rootfig.histograms.Histogram` objects drawn (each wraps a
        ``hist.Hist``).
    stack
        The summed stacked histograms, labelled ``"Total"``, including variations,
        or ``None``. The band and automatic panels use this histogram.
    comparisons
        The :class:`~rootfig.histograms.Comparison` objects drawn in the lower
        panel, one per numerator.
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
    panel_ax: Axes | None = None
    ax_right: Axes | None = None
    panel_ax_right: Axes | None = None
    histograms: list[Histogram] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    variable: Variable | None = None
    matrix: FloatArray | None = None
    efficiencies: list[Efficiency] = field(default_factory=list)
    profiles: list[Profile] = field(default_factory=list)
    stack: Histogram | None = None

    @property
    def axes(self) -> tuple[Axes, ...]:
        """All axes in reading order: main (left, right), then the lower panel (left, right)."""
        candidates = (self.ax, self.ax_right, self.panel_ax, self.panel_ax_right)
        return tuple(a for a in candidates if a is not None)

    @property
    def hists(self) -> list[Any]:
        """The underlying ``hist.Hist`` objects, in input order."""
        return [h.hist for h in self.histograms]

    def uncertainty(self, label: str | None = None) -> Uncertainty:
        """Statistical and systematic uncertainties of the stack or one histogram.

        Parameters
        ----------
        label
            The label of one histogram. ``None`` selects the stack total, with
            same-named systematic sources added linearly, or the sole non-data
            histogram when there is no stack.

        Raises
        ------
        KeyError
            If no histogram has ``label``.
        ValueError
            If ``label`` is ``None`` and there is no stack and either zero or
            several non-data histograms.
        """
        if label is None:
            if self.stack is not None:
                return uncertainty(self.stack)
            simulated = [h for h in self.histograms if not h.is_data]
            if not simulated:
                msg = "the plot has no non-data histograms; pass the label of a histogram"
                raise ValueError(msg)
            if len(simulated) > 1:
                msg = (
                    f"the plot has no stack and {len(simulated)} overlaid histograms; "
                    f"pass the label of one, e.g. uncertainty({simulated[0].label!r})"
                )
                raise ValueError(msg)
            return uncertainty(simulated[0])
        for histogram in self.histograms:
            if histogram.label == label:
                return uncertainty(histogram)
        msg = f"no histogram labelled {label!r}; labels: {[h.label for h in self.histograms]}"
        raise KeyError(msg)

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
            Optional list of formats (``["pdf", "png"]``). Each replaces the
            suffix of ``path``; only the listed formats are written.
        **kwargs
            Forwarded to :meth:`matplotlib.figure.Figure.savefig`. Figures made
            by rootfig use constrained layout and are saved at exactly their
            ``figsize``. For figures drawn into user axes without a layout
            engine, ``bbox_inches="tight"`` is used unless given. The background
            is the figure's own (``facecolor="auto"``) unless given: saving runs
            after the style has been undone, so the global ``savefig.facecolor``
            would otherwise paint over it (a :func:`~rootfig.dark_theme` figure
            would lose its transparency).
        """
        kwargs.setdefault("facecolor", "auto")
        kwargs.setdefault("edgecolor", "auto")
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
        panels = "main+panel" if self.panel_ax is not None else "main"
        return f"Plot(histograms=[{labels}], panels={panels})"


def normalize_formats(formats: str | Sequence[str]) -> tuple[str, ...]:
    """Validate output formats and return them lower-cased, in order, without duplicates.

    One format may be given bare (``"png"``), a leading dot is accepted
    (``".png"``) and the order is kept. Anything matplotlib cannot write, and an
    empty list, raises :class:`ValueError` here rather than part way through a
    batch of figures.
    """
    names: list[object] = [formats] if isinstance(formats, str) else list(formats)
    supported = FigureCanvasBase.get_supported_filetypes()
    chosen: dict[str, None] = {}
    for name in names:
        if not isinstance(name, str):
            msg = f"formats= must contain strings, got {type(name).__name__}"
            raise TypeError(msg)
        suffix = name.lstrip(".").lower()
        if suffix not in supported:
            known = ", ".join(sorted(supported))
            msg = f"unknown output format {name!r}; matplotlib writes {known}"
            raise ValueError(msg)
        chosen[suffix] = None
    if not chosen:
        msg = "formats= must name at least one output format, e.g. formats=['pdf', 'png']"
        raise ValueError(msg)
    return tuple(chosen)
