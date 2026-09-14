"""The :class:`Style` description of how figures look.

rootfig is experiment-independent: the default style is neutral and no
experiment label is drawn unless requested. Experiment styles and labels are
provided through mplhep and switched on with ``Style(experiment="ATLAS")``,
``Style("CMS")`` and so on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

from rootfig.model.units import split_quantity

__all__ = ["EXPERIMENT_STYLES", "Style", "StyleLike", "as_style"]

EXPERIMENT_STYLES: tuple[str, ...] = ("ATLAS", "CMS", "LHCb", "ALICE", "DUNE")
"""Experiments for which mplhep ships a style and a label helper."""


@dataclass(frozen=True)
class Style:
    r"""Appearance settings applied while a figure is drawn.

    All fields are optional. Passing ``Style()`` (or nothing) gives the neutral
    rootfig look; setting ``experiment`` switches to that experiment's mplhep
    style and adds its label.

    Parameters
    ----------
    experiment
        Experiment name written in the label, e.g. ``"ATLAS"``. When it
        matches one of :data:`EXPERIMENT_STYLES` the corresponding mplhep style
        sheet is used as ``base`` unless ``base`` is set explicitly.
    status
        Text after the experiment name: ``"Internal"``, ``"Preliminary"``,
        ``"Work in Progress"``, ``"Simulation Preliminary"``, ...
    text
        Extra line(s) below the label, e.g. ``r"$E_\\gamma = 65$ GeV"`` or a
        list of strings.
    lumi
        Integrated luminosity shown in the label: a number in ``lumi_unit``
        (default fb^-1) or a string carrying its own unit, ``"10.8 ab^-1"``.
    com
        Centre-of-mass energy shown in the label: a number in ``com_unit``
        (default TeV) or a string carrying its own unit, ``"240 GeV"``.
    lumi_unit, com_unit
        Units for numeric ``lumi`` and ``com`` (``"fb^{-1}"``, ``"ab^{-1}"``,
        ``"TeV"``, ``"GeV"``, ...). Lepton-collider analyses typically set
        ``lumi_unit="ab^{-1}", com_unit="GeV"``.
    simulation
        Whether to add "Simulation" to the label. ``None`` (default) adds it
        when the plot contains no data sample.
    label_loc
        Placement of the experiment label (mplhep ``loc``, 0-4). ``None`` uses
        the experiment's own convention (e.g. inside the frame for ATLAS,
        above it for CMS).
    base
        Name of an mplhep style (``"ATLAS"``, ``"CMS"``, ``"LHCb2"``, ``"ALICE"``,
        ``"DUNE"``, ``"ROOT"``), a matplotlib style name, or a mapping of
        rcParams. ``None`` selects the neutral rootfig defaults.
    rc
        Additional rcParams overrides applied on top of ``base``.
    figsize
        Figure size in inches. Defaults to the style's ``figure.figsize``,
        enlarged vertically when a ratio panel is present.
    colors
        Colour cycle for samples without an explicit colour.
    legend
        Draw a legend (``True``), suppress it (``False``), or pass a matplotlib
        legend location string such as ``"upper left"``.
    legend_kwargs
        Extra keyword arguments forwarded to :meth:`matplotlib.axes.Axes.legend`.
    """

    experiment: str | None = None
    status: str | None = None
    text: str | Sequence[str] | None = None
    lumi: float | str | None = None
    com: float | str | None = None
    lumi_unit: str = "fb^{-1}"
    com_unit: str = "TeV"
    simulation: bool | None = None
    label_loc: int | None = None
    base: str | Mapping[str, Any] | None = None
    rc: Mapping[str, Any] = field(default_factory=dict)
    figsize: tuple[float, float] | None = None
    colors: Sequence[str] | None = None
    legend: bool | str = True
    legend_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_label(self) -> bool:
        """True if an experiment label (or free text) should be drawn."""
        return bool(self.experiment or self.status or self.text or self.lumi or self.com)

    @property
    def lumi_parts(self) -> tuple[str, str] | None:
        """``(value, unit)`` of the luminosity, e.g. ``("10.8", "ab^{-1}")``, or ``None``."""
        return split_quantity(self.lumi, self.lumi_unit, inverse=True)

    @property
    def com_parts(self) -> tuple[str, str] | None:
        """``(value, unit)`` of the centre-of-mass energy, e.g. ``("240", "GeV")``, or ``None``."""
        return split_quantity(self.com, self.com_unit)

    @property
    def text_lines(self) -> tuple[str, ...]:
        """``text`` normalised to a tuple of lines."""
        if self.text is None:
            return ()
        if isinstance(self.text, str):
            return tuple(self.text.split("\n"))
        return tuple(self.text)

    def replace(self, **changes: Any) -> Style:
        """Return a copy with the given fields changed, e.g. ``style.replace(lumi=140)``."""
        return replace(self, **changes)


StyleLike: TypeAlias = Style | str | None
"""A :class:`Style`, an experiment/style name, or ``None`` for the default."""


def as_style(style: StyleLike) -> Style:
    """Coerce ``None``, a name, or a ``Style`` into a ``Style``.

    A bare string is interpreted as an experiment name when it matches one of
    :data:`EXPERIMENT_STYLES` (case-insensitively) and as an mplhep/matplotlib
    style name otherwise.
    """
    if style is None:
        return Style()
    if isinstance(style, Style):
        return style
    if isinstance(style, str):
        for experiment in EXPERIMENT_STYLES:
            if style.upper() == experiment.upper():
                return Style(experiment=experiment)
        return Style(base=style)
    msg = f"style must be a Style, a name, or None, got {type(style).__name__}"  # type: ignore[unreachable]
    raise TypeError(msg)
