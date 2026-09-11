"""Applying :class:`~rootfig.model.Style` to matplotlib: rcParams context, colours, labels.

rootfig never changes global matplotlib state on import. While a figure is
drawn, the style's rcParams are applied through :func:`style_context`; users
who want a persistent style can call :func:`use_style` explicitly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredText

from rootfig.model.style import EXPERIMENT_STYLES, Style, StyleLike, as_style

__all__ = [
    "DEFAULT_COLORS",
    "ROOTFIG_STYLE",
    "add_experiment_label",
    "color_cycle",
    "resolve_rc",
    "style_context",
    "use_style",
]

_TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)  # fmt: skip
_TAB20_LIGHT = (
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
)  # fmt: skip
_TAB20B = (
    "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194",
    "#6b6ecf", "#b5cf6b", "#e7ba52", "#d6616b", "#ce6dbd",
)  # fmt: skip
_TAB20C = (
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#969696",
)  # fmt: skip

DEFAULT_COLORS: tuple[str, ...] = (*_TAB10, *_TAB20_LIGHT, *_TAB20B, *_TAB20C)
"""Default colour cycle: matplotlib's ``tab10``, extended to 40 colours.

The first ten are the familiar matplotlib defaults, followed by their lighter
``tab20`` companions and ten colours each from ``tab20b`` and ``tab20c``, so
plots with many samples get distinct colours instead of a repeating cycle
(the scheme used by k4Bench)."""

ROOTFIG_STYLE: Mapping[str, Any] = {
    "figure.figsize": (7.0, 5.6),
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.subplot.left": 0.13,
    "figure.subplot.right": 0.97,
    "figure.subplot.top": 0.95,
    "figure.subplot.bottom": 0.12,
    "font.size": 13,
    "font.family": "sans-serif",
    # matplotlib's default face, so text and math text ($p_T$) share one design;
    # experiment styles bring their own fonts (e.g. TeX Gyre Heros for ATLAS)
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "axes.linewidth": 1.1,
    "axes.labelpad": 6,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 4),
    "axes.prop_cycle": cycler(color=list(DEFAULT_COLORS)),
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "errorbar.capsize": 0,
    "hatch.linewidth": 0.8,
}
"""Experiment-neutral defaults used when a :class:`Style` has no ``base``."""


def _mplhep_style(name: str) -> Mapping[str, Any] | None:
    """Return the mplhep style sheet ``name`` if it exists (case-insensitive)."""
    for attribute in dir(hep.style):
        if attribute.upper() == name.upper():
            sheet = getattr(hep.style, attribute)
            if isinstance(sheet, Mapping):
                return sheet
    return None


def resolve_rc(style: Style) -> list[Mapping[str, Any] | str]:
    """Return the list of style sheets to apply for ``style`` (later entries win)."""
    base = style.base
    if base is None and style.experiment is not None:
        base = (
            style.experiment
            if style.experiment.upper() in {e.upper() for e in EXPERIMENT_STYLES}
            else None
        )
    sheets: list[Mapping[str, Any] | str] = []
    if base is None:
        sheets.append(ROOTFIG_STYLE)
    elif isinstance(base, Mapping):
        sheets.append(ROOTFIG_STYLE)
        sheets.append(base)
    else:
        mplhep_sheet = _mplhep_style(base)
        if mplhep_sheet is not None:
            sheets.append(mplhep_sheet)
        elif base in plt.style.available or base == "default":
            sheets.append(base)
        else:
            known = ", ".join(EXPERIMENT_STYLES)
            msg = (
                f"unknown style {base!r}; expected an mplhep style ({known}, ROOT, ...), "
                "a matplotlib style, or a mapping of rcParams"
            )
            raise ValueError(msg)
    if style.colors:
        sheets.append({"axes.prop_cycle": cycler(color=list(style.colors))})
    if style.rc:
        sheets.append(dict(style.rc))
    return sheets


@contextlib.contextmanager
def style_context(style: StyleLike = None) -> Iterator[Style]:
    """Temporarily apply ``style``'s rcParams; yields the resolved :class:`Style`."""
    resolved = as_style(style)
    sheets: Any = resolve_rc(resolved)
    with plt.style.context(sheets):
        yield resolved


def use_style(style: StyleLike = None) -> Style:
    """Apply ``style`` globally (until :func:`matplotlib.pyplot.rcdefaults`), return it.

    This is the one function in rootfig that changes global matplotlib state;
    everything else uses :func:`style_context`.
    """
    resolved = as_style(style)
    for sheet in resolve_rc(resolved):
        if isinstance(sheet, str):
            plt.style.use(sheet)
        else:
            mpl.rcParams.update(dict(sheet))  # type: ignore[arg-type]
    return resolved


def color_cycle(n: int, style: Style | None = None) -> list[str]:
    """Return ``n`` colours from the style's cycle (repeating if necessary)."""
    if style is not None and style.colors:
        colors = list(style.colors)
    else:
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color")
        colors = list(cycle) if cycle else list(DEFAULT_COLORS)
    return [colors[i % len(colors)] for i in range(n)]


def add_experiment_label(ax: Axes, style: Style, *, has_data: bool) -> None:
    """Draw the experiment label and/or free text described by ``style`` on ``ax``."""
    if not style.has_label:
        return
    text_lines = list(style.text_lines)
    simulation = (not has_data) if style.simulation is None else style.simulation
    status_words = (style.status or "").split()
    if "Simulation" in status_words:
        # mplhep adds the word itself for simulation; avoid "Simulation Simulation"
        status_words.remove("Simulation")
        simulation = True
    status = " ".join(status_words)
    lumi = style.lumi_parts
    com = style.com_parts
    if style.experiment:
        supplementary = "\n".join(text_lines) if text_lines else None
        helper = getattr(hep, style.experiment.lower(), None)
        label_fn = getattr(helper, "label", None) if helper is not None else None
        loc = style.label_loc if style.label_loc is not None else _default_label_loc(style)
        kwargs: dict[str, Any] = {
            "exp": style.experiment,
            "text": status,
            "data": not simulation,
            "ax": ax,
            # mplhep defaults to 13 TeV when com is left out; pass None to omit the energy
            "com": None if com is None else f"{com[0]} {com[1]}".strip(),
        }
        if lumi is not None:
            if lumi[1] == "fb^{-1}":
                kwargs["lumi"] = lumi[0]
            else:
                # mplhep hard-codes fb^-1; build the whole luminosity line ourselves
                kwargs["rlabel"] = _lumi_line(lumi, com, atlas_style=loc == 4)
        if style.label_loc is not None:
            kwargs["loc"] = style.label_loc
        if supplementary:
            kwargs["supp"] = supplementary
        if callable(label_fn):
            # mplhep's per-experiment helpers apply that experiment's conventions.
            kwargs.pop("exp", None)
            texts = label_fn(**kwargs)
        else:
            texts = hep.label.exp_label(**kwargs)
        _separate_label_words(ax, texts)
        return
    # No experiment: draw status/lumi/energy/text as a plain block of text.
    lines: list[str] = []
    header = " ".join(part for part in ["Simulation" if simulation else "", status] if part)
    if header:
        lines.append(header)
    energy = f"$\\sqrt{{s}} = {com[0]}$ {com[1]}".strip() if com is not None else ""
    lumi_text = f"{lumi[0]} $\\mathrm{{{lumi[1]}}}$" if lumi is not None else ""
    if energy or lumi_text:
        lines.append(", ".join(part for part in [energy, lumi_text] if part))
    lines.extend(text_lines)
    if lines:
        box = AnchoredText(
            "\n".join(lines),
            loc="upper left",
            frameon=False,
            prop={"fontsize": mpl.rcParams["legend.fontsize"]},
            pad=0.3,
            borderpad=0.6,
        )
        ax.add_artist(box)


def _default_label_loc(style: Style) -> int:
    """Return the label position mplhep's helper for this experiment uses by default."""
    experiment = (style.experiment or "").upper()
    if experiment == "ATLAS":
        return 4
    if experiment in ("LHCB", "ALICE"):
        return 1
    return 0


def _lumi_line(lumi: tuple[str, str], com: tuple[str, str] | None, *, atlas_style: bool) -> str:
    """Luminosity/energy text in mplhep's two layouts, with arbitrary units."""
    lumi_text = f"{lumi[0]} $\\mathrm{{{lumi[1]}}}$"
    if com is None:
        return lumi_text
    if atlas_style:
        energy = "\\ ".join(part for part in com if part)
        return f"$\\sqrt{{s}} = \\mathrm{{{energy}}}$, {lumi_text}"
    return f"{lumi_text} ({' '.join(part for part in com if part)})"


LABEL_WORD_GAP_EM = 0.3
"""Extra horizontal gap, in units of the font size, inserted between the experiment
name and the status text (mplhep places them nearly touching)."""


def _separate_label_words(ax: Axes, texts: Any) -> None:
    """Nudge the status text right when mplhep put it on the same line as the experiment."""
    try:
        exp_txt, suffix = texts[0], texts[1]
    except (TypeError, IndexError):  # pragma: no cover - defensive against mplhep changes
        return
    if exp_txt is None or suffix is None or not suffix.get_text():
        return
    fig = ax.get_figure(root=True)
    if fig is None:  # pragma: no cover
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    exp_box = exp_txt.get_window_extent(renderer)
    suffix_box = suffix.get_window_extent(renderer)
    same_line = suffix_box.y0 < exp_box.y1 and suffix_box.y1 > exp_box.y0
    if not same_line or suffix_box.x0 < exp_box.x0:
        return
    shift_px = LABEL_WORD_GAP_EM * suffix.get_fontsize() / 72.0 * fig.dpi
    x, y = suffix.get_position()
    x_display, _ = suffix.get_transform().transform((x, y))
    new_x, _ = suffix.get_transform().inverted().transform((x_display + shift_px, 0.0))
    suffix.set_position((new_x, y))


def legend_location(style: Style) -> str | None:
    """Return the legend location string for ``style`` or ``None`` to skip the legend."""
    if style.legend is False:
        return None
    return style.legend if isinstance(style.legend, str) else "best"
