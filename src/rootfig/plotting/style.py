"""Applying :class:`~rootfig.model.Style` to matplotlib: rcParams context, colours, labels.

rootfig never changes global matplotlib state on import. While a figure is
drawn, the style's rcParams are applied through :func:`style_context`; users
who want a persistent style can call :func:`use_style` explicitly.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any
from weakref import WeakKeyDictionary

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont, fontManager
from matplotlib.offsetbox import AnchoredText
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation

from rootfig.errors import RootfigWarning
from rootfig.model.style import EXPERIMENT_STYLES, Style, StyleLike, as_style
from rootfig.plotting.figure import fit_ylabel

__all__ = [
    "DARK_THEME",
    "DEFAULT_COLORS",
    "ROOTFIG_STYLE",
    "add_experiment_label",
    "align_experiment_label",
    "color_cycle",
    "dark_theme",
    "finalize_figure",
    "foreground",
    "pin_fonts",
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

_DARK_INK = "#e6edf3"

DARK_THEME: Mapping[str, Any] = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "text.color": _DARK_INK,
    "axes.labelcolor": _DARK_INK,
    "axes.edgecolor": _DARK_INK,
    "xtick.color": _DARK_INK,
    "ytick.color": _DARK_INK,
    "patch.edgecolor": _DARK_INK,
    "hatch.color": _DARK_INK,
    "grid.color": "#30363d",
    # take the figure's own (transparent) colour when saved, whatever the global setting
    "savefig.facecolor": "auto",
    "savefig.edgecolor": "auto",
}
"""Colours applied by :func:`dark_theme`: light ink on a transparent background."""

_theme: ContextVar[Mapping[str, Any] | None] = ContextVar("rootfig_theme", default=None)


@contextlib.contextmanager
def dark_theme() -> Iterator[None]:
    """Draw everything made inside the block for a dark page.

    The block runs under :data:`DARK_THEME`, so figures, axes and artists made
    with plain matplotlib (``plt.subplots()`` for ``ax=``, ``p.ax.text(...)``)
    match. rootfig figures apply it again after their style sheet, so it also
    holds for experiment styles that fix a white background (ATLAS, LHCb,
    ALICE, DUNE); ``Style.rc`` still wins. The background is transparent, and
    :meth:`Plot.save <rootfig.Plot.save>` keeps it so, so one image suits any
    dark page.
    """
    _activate_backend()  # outside the rc context, as in style_context
    token = _theme.set(DARK_THEME)
    try:
        with plt.style.context(dict(DARK_THEME)):
            yield
    finally:
        _theme.reset(token)


def foreground() -> str:
    """Return the ink colour of the active style (``text.color``) for points and outlines.

    A colour name or hex string is returned as set; an RGB(A) tuple, which the
    rcParam also accepts, is converted to hex so it stays usable as a string.
    """
    color = mpl.rcParams["text.color"]
    return color if isinstance(color, str) else to_hex(color, keep_alpha=True)


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
    theme = _theme.get()
    if theme is not None:
        sheets.append(theme)
    if style.rc:
        sheets.append(dict(style.rc))
    return sheets


def _activate_backend() -> None:
    """Import and activate the pyplot backend, if that has not happened yet.

    matplotlib resolves its backend lazily, on the first pyplot call. In Jupyter
    that activation sets ``matplotlib.interactive(True)`` and the inline
    backend's rcParams. If it happened inside :func:`style_context`, the
    surrounding :func:`matplotlib.pyplot.style.context` would restore the
    pre-activation snapshot on exit and turn interactive mode back off, which
    stops the inline backend from displaying anything for the rest of the
    session. Resolving the backend first keeps those changes outside the
    context. Creates no figure and is a no-op for non-GUI backends.
    """
    if getattr(plt, "_backend_mod", None) is None:
        plt.draw_if_interactive()


@contextlib.contextmanager
def style_context(style: StyleLike = None) -> Iterator[Style]:
    """Temporarily apply ``style``'s rcParams; yields the resolved :class:`Style`."""
    _activate_backend()
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


def add_experiment_label(
    ax: Axes,
    style: Style,
    *,
    has_data: bool,
    above: bool = False,
    right: Axes | None = None,
) -> None:
    """Draw the experiment label and/or free text described by ``style`` on ``ax``.

    ``above=True`` puts an experiment label above the frame unless
    ``style.label_loc`` is set, for plots whose data fill the frame (2D
    histograms, matrices) and so leave no room for a label inside it. ``right``
    is the right segment of a broken x axis: the luminosity text that belongs
    above the right end of the frame is placed above that segment's end.
    """
    if not style.has_label:
        return
    text_lines = list(style.text_lines)
    simulation = (not has_data) if style.simulation is None else style.simulation
    status_words = (style.status or "").split()
    if "Simulation" in status_words:
        # the word comes from the data flag, once, wherever the status mentions it
        status_words.remove("Simulation")
        simulation = True
    status = " ".join(status_words)
    header = " ".join(part for part in ["Simulation" if simulation else "", status] if part)
    lumi = style.lumi_parts
    com = style.com_parts
    if style.experiment:
        helper = getattr(hep, style.experiment.lower(), None)
        label_fn = getattr(helper, "label", None) if helper is not None else None
        label_loc = 0 if style.label_loc is None and above else style.label_loc
        loc = label_loc if label_loc is not None else _default_label_loc(style)
        kwargs: dict[str, Any] = {
            "exp": style.experiment,
            # The finished words after the experiment name. mplhep would build them from
            # text= and data=, but it also prefixes "Supplementary" whenever supp= is set.
            "llabel": header,
            "ax": ax,
            # mplhep defaults to 13 TeV when com is left out; pass None to omit the energy
            "com": None if com is None else f"{com[0]} {com[1]}".strip(),
            # the offset-text dodge is redone in align_experiment_label once the axes are final
            "scilocator_adjust": False,
        }
        if lumi is not None:
            if lumi[1] == "fb^{-1}":
                kwargs["lumi"] = lumi[0]
            else:
                # mplhep hard-codes fb^-1; build the whole luminosity line ourselves
                kwargs["rlabel"] = _lumi_line(lumi, com, atlas_style=loc == 4)
        if label_loc is not None:
            kwargs["loc"] = label_loc
        if text_lines and loc != 0:
            # below the label, inside the frame
            kwargs["supp"] = "\n".join(text_lines)
        if callable(label_fn):
            # mplhep's per-experiment helpers apply that experiment's conventions.
            kwargs.pop("exp", None)
            label_fn(**kwargs)
        else:
            hep.label.exp_label(**kwargs)
        if text_lines and loc == 0:
            # With the label above the frame mplhep would turn supp= into a note rotated
            # along the frame's right edge; the lines belong inside, like any free text.
            _add_text_block(ax, text_lines)
        if right is not None:
            for text in [t for t in ax.texts if _is_lumi_above(t)]:
                # moved to the right segment's axes, so the layout reserves room there
                text.remove()
                text.set_transform(right.transAxes)
                text.set_clip_on(False)
                right.add_artist(text)
        align_experiment_label(ax, right=right)
        return
    # No experiment: draw status/lumi/energy/text as a plain block of text.
    lines = [header] if header else []
    energy = f"$\\sqrt{{s}} = {com[0]}$ {com[1]}".strip() if com is not None else ""
    lumi_text = f"{lumi[0]} $\\mathrm{{{lumi[1]}}}$" if lumi is not None else ""
    if energy or lumi_text:
        lines.append(", ".join(part for part in [energy, lumi_text] if part))
    lines.extend(text_lines)
    if lines:
        _add_text_block(ax, lines)


def _add_text_block(ax: Axes, lines: Sequence[str]) -> None:
    """Draw ``lines`` as one left-aligned block in the upper left corner inside the frame."""
    box = AnchoredText(
        "\n".join(lines),
        loc="upper left",
        frameon=False,
        prop={"fontsize": mpl.rcParams["legend.fontsize"]},
        pad=0.3,
        borderpad=0.6,
    )
    ax.add_artist(box)


def _is_lumi_above(text: Text) -> bool:
    """Whether ``text`` is mplhep's luminosity text on the line above the frame."""
    return isinstance(text, hep.label.LumiText) and text.get_position()[1] >= 1.0


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


def _generic_font_lists() -> dict[str, list[str]]:
    """Return the concrete font list behind each generic family in the current rcParams."""
    sans = list(mpl.rcParams["font.sans-serif"])
    return {
        # matplotlib accepts "sans" and "sans serif" as aliases of "sans-serif"
        "sans-serif": sans,
        "sans": sans,
        "sans serif": sans,
        "serif": list(mpl.rcParams["font.serif"]),
        "monospace": list(mpl.rcParams["font.monospace"]),
        "cursive": list(mpl.rcParams["font.cursive"]),
        "fantasy": list(mpl.rcParams["font.fantasy"]),
    }


def _font_is_available(name: str) -> bool:
    """Return whether matplotlib can resolve the font family ``name``.

    Asks ``findfont`` rather than testing membership in the installed font names,
    because matching is case- and spelling-insensitive: "Tex Gyre Termes" (how the
    mplhep LHCb2 sheet spells it) resolves fine but is not among the names. The
    result is not cached here -- matplotlib memoises the lookup itself and clears
    that cache when a font is registered, which a cache of ours would miss.
    """
    try:
        findfont(FontProperties(family=name), fallback_to_default=False)
    except ValueError:
        return False
    return True


def pin_fonts(fig: Figure) -> None:
    """Replace generic font families on every text of ``fig`` by the style's font list.

    A generic family such as ``"sans-serif"`` is resolved from rcParams each time
    a text is drawn, and matplotlib caches text metrics by the generic name. A
    figure rendered after its style context has ended (saving, inline display)
    would therefore be painted in different fonts from those its layout and
    label positions were computed with: labels shift and get clipped. Call this
    inside the style context once drawing is complete.

    Only families that are actually installed are pinned. matplotlib walks the
    whole family list on *every* draw (it builds the glyph fallback chain from
    it, uncached) and logs a ``findfont: Font family 'X' not found.`` warning for
    each miss, so pinning a font that is absent means hundreds of log lines per
    figure for no benefit: the first family that resolves is the one used.
    Fully unavailable family lists warn once per distinct expanded list per call.
    """
    resolved = _generic_font_lists()
    missing: dict[tuple[str, ...], None] = {}
    fallback = fontManager.defaultFamily["ttf"]

    def concrete(families: Any) -> list[str]:
        names = [families] if isinstance(families, str) else list(families)
        out: list[str] = []
        for family in names:
            out.extend(resolved.get(family, [family]))
        unique = list(dict.fromkeys(out))
        available = [name for name in unique if _font_is_available(name)]
        if available:
            return available
        missing[tuple(unique)] = None
        return [fallback]

    for text in fig.findobj(Text):
        text.set_fontfamily(concrete(text.get_fontfamily()))
    tick_family = concrete(mpl.rcParams["font.family"])
    for axes in fig.axes:
        # tick labels are re-created on every draw; give them the family explicitly
        axes.tick_params(axis="both", which="both", labelfontfamily=tick_family)

    for families in missing:
        warnings.warn(
            f"none of the requested fonts are installed ({', '.join(families)}); "
            f"falling back to {fallback!r}. Install one of them, or set another "
            "via Style(rc={'font.sans-serif': [...]}), to control the figure's font.",
            RootfigWarning,
            stacklevel=2,
        )


def finalize_figure(fig: Figure, *, panels: Sequence[Axes] = ()) -> None:
    """Fix what the figure's look depends on while its style context is still active.

    Call as the last drawing step inside the style context: pins the fonts (see
    :func:`pin_fonts`) and fits the y label of each lower panel in ``panels`` (see
    :func:`~rootfig.plotting.figure.fit_ylabel`), which measures text and so must
    follow the font pinning that decides which font is drawn. The experiment label
    is anchored afterwards, outside the context (see :func:`align_experiment_label`).
    """
    pin_fonts(fig)
    for panel in panels:
        fit_ylabel(panel)


LABEL_WORD_GAP_EM = 0.5
"""Horizontal gap, in units of the status text's font size, between the experiment
name and the status word (mplhep places them nearly touching)."""

LABEL_LUMI_GAP_EM = 1.0
"""Smallest horizontal gap, in units of the status text's font size, between the
words after the experiment name and the luminosity text on the line above the frame."""

LABEL_MIN_SCALE = 0.6
"""Smallest size of the label texts above the frame shrunk to fit its width, relative to
the sizes mplhep gave them."""

LABEL_FIT_PASSES = 3
"""Most draws :func:`align_experiment_label` makes to fit a label to a layout that
changes as the label does."""

_label_sizes: WeakKeyDictionary[Text, float] = WeakKeyDictionary()
_title_pads: WeakKeyDictionary[Text, float] = WeakKeyDictionary()


def _original_size(text: Text) -> float:
    """Return the font size ``text`` had before its first fit, from which every fit starts."""
    if text not in _label_sizes:
        _label_sizes[text] = float(text.get_fontproperties().get_size_in_points())
    return _label_sizes[text]


def _is_stacked(text: Text) -> bool:
    """Whether a luminosity text has a line of its own above the label line."""
    axes = text.axes
    return axes is not None and text.get_transform() is not axes.transAxes


def align_experiment_label(ax: Axes, *, right: Axes | None = None) -> None:
    """Anchor mplhep's experiment label to the finished figure.

    mplhep positions the status word ("Simulation", "Internal", ...) as an axes
    fraction computed when the label is drawn, and shifts a label above the axes
    right by the width of the y axis' scientific-notation offset text measured at
    that moment. Both go stale once the axes are resized (constrained layout) or
    the y scale changes (a log axis has no offset text). This puts a label above
    the axes flush with the frame unless an offset text is really shown, fits the
    line above the frame to its width (see :func:`_fit_label_line`), lifts the
    title above the label texts over the frame (see :func:`_clear_title`), and
    places the status word a fixed gap after the experiment name with a point-based
    offset, so it stays put at any axes size.

    Call it on the finished figure once its style context has ended, as rootfig's
    plotting functions do: tick labels and colour bars are laid out when the figure
    is drawn, from the rcParams active then, so only that layout is the one the
    figure is shown with. Fitting the label changes the layout in turn, so the
    figure is drawn and measured again, up to :data:`LABEL_FIT_PASSES` times, until
    nothing changes. Sizes follow from the texts' original sizes, so aligning again
    leaves the label as it is. ``right`` is the right segment of a broken x axis,
    which carries the luminosity text (see :func:`add_experiment_label`). Does
    nothing when ``ax`` carries no mplhep label.
    """
    exp_txt = next((t for t in ax.texts if isinstance(t, hep.label.ExpLabel)), None)
    if exp_txt is None:
        return
    fig = ax.get_figure(root=True)
    if fig is None:  # pragma: no cover
        return
    texts = [*ax.texts, *(right.texts if right is not None else ())]
    x_exp, y_exp = exp_txt.get_position()
    above = y_exp >= 1.0 and exp_txt.get_horizontalalignment() == "left"  # mplhep loc 0/3
    suffix = next((t for t in ax.texts if isinstance(t, hep.label.ExpText) and t.get_text()), None)
    lumi = next((t for t in texts if _is_lumi_above(t) and t.get_text()), None)
    # The status word and the luminosity text can be wider than a small axes. Left in the
    # layout, constrained layout would shrink the axes to make room for them until it
    # collapses; the short experiment name stays in and reserves the space above the axes.
    # A name inside the frame reserves nothing up there, so a luminosity text above the
    # frame then stays in the layout, which it cannot collapse on its own, and so does one
    # on a line of its own (see _fit_label_line).
    for text in texts:
        if isinstance(text, hep.label.ExpText) or (
            above and _is_lumi_above(text) and not _is_stacked(text)
        ):
            text.set_in_layout(False)
    for _ in range(LABEL_FIT_PASSES):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        if above:
            x_exp = 0.0
            offset_text = ax.yaxis.offsetText
            if offset_text.get_visible() and offset_text.get_text():
                width = offset_text.get_window_extent(renderer).width
                x_exp = 1.1 * width / ax.get_window_extent(renderer).width
            exp_txt.set_position((x_exp, y_exp))
        fitted = above and lumi is not None and _fit_label_line(exp_txt, suffix, lumi, renderer)
        if not (_clear_title(ax, texts, renderer) or fitted):
            break
    else:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    if suffix is None:
        return
    exp_box = exp_txt.get_window_extent(renderer)
    suffix_box = suffix.get_window_extent(renderer)
    same_line = suffix_box.y0 < exp_box.y1 and suffix_box.y1 > exp_box.y0
    if not same_line:
        return
    em_pt = suffix.get_fontproperties().get_size_in_points()
    offset_pt = exp_box.width / fig.dpi * 72.0 + LABEL_WORD_GAP_EM * em_pt
    _, y_suffix = suffix.get_position()
    suffix.set_position((x_exp, y_suffix))
    # ScaledTranslation is evaluated at draw time, so the offset is right at any dpi
    # (offset_copy would freeze it in pixels of the current dpi).
    suffix.set_transform(
        ax.transAxes + ScaledTranslation(offset_pt / 72.0, 0.0, fig.dpi_scale_trans)
    )


def _fit_label_line(name: Text, status: Text | None, lumi: Text, renderer: Any) -> bool:
    """Fit the label line above the frame to the frame's width; return whether it changed.

    The experiment name, the status word after it and the luminosity text at the
    right end share that line, and a narrow axes (a colour bar takes its share)
    leaves too little room for all three. The texts are measured as if at their
    original sizes (see :func:`_original_size`) and shrunk by one common factor
    until they fit. A line that would need less than :data:`LABEL_MIN_SCALE` gives
    the luminosity text a line of its own above the others instead, at that
    smallest size and in the layout, so constrained layout reserves the height
    (unless the text is wider than its axes, which only tiny figures make it).
    """
    fig = name.get_figure(root=True)
    axes = lumi.axes
    if fig is None or axes is None:  # pragma: no cover
        return False
    words = [name] if status is None or status.get_position()[1] < 1.0 else [name, status]
    parts = [*words, lumi]
    current = {text: float(text.get_fontproperties().get_size_in_points()) for text in parts}

    def at_original(text: Text, pixels: float) -> float:
        # text extents are proportional to the font size
        return pixels * _original_size(text) / current[text]

    em = _original_size(words[-1]) / 72.0 * fig.dpi
    words_width = sum(at_original(t, t.get_window_extent(renderer).width) for t in words)
    words_width += (len(words) - 1) * LABEL_WORD_GAP_EM * em
    lumi_width = at_original(lumi, lumi.get_window_extent(renderer).width)
    name_height = at_original(name, name.get_window_extent(renderer).height)
    together = words_width + LABEL_LUMI_GAP_EM * em + lumi_width
    available = float(lumi.get_window_extent(renderer).x1 - name.get_window_extent(renderer).x0)
    stacked = together * LABEL_MIN_SCALE > available
    # stacked lines keep the smallest size, so a label never grows as its axes narrow
    scale = LABEL_MIN_SCALE if stacked else min(1.0, available / together)
    changed = stacked != _is_stacked(lumi)
    for text in parts:
        size = _original_size(text) * scale
        if abs(size - current[text]) > 0.01 * _original_size(text):
            text.set_fontsize(size)
            changed = True
    if stacked:
        lift = ScaledTranslation(0.0, name_height * scale / fig.dpi, fig.dpi_scale_trans)
        lumi.set_transform(axes.transAxes + lift)
    else:
        lumi.set_transform(axes.transAxes)
    # wider than its axes, it would make constrained layout collapse them (see above)
    lumi.set_in_layout(stacked and lumi_width * scale <= axes.get_window_extent(renderer).width)
    return changed


def _clear_title(ax: Axes, texts: Sequence[Text], renderer: Any) -> bool:
    """Lift the title of ``ax`` above the label texts over its frame; return whether it moved.

    A label above the frame (CMS and DUNE, 2D plots, correlation matrices, and the
    luminosity of LHCb and ALICE) shares that space with the title, so the title
    goes above the highest of those texts, the style's title pad apart. It stays in
    the layout, which then reserves room for both.
    """
    fig = ax.get_figure(root=True)
    # Matplotlib's title artists and shared offset are not exposed in its type stubs.
    all_titles = (ax.title, ax._left_title, ax._right_title)  # type: ignore[attr-defined]
    titles = [title for title in all_titles if title.get_text()]
    label = [
        text
        for text in texts
        if isinstance(text, hep.label.ExpLabel | hep.label.ExpText | hep.label.LumiText)
        and text.get_text()
        and text.get_position()[1] >= 1.0
    ]
    if fig is None or not titles or not label:
        return False
    # Read the pad from the axes, not the current rcParams: this also runs after
    # leaving the style context. Keep the original gap across repeated fits.
    current_pad = float(ax.titleOffsetTrans.transform((0, 0))[1])  # type: ignore[attr-defined]
    pad_pt = _title_pads.setdefault(ax.title, current_pad / fig.dpi * 72.0)
    pad_px = pad_pt / 72.0 * fig.dpi
    bottom = max(text.get_window_extent(renderer).y1 for text in label) + pad_px
    lift = bottom - min(title.get_window_extent(renderer).y0 for title in titles)
    if lift <= 1.0:
        return False
    # All three title locations share this offset. Moving it preserves the
    # existing artists, their styles and explicit positions without set_title's
    # rcParam defaults selecting or restyling another title.
    ax._set_title_offset_trans((current_pad + lift) / fig.dpi * 72.0)  # type: ignore[attr-defined]
    return True


def legend_location(style: Style) -> str | None:
    """Return the legend location string for ``style`` or ``None`` to skip the legend."""
    if style.legend is False:
        return None
    return style.legend if isinstance(style.legend, str) else "best"
