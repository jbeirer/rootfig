"""Drawing the lower panel: the comparisons of histograms with a reference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

from rootfig._storage import same_edges
from rootfig.histograms.build import compatible_binning
from rootfig.histograms.comparison import SIGNIFICANCE_KINDS, Comparison, ComparisonKind
from rootfig.histograms.efficiency import Efficiency, Profile
from rootfig.plotting.hist1d import band_label, in_view
from rootfig.plotting.style import color_cycle, foreground

__all__ = ["comparison_label", "draw_panel", "panel_ylim"]

DEFAULT_RATIO_YLIM = (0.5, 1.5)
"""The smallest automatic range of a ratio panel; a relative difference's is this minus one."""

PULL_YLIM = (3.0, 5.0)
"""The smallest and largest automatic half-range of a pull panel."""

ASYMMETRY_YLIM = 1.1
"""The largest automatic half-range of an asymmetry panel, keeping its bounds ±1 in the frame."""

OFF_SCALE_INSET = 4.0
"""How far inside the frame an off-scale marker sits, in points."""

_BASELINES: dict[str, float] = {
    "ratio": 1.0,
    "relative_difference": 0.0,
    "difference": 0.0,
    "asymmetry": 0.0,
}

_LABELS: dict[str, str] = {
    "ratio": "Ratio to {}",
    "relative_difference": "Rel. difference to {}",
    "difference": "Difference to {}",
    "pull": "Pull",
    "asymmetry": "Asymmetry to {}",
    "s/sqrt(b)": r"$S/\sqrt{{B}}$",
    "s/sqrt(s+b)": r"$S/\sqrt{{S+B}}$",
}


def comparison_label(kind: ComparisonKind, reference: str, *, data: bool = False) -> str:
    """Return the default y label of a ``kind`` panel comparing with ``reference``.

    ``data`` words it for a panel of observed data alone over a simulated
    reference (``"Data / MC"``); a pull and a significance are labelled by
    their kind alone.
    """
    minus = "\N{MINUS SIGN}"
    if data:
        match kind:
            case "ratio":
                return f"Data / {reference}"
            case "relative_difference":
                return f"(Data {minus} {reference}) / {reference}"
            case "difference":
                return f"Data {minus} {reference}"
            case "asymmetry":
                return f"(Data {minus} {reference}) / (Data + {reference})"
    return _LABELS[kind].format(reference)


def draw_panel(
    comparisons: Sequence[Comparison],
    ax: Axes,
    *,
    colors: Sequence[str] | None = None,
    observed: Sequence[bool] | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: str | None = None,
    band: bool | None = None,
    view: Sequence[tuple[float, float]] | None = None,
) -> None:
    """Draw ``comparisons`` of one kind into the lower panel ``ax``.

    Ratios, relative differences, differences and asymmetries are points with
    error bars around their baseline (1 or 0, a dashed line), over the
    reference's uncertainty band where there is one; pulls are filled bars from
    0, without error bars; a significance is points with error bars and no
    baseline. A point beyond the y range is marked by a triangle at the edge it
    left through, in its colour; the markers follow later changes of the range.

    Parameters
    ----------
    comparisons
        What :func:`~rootfig.histograms.compare` returned, one per numerator,
        all of one kind.
    ax
        The panel's axes.
    colors
        One colour per comparison; defaults to the foreground colour for
        observed data and the style's cycle for the rest.
    observed
        Which comparisons have observed data as numerator; their markers are
        drawn larger. Defaults to none.
    ylim
        Vertical range; defaults to :func:`panel_ylim`.
    ylabel
        Label; defaults to :func:`comparison_label` for the first comparison's reference.
    band
        Draw the reference's uncertainty band of the first comparison that has
        one. Defaults to drawing it when a comparison's error bars leave the
        reference out (``uncertainty="numerator"``); with ``"propagate"`` they
        already include it.
    view
        The x windows whose bins set the automatic y range.

    Raises
    ------
    ValueError
        Without comparisons, or with comparisons of different kinds, references
        or binnings: one panel shows one kind against one reference, whose
        label and uncertainty band it draws, so references whose bands differ
        are refused too.
    """
    if not comparisons:
        msg = "draw_panel needs at least one comparison"
        raise ValueError(msg)
    kind = comparisons[0].kind
    if any(c.kind != kind for c in comparisons):
        msg = f"draw_panel draws one kind; got {sorted({c.kind for c in comparisons})}"
        raise ValueError(msg)
    first = comparisons[0]
    for comparison in comparisons[1:]:
        if not _same_reference(first, comparison):
            msg = (
                "draw_panel draws one reference, whose label and uncertainty band the panel "
                f"shows; got {first.reference!r} and {comparison.reference!r}, which differ in "
                "their bins, their contents or their uncertainty bands"
            )
            raise ValueError(msg)
        if not same_edges(comparison.edges, first.edges):
            msg = "draw_panel draws one binning; the comparisons have different bin edges"
            raise ValueError(msg)
    flags = [False] * len(comparisons) if observed is None else list(observed)
    if colors is None:
        cycle = iter(color_cycle(len(comparisons)))
        colors = [foreground() if is_data else next(cycle) for is_data in flags]

    band_edges: tuple[np.ndarray, np.ndarray] | None = None
    if kind in SIGNIFICANCE_KINDS:
        _draw_points(comparisons, ax, colors, [False] * len(comparisons), clip_errors=True)
    elif kind == "pull":
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
        for comparison, color in zip(comparisons, colors, strict=True):
            values = np.where(np.isfinite(comparison.values), comparison.values, 0.0)
            ax.stairs(values, comparison.edges, baseline=0.0, fill=True, color=color, alpha=0.6)
    else:
        baseline = _BASELINES[kind]
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.0, zorder=1)
        with_band = next((c for c in comparisons if c.band is not None), None)
        total = with_band.total_band() if with_band is not None else None
        if band is None:
            band = any(c.uncertainty == "numerator" for c in comparisons)
        if band and with_band is not None and total is not None:
            band_down, band_up = total
            has_systematics = with_band.syst_band is not None
            lower = np.where(np.isfinite(band_down), baseline - band_down, baseline)
            upper = np.where(np.isfinite(band_up), baseline + band_up, baseline)
            band_edges = (lower, upper)
            ax.fill_between(
                with_band.edges,
                np.append(lower, lower[-1]),
                np.append(upper, upper[-1]),
                step="post",
                facecolor="gray",
                alpha=0.3,
                linewidth=0,
                zorder=0,
                label=f"{with_band.reference} {band_label(systematics=has_systematics).lower()}",
            )
        _draw_points(comparisons, ax, colors, flags, clip_errors=False)

    ax.set_ylim(
        *(ylim if ylim is not None else panel_ylim(comparisons, band=band_edges, view=view))
    )
    edges = comparisons[0].edges
    ax.set_xlim(edges[0], edges[-1])
    if ylabel is None:
        ylabel = comparison_label(kind, comparisons[0].reference)
    ax.set_ylabel(ylabel, loc="center")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="upper"))
    if kind != "pull":  # a pull's bars end at the edge they leave through
        for comparison, color in zip(comparisons, colors, strict=True):
            for above in (True, False):
                ax.add_artist(_OffScale(ax, comparison, above=above, color=color))


class _OffScale(Line2D):
    """Triangles at the top (``above``) or bottom edge of ``ax`` for points beyond its y range.

    The points are chosen from the limits the axes have when drawn, so they
    follow a later ``set_ylim``; x is in data coordinates and clipped to the
    axes, so bins outside the x range (or the other segment of a broken axis)
    are not marked. They sit :data:`OFF_SCALE_INSET` points inside the frame.
    """

    def __init__(self, ax: Axes, comparison: Comparison, *, above: bool, color: str) -> None:
        inset = -OFF_SCALE_INSET if above else OFF_SCALE_INSET
        super().__init__(
            [],
            [],
            linestyle="none",
            marker="^" if above else "v",
            markersize=5,
            color=color,
            zorder=3,
            transform=blended_transform_factory(ax.transData, ax.transAxes)
            + ScaledTranslation(0.0, inset / 72.0, ax.figure.dpi_scale_trans),
        )
        self._centers = comparison.centers
        self._values = comparison.values
        self._above = above
        self.set_in_layout(False)

    def draw(self, renderer: RendererBase) -> None:
        assert self.axes is not None
        low, high = sorted(self.axes.get_ylim())
        values = self._values
        beyond = np.isfinite(values) & ((values > high) if self._above else (values < low))
        self.set_data(self._centers[beyond], np.full(int(beyond.sum()), float(self._above)))
        super().draw(renderer)


def _same_reference(comparison: Comparison, other: Comparison) -> bool:
    """Whether two comparisons were made against the same reference, uncertainty included.

    The panel draws one band, so the references must agree on it: the
    histograms themselves when they carry them (:func:`compare` keeps the
    reference), one standing for the other when they bin alike and their
    contents agree, and in either case the band the comparisons carry. That
    band is the drawn uncertainty, statistical and systematic together, not the
    sources behind it: references varied by different sources that come to the
    same band are the same reference to draw. Binning alike is what
    :func:`~rootfig.histograms.compatible_binning` means, so two category axes
    must list the same categories, which their numeric edges do not say. An
    efficiency or a profile has no band: it must bin alike and agree in its
    values and errors, and a histogram never stands for one. A hand-built
    :class:`~rootfig.histograms.Comparison` has only its label and its band.
    """
    if not _same_band(comparison, other):  # the same nominal contents, varied differently
        return False
    first = _reference_of(comparison)
    second = _reference_of(other)
    if first is None or second is None:
        return comparison.reference == other.reference
    if first is second:
        return True
    if isinstance(first, Efficiency | Profile) or isinstance(second, Efficiency | Profile):
        return _same_points(first, second)
    if not compatible_binning(first, second):
        return False
    first_variances, second_variances = first.variances(), second.variances()
    same_variances = (
        np.array_equal(first_variances, second_variances)
        if first_variances is not None and second_variances is not None
        else first_variances is second_variances  # a storage without variances, on both sides
    )
    return bool(np.array_equal(first.values(), second.values()) and same_variances)


def _reference_of(comparison: Comparison) -> Any:
    """Return what ``comparison`` was made against, or ``None`` if it was built by hand."""
    if comparison.reference_hist is not None:
        return comparison.reference_hist
    return comparison.reference_points


def _same_points(first: object, second: object) -> bool:
    """Whether two efficiencies, or profiles of one statistic, agree bin by bin, ``nan`` too."""
    arrays: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]
    if isinstance(first, Efficiency) and isinstance(second, Efficiency):
        arrays = (
            (first.values, first.lower, first.upper),
            (second.values, second.lower, second.upper),
        )
    elif (
        isinstance(first, Profile)
        and isinstance(second, Profile)
        and first.statistic == second.statistic
    ):
        arrays = (first.values, first.errors), (second.values, second.errors)
    else:
        return False
    return same_edges(first.edges, second.edges) and all(
        a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
        for a, b in zip(*arrays, strict=True)
    )


def _same_band(comparison: Comparison, other: Comparison) -> bool:
    """Whether two comparisons carry the same reference uncertainty, ``nan`` bins included."""
    return all(
        (first is None and second is None)
        or (
            first is not None
            and second is not None
            and first.shape == second.shape
            and bool(np.array_equal(first, second, equal_nan=True))
        )
        for first, second in zip(_band_arrays(comparison), _band_arrays(other), strict=True)
    )


def _band_arrays(comparison: Comparison) -> tuple[np.ndarray | None, ...]:
    """Return the reference's statistical band and its systematic sides, ``None`` where absent."""
    down, up = comparison.syst_band if comparison.syst_band is not None else (None, None)
    return (comparison.band, down, up)


def _draw_points(
    comparisons: Sequence[Comparison],
    ax: Axes,
    colors: Sequence[str],
    observed: Sequence[bool],
    *,
    clip_errors: bool,
) -> None:
    """Draw every comparison as markers with error bars, skipping undefined bins.

    ``clip_errors`` draws undefined uncertainties of defined values as zero (a
    significance); otherwise the statistical and systematic errors are combined.
    """
    for comparison, color, is_data in zip(comparisons, colors, observed, strict=True):
        ok = np.isfinite(comparison.values)
        yerr: Any
        if clip_errors:
            yerr = np.where(np.isfinite(comparison.errors), comparison.errors, 0.0)[ok]
        elif comparison.syst_errors is None and not isinstance(comparison.errors, tuple):
            yerr = comparison.errors[ok]
        else:
            errors_down, errors_up = comparison.total_errors()
            yerr = [errors_down[ok], errors_up[ok]]
        ax.errorbar(
            comparison.centers[ok],
            comparison.values[ok],
            yerr=yerr,
            xerr=comparison.half_widths[ok],
            fmt="o",
            markersize=5 if is_data else 4,
            color=color,
            elinewidth=1.0,
            capsize=0,
        )


def panel_ylim(
    comparisons: Sequence[Comparison],
    *,
    band: tuple[np.ndarray, np.ndarray] | None = None,
    view: Sequence[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Choose the automatic y range of a panel from the bins in the x windows ``view``.

    ``band`` is the lower and upper edge of the drawn reference band, if any.
    Statistical error bars do not widen the range, so a few low-statistics bins
    with huge uncertainties cannot squash the panel; a significance is the
    exception, its top following the highest point plus its error. Per kind:

    * ratio: at least (0.5, 1.5), widened to the 5th to 95th percentile, padded
      by 10 percent, of the values, their systematic extent and the band edges,
      each judged on its own so it can widen the range but never narrow it;
      clipped to ``[0, 3]``, or to ``[-3, 3]`` with negative ratios (signed
      weights);
    * relative difference: the ratio's range of the values plus one, shifted back;
    * difference: symmetric, 1.1 times the largest magnitude of those
      percentiles, or ``(-1, 1)`` when everything is zero;
    * asymmetry: the difference's range, at most :data:`ASYMMETRY_YLIM` either way;
    * pull: symmetric, 1.1 times the 95th percentile of the magnitudes, at least
      3 and at most 5;
    * significance: from zero to 1.25 times the highest value plus its error.
    """
    kind = comparisons[0].kind if comparisons else "ratio"
    visible = [in_view(c.edges, view) for c in comparisons]
    if kind in SIGNIFICANCE_KINDS:
        tops = []
        for comparison, shown in zip(comparisons, visible, strict=True):
            errors = np.where(np.isfinite(comparison.errors), comparison.errors, 0.0)
            ok = np.isfinite(comparison.values) & shown
            if ok.any():
                tops.append(float(np.max(comparison.values[ok] + errors[ok])))
        top = max(tops, default=1.0)
        return (0.0, 1.25 * top if top > 0 else 1.0)
    if kind == "pull":
        magnitudes = _finite(
            [np.abs(c.values[shown]) for c, shown in zip(comparisons, visible, strict=True)]
        )
        smallest, largest = PULL_YLIM
        if not magnitudes.size:
            return (-smallest, smallest)
        half = min(max(smallest, 1.1 * float(np.percentile(magnitudes, 95))), largest)
        return (-half, half)
    shift = 1.0 if kind == "relative_difference" else 0.0
    ranges = _extents(comparisons, visible, band, view, shift=shift)
    if kind in ("difference", "asymmetry"):
        bound = 0.0
        for lower, upper in ranges:
            if lower.size and upper.size:
                q_low, q_high = np.percentile(lower, 5), np.percentile(upper, 95)
                bound = max(bound, abs(float(q_low)), abs(float(q_high)))
        half = 1.1 * bound if bound > 0 else 1.0
        if kind == "asymmetry":
            half = min(half, ASYMMETRY_YLIM)
        return (-half, half)
    low, high = _ratio_range(ranges)
    return (low - shift, high - shift)


def _extents(
    comparisons: Sequence[Comparison],
    visible: Sequence[np.ndarray],
    band: tuple[np.ndarray, np.ndarray] | None,
    view: Sequence[tuple[float, float]] | None,
    *,
    shift: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the ``(lower, upper)`` finite values of what an automatic range must cover.

    The values, their systematic extent and the band edges, each a pair, in the
    bins shown and moved by ``shift``.
    """
    values = _finite(
        [c.values[shown] + shift for c, shown in zip(comparisons, visible, strict=True)]
    )
    ranges = [(values, values)]
    with_syst = [
        (c, shown, c.syst_errors)
        for c, shown in zip(comparisons, visible, strict=True)
        if c.syst_errors is not None
    ]
    if with_syst:
        ranges.append(
            (
                _finite([(c.values - syst[0])[shown] + shift for c, shown, syst in with_syst]),
                _finite([(c.values + syst[1])[shown] + shift for c, shown, syst in with_syst]),
            )
        )
    if band is not None:
        shown = in_view(comparisons[0].edges, view) if comparisons else np.ones_like(band[0], bool)
        ranges.append((_finite([band[0][shown] + shift]), _finite([band[1][shown] + shift])))
    return ranges


def _ratio_range(ranges: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Return the range of a ratio panel covering ``ranges``; the first holds the values."""
    low, high = DEFAULT_RATIO_YLIM
    for lower, upper in ranges:
        if lower.size and upper.size:
            q_low = float(np.percentile(lower, 5))
            q_high = float(np.percentile(upper, 95))
            pad = 0.1 * max(q_high - q_low, 0.2)
            low = min(low, q_low - pad)
            high = max(high, q_high + pad)
    values = ranges[0][0]
    floor = 0.0
    if values.size and values.min() < 0:
        floor = -3.0
        low = min(low, float(values.min()) - 0.1 * max(high - float(values.min()), 0.2))
    return (max(low, floor), min(high, 3.0))


def _finite(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0)
    combined = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    return combined[np.isfinite(combined)]
