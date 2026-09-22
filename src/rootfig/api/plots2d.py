"""Two-dimensional histograms and correlation matrices: :func:`plot2d`, :func:`correlation`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rootfig._typing import FloatArray
from rootfig.api._common import normalize_for_plot, single_sample, style_for
from rootfig.api._hists import (
    histogram_objects,
    rebin_ready_made,
    reject_fill_options,
    require_dimension,
    wrap_histograms,
)
from rootfig.errors import BinningError, SelectionError
from rootfig.histograms import (
    NormalizeSpec,
    build_histograms_2d,
    correlation_matrix,
    describe_axes,
    load_columns,
    stored_mode,
)
from rootfig.model import (
    Bins,
    CutLike,
    StyleLike,
    Variable,
    as_variable,
)
from rootfig.plotting import (
    AxesLike,
    Finish,
    Plot,
    add_experiment_label,
    correlation_figsize,
    draw_correlation,
    draw_hist2d,
    finish_figure,
    make_figure,
    pin_fonts,
    style_context,
)
from rootfig.selection import Columns, NonFinitePolicy

__all__ = ["correlation", "plot2d"]


def plot2d(
    data: Any,
    x: str | Variable | None = None,
    y: str | Variable | None = None,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | tuple[Bins, Bins] | None = None,
    normalize: NormalizeSpec = None,
    logz: bool = False,
    logx: bool | None = None,
    logy: bool | None = None,
    cmap: str | Any = "viridis",
    colorbar: bool = True,
    zlabel: str | None = None,
    title: str | None = None,
    text: str | Sequence[str] | None = None,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    nonfinite: NonFinitePolicy = "drop",
    assume_poisson: bool = False,
    save: str | None = None,
) -> Plot:
    """Draw a two-dimensional histogram of ``y`` versus ``x`` for one sample.

    ``x`` and ``y`` must have the same structure (both per-event, or both
    per-object from the same collection). ``bins`` applies to both axes unless
    it is a pair of binning specifications, one per axis (so ``(40, 20)`` is
    two bin counts, never a range; a range needs ``(n, low, high)``); per-axis
    ranges, labels and logarithmic scales (``logx``/``logy`` default to the
    variables' ``log`` flags) are best given through
    :class:`~rootfig.model.Variable` objects. The two axes infer their ranges
    robustly and independently, and unlike the 1D plots there is no flow
    indicator, so give every axis that needs its full extent its own
    ``range="auto"``: ``rf.Variable(x, range="auto")`` leaves the y axis
    inferring robustly.

    Like :func:`plot`, it also draws histograms that already exist: ``x`` alone
    may name a ``TH2`` stored in the file (its label, unit and ``log`` flag then
    describe the x axis; the y axis keeps the stored title, and shows its axis
    name when the file has none), and ``data`` may be a 2D ``hist.Hist`` or
    :class:`~rootfig.histograms.Histogram`. For such an object ``x`` and ``y``
    are optional and describe its axes as ``variable`` does in :func:`plot`:
    label, unit, ``log`` flag, a ``name`` for the axis and ``bins``/``range`` to crop
    and merge to;
    the object itself is left untouched. For both, ``bins`` crops and merges bins per axis
    as in :func:`plot`: an integer count, or edges that coincide with the
    existing ones. A range without bins keeps the bins between its ends, which
    must be existing edges too; cropped
    content moves into the flow bins. ``assume_poisson`` accepts such a histogram without
    variances, as in :func:`plot`.
    """
    objects = histogram_objects(data)
    x_bins, y_bins = _split_bins(bins)
    var_x: Variable | None
    if objects is not None:
        reject_fill_options(
            "histogram objects",
            tree=tree,
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
        )
        if len(objects) != 1:
            msg = f"plot2d() draws a single histogram, got {len(objects)}"
            raise ValueError(msg)
        # x and y describe the axes of the histogram given; an explicit bins= overrides theirs
        var_x = None if x is None else as_variable(x, bins=x_bins)
        var_y = None if y is None else as_variable(y, bins=y_bins)
        [histogram_] = wrap_histograms(objects, assume_poisson=assume_poisson)
        require_dimension([histogram_], 2, "plot2d")
        if var_x is not None or var_y is not None:
            described = [var_x, var_y]
            histogram_ = histogram_.map_hists(lambda h: describe_axes(h, described))
        [histogram_] = rebin_ready_made(
            [histogram_],
            [x_bins if var_x is None else var_x.bins, y_bins if var_y is None else var_y.bins],
            [None if var_x is None else var_x.range, None if var_y is None else var_y.range],
        )
        is_data = histogram_.is_data
        logx = (var_x is not None and var_x.log) if logx is None else logx
        logy = (var_y is not None and var_y.log) if logy is None else logy
    else:
        if x is None:
            msg = "plot2d() needs the x and y variables, or the name of a stored 2D histogram"
            raise TypeError(msg)
        sample = single_sample(data, function="plot2d()", tree=tree)
        var_x = as_variable(x, bins=x_bins)
        if y is not None:
            var_y = as_variable(y, bins=y_bins)
        else:
            # the y axis of a stored 2D histogram: the same name (so the axes are told apart by
            # a suffix), its own bin count, and none of x's label, unit or log flag, which
            # describe the x axis only
            var_y = Variable(var_x.expression, bins=y_bins, name=var_x.name)
            if not stored_mode([sample], [var_x, var_y]):
                msg = (
                    "plot2d() needs the x and y variables, or the name of a 2D histogram "
                    f"stored in the file; {var_x.expression!r} is neither"
                )
                raise TypeError(msg)
        logx = var_x.log if logx is None else logx
        logy = var_y.log if logy is None else logy
        [histogram_] = build_histograms_2d(
            [sample],
            var_x,
            var_y,
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
            assume_poisson=assume_poisson,
        )
        is_data = sample.is_data
    if histogram_.ndim != 2:
        msg = f"plot2d() draws two-dimensional histograms, got {histogram_.ndim}D; use plot()"
        raise ValueError(msg)
    if normalize is not None and normalize is not False:
        histogram_ = normalize_for_plot(histogram_, normalize)
    resolved_style = style_for(style, text, lumi)
    with style_context(resolved_style) as st:
        # Same canvas as a 1D plot; the colour bar takes its space from the main axes
        # (draw_hist2d stops mplhep from widening the figure).
        layout = make_figure(st, ratio=False, ax=ax, figsize=figsize or st.figsize)
        fig, main_ax = layout.fig, layout.main
        draw_hist2d(
            histogram_,
            main_ax,
            logz=logz,
            cmap=cmap,
            colorbar=colorbar,
            zlabel=zlabel or (histogram_.normalization or "Events"),
        )
        if logx:
            main_ax.set_xscale("log")
        if logy:
            main_ax.set_yscale("log")
        if title:
            main_ax.set_title(title)
        # the bins fill the frame: an experiment label goes above it
        add_experiment_label(main_ax, st, has_data=is_data, above=True)
        pin_fonts(fig)  # last: fonts
    # outside the style context, against the layout the figure is drawn with
    finish_figure(fig, [Finish(main_ax, xlabel=main_ax)])
    result = Plot(fig=fig, ax=main_ax, histograms=[histogram_], variable=var_x)
    if save:
        result.save(save)
    return result


def correlation(
    data: Any,
    variables: Sequence[str | Variable],
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    labels: Sequence[str] | None = None,
    percent: bool = False,
    cmap: str | Any = "RdBu_r",
    annotate: bool = True,
    title: str | None = None,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    nonfinite: NonFinitePolicy = "drop",
    save: str | None = None,
) -> Plot:
    """Draw the correlation matrix of several variables for one sample.

    All variables must share the same structure (all per-event, or all
    per-object from one collection). The matrix is available as
    ``Plot.matrix``. The matrix is titled ``"<sample>: correlation"``; a style
    with an ``experiment`` draws that experiment's label above the matrix
    instead, and an explicit ``title`` is always shown.
    """
    sample = single_sample(data, function="correlation()", tree=tree)
    var_list = [as_variable(v) for v in variables]
    if len(var_list) < 2:
        msg = "correlation() needs at least two variables"
        raise SelectionError(msg)
    columns: Columns = load_columns(
        sample, var_list, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
    )
    matrix: FloatArray = correlation_matrix(columns)
    tick_labels = (
        list(labels) if labels is not None else [v.label or v.expression for v in var_list]
    )
    with style_context(style) as st:
        size = figsize or st.figsize or correlation_figsize(len(var_list))
        layout = make_figure(st, ratio=False, ax=ax, figsize=size)
        fig, main_ax = layout.fig, layout.main
        draw_correlation(
            matrix, tick_labels, main_ax, cmap=cmap, annotate=annotate, percent=percent
        )
        if st.experiment:
            # above the matrix, where the automatic title would go
            add_experiment_label(main_ax, st, has_data=sample.is_data, above=True)
        if title is not None or not st.experiment:
            main_ax.set_title(title if title is not None else f"{sample.label}: correlation")
        pin_fonts(fig)  # last: fonts
    # outside the style context, against the layout the figure is drawn with
    finish_figure(fig, [Finish(main_ax)])
    result = Plot(fig=fig, ax=main_ax, matrix=matrix)
    if save:
        result.save(save)
    return result


def _split_bins(bins: Bins | tuple[Bins, Bins] | None) -> tuple[Any, Any]:
    """Interpret ``bins`` for two axes: a pair of specifications, or one spec for both.

    Raises
    ------
    BinningError
        If the pair consists of two numbers that are not both integers: that
        reads like a ``(low, high)`` range, which a pair never is.
    """
    if bins is None:
        return None, None
    if isinstance(bins, tuple) and len(bins) == 2:
        numbers = all(
            isinstance(b, int | float | np.number) and not isinstance(b, bool) for b in bins
        )
        if numbers and not all(isinstance(b, int | np.integer) for b in bins):
            msg = (
                f"bins={bins!r}: for plot2d a 2-tuple is (x_bins, y_bins), one specification "
                "per axis, so two numbers are two bin counts, not a range. Give the range per "
                f"axis, e.g. bins=((50, {bins[0]}, {bins[1]}), (50, {bins[0]}, {bins[1]})), or "
                "use rf.Variable(x, bins=50, range=(low, high))"
            )
            raise BinningError(msg)
        return bins[0], bins[1]
    return bins, bins
