"""The high-level API: :func:`plot`, :func:`histogram`, :func:`load` and friends.

Everything here is a thin orchestration of the lower layers
(:mod:`rootfig.io`, :mod:`rootfig.expressions`, :mod:`rootfig.selection`,
:mod:`rootfig.histograms`, :mod:`rootfig.plotting`), which remain usable on
their own.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import awkward as ak
import numpy as np

from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError, RootfigWarning, SelectionError, SourceError
from rootfig.expressions import parse
from rootfig.histograms import (
    SIGNIFICANCE_KINDS,
    CutflowTable,
    Histogram,
    NormalizeSpec,
    ProfileStatistic,
    RatioUncertainty,
    SignificanceKind,
    Summary,
    as_weight_storage,
    build_histograms,
    build_histograms_2d,
    combined_selection,
    correlation_matrix,
    describe_table,
    load_columns,
    load_columns_each,
    read_arrays,
    significance,
)
from rootfig.histograms import cutflow as cutflow_of
from rootfig.histograms import efficiency as efficiency_of
from rootfig.histograms import normalize as normalize_histogram
from rootfig.histograms import profile as profile_of
from rootfig.histograms import summarize as summarize_columns
from rootfig.model import (
    Bins,
    CutLike,
    RangeSpec,
    Sample,
    StyleLike,
    Variable,
    as_cut,
    as_samples,
    as_style,
    as_variable,
    resolve_axis,
)
from rootfig.model.samples import HistType
from rootfig.plotting import (
    AxesLike,
    FlowSpec,
    Plot,
    add_experiment_label,
    add_legend,
    add_stats_box,
    apply_xbreak,
    break_segments,
    color_cycle,
    draw_correlation,
    draw_efficiencies,
    draw_hist2d,
    draw_histograms,
    draw_profiles,
    draw_ratio_panel,
    draw_significance_panel,
    envelope,
    finalize_figure,
    finish_axes,
    fold_flow_bins,
    label_flow_bins,
    legend_location,
    make_figure,
    overlay_artists,
    raise_ylim_above,
    show_flow_bins,
    style_context,
    ylabel_for,
)
from rootfig.selection import Columns, NonFinitePolicy, boolean_mask, depth_of

__all__ = [
    "SummaryTable",
    "correlation",
    "histogram",
    "histograms",
    "load",
    "plot",
    "plot2d",
    "plot_histograms",
    "summarize",
]


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------


def load(
    data: Any,
    expressions: str | Sequence[str] | Mapping[str, str] | None = None,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> ak.Array:
    """Read branches (or evaluate expressions) into an Awkward record array.

    Parameters
    ----------
    data
        File path(s), glob, ``"path:tree"``, a :class:`~rootfig.model.Sample`,
        or in-memory arrays.
    expressions
        Branch names or expressions to evaluate. A mapping gives the output
        field names explicitly (``{"pt": "Muon_pt / 1000"}``). ``None`` reads
        every branch.
    tree
        Tree name when ``data`` is a file specification.
    selection
        An event-level boolean expression; events failing it are dropped. A
        per-object selection raises :class:`~rootfig.errors.SelectionError`
        (apply object cuts inside the expressions instead, e.g.
        ``"Muon_pt[Muon_pt > 20]"``), and so does a numeric one (an integer flag
        would otherwise be taken as an index array; write ``"flag != 0"``).
    entry_start, entry_stop
        Entry range to read (ignored for a ``Sample``, which carries its own).

    Returns
    -------
    awkward.Array
        A record array with one field per expression.
    """
    sample = _single_sample(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
    available = sample.source.branches()
    if expressions is None:
        fields: dict[str, str] = {name: f"`{name}`" for name in available}
    elif isinstance(expressions, str):
        fields = {expressions: expressions}
    elif isinstance(expressions, Mapping):
        fields = dict(expressions)
    else:
        fields = {e: e for e in expressions}
    if not fields:
        msg = "no expressions to load"
        raise SourceError(msg)

    parsed = {name: parse(text) for name, text in fields.items()}
    cut = combined_selection(sample, selection)
    arrays, n_events = read_arrays(sample, [*parsed.values(), *([cut.parsed()] if cut else [])])
    result = {
        name: expression.evaluate(arrays, length=n_events) for name, expression in parsed.items()
    }
    if cut is not None:
        mask = boolean_mask(cut.parsed(), arrays, length=n_events)
        if depth_of(mask) != 1:
            msg = (
                f"selection {cut.expression!r} is per-object; load() only supports per-event "
                "selections. Reduce it with any()/all()/count() or apply it inside the "
                "expressions, e.g. 'Muon_pt[Muon_pt > 20]'"
            )
            raise SelectionError(msg)
        result = {name: array[mask] for name, array in result.items()}
    return ak.Array(result)


# --------------------------------------------------------------------------------------
# Histograms
# --------------------------------------------------------------------------------------


def histograms(
    data: Any,
    variable: str | Variable,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | None = None,
    range: RangeSpec = None,
    label: str | Sequence[str] | None = None,
    normalize: NormalizeSpec = None,
    nonfinite: NonFinitePolicy = "drop",
) -> list[Histogram]:
    """Fill one :class:`~rootfig.histograms.Histogram` per sample with shared binning.

    See :func:`plot` for the meaning of the arguments; this function stops
    before drawing.
    """
    samples = as_samples(data, tree=tree, labels=label)
    var = as_variable(variable, bins=bins, range=range)
    hists = build_histograms(
        samples, var, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
    )
    if normalize is None or normalize is False:
        return hists
    return [normalize_histogram(h, normalize) for h in hists]


def histogram(
    data: Any,
    variable: str | Variable,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | None = None,
    range: RangeSpec = None,
    normalize: NormalizeSpec = None,
    nonfinite: NonFinitePolicy = "drop",
) -> Hist:
    """Fill a single histogram and return it as a plain ``hist.Hist``.

    Examples
    --------
    >>> h = rf.histogram(
    ...     "events.root", "Muon_pt", tree="events", selection="Muon_pt > 20", bins=(50, 0, 200)
    ... )  # doctest: +SKIP
    >>> h.values().sum()  # doctest: +SKIP
    """
    results = histograms(
        data,
        variable,
        tree=tree,
        selection=selection,
        weight=weight,
        lumi=lumi,
        bins=bins,
        range=range,
        normalize=normalize,
        nonfinite=nonfinite,
    )
    if len(results) != 1:
        msg = f"histogram() takes a single sample, got {len(results)}; use histograms() instead"
        raise SourceError(msg)
    return results[0].hist


# --------------------------------------------------------------------------------------
# 1D plots
# --------------------------------------------------------------------------------------


def plot(
    data: Any,
    variable: str | Variable,
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | None = None,
    range: RangeSpec = None,
    label: str | Sequence[str] | None = None,
    observed: Any = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    unit: str | None = None,
    title: str | None = None,
    normalize: NormalizeSpec = None,
    stack: bool = False,
    ratio: RatioSpec = False,
    ratio_ylim: tuple[float, float] | None = None,
    ratio_label: str | None = None,
    ratio_uncertainty: RatioUncertainty | None = None,
    logx: bool | None = None,
    logy: bool = False,
    flow: FlowSpec = "hint",
    histtype: HistType | None = None,
    errorbars: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xbreak: tuple[float, float] | None = None,
    legend: bool | str | None = None,
    stats: bool | str = False,
    text: str | Sequence[str] | None = None,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    nonfinite: NonFinitePolicy = "drop",
    save: str | None = None,
) -> Plot:
    """Histogram a variable from one or more samples and draw it.

    This is the ``TTree::Draw`` of rootfig: read only the branches needed,
    evaluate the expressions, apply the selection with the documented
    per-event/per-object semantics, fill ``hist.Hist`` objects with a binning
    shared by all samples, and render them with mplhep.

    Parameters
    ----------
    data
        What to plot: a file path or glob, ``"path:tree"``, a list of those (one
        sample each), a ``{label: files}`` mapping, one or more
        :class:`~rootfig.model.Sample` objects, or in-memory arrays (a mapping
        of arrays or an Awkward record array).
    variable
        Branch name or expression (see :mod:`rootfig.expressions`), or a
        :class:`~rootfig.model.Variable` carrying binning and labels.
    tree
        Tree name for file inputs; auto-detected when a file holds one tree.
    selection
        Boolean expression or :class:`~rootfig.model.Cut` applied before
        filling. Per-object selections mask objects of jagged variables;
        per-event selections drop events.
    weight
        Weight expression, e.g. ``"mc_weight * sf"``; per-event weights are
        broadcast onto objects.
    lumi
        Integrated luminosity to scale simulated samples to, in fb^-1 or as a
        string with a unit (``"10.8 ab^-1"``). Applies to samples with a cross
        section (``Sample(xsec=..., ngen=...)``): their weights are multiplied
        by ``xsec * lumi / ngen``. Also written into the label unless the style
        already has a luminosity.
    bins
        Binning: an ``int`` (range inferred from the data), ``(n, low, high)``,
        bin edges, or a ``hist`` axis. Overrides the ``Variable``'s binning.
    range
        Range for integer ``bins``: ``(low, high)``, ``"auto"`` or ``"robust"``.
    label
        Legend label(s) for samples given as plain files.
    observed
        A sample of observed data (or the file(s) for one) drawn as black points,
        excluded from stacks and used as numerator of the ratio.
    xlabel, ylabel, unit, title
        Axis labels; defaults come from the variable, the normalisation and the
        bin width (``Events / 2 GeV``).
    normalize
        ``True``/``"unity"`` (sum to one), ``"density"``, ``"width"`` (divide by
        bin width) or a number to normalise to.
    stack
        Stack the non-data samples.
    ratio
        ``True`` for a ratio panel: data / total MC for a stack (needs
        ``observed=``), data / the first non-data sample when data is overlaid,
        otherwise every further sample over the first; a sample label to use
        as the reference (all other histograms, data included, are divided by
        it); or a significance panel: ``"significance"`` (``S/sqrt(B)``),
        ``"s/sqrt(b)"`` or ``"s/sqrt(s+b)"``, where the signal is the last
        non-data sample (the top of a stack) and the background the sum of the
        others; ``("s/sqrt(b)", "Signal")`` names the signal sample.
    ratio_ylim, ratio_label, ratio_uncertainty
        Ratio panel range, y label, and uncertainty treatment
        (``"propagate"`` or ``"numerator"`` with a reference band).
    logx, logy
        Logarithmic axes. ``logx=None`` (default) follows the ``Variable``'s
        ``log`` flag; ``True``/``False`` override it.
    flow
        Under/overflow display: ``"hint"`` (arrows), ``"show"`` (extra bins),
        ``"sum"`` (added to the edge bins, also for ratios and limits), ``"none"``.
    histtype
        Default drawing style for non-data samples: ``"step"``, ``"fill"``,
        ``"errorbar"`` or ``"band"``.
    errorbars
        Draw statistical error bars on non-data histograms.
    xlim, ylim
        Axis limits; ``ylim`` entries may be ``None`` to keep the automatic value.
    xbreak
        ``(a, b)`` to cut the x axis: the range between ``a`` and ``b`` is
        removed and the two remaining segments are drawn side by side with a
        break mark (e.g. a peak and a far tail, or a sentinel region). Works
        with ratio panels; not with ``ax=``.
    legend
        ``False`` to suppress, or a matplotlib location string.
    stats
        Add a box with entries, mean and standard deviation per sample;
        ``True`` or a location string.
    text
        Extra text line(s) drawn with the experiment label.
    style
        :class:`~rootfig.model.Style`, an experiment name (``"ATLAS"``, ...),
        or ``None`` for the neutral rootfig style.
    figsize
        Figure size in inches.
    ax
        Draw into existing axes (``Axes`` or ``(main, ratio)`` pair).
    nonfinite
        ``"drop"`` (warn) or ``"error"`` for ``nan``/``inf`` values.
    save
        Path to save the figure to (also returned in the :class:`Plot`).

    Returns
    -------
    Plot
        The figure, axes, histograms and ratios.
    """
    samples = as_samples(data, tree=tree, labels=label)
    if observed is not None:
        observed_samples = [
            s if s.is_data else s.with_(is_data=True) for s in as_samples(observed, tree=tree)
        ]
        samples = [*samples, *observed_samples]
    var = as_variable(variable, bins=bins, range=range, label=xlabel, unit=unit)
    hists = build_histograms(
        samples, var, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
    )
    resolved_style = _style_for(style, text, lumi)
    return plot_histograms(
        hists,
        variable=var,
        ylabel=ylabel,
        title=title,
        normalize=normalize,
        stack=stack,
        ratio=ratio,
        ratio_ylim=ratio_ylim,
        ratio_label=ratio_label,
        ratio_uncertainty=ratio_uncertainty,
        logx=logx,
        logy=logy,
        flow=flow,
        histtype=histtype,
        errorbars=errorbars,
        xlim=xlim,
        ylim=ylim,
        xbreak=xbreak,
        legend=legend,
        stats=stats,
        style=resolved_style,
        figsize=figsize,
        ax=ax,
        save=save,
    )


def plot_histograms(
    hists: Sequence[Histogram | Hist],
    *,
    variable: Variable | None = None,
    labels: Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    normalize: NormalizeSpec = None,
    stack: bool = False,
    ratio: RatioSpec = False,
    ratio_ylim: tuple[float, float] | None = None,
    ratio_label: str | None = None,
    ratio_uncertainty: RatioUncertainty | None = None,
    logx: bool | None = None,
    logy: bool = False,
    flow: FlowSpec = "hint",
    histtype: HistType | None = None,
    errorbars: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xbreak: tuple[float, float] | None = None,
    legend: bool | str | None = None,
    stats: bool | str = False,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    save: str | None = None,
) -> Plot:
    """Draw already-filled histograms (``hist.Hist`` or :class:`~rootfig.histograms.Histogram`).

    Accepts the same drawing options as :func:`plot`. Plain ``hist.Hist``
    objects are labelled from ``labels`` (or numbered) and converted to
    ``Weight`` storage if they have a plain count storage; mark data by passing
    :class:`~rootfig.histograms.Histogram` objects with ``is_data=True``.
    Overlaid histograms may have different binnings; stacks, ratio panels and
    ``flow="show"`` need identical bin edges.
    """
    if logx is None:
        logx = variable.log if variable is not None else False
    histograms_ = _wrap_hists(hists, labels)
    if not histograms_:
        msg = "no histograms to draw"
        raise ValueError(msg)
    if any(h.ndim != 1 for h in histograms_):
        msg = "plot_histograms() draws one-dimensional histograms; use plot2d() for 2D"
        raise ValueError(msg)
    if normalize is not None and normalize is not False:
        histograms_ = [_normalize_for_plot(h, normalize) for h in histograms_]
    resolved_style = as_style(style)
    if legend is not None:
        resolved_style = resolved_style.with_(legend=legend)

    # The y label quotes the bin width of the histogram as filled, before flow bins are added.
    label_widths = histograms_[0].widths
    flow_shown = (False, False)
    if flow == "show":
        if xbreak is not None and xlim is None:
            msg = "xbreak cannot be combined with flow='show'; pass xlim as well"
            raise ValueError(msg)
        # Done here rather than in mplhep so every histogram, the ratio panel and the
        # x range agree on the extra bins (mplhep adds them per histogram).
        histograms_, flow_shown = show_flow_bins(histograms_)
        flow = "none"
    elif flow == "sum":
        # Fold once, up front, so ratios, bands and limits see the same bins as the drawing.
        histograms_ = fold_flow_bins(histograms_)
        flow = "none"

    reference_hist = histograms_[0]
    outer: tuple[float, float] | None = xlim or (
        float(reference_hist.edges[0]),
        float(reference_hist.edges[-1]),
    )
    if logx and outer is not None and outer[0] <= 0:
        positive = reference_hist.edges[reference_hist.edges > 0]
        outer = (float(positive[0]), outer[1]) if positive.size else None
    segments = None
    break_widths = None
    if xbreak is not None:
        assert outer is not None
        left_range, right_range, break_widths = break_segments(outer, xbreak, logx=logx)
        segments = (left_range, right_range)

    with style_context(resolved_style) as st:
        want_ratio = bool(ratio)
        layout = make_figure(
            st, ratio=want_ratio, ax=ax, figsize=figsize, break_widths=break_widths
        )
        drawn = None
        for axis in layout.main_axes:
            drawn = draw_histograms(
                histograms_,
                axis,
                style=st,
                stack=stack,
                histtype=histtype,
                errorbars=errorbars,
                flow=flow,
            )
        assert drawn is not None
        color_of = dict(zip(map(id, histograms_), drawn.histogram_colors, strict=True))
        has_data = any(h.is_data for h in histograms_)
        add_experiment_label(layout.main, st, has_data=has_data)

        per_object = any(h.stats is not None and h.stats.per_object for h in histograms_)
        unit = variable.unit if variable is not None else None
        y_label = ylabel or ylabel_for(
            normalization=histograms_[0].normalization,
            unit=unit,
            widths=label_widths,
            per_object=per_object,
        )
        x_label = xlabel or (variable.axis_label if variable is not None else None)
        if x_label is None:
            x_label = reference_hist.axis.label or ""
        data_low = drawn.ymin_positive if logy else drawn.ymin
        for index, axis in enumerate(layout.main_axes):
            finish_axes(
                axis,
                data_range=(data_low, drawn.ymax),
                xlabel=None,
                ylabel=y_label if index == 0 else None,
                xlim=outer,
                ylim=ylim,
                logx=logx,
                logy=logy,
            )
            # mplhep labels the axis from the hist; the label goes on one axes only
            axis.set_xlabel("")
        if title:
            layout.main.set_title(title)
        legend_loc = "upper right" if (stats and st.legend is True) else None
        legend_artist = add_legend(layout.legend_axes, st, loc=legend_loc)
        if stats:
            add_stats_box(
                layout.legend_axes,
                histograms_,
                loc=stats if isinstance(stats, str) else "auto",
                colors=drawn.colors,
                legend=legend_artist,
            )
        if ylim is None or ylim[1] is None:
            # keep legend, labels and text boxes clear of the histograms
            floating = legend_location(st) == "best" and legend_loc is None
            obstacles = overlay_artists(layout.legend_axes, None if floating else legend_artist)
            if layout.legend_axes is not layout.main:
                obstacles += overlay_artists(layout.main, None)
            env_edges, env_heights = envelope(histograms_, stack=stack)
            raise_ylim_above(
                layout.main_axes,
                obstacles,
                edges=env_edges,
                heights=env_heights,
                logy=logy,
                floating=[legend_artist] if floating and legend_artist is not None else [],
            )

        ratios = []
        significance_spec = _significance_spec(ratio)
        if layout.ratio is not None and significance_spec is not None:
            kind, signal_label = significance_spec
            signal_h, background_h = _significance_setup(histograms_, signal_label)
            sig_result = significance(signal_h.hist, background_h.hist, kind=kind)
            ratios = [sig_result]
            for index, axis in enumerate(layout.ratio_axes):
                draw_significance_panel(
                    sig_result,
                    axis,
                    kind=kind,
                    color=drawn.colors.get(signal_h.label),
                    ylim=ratio_ylim,
                    ylabel=ratio_label if index == 0 else "",
                )
                if logx:
                    axis.set_xscale("log")
                if outer is not None:
                    axis.set_xlim(*outer)
        elif layout.ratio is not None:
            assert not isinstance(ratio, tuple)  # tuples are significance specs, handled above
            numerators, reference, uncertainty = _ratio_setup(
                histograms_, ratio, stack=stack, uncertainty=ratio_uncertainty
            )
            for index, axis in enumerate(layout.ratio_axes):
                ratios = draw_ratio_panel(
                    numerators,
                    reference,
                    axis,
                    style=st,
                    uncertainty=uncertainty,
                    colors=[color_of.get(id(h), h.color or "black") for h in numerators],
                    ylim=ratio_ylim,
                    ylabel=ratio_label if index == 0 else "",
                )
                if logx:
                    axis.set_xscale("log")
                if outer is not None:
                    axis.set_xlim(*outer)
        layout.xlabel_axes.set_xlabel(x_label, loc="right")
        if any(flow_shown):
            label_flow_bins(
                layout.xlabel_axes, reference_hist.edges, under=flow_shown[0], over=flow_shown[1]
            )
        if segments is not None:
            assert layout.main_right is not None
            apply_xbreak(layout.main, layout.main_right, *segments)
            if layout.ratio is not None and layout.ratio_right is not None:
                apply_xbreak(layout.ratio, layout.ratio_right, *segments)

        finalize_figure(layout.fig, layout.main)  # last: fonts and label anchoring
    result = Plot(
        fig=layout.fig,
        ax=layout.main,
        ratio_ax=layout.ratio,
        ax_right=layout.main_right,
        ratio_ax_right=layout.ratio_right,
        histograms=list(histograms_),
        ratios=ratios,
        variable=variable,
    )
    if save:
        result.save(save)
    return result


# --------------------------------------------------------------------------------------
# 2D plots
# --------------------------------------------------------------------------------------


def plot2d(
    data: Any,
    x: str | Variable,
    y: str | Variable,
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
    save: str | None = None,
) -> Plot:
    """Draw a two-dimensional histogram of ``y`` versus ``x`` for one sample.

    ``x`` and ``y`` must have the same structure (both per-event, or both
    per-object from the same collection). ``bins`` applies to both axes unless
    it is a pair of binning specifications, one per axis (so ``(40, 20)`` is
    two bin counts, never a range; a range needs ``(n, low, high)``); per-axis
    ranges, labels and logarithmic scales (``logx``/``logy`` default to the
    variables' ``log`` flags) are best given through
    :class:`~rootfig.model.Variable` objects.
    """
    sample = _single_sample(data, tree=tree)
    x_bins, y_bins = _split_bins(bins)
    var_x = as_variable(x, bins=x_bins)
    var_y = as_variable(y, bins=y_bins)
    logx = var_x.log if logx is None else logx
    logy = var_y.log if logy is None else logy
    [histogram_] = build_histograms_2d(
        [sample], var_x, var_y, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
    )
    if normalize is not None and normalize is not False:
        histogram_ = _normalize_for_plot(histogram_, normalize)
    resolved_style = _style_for(style, text, lumi)
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
        add_experiment_label(main_ax, st, has_data=sample.is_data)
        finalize_figure(fig, main_ax)  # last: fonts and label anchoring
    result = Plot(fig=fig, ax=main_ax, histograms=[histogram_], variable=var_x)
    if save:
        result.save(save)
    return result


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryTable:
    """Summary statistics for several variables and samples.

    ``str(table)`` gives an aligned text table; :meth:`get` returns a single
    :class:`~rootfig.histograms.Summary`.
    """

    rows: tuple[tuple[str, str, Summary], ...]
    """``(sample label, variable expression, summary)`` triples."""

    def get(self, variable: str, sample: str | None = None) -> Summary:
        """Return the summary for ``variable`` (and ``sample``, if several)."""
        matches = [
            s
            for label, var, s in self.rows
            if var == variable and (sample is None or label == sample)
        ]
        if not matches:
            msg = f"no summary for variable {variable!r}" + (
                f" and sample {sample!r}" if sample else ""
            )
            raise KeyError(msg)
        if len(matches) > 1:
            msg = f"several samples have variable {variable!r}; pass sample=..."
            raise KeyError(msg)
        return matches[0]

    @property
    def samples(self) -> list[str]:
        """Distinct sample labels in order of appearance."""
        return list(dict.fromkeys(label for label, _, _ in self.rows))

    @property
    def variables(self) -> list[str]:
        """Distinct variable expressions in order of appearance."""
        return list(dict.fromkeys(var for _, var, _ in self.rows))

    def __str__(self) -> str:
        multi = len(self.samples) > 1
        entries = [(f"{label}: {var}" if multi else var, s) for label, var, s in self.rows]
        return describe_table(entries)


def summarize(
    data: Any,
    variables: str | Variable | Sequence[str | Variable],
    *,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    label: str | Sequence[str] | None = None,
    nonfinite: NonFinitePolicy = "drop",
) -> SummaryTable:
    """Compute entries, mean, standard deviation, skewness, ... for variables and samples.

    Examples
    --------
    >>> table = rf.summarize(
    ...     "events.root", ["MET", "Muon_pt"], tree="events", selection="nMuon > 0"
    ... )  # doctest: +SKIP
    >>> print(table)  # doctest: +SKIP
    >>> table.get("MET").mean  # doctest: +SKIP
    """
    samples = as_samples(data, tree=tree, labels=label)
    var_list = [variables] if isinstance(variables, str | Variable) else list(variables)
    rows: list[tuple[str, str, Summary]] = []
    for sample in samples:
        # One read per sample: the branches of all variables are fetched together.
        per_variable = load_columns_each(
            sample, var_list, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
        )
        for var, columns in zip(var_list, per_variable, strict=True):
            rows.append((sample.label, as_variable(var).expression, summarize_columns(columns)))
    return SummaryTable(tuple(rows))


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
    ``Plot.matrix``.
    """
    sample = _single_sample(data, tree=tree)
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
        size = figsize or st.figsize
        if size is None:
            side = max(4.5, 0.75 * len(var_list) + 2.5)
            size = (side * 1.15, side)
        layout = make_figure(st, ratio=False, ax=ax, figsize=size)
        fig, main_ax = layout.fig, layout.main
        draw_correlation(
            matrix, tick_labels, main_ax, cmap=cmap, annotate=annotate, percent=percent
        )
        main_ax.set_title(title if title is not None else f"{sample.label}: correlation")
        finalize_figure(fig, main_ax)
    result = Plot(fig=fig, ax=main_ax, matrix=matrix)
    if save:
        result.save(save)
    return result


# --------------------------------------------------------------------------------------
# Cut flows, efficiencies and profiles
# --------------------------------------------------------------------------------------


def cutflow(
    data: Any,
    cuts: Sequence[CutLike],
    *,
    tree: str | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    label: str | Sequence[str] | None = None,
    nonfinite: NonFinitePolicy = "drop",
) -> CutflowTable:
    """Count events and weighted yields after each successive cut, per sample.

    The first row holds all events (after the sample's own selection, if any);
    every further row applies one more cut. Per-object cuts pass an event when
    any object passes. ``weight``, ``lumi`` and ``nonfinite`` work as in
    :func:`plot`: events with a ``nan``/``inf`` weight are excluded from all
    steps with a warning, or raise for ``nonfinite="error"``.

    Examples
    --------
    >>> table = rf.cutflow(
    ...     [sig, bkg], ["nMuon >= 2", rf.Cut("MET > 50", label="MET"), "any(Jet_btag > 0.8)"]
    ... )  # doctest: +SKIP
    >>> print(table)  # doctest: +SKIP
    >>> table.get("Signal").efficiencies  # doctest: +SKIP
    """
    samples = as_samples(data, tree=tree, labels=label)
    return CutflowTable(
        tuple(cutflow_of(s, cuts, weight=weight, lumi=lumi, nonfinite=nonfinite) for s in samples)
    )


def efficiency(
    data: Any,
    variable: str | Variable,
    *,
    passed: CutLike,
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | None = None,
    range: RangeSpec = None,
    label: str | Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    unit: str | None = None,
    title: str | None = None,
    logx: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    legend: bool | str | None = None,
    text: str | Sequence[str] | None = None,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    z: float = 1.0,
    nonfinite: NonFinitePolicy = "drop",
    save: str | None = None,
) -> Plot:
    """Plot the fraction of entries passing ``passed`` as a function of ``variable``.

    For every sample two histograms are filled with the same binning, all
    entries satisfying ``selection`` (the denominator) and those also
    satisfying ``passed`` (the numerator); the ratio is drawn as points with
    Wilson score intervals (``z`` standard deviations, effective entries for
    weighted samples; see :func:`rootfig.histograms.efficiency` for the
    treatment of negative weights). The :class:`~rootfig.histograms.Efficiency`
    objects are returned in ``Plot.efficiencies``.

    Examples
    --------
    >>> rf.efficiency(
    ...     "reco.root", "TrueMuon_pt", passed="TrueMuon_matched", bins=(20, 0, 100)
    ... )  # doctest: +SKIP
    """
    samples = as_samples(data, tree=tree, labels=label)
    var = as_variable(variable, bins=bins, range=range, label=xlabel, unit=unit)
    logx = var.log if logx is None else logx
    totals = build_histograms(
        samples, var, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
    )
    fixed = var.with_(bins=totals[0].axis)  # same binning for the numerators
    pass_cut = as_cut(passed)
    if pass_cut is None:
        msg = "efficiency() needs a 'passed' selection"
        raise ValueError(msg)
    base = as_cut(selection)
    numerator_cut = pass_cut if base is None else base & pass_cut
    passes = build_histograms(
        samples, fixed, selection=numerator_cut, weight=weight, lumi=lumi, nonfinite=nonfinite
    )
    efficiencies = [
        efficiency_of(p.hist, t.hist, z=z, label=sample.label)
        for p, t, sample in zip(passes, totals, samples, strict=True)
    ]
    resolved_style = _style_for(style, text, lumi)
    if legend is not None:
        resolved_style = resolved_style.with_(legend=legend)
    with style_context(resolved_style) as st:
        layout = make_figure(st, ratio=False, ax=ax, figsize=figsize)
        cycle = iter(color_cycle(len(samples), st))
        colors = [s.color or next(cycle) for s in samples]
        low, high = draw_efficiencies(efficiencies, layout.main, style=st, colors=colors)
        outer = xlim or (float(fixed.bins.edges[0]), float(fixed.bins.edges[-1]))  # type: ignore[union-attr]
        finish_axes(
            layout.main,
            data_range=(low, high),
            xlabel=fixed.axis_label,
            ylabel=ylabel or "Efficiency",
            xlim=outer,
            ylim=ylim,
            logx=logx,
            logy=False,
            headroom=1.08,
        )
        if title:
            layout.main.set_title(title)
        add_experiment_label(layout.main, st, has_data=any(s.is_data for s in samples))
        legend_artist = add_legend(layout.main, st)
        if ylim is None or ylim[1] is None:
            edges = efficiencies[0].edges
            heights = np.nanmax(
                np.vstack([np.nan_to_num(e.upper, nan=0.0) for e in efficiencies]), axis=0
            )
            floating = legend_location(st) == "best"
            raise_ylim_above(
                [layout.main],
                overlay_artists(layout.main, None if floating else legend_artist),
                edges=edges,
                heights=heights,
                logy=False,
                floating=[legend_artist] if floating and legend_artist is not None else [],
            )
        finalize_figure(layout.fig, layout.main)  # last: fonts and label anchoring
    result = Plot(
        fig=layout.fig,
        ax=layout.main,
        histograms=list(passes),
        variable=fixed,
        efficiencies=efficiencies,
    )
    if save:
        result.save(save)
    return result


def profile(
    data: Any,
    x: str | Variable,
    y: str | Variable,
    *,
    statistic: ProfileStatistic = "mean",
    tree: str | None = None,
    selection: CutLike | None = None,
    weight: str | None = None,
    lumi: float | str | None = None,
    bins: Bins | None = None,
    range: RangeSpec = None,
    label: str | Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    unit: str | None = None,
    title: str | None = None,
    logx: bool | None = None,
    logy: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    legend: bool | str | None = None,
    text: str | Sequence[str] | None = None,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    nonfinite: NonFinitePolicy = "drop",
    save: str | None = None,
) -> Plot:
    """Plot the mean (or standard deviation) of ``y`` in bins of ``x``, per sample.

    ``statistic="mean"`` draws the weighted mean with its standard error, the
    profile histogram of ROOT; ``"std"`` draws the standard deviation with its
    error, the usual resolution-versus-variable plot (make ``y`` the residual,
    e.g. ``"(reco_pt - true_pt) / true_pt"``). ``x`` and ``y`` must have the same
    structure (both per-event or both per-object of one collection). ``xlabel``
    and ``unit`` describe the x axis; ``logx``/``logy`` default to the
    variables' ``log`` flags. With negative weights a bin whose total weight
    is negative keeps its mean but has no error, and a bin whose weighted
    variance is negative has no standard deviation (``nan``). The
    :class:`~rootfig.histograms.Profile` objects are returned in ``Plot.profiles``.

    Examples
    --------
    >>> rf.profile(
    ...     "reco.root",
    ...     "true_pt",
    ...     "(reco_pt - true_pt) / true_pt",
    ...     statistic="std",
    ...     bins=(20, 0, 100),
    ...     unit="GeV",
    ... )  # doctest: +SKIP
    """
    samples = as_samples(data, tree=tree, labels=label)
    var_x = as_variable(x, bins=bins, range=range, label=xlabel, unit=unit)
    var_y = as_variable(y)
    logx = var_x.log if logx is None else logx
    logy = var_y.log if logy is None else logy
    columns = [
        load_columns(
            s, [var_x, var_y], selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
        )
        for s in samples
    ]
    axis = resolve_axis(var_x, [c.arrays[0] for c in columns], name=var_x.safe_name)
    edges = np.asarray(axis.edges, dtype=float)
    profiles = [
        profile_of(
            c.arrays[0], c.arrays[1], edges, weights=c.weights, statistic=statistic, label=s.label
        )
        for c, s in zip(columns, samples, strict=True)
    ]
    resolved_style = _style_for(style, text, lumi)
    if legend is not None:
        resolved_style = resolved_style.with_(legend=legend)
    if ylabel is None:
        ylabel = var_y.axis_label if statistic == "mean" else f"Std. dev. of {var_y.axis_label}"
    with style_context(resolved_style) as st:
        layout = make_figure(st, ratio=False, ax=ax, figsize=figsize)
        cycle = iter(color_cycle(len(samples), st))
        colors = [s.color or next(cycle) for s in samples]
        low, high = draw_profiles(profiles, layout.main, style=st, colors=colors)
        finish_axes(
            layout.main,
            data_range=(low, high),
            xlabel=var_x.with_(bins=axis).axis_label,
            ylabel=ylabel,
            xlim=xlim or (float(edges[0]), float(edges[-1])),
            ylim=ylim,
            logx=logx,
            logy=logy,
            headroom=1.25,
            ymin_linear="auto",
        )
        if title:
            layout.main.set_title(title)
        add_experiment_label(layout.main, st, has_data=any(s.is_data for s in samples))
        legend_artist = add_legend(layout.main, st)
        if ylim is None or ylim[1] is None:
            heights = np.nanmax(
                np.vstack(
                    [
                        np.nan_to_num(pr.values + np.nan_to_num(pr.errors), nan=-np.inf)
                        for pr in profiles
                    ]
                ),
                axis=0,
            )
            heights = np.where(np.isfinite(heights), heights, 0.0)
            floating = legend_location(st) == "best"
            raise_ylim_above(
                [layout.main],
                overlay_artists(layout.main, None if floating else legend_artist),
                edges=edges,
                heights=heights,
                logy=logy,
                floating=[legend_artist] if floating and legend_artist is not None else [],
            )
        finalize_figure(layout.fig, layout.main)  # last: fonts and label anchoring
    result = Plot(fig=layout.fig, ax=layout.main, variable=var_x, profiles=profiles)
    if save:
        result.save(save)
    return result


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _single_sample(
    data: Any,
    *,
    tree: str | None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> Sample:
    samples = as_samples(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
    if len(samples) != 1:
        msg = f"expected a single sample, got {len(samples)}"
        raise SourceError(msg)
    return samples[0]


def _style_for(
    style: StyleLike, text: str | Sequence[str] | None, lumi: float | str | None = None
) -> Any:
    """Resolve ``style`` and add free text lines and the luminosity used for scaling."""
    resolved = as_style(style)
    if lumi is not None and resolved.lumi is None:
        resolved = resolved.with_(lumi=lumi)
    if text is None:
        return resolved
    existing = list(resolved.text_lines)
    extra = [text] if isinstance(text, str) else list(text)
    return resolved.with_(text=[*existing, *extra])


def _significance_spec(ratio: RatioSpec) -> tuple[SignificanceKind, str | None] | None:
    """``(kind, signal label)`` when ``ratio`` asks for a significance panel, else ``None``."""
    if isinstance(ratio, tuple):
        kind, label = ratio
        if kind not in SIGNIFICANCE_KINDS:
            msg = f"ratio=({kind!r}, ...) must use one of {SIGNIFICANCE_KINDS}"
            raise ValueError(msg)
        return ("s/sqrt(b)" if kind == "significance" else kind, label)  # type: ignore[return-value]
    if isinstance(ratio, str) and ratio in SIGNIFICANCE_KINDS:
        return ("s/sqrt(b)" if ratio == "significance" else ratio, None)  # type: ignore[return-value]
    return None


def _significance_setup(
    hists: Sequence[Histogram], signal_label: str | None
) -> tuple[Histogram, Histogram]:
    """Pick the signal histogram and sum the other non-data ones into the background."""
    mc = [h for h in hists if not h.is_data]
    if len(mc) < 2:
        msg = "a significance panel needs at least two non-data histograms (signal and background)"
        raise ValueError(msg)
    if signal_label is None:
        signal = mc[-1]
    else:
        matches = [h for h in mc if h.label == signal_label]
        if not matches:
            msg = f"signal {signal_label!r} is not one of {[h.label for h in mc]}"
            raise ValueError(msg)
        signal = matches[0]
    others = [h for h in mc if h is not signal]
    total = others[0].hist.copy()
    for h in others[1:]:
        total = total + h.hist
    return signal, Histogram(total, label="Background", normalization=others[0].normalization)


RatioSpec = bool | str | tuple[str, str]
"""What ``ratio=`` accepts: a flag, a reference label, a significance kind, or (kind, signal)."""


def _wrap_hists(hists: Sequence[Histogram | Hist], labels: Sequence[str] | None) -> list[Histogram]:
    if labels is not None and len(labels) != len(hists):
        msg = f"got {len(labels)} labels for {len(hists)} histograms"
        raise ValueError(msg)
    wrapped: list[Histogram] = []
    for index, item in enumerate(hists):
        if isinstance(item, Histogram):
            wrapped.append(item if labels is None else item.with_(label=labels[index]))
            continue
        if labels is not None:
            label = labels[index]
        else:
            axis_name = item.axes[0].name if item.ndim == 1 else ""
            label = axis_name or f"hist {index + 1}"
        wrapped.append(Histogram(as_weight_storage(item), label=str(label)))
    return wrapped


def _normalize_for_plot(histogram_: Histogram, spec: NormalizeSpec) -> Histogram:
    if histogram_.normalization is not None:
        warnings.warn(
            f"histogram {histogram_.label!r} is already normalised ({histogram_.normalization}); "
            "normalising again",
            RootfigWarning,
            stacklevel=3,
        )
    return normalize_histogram(histogram_, spec)


def _ratio_setup(
    hists: Sequence[Histogram],
    ratio: bool | str,
    *,
    stack: bool,
    uncertainty: RatioUncertainty | None,
) -> tuple[list[Histogram], Histogram, RatioUncertainty]:
    data = [h for h in hists if h.is_data]
    mc = [h for h in hists if not h.is_data]
    if isinstance(ratio, str):
        matches = [h for h in hists if h.label == ratio]
        if not matches:
            msg = f"ratio reference {ratio!r} is not one of {[h.label for h in hists]}"
            raise ValueError(msg)
        reference = matches[0]
        numerators = [h for h in hists if h is not reference]
        default_uncertainty: RatioUncertainty = (
            "numerator" if data and reference in mc else "propagate"
        )
        return numerators, reference, uncertainty or default_uncertainty
    if stack and mc:
        total = mc[0].hist.copy()
        for h in mc[1:]:
            total = total + h.hist
        reference = Histogram(total, label="Total", normalization=mc[0].normalization)
        numerators = data if data else []
        if not numerators:
            msg = "ratio=True with stack=True needs an observed data sample (observed=...)"
            raise ValueError(msg)
        return numerators, reference, uncertainty or "numerator"
    if data and mc:
        return data, mc[0], uncertainty or "numerator"
    if len(hists) < 2:
        msg = "a ratio panel needs at least two histograms"
        raise ValueError(msg)
    return list(hists[1:]), hists[0], uncertainty or "propagate"


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
