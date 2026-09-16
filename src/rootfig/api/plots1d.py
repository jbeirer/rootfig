"""One-dimensional plots: :func:`plot`, from trees, stored histograms or histogram objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rootfig.api._common import normalize_for_plot, style_for
from rootfig.api._hists import (
    histogram_objects,
    rebin_ready_made,
    reject_fill_options,
    unit_of,
    wrap_histograms,
)
from rootfig.histograms import (
    SIGNIFICANCE_KINDS,
    Histogram,
    NormalizeSpec,
    RatioUncertainty,
    SignificanceKind,
    build_histograms,
    significance,
    sum_histograms,
)
from rootfig.model import (
    Bins,
    CutLike,
    RangeSpec,
    StyleLike,
    SystematicLike,
    Variable,
    as_samples,
    as_style,
    as_variable,
)
from rootfig.model.samples import HistType
from rootfig.plotting import (
    AxesLike,
    FlowSpec,
    Plot,
    add_experiment_label,
    add_legend,
    add_stats_box,
    align_experiment_label,
    apply_xbreak,
    break_segments,
    draw_histograms,
    draw_ratio_panel,
    draw_significance_panel,
    envelope,
    finalize_figure,
    finish_axes,
    fold_flow_bins,
    foreground,
    label_flow_bins,
    legend_location,
    make_figure,
    overlay_artists,
    raise_ylim_above,
    show_flow_bins,
    style_context,
    ylabel_for,
)
from rootfig.selection import NonFinitePolicy

__all__ = ["plot"]

RatioSpec = bool | str | tuple[str, str]
"""What ``ratio=`` accepts: a flag, a reference label, a significance kind, or (kind, signal)."""


def plot(
    data: Any,
    variable: str | Variable | None = None,
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
    systematics: Mapping[str, SystematicLike] | None = None,
    assume_poisson: bool = False,
    save: str | None = None,
) -> Plot:
    """Histogram a variable from one or more samples and draw it.

    This is the ``TTree::Draw`` of rootfig: read only the branches needed,
    evaluate the expressions, apply the selection with the documented
    per-event/per-object semantics, fill ``hist.Hist`` objects with a binning
    shared by all samples, and render them with mplhep.

    The same call draws histograms that already exist. A ``variable`` that is a
    bare name and addresses a ``TH1`` stored in the files (a histogram written by
    an analysis framework) is read instead of filled, summed over each sample's
    files and scaled like a filled one; an explicit ``tree=`` always means a
    branch. Histogram objects (``hist.Hist`` or
    :class:`~rootfig.histograms.Histogram`, one or a list) passed as ``data``
    are drawn as they are, with ``label`` naming them and ``variable`` optional.
    Options that act on event data (``selection``, ``weight``, ``range``, ...)
    raise for both.

    Parameters
    ----------
    data
        What to plot: a file path or glob, ``"path:tree"``, a list of those (one
        sample each), a ``{label: files}`` mapping, one or more
        :class:`~rootfig.model.Sample` objects, in-memory arrays (a mapping of
        arrays or an Awkward record array), or histogram objects.
    variable
        Branch name or expression (see :mod:`rootfig.expressions`), the name of
        a histogram stored in the files, or a :class:`~rootfig.model.Variable`
        carrying binning and labels. Optional for histogram objects, where it
        only supplies the labels, unit and ``log`` flag.
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
        bin edges, or a ``hist`` axis. Overrides the ``Variable``'s binning. A
        histogram that already exists (stored or object) keeps its binning
        unless an ``int`` asks for fewer bins, which must divide its count.
    range
        Range for integer ``bins``: ``(low, high)``, ``"robust"`` (the default)
        or ``"auto"``. ``"robust"`` ignores values far from the bulk of the data,
        so sentinels such as ``-999`` do not set the axis, and cuts the thin end
        of a tail so a long one does not leave the rest of the distribution in a
        corner of the axis. Nothing is dropped: those values land in the
        under/overflow, which ``flow`` shows. Use ``"auto"`` for the full finite
        minimum and maximum.
    label
        Legend label(s) for samples given as plain files or as histogram objects.
    observed
        A sample of observed data (or the file(s) for one; histogram objects
        when ``data`` are) drawn as points, excluded from stacks and used as
        numerator of the ratio.
    xlabel, ylabel, unit, title
        Axis labels; defaults come from the variable (or the stored axis title),
        the normalisation and the bin width (``Events / 2 GeV``).
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
        (``"propagate"`` or ``"numerator"`` with a reference band; by default
        data uses ``"numerator"`` and simulation ``"propagate"``, so systematic
        sources shared with the reference cancel). The label is
        shrunk, and if needed wrapped onto two lines, to fit the short panel;
        pass a shorter ``ratio_label`` (``"Ratio"``) to keep it at full size.
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
        ``True`` or a location string. Needs the unbinned statistics collected
        while filling, which stored histograms and histogram objects lack.
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
    systematics
        Sources of systematic uncertainty applied to every simulated sample, in
        the forms ``Sample(systematics=...)`` accepts (a sample's own source of
        the same name takes precedence), e.g. ``{"lumi": 0.017}``. Stacks draw
        the statistical and systematic uncertainty as one band, overlaid samples
        with variations a light band in their colour, and the ratio panel
        includes them in its band and error bars. Sources of the same name are
        fully correlated across samples, different ones added in quadrature;
        ``Plot.uncertainty()`` returns the components. Histogram objects carry
        theirs in :attr:`~rootfig.histograms.Histogram.variations`.
    assume_poisson
        Accept a histogram (stored or object) with a plain count storage that
        was filled with weights or rescaled, so ``hist`` reports no variances:
        the absolute bin contents are used instead (a warning says so). Fill
        with ``hist.storage.Weight()`` to keep the real uncertainties.
    save
        Path to save the figure to (also returned in the :class:`Plot`).

    Returns
    -------
    Plot
        The figure, axes, histograms and ratios.
    """
    objects = histogram_objects(data)
    if objects is not None:
        reject_fill_options(
            "histogram objects",
            tree=tree,
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
            systematics=systematics,
        )
        var = (
            None
            if variable is None
            else as_variable(variable, bins=bins, range=range, label=xlabel, unit=unit)
        )
        hists = wrap_histograms(objects, label, assume_poisson=assume_poisson)
        if observed is not None:
            observed_objects = histogram_objects(observed)
            if observed_objects is None:
                msg = "observed= must be histogram objects when data are histogram objects"
                raise TypeError(msg)
            hists += wrap_histograms(observed_objects, assume_poisson=assume_poisson, is_data=True)
        hists = rebin_ready_made(
            hists,
            [bins if var is None else var.bins],
            range_=range if var is None else var.range,
        )
    else:
        if variable is None:
            msg = "plot() needs a variable (a branch, expression or stored histogram name)"
            raise TypeError(msg)
        samples = as_samples(data, tree=tree, labels=label)
        if observed is not None:
            observed_samples = [
                s if s.is_data else s.replace(is_data=True) for s in as_samples(observed, tree=tree)
            ]
            samples = [*samples, *observed_samples]
        var = as_variable(variable, bins=bins, range=range, label=xlabel, unit=unit)
        hists = build_histograms(
            samples,
            var,
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
            systematics=systematics,
            assume_poisson=assume_poisson,
        )
    return _draw(
        hists,
        variable=var,
        xlabel=xlabel,
        unit=unit,
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
        style=style_for(style, text, lumi),
        figsize=figsize,
        ax=ax,
        save=save,
    )


def _draw(
    histograms_: list[Histogram],
    *,
    variable: Variable | None,
    xlabel: str | None,
    unit: str | None,
    ylabel: str | None,
    title: str | None,
    normalize: NormalizeSpec,
    stack: bool,
    ratio: RatioSpec,
    ratio_ylim: tuple[float, float] | None,
    ratio_label: str | None,
    ratio_uncertainty: RatioUncertainty | None,
    logx: bool | None,
    logy: bool,
    flow: FlowSpec,
    histtype: HistType | None,
    errorbars: bool | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float | None, float | None] | None,
    xbreak: tuple[float, float] | None,
    legend: bool | str | None,
    stats: bool | str,
    style: StyleLike,
    figsize: tuple[float, float] | None,
    ax: AxesLike,
    save: str | None,
) -> Plot:
    """Draw filled :class:`Histogram` objects; see :func:`plot` for the options."""
    if logx is None:
        logx = variable.log if variable is not None else False
    if not histograms_:
        msg = "no histograms to draw"
        raise ValueError(msg)
    if any(h.ndim != 1 for h in histograms_):
        msg = "plot() draws one-dimensional histograms; use plot2d() for 2D"
        raise ValueError(msg)
    if stats and all(h.stats is None for h in histograms_):
        msg = (
            "stats= needs the unbinned statistics collected while filling from event data; "
            "stored histograms and histogram objects have none"
        )
        raise ValueError(msg)
    if normalize is not None and normalize is not False:
        histograms_ = [normalize_for_plot(h, normalize) for h in histograms_]
    resolved_style = as_style(style)
    if legend is not None:
        resolved_style = resolved_style.replace(legend=legend)

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
        add_experiment_label(layout.main, st, has_data=has_data, right=layout.main_right)

        per_object = any(h.stats is not None and h.stats.per_object for h in histograms_)
        x_label, bin_unit = _axis_labels(xlabel, unit, variable, reference_hist)
        y_label = ylabel or ylabel_for(
            normalization=histograms_[0].normalization,
            unit=bin_unit,
            widths=label_widths,
            per_object=per_object,
        )
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
                    colors=[color_of.get(id(h), h.color or foreground()) for h in numerators],
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

        finalize_figure(layout.fig, panels=layout.ratio_axes)  # last: fonts, panel labels
    # outside the style context, against the layout the figure is drawn with
    align_experiment_label(layout.main, right=layout.main_right)
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


def _axis_labels(
    xlabel: str | None, unit: str | None, variable: Variable | None, reference: Histogram
) -> tuple[str, str | None]:
    """Return the x label and the unit quoted in the bin-width y label.

    The label is the explicit ``xlabel``, else the variable's own, else the
    histogram's axis label (a stored title or the expression). The unit comes
    from ``unit``, the variable or a ``[unit]`` the label ends with, and is
    appended to the label once.
    """
    if xlabel:
        base = xlabel
    elif variable is not None and variable.label is not None:
        base = variable.label
    else:
        base = str(reference.axis.label or "") or (variable.expression if variable else "")
    unit = unit or (variable.unit if variable is not None else None)
    if unit and not base.endswith(f"[{unit}]"):
        base = f"{base} [{unit}]"
    return base, unit or unit_of(base)


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


def _ratio_setup(
    hists: Sequence[Histogram],
    ratio: bool | str,
    *,
    stack: bool,
    uncertainty: RatioUncertainty | None,
) -> tuple[list[Histogram], Histogram, RatioUncertainty | list[RatioUncertainty]]:
    data = [h for h in hists if h.is_data]
    mc = [h for h in hists if not h.is_data]
    if isinstance(ratio, str):
        matches = [h for h in hists if h.label == ratio]
        if not matches:
            msg = f"ratio reference {ratio!r} is not one of {[h.label for h in hists]}"
            raise ValueError(msg)
        reference = matches[0]
        numerators = [h for h in hists if h is not reference]
        if uncertainty is not None:
            return numerators, reference, uncertainty
        # chosen per numerator: data over simulation keeps the reference as a band, while
        # simulation over simulation propagates both, so shared systematic sources cancel
        per_numerator: list[RatioUncertainty] = [
            "numerator" if h.is_data and not reference.is_data else "propagate" for h in numerators
        ]
        return numerators, reference, per_numerator
    if stack and mc:
        reference = sum_histograms(mc)
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
