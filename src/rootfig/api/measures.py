"""Efficiencies and profiles in bins of a variable: :func:`efficiency`, :func:`profile`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np

from rootfig.api._common import style_for
from rootfig.api._panel import PanelPlan, resolve_points
from rootfig.histograms import (
    ONE_SIGMA,
    Comparison,
    ComparisonKind,
    Efficiency,
    EfficiencyInterval,
    Profile,
    ProfileStatistic,
    fill,
    from_sample,
    load_columns,
    negative_bins,
    summarize,
)
from rootfig.histograms import efficiency as efficiency_of
from rootfig.histograms import profile as profile_of
from rootfig.model import (
    Bins,
    CutLike,
    RangeSpec,
    StyleLike,
    Variable,
    as_cut,
    as_samples,
    as_variable,
    resolve_axis,
)
from rootfig.plotting import (
    AxesLike,
    Finish,
    Layout,
    Plot,
    add_experiment_label,
    add_legend,
    color_cycle,
    draw_efficiencies,
    draw_panel,
    draw_profiles,
    finish_axes,
    finish_figure,
    legend_location,
    make_figure,
    overlay_artists,
    pin_fonts,
    raise_ylim_above,
    style_context,
)
from rootfig.selection import NonFinitePolicy

__all__ = ["efficiency", "profile"]


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
    panel: ComparisonKind | None = None,
    reference: str | None = None,
    panel_ylim: tuple[float, float] | None = None,
    panel_label: str | None = None,
    legend: bool | str | None = None,
    text: str | Sequence[str] | None = None,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    cl: float = ONE_SIGMA,
    interval: EfficiencyInterval = "auto",
    show_empty: bool = False,
    nonfinite: NonFinitePolicy = "drop",
    save: str | None = None,
) -> Plot:
    """Plot the fraction of entries passing ``passed`` as a function of ``variable``.

    For every sample two histograms are filled with the same binning, all
    entries satisfying ``selection`` (the denominator) and those also
    satisfying ``passed`` (the numerator); the ratio is drawn as points with
    confidence intervals of confidence level ``cl`` (one standard deviation,
    68.27 %, by default, as ROOT's). ``interval="auto"`` gives what ROOT's
    ``TEfficiency`` gives: Clopper-Pearson for unweighted entries, the normal
    approximation for weighted ones. The sample's ``scale`` and luminosity
    factor cancel in an efficiency and are left out, so an unweighted sample
    stays unweighted. ``interval`` names another method, as ROOT means it:
    ``"clopper-pearson"``, ``"normal"``, ``"wilson"`` and
    ``"agresti-coull"`` (all but the normal approximation for unweighted
    entries only), the Bayesian
    ``"jeffreys"`` and ``"uniform"`` or any ``rf.Bayesian(alpha, beta, mode=,
    shortest=)`` prior (weighted entries too; the efficiency is then the
    posterior's mean or mode), and ``"wilson-effective"``, the Wilson interval
    of the effective entries for weighted samples (see
    :data:`~rootfig.histograms.binomial.EfficiencyInterval`). A binomial
    interval (any but the normal approximation) is not drawn, with a warning, for
    a bin that an entry with a negative weight falls into, which keeps its
    efficiency. Empty bins are left out; ``show_empty=True`` draws them as
    ROOT's ``"e0"`` does (0 in ``[0, 1]``, or the prior for a Bayesian interval),
    while a bin whose signed weights cancel holds entries and stays undefined.
    The :class:`~rootfig.histograms.Efficiency`
    objects are returned in ``Plot.efficiencies``. An integer ``bins`` without a
    ``range`` infers one robustly, shared by numerator and denominator (see
    :func:`plot`).

    ``panel`` compares the efficiencies in a lower panel (``"ratio"`` for a
    scale factor, ``"relative_difference"``, ``"difference"``, ``"pull"``,
    ``"asymmetry"``): data over the first simulated sample, or every further
    sample over the first, or every other one over the sample ``reference``
    names; the intervals are propagated as independent, keeping their
    asymmetry. ``panel_ylim`` and ``panel_label`` set its range and y label, and
    ``Plot.comparisons`` holds the :class:`~rootfig.histograms.Comparison` objects.

    Examples
    --------
    >>> rf.efficiency(
    ...     "reco.root", "TrueMuon_pt", passed="TrueMuon_matched", bins=(20, 0, 100)
    ... )  # doctest: +SKIP
    """
    # efficiencies are statistical only: load_columns reads no systematic variations
    samples = as_samples(data, tree=tree, labels=label)
    var = as_variable(variable, bins=bins, range=range, label=xlabel, unit=unit)
    logx = var.log if logx is None else logx
    pass_cut = as_cut(passed)
    if pass_cut is None:
        msg = "efficiency() needs a 'passed' selection"
        raise ValueError(msg)
    base = as_cut(selection)
    numerator_cut = pass_cut if base is None else base & pass_cut
    # the sample's scale and luminosity factor cancel in a ratio: left out, unweighted entries
    # stay unweighted, and a zero or negative factor changes nothing
    options: dict[str, Any] = {"weight": weight, "nonfinite": nonfinite, "scaled": False}
    totals = [load_columns(s, [var], selection=selection, **options) for s in samples]
    axis = resolve_axis(
        var, [c.values for c in totals], name=var.safe_name, weights=[c.weights for c in totals]
    )
    fixed = var.replace(bins=axis)  # the same binning for the numerators
    passing = [load_columns(s, [fixed], selection=numerator_cut, **options) for s in samples]
    pass_hists = [fill([axis], c) for c in passing]
    efficiencies = [
        efficiency_of(
            h,
            fill([axis], t),
            cl=cl,
            label=sample.label,
            negative_weights=negative_bins(axis, t),  # which the sums cannot always tell
            interval=interval,
            show_empty=show_empty,
        )
        for h, t, sample in zip(pass_hists, totals, samples, strict=True)
    ]
    # the numerators returned in Plot.histograms are yields, with the factor like any histogram
    passes = [
        from_sample(s, h, stats=summarize(c), per_object=c.per_object, weighted=c.weighted).scaled(
            s.scale * s.lumi_scale(lumi)
        )
        for s, h, c in zip(samples, pass_hists, passing, strict=True)
    ]
    plan = resolve_points(
        efficiencies,
        [s.is_data for s in samples],
        panel=panel,
        reference=reference,
        label=panel_label,
    )
    resolved_style = style_for(style, text, lumi)
    if legend is not None:
        resolved_style = resolved_style.replace(legend=legend)
    with style_context(resolved_style) as st:
        layout = make_figure(st, panel=plan is not None, ax=ax, figsize=figsize)
        cycle = iter(color_cycle(len(samples), st))
        colors = [s.color or next(cycle) for s in samples]
        low, high = draw_efficiencies(efficiencies, layout.main, style=st, colors=colors)
        outer = xlim or (float(fixed.bins.edges[0]), float(fixed.bins.edges[-1]))  # type: ignore[union-attr]
        finish_axes(
            layout.main,
            data_range=(low, high),
            xlabel=None,
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
        comparisons = _draw_panel(
            layout, plan, efficiencies, colors, outer=outer, logx=logx, ylim=panel_ylim
        )
        layout.xlabel_axes.set_xlabel(fixed.axis_label, loc="right")
        headroom = None
        if ylim is None or ylim[1] is None:
            edges = efficiencies[0].edges
            heights = np.nanmax(
                np.vstack([np.nan_to_num(e.upper, nan=0.0) for e in efficiencies]), axis=0
            )
            floating = legend_location(st) == "best"
            headroom = partial(
                raise_ylim_above,
                [layout.main],
                overlay_artists(layout.main, None if floating else legend_artist),
                edges=edges,
                heights=heights,
                logy=False,
                floating=[legend_artist] if floating and legend_artist is not None else [],
            )
        pin_fonts(layout.fig)  # last: fonts
    # outside the style context, against the layout the figure is drawn with
    finish_figure(layout.fig, [_finish(layout, headroom)])
    result = Plot(
        fig=layout.fig,
        ax=layout.main,
        panel_ax=layout.panel,
        histograms=list(passes),
        comparisons=comparisons,
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
    panel: ComparisonKind | None = None,
    reference: str | None = None,
    panel_ylim: tuple[float, float] | None = None,
    panel_label: str | None = None,
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
    variance is negative has no standard deviation (``nan``). An integer
    ``bins`` without a ``range`` infers the x range robustly (see :func:`plot`).
    The :class:`~rootfig.histograms.Profile` objects are returned in
    ``Plot.profiles``. ``panel``, ``reference``, ``panel_ylim`` and
    ``panel_label`` add a lower panel comparing the profiles, as for
    :func:`efficiency` (``panel="difference"`` compares the response or
    resolution of two configurations).

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
    axis = resolve_axis(
        var_x,
        [c.arrays[0] for c in columns],
        name=var_x.safe_name,
        weights=[c.weights for c in columns],
    )
    edges = np.asarray(axis.edges, dtype=float)
    profiles = [
        profile_of(
            c.arrays[0], c.arrays[1], edges, weights=c.weights, statistic=statistic, label=s.label
        )
        for c, s in zip(columns, samples, strict=True)
    ]
    plan = resolve_points(
        profiles, [s.is_data for s in samples], panel=panel, reference=reference, label=panel_label
    )
    resolved_style = style_for(style, text, lumi)
    if legend is not None:
        resolved_style = resolved_style.replace(legend=legend)
    if ylabel is None:
        ylabel = var_y.axis_label if statistic == "mean" else f"Std. dev. of {var_y.axis_label}"
    with style_context(resolved_style) as st:
        layout = make_figure(st, panel=plan is not None, ax=ax, figsize=figsize)
        cycle = iter(color_cycle(len(samples), st))
        colors = [s.color or next(cycle) for s in samples]
        low, high = draw_profiles(profiles, layout.main, style=st, colors=colors)
        outer = xlim or (float(edges[0]), float(edges[-1]))
        finish_axes(
            layout.main,
            data_range=(low, high),
            xlabel=None,
            ylabel=ylabel,
            xlim=outer,
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
        comparisons = _draw_panel(
            layout, plan, profiles, colors, outer=outer, logx=logx, ylim=panel_ylim
        )
        layout.xlabel_axes.set_xlabel(var_x.replace(bins=axis).axis_label, loc="right")
        headroom = None
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
            headroom = partial(
                raise_ylim_above,
                [layout.main],
                overlay_artists(layout.main, None if floating else legend_artist),
                edges=edges,
                heights=heights,
                logy=logy,
                floating=[legend_artist] if floating and legend_artist is not None else [],
            )
        pin_fonts(layout.fig)  # last: fonts
    # outside the style context, against the layout the figure is drawn with
    finish_figure(layout.fig, [_finish(layout, headroom)])
    result = Plot(
        fig=layout.fig,
        ax=layout.main,
        panel_ax=layout.panel,
        comparisons=comparisons,
        variable=var_x,
        profiles=profiles,
    )
    if save:
        result.save(save)
    return result


def _draw_panel(
    layout: Layout,
    plan: PanelPlan | None,
    points: Sequence[Efficiency] | Sequence[Profile],
    colors: Sequence[str],
    *,
    outer: tuple[float, float],
    logx: bool,
    ylim: tuple[float, float] | None,
) -> list[Comparison]:
    """Draw the lower panel of ``plan`` into ``layout``; return its comparisons."""
    if plan is None or layout.panel is None:
        return []
    comparisons = plan.comparisons()
    color_of = dict(zip(map(id, points), colors, strict=True))
    draw_panel(
        comparisons,
        layout.panel,
        colors=[color_of[id(p)] for p in plan.numerators],
        observed=plan.observed,
        ylim=ylim,
        ylabel=plan.label,
        view=[outer],
    )
    if logx:
        layout.panel.set_xscale("log")
    layout.panel.set_xlim(*outer)
    return comparisons


def _finish(layout: Layout, headroom: Callable[[], None] | None) -> Finish:
    """Return what an efficiency or profile plot does once laid out."""
    return Finish(
        layout.main, panels=layout.panel_axes, xlabel=layout.xlabel_axes, headroom=headroom
    )
