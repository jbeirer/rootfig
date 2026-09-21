"""Efficiencies and profiles in bins of a variable: :func:`efficiency`, :func:`profile`."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from rootfig.api._common import style_for
from rootfig.histograms import (
    ProfileStatistic,
    build_histograms,
    load_columns,
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
    Plot,
    add_experiment_label,
    add_legend,
    color_cycle,
    draw_efficiencies,
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
    objects are returned in ``Plot.efficiencies``. An integer ``bins`` without a
    ``range`` infers one robustly, shared by numerator and denominator (see
    :func:`plot`).

    Examples
    --------
    >>> rf.efficiency(
    ...     "reco.root", "TrueMuon_pt", passed="TrueMuon_matched", bins=(20, 0, 100)
    ... )  # doctest: +SKIP
    """
    # efficiencies are statistical only: systematics are neither evaluated nor needed
    samples = [s.replace(systematics={}) for s in as_samples(data, tree=tree, labels=label)]
    var = as_variable(variable, bins=bins, range=range, label=xlabel, unit=unit)
    logx = var.log if logx is None else logx
    totals = build_histograms(
        samples, var, selection=selection, weight=weight, lumi=lumi, nonfinite=nonfinite
    )
    fixed = var.replace(bins=totals[0].axis)  # same binning for the numerators
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
    resolved_style = style_for(style, text, lumi)
    if legend is not None:
        resolved_style = resolved_style.replace(legend=legend)
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
    finish_figure(layout.fig, [Finish(layout.main, xlabel=layout.main, headroom=headroom)])
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
    variance is negative has no standard deviation (``nan``). An integer
    ``bins`` without a ``range`` infers the x range robustly (see :func:`plot`).
    The :class:`~rootfig.histograms.Profile` objects are returned in
    ``Plot.profiles``.

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
    resolved_style = style_for(style, text, lumi)
    if legend is not None:
        resolved_style = resolved_style.replace(legend=legend)
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
            xlabel=var_x.replace(bins=axis).axis_label,
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
    finish_figure(layout.fig, [Finish(layout.main, xlabel=layout.main, headroom=headroom)])
    result = Plot(fig=layout.fig, ax=layout.main, variable=var_x, profiles=profiles)
    if save:
        result.save(save)
    return result
