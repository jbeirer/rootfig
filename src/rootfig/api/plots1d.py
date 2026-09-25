"""One-dimensional plots: :func:`plot`, from trees, stored histograms or histogram objects.

:func:`plot` is :func:`prepare_plot` (read or fill the histograms) followed by
:func:`draw_plot` (render them). :class:`~rootfig.PlotBook` calls the two apart
to prepare once and draw several variants, with :func:`prefetch_plots` reading
the branches of several variables together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any

from matplotlib.gridspec import SubplotSpec

from rootfig.api._common import as_observed, normalize_for_plot, style_for, with_data_errors
from rootfig.api._hists import (
    histogram_objects,
    rebin_ready_made,
    reject_fill_options,
    require_dimension,
    unit_of,
    wrap_histograms,
)
from rootfig.api._panel import resolve as resolve_panel
from rootfig.histograms import (
    ComparisonKind,
    DataErrors,
    Histogram,
    NormalizeSpec,
    NormalizeUncertainty,
    ReadPlan,
    UncertaintyMode,
    build_histograms,
)
from rootfig.io import ReadCache
from rootfig.model import (
    Bins,
    CutLike,
    PlotItem,
    RangeSpec,
    StyleLike,
    SystematicLike,
    Variable,
    as_plot_items,
    as_variable,
)
from rootfig.model.samples import HistType
from rootfig.plotting import (
    AxesLike,
    Finish,
    FlowSpec,
    Plot,
    StackSpec,
    add_experiment_label,
    add_legend,
    add_stats_box,
    apply_xbreak,
    break_segments,
    draw_histograms,
    draw_panel,
    envelope,
    finish_axes,
    finish_figure,
    fold_flow_bins,
    label_flow_bins,
    legend_location,
    make_figure,
    overlay_artists,
    pin_fonts,
    raise_ylim_above,
    require_same_binning,
    show_flow_bins,
    split_stack,
    style_context,
    ylabel_for,
)
from rootfig.selection import NonFinitePolicy

__all__ = ["plot"]


@dataclass(frozen=True)
class PreparedPlot:
    """The histograms :func:`plot` draws and what it was asked to label them with.

    Attributes
    ----------
    histograms
        One :class:`~rootfig.histograms.Histogram` per sample or group, observed
        data last, as :func:`~rootfig.histograms.build_histograms` returns them,
        or the histogram objects given as ``data``.
    variable
        The :class:`~rootfig.model.Variable` with the ``bins``, ``range``,
        ``xlabel`` and ``unit`` options applied; ``None`` for histogram objects
        drawn without one.
    xlabel, unit
        As given to :func:`plot`; they label histogram objects drawn without a
        variable.
    lumi
        The luminosity the samples were scaled to, written into the experiment
        label.
    """

    histograms: list[Histogram]
    variable: Variable | None
    xlabel: str | None = None
    unit: str | None = None
    lumi: float | str | None = None


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
    normalize_uncertainty: NormalizeUncertainty = "scale",
    stack: StackSpec = False,
    panel: ComparisonKind | None = None,
    reference: str | None = None,
    panel_ylim: tuple[float, float] | None = None,
    panel_label: str | None = None,
    panel_uncertainty: UncertaintyMode | None = None,
    logx: bool | None = None,
    logy: bool = False,
    flow: FlowSpec = "hint",
    histtype: HistType | None = None,
    errorbars: bool | None = None,
    data_errors: DataErrors | None = None,
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
    variances_from_contents: bool = False,
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
    Options that need event data, such as ``selection`` and ``weight``, raise
    for both. Range inference (``"auto"``, ``"robust"``) is a no-op for them; an
    explicit range crops their axis, moving the rest into flow bins.

    Parameters
    ----------
    data
        What to plot: a file path or glob, ``"path:tree"``, a list of those (one
        sample each), a ``{label: files}`` mapping, one or more
        :class:`~rootfig.model.Sample` objects, :class:`~rootfig.model.Group`
        objects (several samples drawn as one histogram), in-memory arrays (a
        mapping of arrays or an Awkward record array), or histogram objects.
    variable
        Branch name or expression (see :mod:`rootfig.expressions`), the name of
        a histogram stored in the files, or a :class:`~rootfig.model.Variable`
        carrying binning and labels. Optional for histogram objects, where it
        supplies the labels, unit and ``log`` flag and, through ``bins`` and
        ``range``, a binning to crop and merge them to.
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
        bin edges, or a ``hist.axis.Regular``/``Variable`` axis. Overrides the
        ``Variable``'s binning. A
        histogram that already exists (stored or object) keeps its binning
        unless asked to crop or merge: an ``int`` must divide its count, and
        explicit edges must coincide with its own (a ``Variable`` written for
        the tree describes the histogram filled from it) and merge the bins
        between them. Content outside the requested edges joins the flow bins.
    range
        Range for integer ``bins``: ``(low, high)``, ``"robust"`` (the default)
        or ``"auto"``. ``"robust"`` ignores values far from the bulk of the data,
        so sentinels such as ``-999`` do not set the axis, and cuts the thin end
        of a tail so a long one does not leave the rest of the distribution in a
        corner of the axis. Nothing is dropped: those values land in the
        under/overflow, which ``flow`` shows. Use ``"auto"`` for the full finite
        minimum and maximum. For existing histograms, an explicit range without
        bins crops to its ends, which must be existing edges, and keeps the bins
        between them.
    label
        Legend label(s) for samples given as plain files or as histogram objects.
    observed
        A sample or group of observed data (or the file(s) for one; histogram
        objects when ``data`` are) drawn as points, excluded from stacks and,
        without ``reference=``, compared with the prediction in a ratio,
        relative difference, difference or pull panel.
    xlabel, ylabel, unit, title
        Axis labels; defaults come from the variable (or the stored axis title),
        the normalisation and the bin width (``Events / 2 GeV``).
    normalize
        ``True``/``"unity"`` (sum to one), ``"density"``, ``"width"`` (divide by
        bin width) or a number to normalise to. With any stacked histograms,
        only ``None``, ``False`` and ``"width"`` are allowed: normalising each
        component separately would not give a normalised total.
    normalize_uncertainty
        What normalising to a histogram's own total (``True``, ``"unity"``,
        ``"density"``, a number) does to its statistical uncertainty:
        ``"scale"`` keeps every bin's relative uncertainty, as ``TH1::Scale``;
        ``"shape"`` lets the total fluctuate with the bins (the uncertainty of
        a shape: a binomial fraction, Clopper-Pearson for counts; see
        :data:`~rootfig.histograms.normalize.NormalizeUncertainty`).
    stack
        ``True`` stacks every non-data histogram, ``False`` or ``[]`` overlays
        all. A legend label or sequence of labels stacks every histogram carrying
        those labels, with the rest overlaid. A group is selected by its own
        label. Unknown labels and data-only labels raise ``ValueError``.
        Stacks are filled in input order, first at the bottom, followed by the
        band, overlays in input order, then data. A histogram keeps its colour
        whether it is stacked or overlaid; explicit colours do not use up
        entries in the colour cycle.
    panel
        What a lower panel shows: ``"ratio"``, ``"relative_difference"``,
        ``"difference"``, ``"pull"``, ``"asymmetry"``, ``"s/sqrt(b)"`` or
        ``"s/sqrt(s+b)"``, each defined in the plotting guide ("Lower panel");
        ``None`` draws none. A point beyond the panel's range is marked at its edge.
    reference
        The label of the one histogram the panel compares with, the background
        of a significance: every other histogram, data included, is compared
        with it (every other non-data histogram, as a signal, for a
        significance). By default data is compared with the stack total, or with
        the first non-data histogram without a stack; with a stack but no data,
        every overlaid histogram with the total, and with neither (or with
        observed data alone), every histogram after the first with the first.
        A significance takes the overlaid non-data histograms as signals over the
        stack, or otherwise the last non-data histogram over the sum of the others.
    panel_ylim, panel_label, panel_uncertainty
        Lower panel range, y label, and uncertainty treatment of a ratio,
        relative difference or difference (``"propagate"``, or ``"numerator"``
        with the reference's uncertainty as a band; by default each numerator
        uses ``"numerator"`` for data over simulation and ``"propagate"``
        otherwise, so shared systematic sources cancel). ``"poisson-ratio"``
        takes the exact interval of the ratio of two Poisson means for a ratio
        of counts, as ROOT's ``TGraphAsymmErrors::Divide(..., "pois")``: every
        compared histogram must hold known counts (see
        :meth:`~rootfig.histograms.Histogram.counts`). The label is shrunk,
        and if needed wrapped onto two lines, to fit the short panel; pass a
        shorter ``panel_label`` (``"Ratio"``) to keep it at full size.
    logx, logy
        Logarithmic axes. ``logx=None`` (default) follows the ``Variable``'s
        ``log`` flag; ``True``/``False`` override it.
    flow
        Under/overflow display: ``"hint"`` (arrows), ``"show"`` (extra bins),
        ``"sum"`` (added to the edge bins, also for ratios and limits), ``"none"``.
    histtype
        Default drawing style for overlaid samples: ``"step"`` (default),
        ``"fill"``, ``"errorbar"`` or ``"band"``. A sample's or group's own
        ``histtype`` takes precedence. Stacked histograms are always filled.
    errorbars
        Draw statistical error bars on overlaid histograms; ignored for stacked
        histograms. ``None`` draws them only for ``"errorbar"`` histtypes.
    data_errors
        The statistical uncertainty of observed data, in the main and the lower
        panel alike. ``None`` (default) keeps each histogram's own: ROOT's
        ``TH1`` default, ``sqrt(sum of squared weights)`` (``sqrt(N)`` for
        counts), unless it carries a Poisson interval or errors of its own (a
        histogram object with ``poisson=`` or ``stat_errors=``, a stored ``TH1``
        saved with ``kPoisson``/``kPoisson2``); ``"sumw2"`` forces ``sqrt(sum of
        squared weights)`` on every data histogram. ``"poisson"`` is ROOT's
        ``TH1::kPoisson``, the Garwood 68 % interval of the counts, asymmetric
        and with an upper error for an empty bin; a confidence level such as
        ``0.95`` (ROOT's ``TH1::kPoisson2``) is the Garwood interval at that
        level. Both refuse anything but unit-weight counts (every bin a whole
        number equal to its variance, before any normalisation). ``"auto"`` is
        Poisson for unit-weight counts and ``sqrt(sum of squared weights)``
        otherwise.
    xlim, ylim
        Axis limits; ``ylim`` entries may be ``None`` to keep the automatic value.
    xbreak
        ``(a, b)`` to cut the x axis: the range between ``a`` and ``b`` is
        removed and the two remaining segments are drawn side by side with a
        break mark (e.g. a peak and a far tail, or a sentinel region). Works
        with a lower panel; not with ``ax=``.
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
        Draw into existing axes (``Axes``, or a ``(main, panel)`` pair with ``panel=``).
    nonfinite
        ``"drop"`` (warn) or ``"error"`` for ``nan``/``inf`` values.
    systematics
        Sources of systematic uncertainty applied to every simulated sample, in
        the forms ``Sample(systematics=...)`` accepts (a sample's own source of
        the same name takes precedence), e.g. ``{"lumi": 0.017}``. Stacks draw
        the statistical and systematic uncertainty as one band, overlaid samples
        with variations a light band in their colour, and the lower panel
        includes them in its band and error bars (in a pull, in its
        denominator; a significance is statistical only). A name is one source:
        sources of the same name are fully correlated across samples (a
        plot-level source is shared by all of them), different names are
        independent and added in quadrature;
        ``Plot.uncertainty()`` returns the components. Histogram objects carry
        theirs in :attr:`~rootfig.histograms.Histogram.variations`.
    variances_from_contents
        Accept a histogram (stored or object) with a plain count storage that
        was filled with weights or rescaled, so ``hist`` reports no variances:
        the absolute bin contents are used instead (a warning says so). Fill
        with ``hist.storage.Weight()`` to keep the real uncertainties.
    save
        Path to save the figure to (also returned in the :class:`Plot`).

    Returns
    -------
    Plot
        The figure, axes, histograms and comparisons.
    """
    prepared = prepare_plot(
        data,
        variable,
        tree=tree,
        selection=selection,
        weight=weight,
        lumi=lumi,
        bins=bins,
        range=range,
        label=label,
        observed=observed,
        xlabel=xlabel,
        unit=unit,
        nonfinite=nonfinite,
        systematics=systematics,
        variances_from_contents=variances_from_contents,
    )
    return draw_plot(
        prepared,
        ylabel=ylabel,
        title=title,
        normalize=normalize,
        normalize_uncertainty=normalize_uncertainty,
        stack=stack,
        panel=panel,
        reference=reference,
        panel_ylim=panel_ylim,
        panel_label=panel_label,
        panel_uncertainty=panel_uncertainty,
        logx=logx,
        logy=logy,
        flow=flow,
        histtype=histtype,
        errorbars=errorbars,
        data_errors=data_errors,
        xlim=xlim,
        ylim=ylim,
        xbreak=xbreak,
        legend=legend,
        stats=stats,
        text=text,
        style=style,
        figsize=figsize,
        ax=ax,
        save=save,
    )


def prepare_plot(
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
    unit: str | None = None,
    nonfinite: NonFinitePolicy = "drop",
    systematics: Mapping[str, SystematicLike] | None = None,
    variances_from_contents: bool = False,
    cache: ReadCache | None = None,
) -> PreparedPlot:
    """Read or fill the histograms :func:`plot` draws; see there for the options.

    Everything that decides what is read and how the histograms are filled
    happens here; :func:`draw_plot` takes the result. A ``cache``
    (:class:`~rootfig.io.ReadCache`) serves the branch arrays and stored
    histograms it holds and reads the rest, for file sources.
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
        hists = wrap_histograms(objects, label, variances_from_contents=variances_from_contents)
        if observed is not None:
            observed_objects = histogram_objects(observed)
            if observed_objects is None:
                msg = "observed= must be histogram objects when data are histogram objects"
                raise TypeError(msg)
            hists += wrap_histograms(
                observed_objects, variances_from_contents=variances_from_contents, is_data=True
            )
        require_dimension(hists, 1, "plot")
        hists = rebin_ready_made(
            hists,
            [bins if var is None else var.bins],
            [range if var is None else var.range],
        )
    else:
        if variable is None:
            msg = "plot() needs a variable (a branch, expression or stored histogram name)"
            raise TypeError(msg)
        items = plot_items(data, tree=tree, label=label, observed=observed)
        var = as_variable(variable, bins=bins, range=range, label=xlabel, unit=unit)
        hists = build_histograms(
            items,
            var,
            selection=selection,
            weight=weight,
            lumi=lumi,
            nonfinite=nonfinite,
            systematics=systematics,
            variances_from_contents=variances_from_contents,
            cache=cache,
        )
    return PreparedPlot(hists, var, xlabel=xlabel, unit=unit, lumi=lumi)


def plot_items(
    data: Any, *, tree: str | None, label: str | Sequence[str] | None, observed: Any
) -> list[PlotItem]:
    """Return the samples and groups :func:`plot` fills from: ``data``, then ``observed``."""
    items = as_plot_items(data, tree=tree, labels=label)
    if observed is not None:
        items += [as_observed(item) for item in as_plot_items(observed, tree=tree)]
    return items


def prefetch_plots(
    cache: ReadCache,
    data: Any,
    variables: Sequence[str | Variable],
    selections: Sequence[CutLike | None],
    options: Sequence[Mapping[str, Any]] = ({},),
) -> None:
    """Read into ``cache`` what :func:`prepare_plot` reads for ``variables`` and ``selections``.

    Each entry of ``options`` is a set of :func:`prepare_plot` keywords to read
    for; those deciding what is read (``tree``, ``label``, ``observed``,
    ``weight``, ``systematics``, ``variances_from_contents``) are used, the others are
    accepted and ignored. All of them are planned before anything is read, so
    sets needing different branches, such as two variants with different
    weights, cost one pass over each file rather than one each.

    An optimisation only, so it never raises: nothing is read for histogram
    objects, and whatever :func:`prepare_plot` will refuse, an unusable
    ``data`` or ``label`` as much as an unknown branch, is left to the call
    that needs it, which raises the error for its own task.
    """
    if histogram_objects(data) is not None:
        return
    plan = ReadPlan(cache)
    for option_set in options:
        # Every exception, not only rootfig's: a bad option of one task must not surface
        # while the histograms of another are read ahead, nor stop them being read.
        with suppress(Exception):
            items = plot_items(
                data,
                tree=option_set.get("tree"),
                label=option_set.get("label"),
                observed=option_set.get("observed"),
            )
            plan.add(
                items,
                [as_variable(variable) for variable in variables],
                selections=selections,
                weight=option_set.get("weight"),
                systematics=option_set.get("systematics"),
                variances_from_contents=bool(option_set.get("variances_from_contents", False)),
            )
    plan.read()


def draw_plot(
    prepared: PreparedPlot,
    *,
    ylabel: str | None = None,
    title: str | None = None,
    normalize: NormalizeSpec = None,
    normalize_uncertainty: NormalizeUncertainty = "scale",
    stack: StackSpec = False,
    panel: ComparisonKind | None = None,
    reference: str | None = None,
    panel_ylim: tuple[float, float] | None = None,
    panel_label: str | None = None,
    panel_uncertainty: UncertaintyMode | None = None,
    logx: bool | None = None,
    logy: bool = False,
    flow: FlowSpec = "hint",
    histtype: HistType | None = None,
    errorbars: bool | None = None,
    data_errors: DataErrors | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xbreak: tuple[float, float] | None = None,
    legend: bool | str | None = None,
    stats: bool | str = False,
    text: str | Sequence[str] | None = None,
    style: StyleLike = None,
    figsize: tuple[float, float] | None = None,
    ax: AxesLike = None,
    save: str | None = None,
    cell: SubplotSpec | None = None,
) -> Plot:
    """Draw prepared histograms; see :func:`plot` for the options.

    The histograms are not modified: every transformation for display
    (normalisation, flow bins, sums) works on copies, so one
    :class:`PreparedPlot` can be drawn several ways. ``cell`` is a cell of a
    grid on an existing figure to build the panels in (see
    :func:`~rootfig.plotting.make_figure`), for several plots on one page; the
    figure then belongs to the page, which sets its size and saves it.
    """
    histograms_ = prepared.histograms
    variable = prepared.variable
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
            "stored histograms, histogram objects and groups have none"
        )
        raise ValueError(msg)
    stacked, _, _ = split_stack(histograms_, stack)
    if stacked and (
        normalize is True
        or normalize in ("unity", "density")
        or (isinstance(normalize, int | float) and not isinstance(normalize, bool))
    ):
        msg = (
            f"normalize={normalize!r} would normalise every stacked histogram on its own, "
            "so the stack would not add up to a normalised total; compare shapes with "
            "stack=False, draw a combination as one histogram with rf.Group, "
            "or use normalize='width'"
        )
        raise ValueError(msg)
    histograms_ = with_data_errors(histograms_, data_errors)
    if normalize is not None and normalize is not False:
        histograms_ = [normalize_for_plot(h, normalize, normalize_uncertainty) for h in histograms_]
    elif normalize_uncertainty != "scale":
        msg = (
            f"normalize_uncertainty={normalize_uncertainty!r} needs a normalisation to the "
            "histogram's own total: normalize=True, 'unity', 'density' or a number"
        )
        raise ValueError(msg)
    resolved_style = style_for(style, text, prepared.lumi)
    if legend is not None:
        resolved_style = resolved_style.replace(legend=legend)

    # The y label quotes the bin width of the histogram as filled, before flow bins are added.
    label_widths = histograms_[0].widths
    flow_shown = (False, False)
    if flow == "show":
        if xbreak is not None and xlim is None:
            msg = "xbreak cannot be combined with flow='show'; pass xlim as well"
            raise ValueError(msg)
        # Done here rather than in mplhep so every histogram, the lower panel and the
        # x range agree on the extra bins (mplhep adds them per histogram).
        histograms_, flow_shown = show_flow_bins(histograms_)
        flow = "none"
    elif flow == "sum":
        # Fold once, up front, so comparisons, bands and limits see the same bins as the drawing.
        histograms_ = fold_flow_bins(histograms_)
        flow = "none"

    stacked, overlaid, data = split_stack(histograms_, stack)
    # the stack and the roles of the lower panel are checked before a figure exists
    require_same_binning(stacked, "a stack", flow=True)  # summed: the flow bins must agree too
    plan = resolve_panel(
        histograms_,
        panel=panel,
        reference=reference,
        uncertainty=panel_uncertainty,
        label=panel_label,
        stacked=stacked,
        overlaid=overlaid,
        data=data,
    )
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
    view = segments if segments is not None else [outer] if outer is not None else None

    with style_context(resolved_style) as st:
        layout = make_figure(
            st, panel=plan is not None, ax=ax, figsize=figsize, break_widths=break_widths, cell=cell
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
                view=view,
            )
        assert drawn is not None
        color_of = dict(zip(map(id, histograms_), drawn.colors, strict=True))
        has_data = any(h.is_data for h in histograms_)
        add_experiment_label(layout.main, st, has_data=has_data, right=layout.main_right)

        per_object = any(h.per_object for h in histograms_)
        x_label, bin_unit = _axis_labels(prepared.xlabel, prepared.unit, variable, reference_hist)
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
        headroom = None
        if ylim is None or ylim[1] is None:
            # keep legend, labels and text boxes clear of the histograms, once laid out
            floating = legend_location(st) == "best" and legend_loc is None
            obstacles = overlay_artists(layout.legend_axes, None if floating else legend_artist)
            if layout.legend_axes is not layout.main:
                obstacles += overlay_artists(layout.main, None)
            env_edges, env_heights = envelope(histograms_, stack=stack)
            headroom = partial(
                raise_ylim_above,
                layout.main_axes,
                obstacles,
                edges=env_edges,
                heights=env_heights,
                logy=logy,
                floating=[legend_artist] if floating and legend_artist is not None else [],
            )

        comparisons = []
        if plan is not None and layout.panel is not None:
            comparisons = plan.comparisons(drawn.stack)
            for index, axis in enumerate(layout.panel_axes):
                draw_panel(
                    comparisons,
                    axis,
                    colors=[color_of[id(h)] for h in plan.numerators],  # all drawn
                    observed=plan.observed,
                    ylim=panel_ylim,
                    ylabel=plan.label if index == 0 else "",
                    view=view,
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
            if layout.panel is not None and layout.panel_right is not None:
                apply_xbreak(layout.panel, layout.panel_right, *segments)

        # last: fonts, of this plot's axes only (a page holds others)
        pin_fonts(layout.fig, axes=layout.axes)
    # outside the style context, against the layout the figure is drawn with
    finish = Finish(
        layout.main,
        right=layout.main_right,
        panels=layout.panel_axes,
        xlabel=layout.xlabel_axes,
        headroom=headroom,
    )
    finish_figure(layout.fig, [finish])
    result = Plot(
        fig=layout.fig,
        ax=layout.main,
        panel_ax=layout.panel,
        ax_right=layout.main_right,
        panel_ax_right=layout.panel_right,
        histograms=list(histograms_),
        comparisons=comparisons,
        variable=variable,
        stack=drawn.stack,
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
