# Plotting options

All options below are keyword arguments of [`rf.plot`][rootfig.plot], whether
it fills from a tree, reads a histogram stored in the file or draws histogram
objects you already have (see [the end of this page](#histograms-that-already-exist)).

## Overlays, stacks and data

- **Overlay** (default): every sample is drawn as an outline (`histtype="step"`).
  Use `histtype="fill"` for translucent filled areas, `"errorbar"` for points,
  `"band"` for uncertainty bands; a `Sample(histtype=...)` overrides per sample.
- **Stack**: `stack=True` stacks every non-data histogram; a label or a sequence
  of legend labels selects what is stacked, for example
  `rf.plot([ww, zz, zh], "mass", stack=["WW", "ZZ"])`. Every histogram carrying
  a selected label is stacked, in input order with the first at the bottom;
  the order of the selectors does not matter. The stack is drawn first, then
  its hatched uncertainty band, then the remaining histograms as overlays in
  input order, then data. A histogram keeps its colour whether it is stacked
  or overlaid; explicit colours do not use up entries in the colour cycle.
  Stacked histograms are always filled: `histtype=`, a sample's or group's own
  `histtype`, and `errorbars=` apply only to overlays. `False` or `[]` overlays
  everything. Unknown labels and labels belonging only to observed data raise
  `ValueError`; anything other than a bool, string or sequence of strings raises
  `TypeError`. A group is stacked by its own label, not its components' labels.
- **Data**: samples with `is_data=True` (or passed as `observed=...`) are
  points with error bars, drawn on top and never stacked, in the style's text
  colour (black by default) unless the sample sets `color`.
- **Groups**: a [`Group`][rootfig.Group] of samples is one histogram of the
  overlay or stack, the sum of its components filled apart; see
  [Group](composable.md#group).
- `errorbars=True` adds statistical error bars to overlaid non-data histograms.

The [selective-stacking example](gallery/selective_stack.md) shows stacked
backgrounds with a signal drawn over them and a significance panel.

## Systematic uncertainties

Samples carry their sources of systematic uncertainty, `{name: variation}`;
`systematics=` on `plot()` and `histograms()` adds sources to every simulated
sample (a sample's own source of the same name wins):

```python
bkg = rf.Sample(
    "mc.root",
    tree="events",
    label="Background",
    weight="weight",
    systematics={
        "pileup": ("weight_pu_up", "weight_pu_down"),  # weights replacing Sample.weight
        "trigger": "weight_trig_up",  # one-sided: down mirrors up
        "xsec": 0.05,  # ±5 % normalisation
        "shower": (1.10, 0.97),  # (up, down) normalisation factors
        "jes": {"Jet_pt": ("Jet_pt_jesUp", "Jet_pt_jesDown")},  # branches
        "generator": rf.Systematic.samples("mc_herwig.root"),
    },
)
p = rf.plot(
    [bkg, sig], "Jet_pt", observed=data, stack=True, ratio=True, systematics={"lumi": 0.017}
)
```

| Form | Variation |
| --- | --- |
| `"w_up"`, `("w_up", "w_down")` | weight expression(s) replacing `Sample.weight`; the plot `weight=`, `scale` and luminosity scaling still multiply |
| `0.05`, `(1.10, 0.97)` | the nominal histogram scaled by `1 ± 0.05` (a magnitude, at least 0 and below 1), or by the two positive factors, which may point either way |
| `{"Jet_pt": ("Jet_pt_up", "Jet_pt_down")}` | branch names replaced by other branches in the variable, the selection and the weight, so a cut on `Jet_pt` moves with it; replacements are branch names, not expressions |
| `Systematic.samples(up, down)` | other files or arrays with the sample's selection, weight, cross section, tree name and entry range; files look up a string `ngen` themselves, arrays take the nominal sample's; a `Sample` is used as given |

Rules:

- Every tuple is `(up, down)`. Without a down variation, the up shift is
  mirrored: `down = 2 × nominal − up`.
- The binning comes from the nominal values; variations fill the same axis.
  Weight and branch variations are evaluated on the branches read once for the
  nominal histogram.
- Per bin, each source shifts the contents by `up − nominal` and
  `down − nominal`. The larger positive shift enters the upper uncertainty,
  the larger negative one the lower (so two variations moving the same way
  widen one side only). Different sources are independent and added in
  quadrature; the total is statistical ⊕ systematic, per side.
- Sources with the same name are fully correlated across samples: the stack
  total adds their variations linearly (a sample without the source
  contributes its nominal contents), and a ratio varies numerator and
  denominator together, so a shared luminosity uncertainty cancels in an
  MC/MC ratio.
- A [`Group`][rootfig.Group] sums its components' variations by the same rule
  before it is drawn or normalised.
- `normalize=True`, `"unity"`, `"density"` or a numeric target normalises every
  variation by its own total, so the plot shows shape uncertainties; a pure
  normalisation uncertainty drops out. `normalize="width"` only divides by
  bin width and retains normalisation uncertainties. If the nominal cannot be
  normalised, its variations also stay raw. If only a variation has a zero or
  non-finite total, normalization raises `SystematicError`.
  `flow="sum"`/`"show"` treat variations like the nominal histogram.
- Non-finite values are reported for each affected variation, with the source
  name and direction in the warning. `nonfinite="error"` rejects them.
- Data samples cannot carry systematics ([`SystematicError`][rootfig.SystematicError]),
  and neither can pre-filled `Histogram(is_data=True)`; plot-level sources skip them.
- `plot2d`, `correlation`, `summarize`, `cutflow`, `efficiency`, `profile` and
  `histogram()` (a plain `hist.Hist`) ignore systematics.
  Significance panels also use only statistical uncertainties.

Drawing follows mplhep's conventions: a stack's hatched band shows the
statistical and systematic uncertainty of the total (legend `Stat. + syst.
unc.`), overlaid samples with variations get a light band in their own colour,
and the ratio panel includes the systematics in its band around one
(`split_ratio`, data/MC) or in the error bars of the points (`propagate`:
statistical uncertainties uncorrelated, systematic ones propagated source by
source through the varied ratio; a variation that empties a denominator bin
leaves that bin's systematic uncertainty undefined, with a warning). The
automatic ratio range covers the bulk of the band and of the systematic error
bars (robust percentiles, like the points, so a single bin with a huge
uncertainty runs off the panel instead of squashing it; pass `ratio_ylim` to
show it in full). It stays at or above zero unless a central ratio is negative
(signed weights).

The numbers are part of the result. `Plot.stack` holds the sum of the stacked
histograms, labelled `"Total"` and including variations, or `None` without a
stack. `p.uncertainty()` uses that total, or the sole non-data histogram if
there is no stack; several overlays require a label, such as `p.uncertainty("ZH")`:

```python
u = p.uncertainty()  # the stack total (p.stack), or one histogram by label
u.stat, u.syst_down, u.syst_up  # per bin, visible bins
u.total_down, u.total_up  # statistical ⊕ systematic
u.components["jes"]  # signed (up − nominal, down − nominal) shifts
p.histograms[0].variations  # {"jes": (hist_up, hist_down), ...}
p.ratios[0].syst_band  # relative (down, up) band of the reference
```

Pre-filled histograms take variations directly and are drawn the same way:
`rf.Histogram(h, label="MC", variations={"jes": (h_up, h_down)})` passed to
`plot`; [`uncertainty`][rootfig.histograms.uncertainty] and
[`sum_histograms`][rootfig.histograms.sum_histograms] work on them too.

Use `(h_up, None)` for a mirrored variation. `Sample.systematics` and
`Histogram.variations` are read-only; `sample.replace(systematics=...)` and
`histogram.replace(variations=...)` return copies with other ones.

## Luminosity

`lumi=` scales simulated samples that carry a cross section
(`Sample(xsec=..., ngen=...)`) to expected yields, `xsec × lumi / ngen`, and
writes the luminosity into the label. Numbers are in fb⁻¹; strings carry a
unit: `lumi="10.8 ab^-1"`. Data samples and samples without a cross section
are left alone. The same keyword exists on `histogram(s)`, `plot2d`,
`summarize`, `correlation`, `cutflow`, `efficiency` and `profile`. A
luminosity already set on the `Style` wins in the label: `Style(lumi=...)`
describes the figure, `lumi=` the scaling, and they need not agree (a partial
dataset scaled to the full one, for instance).

## Normalisation

`normalize=`

| Value | Meaning | y label |
| --- | --- | --- |
| `False`/`None` | raw sums of weights | `Events` or `Entries / 5 GeV` |
| `True`, `"unity"` | visible bins sum to one | `Normalised to unity` |
| `"density"` | integral over the visible range is one | `Density` |
| `"width"` | divide by bin width, no rescaling | `Entries / GeV` |
| a number | visible bins sum to that number | `Normalised to 100` |

With any histograms stacked, `normalize=True`, `"unity"`, `"density"` and numeric
targets raise: normalising each component separately would not yield a normalised
total. Use `stack=False` to compare shapes, `rf.Group` to draw a combination as one
histogram, or `normalize="width"`; `None` and `False` also pass.

A [`Group`][rootfig.Group] is summed first and normalised as one histogram.
Variances are scaled consistently. Flow bins scale with the same factor; for
`"width"` and `"density"` they are divided by the width of the neighbouring
visible bin. Plain `hist.Hist` objects with a count storage passed to
`plot` are converted to `Weight` storage first. If such a histogram
was filled with weights (or rescaled) its sum of squared weights is lost, and
rootfig refuses it with a `ValueError` rather than invent uncertainties; pass
`assume_poisson=True` to use the absolute bin contents as variances (with a
warning), or fill with `hist.storage.Weight()` in the first place.

The rescaling modes divide by the signed sum of the visible bins: a histogram
dominated by negative weights still sums to the target, its shape flips sign,
and a warning says so. An empty histogram, or one whose positive and negative
weights cancel exactly, is left unchanged with a warning and keeps the plain
`Events` label (`Histogram.normalization` stays `None`).

## Ratio panel

`ratio=True` adds a lower panel sharing the x axis:

- with a stack and data: data / stack total; overlaid histograms are not part
  of the prediction and do not appear in the panel;
- with a stack and no data: every overlaid histogram / stack total; a full
  stack needs observed data or a histogram outside the stack;
- without a stack, with data: data / the first non-data histogram;
- without a stack or data: every histogram after the first / the first.

`ratio="Background"` picks the reference by label (a group's label counts);
all other histograms, data included, are divided by it. For every form, the
uncertainty treatment is chosen per numerator: data over simulation keeps the
reference uncertainty as a grey band (`"numerator"`), and every other ratio
propagates both sides (`"propagate"`), so shared systematic sources cancel.
`ratio_uncertainty=` applies one treatment to all.

`ratio="significance"` (or `"s/sqrt(b)"`, `"s/sqrt(s+b)"`) draws a
**significance panel** instead: per bin, the signal over the square root of
the background (or of signal plus background), with propagated uncertainties.
With histograms overlaid on a stack, the stack is the background and every
overlaid non-data histogram is a signal, so stacking the backgrounds compares
several signals with them. Without a stack, or with everything stacked, the last
non-data histogram is the signal and the others are summed into the background:
with the signal last, drawing it inside the stack or over it shows the same panel.
`ratio=("s/sqrt(b)", "ZH")` names one signal and sums every other non-data histogram
into the background. `Plot.ratios` holds one `Ratio` per signal, drawn in that
histogram's colour.

`ratio_ylim` and
`ratio_label` override the automatic range (at least 0.5 to 1.5, widened to
cover the bulk of the points) and label (`Ratio to X` or `Data / MC`). A
rotated y label is bounded by the height of the short ratio panel, so a long
one is shrunk and, if that is not enough, wrapped onto two lines; pass a
shorter `ratio_label` such as `"Ratio"` to keep it at full size. The
computed values are returned in `Plot.ratios` as
[`Ratio`][rootfig.Ratio] objects (`values`, `errors`, `band`, `edges`, and
`syst_errors`/`syst_band` with [systematic uncertainties](#systematic-uncertainties)).

## Binning and range

`bins` takes an `int`, a `(n, low, high)` triple, a sequence of edges or a
`hist` axis, and is shared by every sample of one plot (and by the numerator
and denominator of an [efficiency](#efficiencies)).

With an integer `bins` the range comes from `range`:

| `range` | Meaning |
| --- | --- |
| `(low, high)` | explicit |
| `"robust"` | **default**: the min/max of the data, ignoring values far from the bulk |
| `"auto"` | the full finite minimum and maximum over all samples |

```python
rf.plot("events.root", "d0_significance", bins=50)  # robust
rf.plot("events.root", "d0_significance", bins=50, range="auto")  # full extent
rf.plot("events.root", "d0_significance", bins=50, range=(-5, 5))  # explicit
```

!!! note "Rejected values are not discarded"

    A value outside the range is **not removed from the data**: it goes to the
    under/overflow, shown by the flow arrows (`flow="show"` turns them into
    visible bins, `flow="sum"` folds them into the edge bins). Statistics boxes
    and `rf.summarize` are computed before binning, so means and entry counts
    cover the full sample whichever range is used.

This happens in two steps. Outliers are rejected by their modified z-score
(`0.6745 * |x - median| / MAD`), with a threshold of 30, which removes sentinels
and anything else far from the bulk. That is done within each sample, and the
ranges they keep are unioned, so a sample is judged against its own median and
spread: a signal offset from a background is not an outlier merely because the
background outnumbers it, and a sample keeps the same values whether it is
plotted alone or in an overlay. The threshold is then tightened for as
long as each step costs no more than an *additional* 1 percent of any one
sample, by entries and by weight. Additional is meant literally: the budget is
measured against what the first step already moved out of the view, which may
be a good deal more than 1 percent. This second step cuts a tail that reaches
far but thins out smoothly, the kind a distance threshold keeps and that leaves
the interesting part of the distribution in a corner of the axis.

The budget is charged per sample rather than over the pooled entries, so a small
signal sitting far from a large background keeps its own place on the axis
instead of being cut as a rounding error, and it is charged against the weight a
cut would remove as well as the entries, so a handful of high-weight entries is
not treated as negligible. A sample with fewer than 20 distinct values is
categorical — counts, flags, multiplicities — has no tail to cut and gets no
budget at all, so the second step takes no value off its axis; this is decided
per sample too, and holds when it is overlaid with a continuous one. If MAD is zero, the mean absolute deviation from
the median is used instead. Every candidate is padded by 5 percent and clamped
to the `"auto"` range (whose upper edge is nudged above the maximum to include
it), so the range never reaches past the data. Degenerate ranges are widened
symmetrically. The threshold cannot distinguish sentinels from valid data.
Cases where you may want `range="auto"`:

- a distribution with a long tail (log-normal, Student-t, or an invariant mass
  with a continuum) has valid tail entries pushed into the flow bins — this is
  what the second step is for, so `"auto"` is the way to see the whole tail;
- a sparse discrete distribution can lose rare valid values from the visible
  range, for example the ones in a binary sample with 999 zeros and one one;
- a lone value far from a bulk of near-identical ones looks exactly like a
  sentinel and is rejected with them, however real it is;
- `xbreak=(a, b)` is validated against the inferred axis, so a break meant to
  span a far tail needs `range="auto"` or an explicit range.

Inferring a robust range requires additional median and deviation calculations
over the combined samples, with additional time and memory costs. An explicit
range avoids range inference.

!!! tip "The plot is mostly empty space"

    The inferred range cuts a thin tail, but only as far as its coverage
    budget allows. A distribution whose tail carries more than that — a heavy
    Student-t, a steeply falling spectrum over several decades — still spreads
    the axis over bins holding a fraction of a percent of the peak. Three ways
    out, in order of how often they are what you want:

    - `range=(a, b)` around the core. Nothing is lost: entries outside go to
      the flow bins, where `flow="hint"` (the default) marks them with arrows
      and `flow="sum"` folds them into the edge bins.
    - `logy=True`, which makes the tail visible instead of hiding it.
    - `xbreak=(a, b)` to cut the empty middle out and keep both ends, with
      `range="auto"` or an explicit range so the break lies inside the axis.

## Axes

Automatic y limits come from the bins overlapping the x range shown: `xlim`,
or both segments of `xbreak`. This applies to the main panel and to ratio and
significance panels alike.

- `logx`, `logy`: logarithmic scales. Log-spaced bins: `bins=rf.log_bins(n, low, high)`.
- `xlim`, `ylim`: limits; `ylim=(None, 1e4)` keeps the automatic lower value.
  Automatic y limits add a small margin above the tallest bin (a factor 1.2 in
  linear scale, 12 in log scale) and then raise it further as the drawn
  legend, label, statistics box and text lines need.
- `xbreak=(a, b)`: cut the range between `a` and `b` out of the x axis and
  draw the two remaining segments side by side with break marks, sharing the
  y axis (and the ratio panel, if any). Useful for a peak plus a far tail or
  a sentinel region. The right segment is `Plot.ax_right`
  (`Plot.ratio_ax_right`). Not available together with `ax=` or `flow="show"`.

  ![Broken x axis with a ratio panel](images/gallery/xbreak_ratio-atlas.png#only-light){ width="60%" }
  ![Broken x axis with a ratio panel](images/gallery/xbreak_ratio-atlas-dark.png#only-dark){ width="60%" }
- `flow`: how under/overflow is shown (this is where entries outside an
  inferred [range](#binning-and-range) end up), `"hint"` (small arrows, default),
  `"show"` (extra bins labelled `<low` / `>high`, added on a side as soon as any
  sample has content there, identical for all samples and the ratio panel),
  `"sum"` (added to the edge bins before anything is computed, so ratios,
  significances, stack bands and y limits use the folded bins), `"none"`.
- `logx`, `logy`: logarithmic axes. `logx=None` (default) follows the
  `Variable`'s `log` flag (`plot2d`, `efficiency` and `profile` do the same for
  their variables); `True`/`False` override it.
- `xlabel`, `ylabel`, `unit`, `title`. The title sits above the axes, where
  the CMS-style label is also drawn; with such a style prefer `text=`. The x
  label ends at the right end of the axis, where matplotlib also puts the
  axis' offset text (`×10⁻⁶` for small values); when both are shown the label
  moves left of it, or below it where the axis is too short for both side by
  side, also with `axes.formatter.useoffset` off, which drops an additive offset
  but still shows the order of magnitude. On a figure rootfig makes the place is
  chosen each time the figure is drawn, so a resized figure keeps the label clear
  for its new size; on axes passed with `ax=` the label goes below the offset
  text, which holds at any size. A `labelpad` or transform set on the label after
  plotting is kept, and the label is kept clear starting from it.
- Automatic y limits leave room for the legend, the experiment label, the
  statistics box and `text` lines: a small fixed margin is added above the
  tallest bin, and the upper limit is then raised until none of them covers a
  histogram (the legend picks a free upper corner of its axes). Room is only
  made for what is actually drawn, so a plot without annotations keeps the
  margin. It is measured on the figure as laid out, as it is shown and saved.
  A `ylim` with an explicit upper value switches this off.

## Legend, labels, text and statistics

- `legend=False` or a location string such as `"upper left"`.
- `style=` an experiment name or a [`Style`][rootfig.Style]; see
  [Samples, variables, cuts and styles](composable.md).
- `with rf.dark_theme():` draws the figures made inside the block for a dark
  page — light ink on a transparent background, applied on top of any style
  (including experiment styles that fix a white background). Colours set in
  `Style.rc`, background and text included, are applied after it and win for
  the figures and axes rootfig creates, so such a style is not rendered dark.
  Data points and outlines follow the style's `text.color`. Plain matplotlib
  inside the block matches too: axes made with `plt.subplots()` for `ax=`, text
  added to `Plot.ax`, and `fig.savefig()` keeps the transparent background.
  Axes passed with `ax=` keep the properties they were created with (their
  background, frame and ticks), so a `Style.rc` passed to that plot colours
  only what rootfig draws into them.
- `text=` extra line(s) drawn inside the frame: in the upper left corner for
  `label_loc=0` (the CMS and DUNE default), below the secondary text for the
  split layout `label_loc=3`, and below the label for locations 1, 2, and 4.
- A label line above the frame (experiment name, status, luminosity) that does
  not fit the width of the axes, next to a colour bar or over a broken x axis,
  is shrunk to fit, down to 60% of its original font size. If that is still too
  wide, the luminosity gets a separate line above the label. The luminosity of
  a broken x axis sits above the right end of the right segment. An explicit
  title is placed above these labels, preserving its font, alignment and padding.
- The experiment name and status share a baseline, including after resizing
  the figure. `label_loc=2` and `3` explicitly put the status on a separate line.
- `stats=True` adds entries, mean and standard deviation per sample below the
  legend (a location string moves it).

## Figure handling

- `figsize=(w, h)`; `ax=some_axes` draws into your own axes (pass a pair
  `(main, ratio)` for ratio plots), so several rootfig plots can share a figure.
- `save="file.pdf"` saves immediately; `Plot.save(path)` accepts a directory
  (file named after the variable) and `formats=["pdf", "png"]`.
- Figures use matplotlib's constrained layout, so labels, legends and colour
  bars fit inside the canvas and a saved file has exactly the `figsize`
  dimensions: 1D, 2D and ratio plots of one size share one shape. Figures
  drawn into your own `ax` are saved with a tight bounding box instead.
- Fonts are fixed on the figure when it is made, so saving or displaying it
  later renders exactly the layout that was computed, in the style's fonts.
- `Plot.fig`, `Plot.ax`, `Plot.ratio_ax` are plain matplotlib objects;
  `Plot.histograms` wrap the `hist.Hist` objects with labels and statistics.
- In a notebook the figure is displayed automatically — it is a pyplot figure,
  flushed by the inline backend at the end of the cell, so `%matplotlib inline`
  is not needed. End the call with `;` to hide the `Plot` repr, and use
  `Plot.close()` in loops that make many figures.

## 2D histograms and correlations

```python
rf.plot2d(sample, x, y, bins=((40, 0, 200), (30, -3, 3)), logz=True, normalize="density")
rf.correlation(sample, ["MET", "nJet", "HT"], selection="nJet >= 2", percent=True)
```

`rf.plot2d` options: `logz`, `logx`, `logy`, `cmap` (any matplotlib colour
map, default `viridis`), `colorbar=False`, `zlabel` (default `Events` or the
normalisation), `normalize`, `title`, `text`, `style`, `figsize`, `ax`, `save`.
The figure has the same size as a 1D plot; the colour bar takes its space from
the axes. For 2D histograms and correlation matrices, `label_loc` defaults to
0, with the experiment and secondary text above the frame. Location 3 splits
them: experiment above, secondary text inside. Only `label_loc=1`, `2`, or `4`
moves both inside. Luminosity remains above for locations 0–3 and inside for 4.

`rf.correlation` options: `labels` (tick labels, default the variable labels),
`percent=True` (integer percentages instead of two-decimal coefficients),
`annotate=False` (colours only), `cmap` (default `RdBu_r`), `title`, `style`,
`figsize`, `ax`, `save`. The matrix is returned as `Plot.matrix`. Without a
`figsize` the figure grows with the number of variables, in proportion to the
style's figure width, so the larger fonts of experiment styles keep their
cells readable. It is titled
`"<sample>: correlation"`; with an experiment style the experiment label replaces
that automatic title, using the placement described above. An explicit `title`
is shown either way.

Both variables of a 2D histogram (and all variables of a correlation matrix)
must share their structure: all per-event, or all per-object from the same
collection.

## Efficiencies

```python
rf.efficiency([reco], "TrueMuon_pt", passed="TrueMuon_matched", bins=(20, 0, 100), unit="GeV")
```

For every sample the entries satisfying `selection` form the denominator and
those also satisfying `passed` the numerator, with the same binning. The
efficiency is drawn as points with Wilson score intervals (`z=1` standard
deviations by default; weighted samples use effective entries). Options are
the usual axis, legend, label and style ones (`xlabel`, `ylabel`, `unit`,
`title`, `logx`, `xlim`, `ylim`, `legend`, `text`, `style`, `figsize`, `ax`,
`save`); the [`Efficiency`][rootfig.Efficiency] objects (`values`, `lower`,
`upper`, `edges`) are in `Plot.efficiencies`.

## Profiles and resolutions

```python
rf.profile(sample, "true_E", "(reco_E - true_E) / true_E", statistic="std", bins=(20, 0, 100))
```

`statistic="mean"` (default) draws the weighted mean of `y` per bin of `x`
with its standard error (ROOT's `TProfile`); `"std"` draws the standard
deviation with its error, the usual resolution-versus-variable plot. `x` and
`y` must have the same structure; `xlabel` and `unit` describe the x axis. The
[`Profile`][rootfig.Profile] objects (`values`, `errors`, `counts`, `edges`)
are in `Plot.profiles`.

Negative weights (NLO samples) can make a weighted variance negative or an
efficiency leave `[0, 1]`. rootfig then reports `nan` for the standard
deviation, the profile error or the confidence interval (with a warning for
efficiencies) rather than a made-up uncertainty; means, yields and histogram
contents are unaffected. Bins whose total weight is negative keep their mean
or efficiency (the plain ratio) but get no uncertainty; bins whose weights
cancel to exactly zero count as empty (`nan`).

## Cut flows

```python
table = rf.cutflow(
    [zh, ww, zz],
    ["nMuon >= 2", rf.Cut("abs(m_ll - 91.2) < 10", label="Z window"), "recoil_mass > 120"],
    lumi="10.8 ab^-1",
)
print(table)  # yields ± error (raw events) and step efficiencies
table.get("ZH").efficiencies  # relative to the previous step
table.get("ZH").absolute_efficiencies
```

Step efficiencies are plain ratios of weighted yields (`nan` after a zero
yield). With signed (NLO) weights a yield can be negative and a ratio can lie
outside `[0, 1]`; it is reported as is.

Cuts apply cumulatively; a sample's own selection is the first row. Per-object
cuts pass an event when any object passes. `weight`, `lumi` and `nonfinite`
work as in `plot()`: events with a missing or non-finite weight are excluded
from every row (with a warning, or an error for `nonfinite="error"`).

## Statistics tables

```python
table = rf.summarize([sig, bkg], ["MET", "Muon_pt"], selection="nMuon > 0", weight="mc_weight")
print(table)  # aligned text table
table.get("MET", "Signal").mean  # a Summary: entries, mean, std, sem, skewness, min, max
```

## Histograms that already exist

`rf.plot`, `rf.histogram(s)` and `rf.plot2d` read histograms stored in ROOT
files; `rf.plot` and `rf.plot2d` also draw histogram objects you already have.
Each keeps its usual drawing options where they apply: the 1D options above
for `rf.plot` (stacks, ratios, `flow`, ...), those of the
[2D section](#2d-histograms-and-correlations) for `rf.plot2d`. Options that
need information a ready-made histogram does not contain (a selection, a weight,
`stats` on one without statistics) are refused.

### Histograms already in ROOT files

Analysis frameworks often write their selections out as `TH1`/`TH2` objects,
one file per process. Name the histogram where a branch would go:

```python
ww = rf.Sample("outputs/p8_ee_WW_ecm240.root", label="WW")
zz = rf.Sample("outputs/p8_ee_ZZ_ecm240.root", label="ZZ")
zh = rf.Sample("outputs/p8_ee_ZH_ecm240.root", label="ZH")
recoil = rf.Variable("zmumu_recoil_m", bins=(200, 120, 140), label="Recoil mass", unit="GeV")
rf.plot(
    [rf.Group([ww, zz], label="VV"), zh],
    recoil,
    stack=["VV"],
    logy=True,
    style=rf.Style(experiment="FCC-ee", com="240 GeV", lumi="5 ab^-1"),
)
h = rf.histogram("outputs/p8_ee_ZH_ecm240.root", "zmumu_recoil_m")  # a hist.Hist
rf.plot2d("outputs/p8_ee_ZH_ecm240.root", "zmumu_m_vs_recoil_m")  # a stored TH2
```

A [`Group`][rootfig.Group] of such samples sums their stored histograms into
one, each scaled by its sample's `scale` and luminosity factor first. See the
[complete histogram-file example](batch.md#a-complete-example-histogram-files)
for an overview book and selected plots from histogram files or ntuples.

The decision is made per call and is deterministic. A variable that is a bare
name is read as a stored histogram when every sample reads files without an
explicit `tree=` or entry range, the first file of each sample holds a `TH1` or
`TH2` of that name, and the file has no tree or its only tree has no branch of
that name. Anything else fills from the tree as usual: an explicit
`tree=` always means a branch, a branch of the same name wins over a histogram,
a file with several trees next to the histogram raises (pass `tree=` for a
branch, or read the histogram with `FileSource.read_histogram`), and samples
that disagree raise with the reason per sample. A histogram inside a directory
is named by its path in backticks, like a branch with odd characters:
`` rf.plot("histo.root", "`selection/mz`") ``.
`rf.PlotBook(file, rf.ALL)` discovers every stored `TH1` of a file, and every
branch of a tree whose values are numbers or booleans (lists and fixed-size
arrays of them included; strings and records are not), from the same metadata;
see [Automatic variable discovery](batch.md#automatic-variable-discovery).

What a stored histogram supports:

- The files of one `Sample` are summed (they must agree on the binning);
  `scale`, `color`, `is_data`, `histtype` and the luminosity scaling
  (`Sample(xsec=..., ngen=...)` with `lumi=`, `ngen` may name a `TParameter`
  or a sum-of-weights histogram in the same file) apply as for trees. A file
  without a tree has no entries to count, so it needs `ngen`.
- The stored axis title is the x label unless `xlabel=`/a `Variable` label is
  given; a `unit=` is appended to it and, as for trees, feeds the bin-width y
  label (`Events / 5 GeV`). A title that ends in `[unit]` already supplies it.
  A placeholder title (none, ROOT's `xaxis`, uproot's `Axis 0`) gives way to the
  variable's name. The y axis of a `TH2` named by a single variable gets no
  label from that name, which describes the histogram rather than the axis; as
  for any unlabelled axis, hist shows the axis name (`mz_recoil_2D_y`) instead.
  To label it, pass a `Variable` for `y` that names the same histogram, since
  both axes must resolve to one stored object:
  `` rf.plot2d(f, "my_hist", rf.Variable("my_hist", name="recoil", label="Recoil")) ``.
  A `Variable` with another expression asks for a branch instead.
- `bins=` and `range=` crop and merge the stored bins as described below.
  Every requested edge must coincide with an existing edge; the range can
  shrink but cannot grow. Content and variances outside the range join the
  flow bins. A crop requires the flow bin on each cropped side. Crop and
  rebin before normalising; asking for the bins already present is a no-op.

  | Specification | Filling a tree | Stored or ready-made histogram |
  | --- | --- | --- |
  | `bins=None`, range unset, `"auto"` or `"robust"` | `DEFAULT_BINS` over an inferred range | Keeps its binning |
  | `bins=20`, range unset, `"auto"` or `"robust"` | 20 bins over an inferred range | Merges the whole axis to 20 bins; the count must divide its size |
  | `bins=(20, 120, 140)`, `bins=20, range=(120, 140)`, explicit edges, or a `Regular`/`Variable` axis | Fills those bins, with entries outside in flow bins | Crops and merges to those edges, with contents outside in flow bins |
  | `bins=None, range=(120, 140)` | `DEFAULT_BINS` between 120 and 140 | Keeps its own bins between those edges |

  `bins=` accepts only `Regular` and `Variable` axes; integer, boolean and
  category axes must be expressed as a count/range or numeric edges instead.
- Normalisation, stacks, ratios, `flow` and the other drawing options work
  unchanged. Systematics of the normalisation kind (`{"lumi": 0.02}`) and
  `Systematic.samples(other_files)` (the same histogram read from other files)
  are supported by `plot` and `histograms`; `plot2d` ignores systematics, for
  stored histograms as for trees.
- `selection=`, `weight=` (on the call or the `Sample`), `nonfinite="error"`,
  weight and branch-replacement systematics and `stats=` need event data and raise with a
  message that says so.
- A `TH1` written with `Sumw2` keeps its uncertainties; one without it arrives
  with its bin contents as variances (uproot cannot know the weights). Negative
  contents without `Sumw2` leave no usable variances: rootfig refuses them unless
  `assume_poisson=True` takes the absolute contents, as for histogram objects.
  The files of one sample may mix both kinds and may differ in their axis
  titles; the sum has `Weight` storage and the first file's titles.

### Histogram objects

`hist.Hist` or [`Histogram`][rootfig.Histogram] objects, one or a list, are
drawn as they are: `rf.plot([h_sig, h_bkg], label=["Signal", "Background"],
ratio=True)`. `label=` names plain `hist.Hist` objects (otherwise their first
axis name is used), `observed=` takes histogram objects for the data,
`variable=` optionally supplies the axis label, unit and `log` flag, and
`rf.plot2d(h2)` draws a 2D one, with `x` and `y` `Variable`s optionally
describing its axes the same way (`rf.plot2d(h2, rf.Variable("mass",
label="Mass", unit="GeV"), rf.Variable("recoil", bins=6))`; a `name=` renames
the axis, the histogram you passed is left untouched). As for stored histograms, `bins=` (given
directly or on the `Variable`) crops and merges bins: an integer count, or edges that
coincide with the existing ones, so the `Variable` a histogram was filled with
can be passed along with it (`rf.plot(rf.histogram(sample, pt), pt)`), also
after normalising it, as long as it asks for the bins the histogram has. An
explicit `(low, high)` range without a bin count crops to those ends, which must
be existing edges, keeping the bins between them and moving the rest into flow. Options that fill from event data
(`tree`, `selection`, `weight`, `lumi`, `systematics`) raise;
`range="auto"`/`"robust"` are no-ops. `Histogram.variations` carries
systematics instead. Stacks, sums and ratios of histograms with category axes
(ROOT bin labels) require the same categories in the same order; the flow bins
of such an axis hold entries of categories it does not list, which
`flow="hint"` marks with an arrow and `flow="show"`/`flow="sum"` refuse, since
they have no bin beyond the last category.
