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
  colour (black by default) unless the sample sets `color`. Their error bars are
  `√N`, as ROOT draws a `TH1`, or Poisson intervals with `data_errors="poisson"`
  ([Uncertainties of data](#uncertainties-of-data)).
- **Groups**: a [`Group`][rootfig.Group] of samples is one histogram of the
  overlay or stack, the sum of its components filled apart; see
  [Group](composable.md#group).
- `errorbars=True` adds statistical error bars to overlaid non-data histograms.

The [selective-stacking example](gallery/selective_stack.md) shows stacked
backgrounds with a signal drawn over them and a significance panel.

## Uncertainties of data

`data_errors=` chooses the statistical uncertainty of observed data, in the main
panel and in the lower panel alike, and in `p.uncertainty("Data")`:

| `data_errors=` | Error bars |
| --- | --- |
| `None` (default) | ROOT's `TH1` default: `√(Σw²)` on both sides, `√N` for counts, so an empty bin is 0 ± 0 |
| `"poisson"` | ROOT's `TH1::kPoisson`: the Garwood 68 % interval of the counts, asymmetric, and 0 +1.84 for an empty bin |
| `"auto"` | `"poisson"` where a data histogram holds unit-weight counts, `√(Σw²)` otherwise: the usual convention for data points |

- The model is decided on each data histogram as filled or read, before
  `normalize=` and `flow=` change it, so normalising data keeps its error model.
- `"poisson"` needs unit-weight counts: every bin a non-negative whole number
  equal to its variance. Unweighted data filled from a tree holds them, and so
  does a stored `TH1` without `Sumw2` holding whole numbers, which ROOT itself
  treats as unweighted. Anything else raises `ValueError`, data scaled by a
  common weight included: the sums of weights (`Σw`, `Σw²`) cannot tell counts
  scaled by one factor from unequal weights (`[1, 1, 4]` sums like two entries
  of weight 3), so no interval of counts describes them. `"auto"` keeps
  `√(Σw²)` for them instead.
- The interval of `n` counts runs from `L` to `U` with
  `P(N ≥ n | L) = P(N ≤ n | U) = 15.87 %` (`L = 0` for `n = 0`), as in ROOT.
- Normalising, rescaling and rebinning keep the interval, scaled like the
  contents: the histogram carries a record of one count per bin through each of
  them, which gives every bin its factor. After `normalize="width"` an empty
  bin gets `0 +1.84` divided by its own width, and an empty histogram scaled by
  3 gets `0 +5.52`; summed flow bins (`flow="sum"`) get the interval of the
  summed count. For a known constant (a luminosity, a bin width) this is the
  interval of the scaled counts. `normalize=True`, `"density"` and numeric
  targets divide by the histogram's own total, which fluctuates too: the bars
  are then the counts' interval scaled by the observed total, not an interval of
  the normalised shape ([Normalisation](#normalisation)). ROOT keeps
  `kPoisson` for unweighted histograms only and falls back to `√(Σw²)` once a
  histogram is scaled.
- The [lower panel](#lower-panel) propagates the two sides separately: a
  data/MC ratio runs from `L / d` to `U / d`, and a pull divides by the data
  error facing the prediction (the upper one where data lie below it).
- The bounds are gamma quantiles from SciPy, exact at any count, as ROOT's.
  [`poisson_interval`][rootfig.histograms.poisson_interval] gives them for any
  counts.
- `rf.Histogram(h, label="Data", is_data=True, poisson=True)` carries the model
  on a histogram of counts you pass yourself, so `rf.compare` uses it too and
  every `data_errors=` keeps it; counts scaled by `c` are
  `rf.Histogram(counts, ..., poisson=True).scaled(c)`. `sum_histograms` keeps it
  when every input carries the same record of counts (two data periods), as
  `TH1::Add` keeps `kPoisson` for unweighted histograms.

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
    [bkg, sig], "Jet_pt", observed=data, stack=True, panel="ratio", systematics={"lumi": 0.017}
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
- A source's name is its identity (its nuisance parameter). Sources with the
  same name are one source, fully correlated across samples: the stack total
  adds their variations linearly (a sample without the source contributes its
  nominal contents), and a comparison in the lower panel varies numerator and
  reference together, so a shared luminosity uncertainty cancels in an MC/MC
  ratio. Sources with different names are independent. To correlate two
  sources, give them one name; to keep them apart, two names. A plot-level
  source (`systematics=`) is one source shared by every simulated sample, which
  suits a luminosity; uncertainties independent between processes, such as
  their cross sections, take a name per sample:
  `Sample(..., systematics={"xsec_zh": 0.05})`.
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
and the [lower panel](#lower-panel) includes the systematics in its band
around the baseline (`split_ratio`, data/MC) or in the error bars of the points
(`propagate`: statistical uncertainties uncorrelated, systematic ones
propagated source by source through the varied comparison; in a ratio or a
relative difference, a variation that empties a reference bin leaves that
bin's systematic uncertainty undefined, with a warning), and a pull divides
by them. The
automatic range covers the bulk of the band and of the systematic error bars
(robust percentiles, like the points, so a single bin with a huge uncertainty
runs off the panel instead of squashing it; pass `panel_ylim` to show it in
full). A ratio stays at or above zero unless a central ratio is negative
(signed weights).

The numbers are part of the result. `Plot.stack` holds the sum of the stacked
histograms, labelled `"Total"` and including variations, or `None` without a
stack. `p.uncertainty()` uses that total, or the sole non-data histogram if
there is no stack; several overlays require a label, such as `p.uncertainty("ZH")`:

```python
u = p.uncertainty()  # the stack total (p.stack), or one histogram by label
u.stat_down, u.stat_up  # statistical, per bin, visible bins
u.syst_down, u.syst_up  # systematic
u.total_down, u.total_up  # statistical ⊕ systematic
u.components["jes"]  # signed (up − nominal, down − nominal) shifts
p.histograms[0].variations  # {"jes": (hist_up, hist_down), ...}
p.comparisons[0].syst_band  # (down, up) band of the reference, relative for a ratio
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

**Uncertainties.** Normalising multiplies each bin's contents by a factor and
its variance by the factor squared, taking the factor as a constant: the
plotting convention, under which a normalised histogram keeps the relative
uncertainties of its bins. The rescaling modes take the factor from the
histogram's own total, which fluctuates with the bins, so the bins of a
normalised shape are correlated and its full covariance differs (for `N`
counts normalised to unity, a bin with fraction `p` has variance `p / N` as
drawn, `p (1 − p) / N` as a multinomial shape). Error bars, bands, lower
panels and `Plot.uncertainty()` all use the scaled variances, and nothing in
rootfig claims the covariance of a shape. Systematic variations are normalised
by their own totals instead, so a pure normalisation uncertainty drops out of
the rescaling modes ([Systematic uncertainties](#systematic-uncertainties)),
and the [Poisson intervals of data](#uncertainties-of-data) scale with the same
factor as its contents, which makes them the counts' interval over the observed
total rather than an interval of the shape.

The rescaling modes divide by the signed sum of the visible bins: a histogram
dominated by negative weights still sums to the target, its shape flips sign,
and a warning says so. An empty histogram, or one whose positive and negative
weights cancel exactly, is left unchanged with a warning and keeps the plain
`Events` label (`Histogram.normalization` stays `None`).

## Lower panel

`panel=` adds a lower panel sharing the x axis and says what it shows;
`reference=` names the histogram it compares with. With `n` and `d` the
contents of a numerator and of the reference, and `vn` and `vd` their
variances (the formulas for symmetric errors; data with [Poisson
intervals](#uncertainties-of-data) enters each side with the error that moves
the result that way):

| `panel=` | Values | Error bars: `propagate` / `numerator` | Band (`numerator`) | Automatic range |
| --- | --- | --- | --- | --- |
| `"ratio"` | `n / d`, around 1 | `√(vn/d² + n²·vd/d⁴)` / `√vn / abs(d)` | `√vd / abs(d)` around 1 | at least 0.5 to 1.5, widened to the bulk of the points, their systematic error bars and the band; within 0 to 3, or −3 to 3 when a ratio is negative |
| `"relative_difference"` | `n / d − 1`, around 0 | the ratio's | the ratio's, around 0 | the ratio's range moved down by one: at least −0.5 to 0.5, within −1 to 2 (−4 to 2) |
| `"difference"` | `n − d`, around 0 | `√(vn + vd)` / `√vn` | `√vd` around 0 | symmetric: ±1.1 times the largest magnitude of the 5th and 95th percentiles of the points, their systematic extent and the band; ±1 when all are zero |
| `"pull"` | `(n − d) / σ`, `σ² = vn + vd + σ_syst²` | none: 1 by construction | none | symmetric: ±1.1 times the 95th percentile of the magnitudes, at least ±3 and at most ±5 |
| `"asymmetry"` | `(n − d) / (n + d)`, around 0 | `2·√(d²·vn + n²·vd) / (n + d)²`; `propagate` only | none | the difference's, at most ±1.1 so that ±1 (an empty side) stays in the frame |
| `"s/sqrt(b)"`, `"s/sqrt(s+b)"` | `S/√B`, `S/√(S+B)` per bin, the reference as background | statistical, propagated | none | 0 to 1.25 times the highest point plus its error |

Ratios, relative and absolute differences and asymmetries are points with
error bars and a dashed line at their baseline, over the grey reference band
where one is drawn (the `"numerator"` mode below); pulls are filled bars from 0
in the numerator's colour; significances are points without a baseline. Every
numerator gets its own series, in its histogram's colour. A bin is left empty
where the value is undefined: an empty reference for a ratio or a relative
difference, `σ = 0` for a pull, `n + d = 0` for an asymmetry, and no background
(or no signal plus background) for a significance. Only the bins inside the
visible x range (`xlim`, both segments of `xbreak`) set the automatic range.
Statistical error bars do not widen it, so a few low-statistics bins cannot
squash the panel; a significance is the exception, its upper limit following
the highest point plus its error. A point beyond the range is marked by a
triangle at the edge it left through, in its colour, so the robust range never
hides a bin; the markers follow a later `p.panel_ax.set_ylim(...)`.

`σ_syst` of a pull is the combined [systematic
uncertainty](#systematic-uncertainties) of `n − d`: a source carried by both
histograms varies them together, and one carried by a single histogram varies
it against the other's nominal contents. The combination is taken on the side
facing the other histogram: the lower one where `n > d`, the upper one
elsewhere (mplhep's rule for Poisson pulls).
Significance panels use statistical uncertainties only.

**Roles.** Without `reference=`:

| Drawn | Ratio, differences, asymmetry, pull: numerators / reference | Significance: signals / background |
| --- | --- | --- |
| stack and data | data / stack total; overlays are not part of the prediction | overlaid non-data histograms / stack total, or, with everything stacked, the last non-data histogram / the sum of the others |
| stack, no data | every overlaid histogram / stack total; a full stack raises | as above |
| no stack, data | data / the first non-data histogram | the last non-data histogram / the sum of the others |
| no stack, no data | every histogram after the first / the first | the last non-data histogram / the sum of the others |
| observed data alone | every data histogram after the first / the first | raises: no non-data histograms |

With the signal last, drawing it inside the stack or over it shows the same
significance panel, and stacking the backgrounds (`stack=["WW", "ZZ"]`)
compares every overlaid signal with them. `reference="Background"` names one
histogram (a group's label counts): the denominator of a ratio or a relative
difference, the `d` of `n − d` for a difference, an asymmetry and a pull, and
the background of a significance. Every other histogram, data included, is compared with it;
for a significance every other non-data histogram is a signal over it. A label
that no drawn histogram or several carry raises `ValueError`, as does observed
data as a background, or `reference=` without `panel=`.

**Uncertainties.** A ratio, relative difference or difference draws its error
bars in one of two modes, chosen per numerator: data over simulation keeps the
reference uncertainty as the grey band (`"numerator"`, mplhep's
`split_ratio`), and everything else propagates both sides (`"propagate"`), so
sources shared by numerator and reference cancel. `panel_uncertainty=` applies
one mode to all numerators; it raises for a pull, an asymmetry or a
significance, which have no band.

**Range and label.** `panel_ylim` and `panel_label` override the automatic
range and label. The label names the reference, `MC` for the stack total, and
is worded for data only when every numerator is observed data and the reference
is simulated; a named reference with simulation and data compared with it keeps
the general label:

| `panel=` | Data alone over simulation | Otherwise |
| --- | --- | --- |
| `"ratio"` | `Data / MC` | `Ratio to X` |
| `"relative_difference"` | `(Data − MC) / MC` | `Rel. difference to X` |
| `"difference"` | `Data − MC` | `Difference to X` |
| `"pull"` | `Pull` | `Pull` |
| `"asymmetry"` | `(Data − MC) / (Data + MC)` | `Asymmetry to X` |
| `"s/sqrt(b)"`, `"s/sqrt(s+b)"` | `S/√B`, `S/√(S+B)` | the same |

A rotated y label is bounded by the height of the short panel, so a long one
is shrunk and, if that is not enough, wrapped onto two lines; pass a shorter
`panel_label` such as `"Ratio"` to keep it at full size.

**Result.** `Plot.comparisons` holds one [`Comparison`][rootfig.Comparison]
per numerator (`kind`, `label`, `reference`, `values`, `errors`, `edges`,
`band`, and `syst_errors`/`syst_band` with systematic uncertainties), in the
order drawn; [`rf.compare`][rootfig.compare] computes one from any two
histograms. Every uncertainty in it is a `(down, up)` pair of arrays,
matplotlib's `yerr` order, so asymmetric errors stay asymmetric: each side of a
numerator and of the reference enters with the error that moves the result the
same way, which with symmetric errors is the formula in the table.

A detector or software comparison, a fast simulation against the full one,
normalised to compare shapes and with the full simulation as the reference
because it comes first:

```python
full = rf.Sample("full_sim.root", tree="events", label="Full simulation")
fast = rf.Sample("fast_sim.root", tree="events", label="Fast simulation")
rf.plot([full, fast], "Muon_pt", normalize=True, panel="relative_difference")
```

The gallery shows a [ratio to a chosen sample](gallery/ratio_reference.md), a
[pull](gallery/pull.md), a [significance panel](gallery/selective_stack.md) and a
[ratio of efficiencies](gallery/efficiency.md).

**Efficiencies and profiles.** [`rf.efficiency`](#efficiencies) and
[`rf.profile`](#profiles-and-resolutions) take `panel=` (every kind but the
significances, which count events), `reference=`, `panel_ylim` and `panel_label`.
The roles are those without a stack: data over the first simulated sample, or
every further sample over the first. Both sides are independent points, so
their intervals are propagated to first order, each side entering with the error
that moves the result the same way: a ratio of efficiency intervals keeps its
asymmetry, and a pull divides by the errors facing the other side. There is no band and no `panel_uncertainty`.

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
or both segments of `xbreak`. This applies to the main panel and to the lower
panel alike.

- `logx`, `logy`: logarithmic scales. Log-spaced bins: `bins=rf.log_bins(n, low, high)`.
- `xlim`, `ylim`: limits; `ylim=(None, 1e4)` keeps the automatic lower value.
  Automatic y limits add a small margin above the tallest bin (a factor 1.2 in
  linear scale, 12 in log scale) and then raise it further as the drawn
  legend, label, statistics box and text lines need.
- `xbreak=(a, b)`: cut the range between `a` and `b` out of the x axis and
  draw the two remaining segments side by side with break marks, sharing the
  y axis (and the lower panel, if any). Useful for a peak plus a far tail or
  a sentinel region. The right segment is `Plot.ax_right`
  (`Plot.panel_ax_right`). Not available together with `ax=` or `flow="show"`.

  ![Broken x axis with a ratio panel](images/gallery/xbreak_ratio-atlas.png#only-light){ width="60%" }
  ![Broken x axis with a ratio panel](images/gallery/xbreak_ratio-atlas-dark.png#only-dark){ width="60%" }
- `flow`: how under/overflow is shown (this is where entries outside an
  inferred [range](#binning-and-range) end up), `"hint"` (small arrows, default),
  `"show"` (extra bins labelled `<low` / `>high`, added on a side as soon as any
  sample has content there, identical for all samples and the lower panel),
  `"sum"` (added to the edge bins before anything is computed, so the lower
  panel, stack bands and y limits use the folded bins), `"none"`.
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
  `(main, panel)` with `panel=`), so several rootfig plots can share a figure.
- `save="file.pdf"` saves immediately; `Plot.save(path)` accepts a directory
  (file named after the variable) and `formats=["pdf", "png"]`.
- Figures use matplotlib's constrained layout, so labels, legends and colour
  bars fit inside the canvas and a saved file has exactly the `figsize`
  dimensions: 1D, 2D and lower-panel plots of one size share one shape. Figures
  drawn into your own `ax` are saved with a tight bounding box instead.
- Fonts are fixed on the figure when it is made, so saving or displaying it
  later renders exactly the layout that was computed, in the style's fonts.
- `Plot.fig`, `Plot.ax`, `Plot.panel_ax` are plain matplotlib objects;
  `Plot.histograms` wrap the `hist.Hist` objects with labels and statistics.
- In a notebook the figure is displayed automatically — it is a pyplot figure,
  flushed by the inline backend at the end of the cell, so `%matplotlib inline`
  is not needed. End the call with `;` to hide the `Plot` repr, and use
  `Plot.close()` in loops that make many figures.

## Large inputs

A plot reads only the branches its variable, selection, weight and systematics
use, a chunk of entries (about 32 MB of arrays) at a time. The chunks are
decompressed and prepared (expressions, selection, weights) in parallel
threads, and only the values that fill the histograms are kept, so the peak
memory follows those values rather than every branch read. The histograms and
their statistics are the same as from reading everything at once.

- Decompressing and preparing share up to eight worker threads, fewer on a
  machine with fewer cores. Set `ROOTFIG_THREADS` to choose the number, e.g. the
  cores a batch job was given; `ROOTFIG_THREADS=1` decompresses and prepares in
  the calling thread. uproot fetches the file contents in a thread or two of its
  own either way, which do little but wait for the data.
- An explicit range (`bins=(50, 0, 200)`, or `range=(low, high)`) skips range
  inference, which takes two medians over all the values of every sample.
- Many plots of the same files are fastest as a [`PlotBook`](batch.md), which
  reads each file once per batch of variables: a tree with thousands of
  branches costs a noticeable fraction of a second to open, every time.
- LZMA-compressed files (the NanoAOD default) take several times longer to
  decompress than ZLIB, LZ4 or ZSTD ones; the threads help most there.
- A sample whose files hold a branch in different types (`float32` in one,
  `float64` in another) is read whole once the difference shows, and evaluated
  in the wider type. The first file's chunks may already have been evaluated in
  their own type by then, so NumPy can warn about an overflow that the wider
  type avoids. The histograms are the same either way, also when NumPy is set
  to raise such errors (`np.errstate(over="raise")`, a warnings filter).

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
efficiency is drawn as points with a confidence interval of `z=1` standard
deviations by default, chosen with `interval=`. The passing entries are a
subset of all of them, so the uncertainty is that of a pass fraction, never
that of two independent yields.

| `interval=` | Interval |
| --- | --- |
| `"auto"` (default) | what ROOT's `TEfficiency` and `TGraphAsymmErrors::Divide` give: `"clopper-pearson"` for unweighted entries, `"normal"` for weighted ones |
| `"clopper-pearson"` | the exact binomial interval of the counts, never covering less than 68 %; unweighted entries only |
| `"normal"` | `ε ± z·σ` clipped to `[0, 1]`, with `σ² = (Σw²_pass (1 − 2ε) + Σw²_all ε²) / (Σw_all)²` (`ε(1 − ε) / n` for counts); no width at 0 and 1 |
| `"wilson"` | the Wilson score interval of the counts; unweighted entries only. It keeps a width at 0 and 1 |
| `"wilson-effective"` | rootfig's extension of `"wilson"` to weighted entries: the Wilson interval of the effective entries `n_eff = (Σw)² / Σw²` of the denominator. Entries passing with probability `ε` give a weighted fraction of variance `ε(1 − ε) / n_eff`, and the interval inverts that. The same as `"wilson"` for counts |

Entries count as unweighted as ROOT decides: when the sum of weights equals the
sum of squared weights (to 10⁻¹², `TEfficiency`'s tolerance for `TH1D`), so
every weight is 1 (or 0 and 1). The sample's `scale` and luminosity factor cancel in an
efficiency and are left out, so they neither make a sample weighted nor, when
zero or negative, change its efficiency; a `weight=`, even one constant for
every entry, makes it weighted.
The normal approximation shrinks to nothing at 0 % and 100 %, which is why the
two Z + jets bins at 100 % in the [gallery](gallery/efficiency.md) have no error bar;
`"wilson-effective"` is the alternative that keeps one for weighted samples.
ROOT has no Clopper–Pearson or Wilson interval for weighted entries: asked for
one, it warns and uses the normal approximation. rootfig raises instead, so an
explicit method never silently changes. The Clopper–Pearson bounds are beta
quantiles from SciPy, as ROOT's, and agree with ROOT's to 10⁻¹¹.
Options are
the usual axis, legend, label and style ones (`xlabel`, `ylabel`, `unit`,
`title`, `logx`, `xlim`, `ylim`, `legend`, `text`, `style`, `figsize`, `ax`,
`save`); the [`Efficiency`][rootfig.Efficiency] objects (`values`, `lower`,
`upper`, `edges`) are in `Plot.efficiencies`. `panel="ratio"` adds the
[lower panel](#lower-panel) of a scale factor, data over simulation:

```python
rf.efficiency([mc, data], "Muon_pt", passed="Muon_isTight", bins=(20, 0, 100), panel="ratio")
```

## Profiles and resolutions

```python
rf.profile(sample, "true_E", "(reco_E - true_E) / true_E", statistic="std", bins=(20, 0, 100))
```

`statistic="mean"` (default) draws the weighted mean of `y` per bin of `x`
with its standard error (ROOT's `TProfile`); `"std"` draws the standard
deviation with its error, the usual resolution-versus-variable plot. `x` and
`y` must have the same structure; `xlabel` and `unit` describe the x axis. The
[`Profile`][rootfig.Profile] objects (`values`, `errors`, `counts`, `edges`)
are in `Plot.profiles`. A [lower panel](#lower-panel) compares configurations,
here the resolution of a new reconstruction against the nominal one:

```python
rf.profile(
    [nominal, new],
    "true_E",
    "(reco_E - true_E) / true_E",
    statistic="std",
    bins=(20, 0, 100),
    panel="difference",
)
```

Negative weights (NLO samples) can make a weighted variance negative or an
efficiency leave `[0, 1]`. rootfig then reports `nan` for the standard
deviation, the profile error or the confidence interval (with a warning for
efficiencies) rather than a made-up uncertainty; means, yields and histogram
contents are unaffected. Bins whose total weight is negative keep their mean
or efficiency (the plain ratio) but get no uncertainty; bins whose weights
cancel to exactly zero count as empty (`nan`). The normal approximation, the
default for weighted entries, propagates the sums to first order, which holds
for signed weights too, as in ROOT. A binomial interval (any other method)
assumes non-negative weights, so with it an efficiency bin that any
entry with a negative weight falls into has no interval even when its ratio
lies in `[0, 1]`: `rf.efficiency` knows them from filling. The lower-level
`rootfig.histograms.efficiency` sees only the sums of
the two histograms, which reveal a negative weight when a part of the bin has
a sum of squared weights above the square of its sum; its `negative_weights=`
flags the bins the sums do not reveal.

## Cut flows

```python
table = rf.cutflow(
    [zh, ww, zz],
    ["nMuon >= 2", rf.Cut("abs(m_ll - 91.2) < 10", label="Z window"), "recoil_mass > 120"],
    lumi="10.8 ab^-1",
)
print(table)  # yields ± error (raw events) and step efficiencies
table.get("ZH").efficiencies  # relative to the previous step
table.get("ZH").efficiency_errors  # (down, up)
table.get("ZH").absolute_efficiencies
table.get("ZH").absolute_efficiency_errors
```

Step efficiencies are ratios of weighted yields (`nan` after a zero yield).
Every step keeps a subset of the events before it, so their uncertainty is
that of a pass fraction, not that of two independent yields:
`efficiency_errors` and `absolute_efficiency_errors` are the `(down, up)`
distances to the confidence interval at one standard deviation that
`interval=` names, as for [efficiencies](#efficiencies), of the summed event
weights. The default is what `TEfficiency` gives for histograms of the steps:
Clopper–Pearson when every step's sum of weights equals its sum of squared
weights (every weight 1, or 0 and 1: an event of weight 0 is no trial), the
normal approximation otherwise; `Cutflow.interval` says which.
Efficiencies and their errors come from the event weights alone
(`CutflowStep.sum_w`, `sum_w2`): the sample's `scale` and luminosity factor
cancel, whatever their sign, and enter only the yields. With
signed (NLO) weights a yield can be negative and a ratio can lie outside
`[0, 1]`; the ratio is reported as is, without errors outside `[0, 1]`, and
with `interval="wilson-effective"` also without errors when measured against events
that include a negative weight (`CutflowStep.negative_weights`), since no
binomial interval describes them. Cut flows are statistical only; systematic
variations are not propagated through them.

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
for `rf.plot` (stacks, lower panels, `flow`, ...), those of the
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
- Normalisation, stacks, lower panels, `flow` and the other drawing options work
  unchanged. Systematics of the normalisation kind (`{"lumi": 0.02}`) and
  `Systematic.samples(other_files)` (the same histogram read from other files)
  are supported by `plot` and `histograms`; `plot2d` ignores systematics, for
  stored histograms as for trees.
- `selection=`, `weight=` (on the call or the `Sample`), `nonfinite="error"`,
  weight and branch-replacement systematics and `stats=` need event data and raise with a
  message that says so.
- A `TH1` written with `Sumw2` keeps its uncertainties; one without it arrives
  with its bin contents as variances (uproot cannot know the weights). As
  observed data, one without `Sumw2` holding whole numbers is taken as
  unweighted counts, which `data_errors="poisson"` or `"auto"` give
  [Poisson intervals](#uncertainties-of-data). Negative
  contents without `Sumw2` leave no usable variances: rootfig refuses them unless
  `assume_poisson=True` takes the absolute contents, as for histogram objects.
  The files of one sample may mix both kinds and may differ in their axis
  titles; the sum has `Weight` storage and the first file's titles.

### Histogram objects

`hist.Hist` or [`Histogram`][rootfig.Histogram] objects, one or a list, are
drawn as they are: `rf.plot([h_sig, h_bkg], label=["Signal", "Background"],
panel="ratio")`. `label=` names plain `hist.Hist` objects (otherwise their first
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
systematics instead. Stacks, sums and comparisons of histograms with category axes
(ROOT bin labels) require the same categories in the same order; the flow bins
of such an axis hold entries of categories it does not list, which
`flow="hint"` marks with an arrow and `flow="show"`/`flow="sum"` refuse, since
they have no bin beyond the last category.
