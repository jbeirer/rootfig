# Plotting options

All options below are keyword arguments of [`rf.plot`][rootfig.plot] (and of
[`rf.plot_histograms`][rootfig.plot_histograms], which draws existing
`hist.Hist` objects with the same options).

## Overlays, stacks and data

- **Overlay** (default): every sample is drawn as an outline (`histtype="step"`).
  Use `histtype="fill"` for translucent filled areas, `"errorbar"` for points,
  `"band"` for uncertainty bands; a `Sample(histtype=...)` overrides per sample.
- **Stack**: `stack=True` stacks all non-data samples as filled histograms in
  the given order (first sample at the bottom) and draws a hatched band for
  the statistical uncertainty of the total.
- **Data**: samples with `is_data=True` (or passed as `observed=...`) are
  black points with error bars, drawn on top and never stacked.
- `errorbars=True` adds statistical error bars to non-data histograms.

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

Variances are scaled consistently. Flow bins scale with the same factor; for
`"width"` and `"density"` they are divided by the width of the neighbouring
visible bin. Plain `hist.Hist` objects with a count storage passed to
`plot_histograms` are converted to `Weight` storage first.

## Ratio panel

`ratio=True` adds a lower panel sharing the x axis:

- with a stack: data / total MC, error bars from the data, grey band for the
  MC statistical uncertainty (`ratio_uncertainty="numerator"`);
- with data and overlaid samples: data / first sample;
- otherwise: every sample / the first sample, uncertainties of both
  propagated in quadrature (`ratio_uncertainty="propagate"`).

`ratio="Background"` picks the reference by label.

`ratio="significance"` (or `"s/sqrt(b)"`, `"s/sqrt(s+b)"`) draws a
**significance panel** instead: per bin, the signal over the square root of
the background (or of signal plus background), with propagated
uncertainties. The signal is the last non-data sample (the top of a stack)
and the background the sum of the others; `ratio=("s/sqrt(b)", "ZH")` names
the signal. The values are returned as a `Ratio` in `Plot.ratios`.

`ratio_ylim` and
`ratio_label` override the automatic range (at least 0.5 to 1.5, widened to
cover the bulk of the points) and label (`Ratio to X` or `Data / MC`). The
computed values are returned in `Plot.ratios` as
[`Ratio`][rootfig.Ratio] objects (`values`, `errors`, `band`, `edges`).

## Axes

- `logx`, `logy`: logarithmic scales. Log-spaced bins: `bins=rf.log_bins(n, low, high)`.
- `xlim`, `ylim`: limits; `ylim=(None, 1e4)` keeps the automatic lower value.
  Automatic y limits leave room for the legend and label (a factor 1.45 in
  linear scale, 30 in log scale).
- `xbreak=(a, b)`: cut the range between `a` and `b` out of the x axis and
  draw the two remaining segments side by side with break marks, sharing the
  y axis (and the ratio panel, if any). Useful for a peak plus a far tail or
  a sentinel region. The right segment is `Plot.ax_right`
  (`Plot.ratio_ax_right`). Not available together with `ax=` or `flow="show"`.

  ![Broken x axis with a ratio panel](images/gallery/xbreak_ratio.png){ width="60%" }
- `flow`: how under/overflow is shown, `"hint"` (small arrows, default),
  `"show"` (extra bins labelled `<low` / `>high`, added on a side as soon as any
  sample has content there, identical for all samples and the ratio panel),
  `"sum"` (added to the edge bins before anything is computed, so ratios,
  significances, stack bands and y limits use the folded bins), `"none"`.
- `logx`, `logy`: logarithmic axes. `logx=None` (default) follows the
  `Variable`'s `log` flag (`plot2d`, `efficiency` and `profile` do the same for
  their variables); `True`/`False` override it.
- `xlabel`, `ylabel`, `unit`, `title`. The title sits above the axes, where
  the CMS-style label is also drawn; with such a style prefer `text=`.
- Automatic y limits leave room for the legend, the experiment label, the
  statistics box and `text` lines: the upper limit is raised until none of
  them covers a histogram (the legend picks a free upper corner). A `ylim`
  with an explicit upper value switches this off.

## Legend, labels, text and statistics

- `legend=False` or a location string such as `"upper left"`.
- `style=` an experiment name or a [`Style`][rootfig.Style]; see
  [Samples, variables, cuts and styles](composable.md).
- `text=` extra line(s) drawn with the experiment label.
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

## 2D histograms and correlations

```python
rf.plot2d(sample, x, y, bins=((40, 0, 200), (30, -3, 3)), logz=True, normalize="density")
rf.correlation(sample, ["MET", "nJet", "HT"], selection="nJet >= 2", percent=True)
```

`rf.plot2d` options: `logz`, `logx`, `logy`, `cmap` (any matplotlib colour
map, default `viridis`), `colorbar=False`, `zlabel` (default `Events` or the
normalisation), `normalize`, `title`, `text`, `style`, `figsize`, `ax`, `save`.
The figure has the same size as a 1D plot; the colour bar takes its space from
the axes.

`rf.correlation` options: `labels` (tick labels, default the variable labels),
`percent=True` (integer percentages instead of two-decimal coefficients),
`annotate=False` (colours only), `cmap` (default `RdBu_r`), `title`, `style`,
`figsize`, `ax`, `save`. The matrix is returned as `Plot.matrix`.

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
contents are unaffected.

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
