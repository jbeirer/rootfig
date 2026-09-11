# Changelog

All notable changes to rootfig are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/): until 1.0 the public API may
change in minor releases, with deprecation warnings where practical.

## [Unreleased]

## [0.1.0] - 2026-09-11

First release.

### Added

- `rf.plot`: one call from ROOT files (TTree or RNTuple, globs, multiple
  files) to a styled matplotlib figure, with selections, weights, shared
  binning, overlays, stacks, observed-data points, normalisation, ratio
  panels, log axes, flow-bin display, a broken x axis (`xbreak`), legends,
  statistics boxes and experiment labels.
- `rf.plot_histograms` for existing `hist.Hist` objects (plain count
  storages are converted to `Weight` storage; overlays may differ in
  binning), `rf.plot2d`, `rf.correlation`, `rf.summarize`,
  `rf.histogram`/`rf.histograms`, `rf.load`, `rf.evaluate`, `rf.ratio`,
  `rf.log_bins`, `rf.use_style`.
- `rf.cutflow()`: weighted yields, raw counts and step efficiencies after
  successive cuts, per sample (`CutflowTable`, `Cutflow`).
- `rf.efficiency()`: pass/total versus a variable with Wilson intervals
  (`Efficiency`); `rf.profile()`: mean or standard deviation of `y` in bins
  of `x` (`Profile`), for resolution plots.
- Significance panel: `ratio="significance"` / `"s/sqrt(b)"` /
  `"s/sqrt(s+b)"` or `("s/sqrt(b)", "Signal")`; `rf.significance()`.
- Luminosity normalisation: `Sample(xsec=..., ngen=...)` plus `lumi=` scales
  simulated samples to expected yields (`xsec × lumi / ngen`, units in
  strings such as `"1.2 fb"`, `"10.8 ab^-1"`); `ngen` may name an object in
  the file (a `TParameter` such as FCCAnalyses' `eventsProcessed`, or a
  sum-of-weights histogram). `LuminosityError`.
- `Sample`, `Variable`, `Cut` and `Style` descriptions for composable
  analysis scripts; `Style.lumi`/`Style.com` accept units and
  `Style(lumi_unit=..., com_unit=...)` set the defaults for numbers.
- Expression language in Python syntax with `and`/`or`/`not`, chained
  comparisons, backtick-quoted and dotted branch names
  (`ReconstructedParticles.momentum.x`, as podio/EDM4hep files write them),
  NumPy functions, per-event reductions (`count`, `sum`, `min`, `max`,
  `mean`, `any`, `all`, `first`, ...) and kinematic helpers `pt`, `p`,
  `theta`, `costheta`, `eta`, `phi`, `mass` from Cartesian components.
- Documented and tested per-event/per-object semantics for selections and
  weights, with explicit errors for ambiguous combinations. Fixed-size
  collections (`float x[3]` branches, two-dimensional NumPy arrays) follow
  the per-object rules like variable-length lists. Split TTree branches and
  nested RNTuple fields are listed and read by their dotted leaf name.
- Experiment-neutral default style with a 40-colour cycle (`tab10` extended
  by the `tab20` light companions and `tab20b`/`tab20c`); mplhep styles and
  labels for ATLAS, CMS, LHCb, ALICE and DUNE via `Style(experiment=...)`.
- Automatic y headroom accounts for the actual legend, experiment label,
  statistics box and text sizes, so they never cover a histogram (unless
  `ylim` sets the upper value).
- A gallery (`examples/gallery.py`, `docs/gallery.md`) with 20 examples
  covering the plotting options, each shown next to its code, doubling as an
  image-regression suite (`tests/test_gallery.py`, pytest-mpl) whose
  baselines are the documentation images.
- Lower-layer entry points for callers building on rootfig:
  `rootfig.selection.prepare`/`boolean_mask`, `rootfig.histograms.fill`,
  `normalize_hist`, `as_weight_storage`, `read_arrays`, `source_length`,
  `rootfig.plotting.draw_histograms`, `show_flow_bins`, `fold_flow_bins`.

### Behaviour worth knowing

- `nan`/`inf` values are dropped with a `RootfigWarning`
  (`nonfinite="error"` raises), also in cut flows; missing values (`None`),
  missing collections and missing event weights drop the entry or event and
  are counted separately in `Summary`.
- Negative weights: a weighted variance that comes out negative gives `nan`
  for the standard deviation of `Summary` and `Profile` (and the profile
  error); an efficiency outside `[0, 1]` has `nan` interval bounds and
  warns. Histogram contents, means and yields are unaffected.
- `flow="show"` and `flow="sum"` are applied by rootfig before ratios,
  significances, stack bands and limits are computed, identically for all
  samples; `normalize="width"`/`"density"` divide flow bins by the
  neighbouring bin width.
- Ratios, significances and efficiencies require identical bin edges (up to
  a millionth of a bin width). `Sample.scale`, `xsec`, `ngen` and `lumi`
  must be finite. `logx`/`logy` default to the `Variable`'s `log` flag.
- Requires Python 3.12 and matplotlib 3.10 or newer.

[Unreleased]: https://github.com/jbeirer/rootfig/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jbeirer/rootfig/releases/tag/v0.1.0
