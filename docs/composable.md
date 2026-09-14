# Samples, variables, cuts and styles

`rf.plot` accepts plain strings everywhere, which is all you need for a quick
look. Analysis scripts that make dozens of plots from the same inputs are
clearer when the pieces are named once. Four small frozen dataclasses do
that; none of them opens a file or holds data.

## `Sample`

Where data comes from, how it is labelled, and how it is drawn.

```python
sig = rf.Sample("sig_*.root", tree="events", label="Signal", weight="mc_weight", color="tab:red")
bkg = rf.Sample(
    ["bkg_a.root", "bkg_b.root"],
    tree="events",
    label="Background",
    selection="passTrigger",
    weight="mc_weight",
    scale=0.98,
)
data = rf.Sample("data.root:events", label="Data", is_data=True)
mem = rf.Sample({"x": awkward_array, "w": weights}, label="in memory")
```

- `data` may be a path, glob, `"path:tree"`, a list of those, a mapping of
  arrays, an Awkward record array, a NumPy structured array, or any object
  implementing the [`Source`][rootfig.io.Source] protocol.
- `selection` and `weight` belong to the sample and combine with the ones
  given to `plot()` (`&` and `*` respectively).
- `is_data=True` draws points with error bars (in the style's text colour
  unless `color` is set), keeps the sample out of stacks and makes it the
  numerator of ratios.
- `xsec` and `ngen` describe simulated processes: the cross section (pb, or a
  string with a unit such as `"1.2 fb"`) and the number of generated events (a
  number, the name of an object in the file holding it, e.g. FCCAnalyses'
  `"eventsProcessed"` `TParameter` or a sum-of-weights histogram, or `None`
  for the number of entries). With `lumi=` given to `plot()`, `cutflow()`,
  `summarize()`, ... every weight is multiplied by `xsec × lumi / ngen`:

  ```python
  zh = rf.Sample("p8_ee_ZH_ecm240.root", label="ZH", xsec="0.201 pb", ngen="eventsProcessed")
  ww = rf.Sample("p8_ee_WW_ecm240.root", label="WW", xsec="16.4 pb", ngen="eventsProcessed")
  rf.plot([ww, zh], "recoil_mass", lumi="10.8 ab^-1", stack=True)  # expected yields
  ```

  Without a luminosity such samples raise a [`LuminosityError`][rootfig.LuminosityError].
- `entry_start`/`entry_stop` restrict reading for quick looks at large files
  (a plot reads every needed branch of every file into memory at once). For a
  ready-made `FileSource`, give the range to the source itself; passing it to
  `Sample` afterwards raises a [`SourceError`][rootfig.SourceError].
- `systematics={name: variation}` lists the sample's sources of systematic
  uncertainty: weight expressions (`("w_up", "w_down")`), normalisation
  uncertainties (`0.05`, `(1.1, 0.95)`), varied branches
  (`{"Jet_pt": ("Jet_pt_up", "Jet_pt_down")}`) and varied files
  ([`Systematic.samples`][rootfig.Systematic]). Sources with
  the same name are correlated across samples; see
  [Systematic uncertainties](plotting.md#systematic-uncertainties).
- `sample.with_(label="...")` returns a modified copy; replacement values are
  validated like constructor arguments.

Passing a list of files to `plot()` creates one sample per file. To merge
several files into *one* sample, use a glob or a `Sample`.

## `Variable`

What to histogram and how to present it.

```python
pt = rf.Variable("Muon_pt", bins=(50, 0, 200), label=r"$p_T^{\mu}$", unit="GeV")
met = rf.Variable(
    "MET / 1000", bins=40, range="auto", label=r"$E_T^{miss}$", unit="TeV", log=True, name="met"
)
```

- `bins`: an `int` (range inferred from the data), `(n, low, high)`, a sequence
  of edges (e.g. `rf.log_bins(30, 1, 1000)`), or a `hist.axis.Regular`/`Variable`.
- `range`: `(low, high)`, `"robust"` (the default: ignores far outliers such as
  `-999` sentinels and cuts a thin tail, both of which then land in the
  under/overflow) or `"auto"` (the finite min/max over all samples). See
  [Binning and range](plotting.md#binning-and-range).
- `label` and `unit` form the axis label `label [unit]`; the unit also appears
  in the automatic y label (`Events / 4 GeV`).
- `name` is used for file names by `Plot.save(directory)`; it must be a plain
  file stem (no path separators).

`bins`, `range`, `xlabel` and `unit` given to `plot()` override the variable.

## `Cut`

```python
base = rf.Cut("nMuon >= 1", label="1 muon")
sr = base & "abs(Muon_eta) < 2.5" & ~rf.Cut("isCosmic")
```

See [Expressions and selections](expressions.md).

## `Style`

Appearance, applied only while a figure is drawn (global matplotlib state is
untouched unless you call `rf.use_style`).

```python
atlas = rf.Style(
    experiment="ATLAS", status="Internal", lumi=140, com=13.6, text=[r"$Z \to \mu\mu$ selection"]
)
neutral = rf.Style(
    figsize=(6, 5), legend="upper left", colors=["#1b9e77", "#d95f02"], rc={"font.size": 12}
)
```

- `experiment` selects the matching mplhep style sheet and label helper
  (ATLAS, CMS, LHCb, ALICE, DUNE); any other name still gets a label with the
  neutral style. Nothing is drawn unless you ask for it.
- `status`, `lumi`, `com`, `text` fill the label; `simulation` controls the
  "Simulation" word (default: shown when there is no data sample; a status
  that already contains it is not doubled).
- `lumi` and `com` are numbers in `lumi_unit` (default fb⁻¹) and `com_unit`
  (default TeV), or strings with their own unit: `Style(experiment="FCC-ee",
  com="240 GeV", lumi="10.8 ab^-1")`. The energy is shown only when `com` is
  given.
- `base` can be any mplhep or matplotlib style name (`"ATLAS"`, `"ggplot"`,
  ...) or a mapping of rcParams; `rc` adds overrides on top; `colors` replaces
  the colour cycle. `label_loc` overrides the experiment's convention: 0 puts
  the experiment and secondary text above the frame; 3 puts the experiment
  above and secondary text inside; 1, 2, and 4 put both inside. Luminosity stays
  above for locations 0–3 and inside for 4. 2D histograms and correlation
  matrices default to location 0; only `label_loc=1`, `2`, or `4` moves the
  experiment inside the frame.
- The centre-of-mass energy and luminosity appear only when `com`/`lumi` are
  given; nothing is invented for you.
- `legend` is `True`, `False` or a location string; `legend_kwargs` are
  forwarded to `Axes.legend`.

A bare string is accepted too: `style="CMS"`.

## Putting it together

```python
samples = [bkg, sig]
variables = [pt, met, rf.Variable("nMuon", bins=(8, -0.5, 7.5))]

for var in variables:
    p = rf.plot(samples, var, observed=data, selection=sr, stack=True, ratio=True, style=atlas)
    p.save("plots/")  # plots/Muon_pt.pdf, plots/met.pdf, plots/nMuon.pdf
    p.close()
```
