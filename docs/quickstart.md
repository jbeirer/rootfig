# Quick start

## Install

```bash
pip install rootfig        # or: uv add rootfig
```

## The one-liner

```python
import rootfig as rf

rf.plot("events.root", "Muon_pt", tree="events", selection="Muon_pt > 20", bins=50)
```

`rf.plot` reads the branches referenced by the variable, the selection and
the weight (nothing else), evaluates them, fills a `hist.Hist` and draws it.
If the file contains exactly one tree you can leave `tree` out. The file can
also be a glob (`"run_*.root"`), a list of files, or `"file.root:tree"`.

A bare `bins=50` infers the range from the data, ignoring far outliers so that
sentinel values such as `-999` do not set the axis; pass `range=(low, high)` to
be explicit or `range="auto"` for the full extent (see
[Binning and range](plotting.md#binning-and-range)).

The return value is a [`Plot`][rootfig.Plot] with `fig`, `ax`, `hists` and a
`save()` method:

```python
p = rf.plot("events.root", "MET", tree="events", bins=(40, 0, 400), unit="GeV")
p.ax.axvline(100, color="gray", linestyle="--")
p.save("met.pdf")
```

## Selections and weights

```python
rf.plot(
    "events.root",
    "Muon_pt",
    tree="events",
    selection="Muon_pt > 20 and abs(Muon_eta) < 2.5",
    weight="mc_weight * pileup_weight",
    bins=(50, 0, 200),
)
```

`Muon_pt` is jagged (a list of muons per event), so the selection masks
individual muons and every muon inherits its event's weight. See
[Expressions and selections](expressions.md) for the full rules.

## Several samples

A list of files gives one histogram per file, with a binning shared by all:

```python
rf.plot(
    ["signal.root", "background.root"],
    "Muon_pt",
    tree="events",
    bins=(50, 0, 200),
    normalize=True,
    ratio=True,
)
```

Use a mapping to name them, or [`Sample`][rootfig.Sample] objects for full
control:

```python
rf.plot({"Signal": "sig.root", "Background": "bkg_*.root"}, "Muon_pt", tree="events")
```

## Stacked MC with data and a ratio panel

```python
mc = [
    rf.Sample("ttbar.root", tree="events", label=r"$t\bar{t}$", weight="mc_weight"),
    rf.Sample("wjets.root", tree="events", label="W + jets", weight="mc_weight"),
]
data = rf.Sample("data.root", tree="events", label="Data", is_data=True)

rf.plot(
    mc,
    "MET",
    observed=data,
    stack=True,
    ratio=True,
    logy=True,
    bins=(40, 0, 400),
    unit="GeV",
    style=rf.Style(experiment="CMS", status="Preliminary", lumi=138, com=13),
)
```

## Histograms and arrays without plotting

```python
h = rf.histogram("events.root", "MET", tree="events", selection="nJet >= 2", bins=(40, 0, 400))
h.values(), h.variances()  # a hist.Hist with Weight storage

arrays = rf.load(
    "events.root", ["MET", "Jet_pt", "count(Jet_pt)"], tree="events", selection="nJet >= 2"
)  # an Awkward record array
```

## Statistics, 2D histograms, correlations

```python
print(rf.summarize("events.root", ["MET", "Muon_pt"], tree="events", selection="nMuon > 0"))

rf.plot2d("events.root", "Muon_pt", "Muon_eta", tree="events", bins=(40, 20), logz=True)

rf.correlation("events.root", ["MET", "nJet", "HT"], tree="events")
```
