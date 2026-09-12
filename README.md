# rootfig

**Publication-quality figures straight from ROOT trees, without ROOT.**

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://jbeirer.github.io/rootfig/)
[![CI](https://github.com/jbeirer/rootfig/actions/workflows/ci.yml/badge.svg)](https://github.com/jbeirer/rootfig/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jbeirer/rootfig/branch/main/graph/badge.svg)](https://codecov.io/gh/jbeirer/rootfig)
[![PyPI](https://img.shields.io/pypi/v/rootfig.svg)](https://pypi.org/project/rootfig/)
[![Python](https://img.shields.io/pypi/pyversions/rootfig.svg)](https://pypi.org/project/rootfig/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Documentation](https://jbeirer.github.io/rootfig/) ·
[Gallery](https://jbeirer.github.io/rootfig/gallery/) ·
[Quick start](https://jbeirer.github.io/rootfig/quickstart/)**

Go from ROOT files to a styled figure in one call. Choose a variable, add a
selection, and plot:

```python
import rootfig as rf

rf.plot("events.root", "Muon_pt", tree="events", selection="Muon_pt > 20", bins=50)
```

Start with a single distribution; add samples, weights, stacks and ratio
panels as your analysis grows. Every plot gives you a matplotlib figure to
customise and save. No ROOT installation required.

<p align="center">
  <a href="https://jbeirer.github.io/rootfig/gallery/"><img src="docs/images/gallery/stack_data.png" alt="Stacked simulation with data and a ratio panel" width="48%"></a>
  <a href="https://jbeirer.github.io/rootfig/gallery/"><img src="docs/images/gallery/xbreak_ratio.png" alt="Broken x axis with a ratio panel" width="48%"></a>
</p>
<p align="center">
  <a href="https://jbeirer.github.io/rootfig/gallery/"><img src="docs/images/gallery/object_vs_event.png" alt="Per-object versus per-event selections" width="48%"></a>
  <a href="https://jbeirer.github.io/rootfig/gallery/"><img src="docs/images/gallery/hist2d.png" alt="Two-dimensional histogram" width="48%"></a>
</p>

**[Explore the gallery →](https://jbeirer.github.io/rootfig/gallery/)**
See each figure alongside the code that makes it, from simple overlays to
stacked data/MC comparisons, broken axes and 2D histograms.

## Installation

```bash
pip install rootfig
# or
uv add rootfig
```

Python 3.12 or newer. No ROOT installation is needed; `TTree` and `RNTuple`
files are both supported.

## Compare samples in one call

```python
import rootfig as rf

# Overlay two samples, normalised to unity, with a ratio panel.
rf.plot(
    ["signal.root", "background.root"],
    "Muon_pt",
    tree="events",
    selection="abs(Muon_eta) < 2.5",
    weight="event_weight",
    bins=(50, 0, 200),
    normalize=True,
    ratio=True,
)
```

## Build up to a full analysis

Define samples, variables, cuts and styles once, then reuse them across plots:

```python
import rootfig as rf

signal = rf.Sample("sig_*.root", tree="events", label="Signal", weight="mc_weight")
background = rf.Sample("bkg.root", tree="events", label="Background", weight="mc_weight")
data = rf.Sample("data.root", tree="events", label="Data", is_data=True)

pt = rf.Variable("Muon_pt", bins=(50, 0, 200), label=r"$p_T^{\mu}$", unit="GeV")
baseline = rf.Cut("nMuon >= 1") & "abs(Muon_eta) < 2.5"
style = rf.Style(experiment="ATLAS", status="Internal", lumi=140, com=13.6)

p = rf.plot(
    [background, signal],
    pt,
    observed=data,
    selection=baseline,
    stack=True,
    ratio=True,
    logy=True,
    style=style,
)
p.ax.set_ylim(top=1e5)  # it is a normal matplotlib Axes
p.save("muon_pt.pdf")
```

Everything you get back is a standard object: `p.fig` and `p.ax` are
matplotlib `Figure`/`Axes`, `p.hists` are `hist.Hist` objects, and
`rf.load(...)` returns Awkward arrays.

## What you can do

- **Select events and objects with readable expressions.** Write cuts such as
  `count(Jet_pt) >= 2` or `Muon_pt > 20`; event and object selections have
  explicit rules, and event weights carry through to each selected object.
- **Compare samples with a few keywords.** Overlays, stacks, data points and
  ratio panels share binning and propagate histogram uncertainties.
  Normalise to unity, density, bin width or luminosity.
- **Style figures for your analysis.** Add experiment labels, units, log axes
  and broken axes, then refine the result with matplotlib.
- **Go beyond 1D plots.** Draw 2D histograms, correlations, efficiencies,
  profiles, resolutions and significance panels; produce cut flows and
  summary statistics from the same inputs.
- **Work directly with your files.** Read `TTree` and `RNTuple` data, combine
  files with globs, limit entry ranges for quick checks, and use EDM4hep
  split collections. Only the branches your expressions need are read.

## Documentation

**[Read the docs](https://jbeirer.github.io/rootfig/)** or
**[browse the gallery](https://jbeirer.github.io/rootfig/gallery/)** for examples
with figures and code.

- [Quick start](https://jbeirer.github.io/rootfig/quickstart/): your first plot, selections and weights.
- [Expressions and selections](https://jbeirer.github.io/rootfig/expressions/): syntax and event/object rules.
- [Samples, variables, cuts and styles](https://jbeirer.github.io/rootfig/composable/): reusable analysis definitions.
- [Plotting options](https://jbeirer.github.io/rootfig/plotting/): binning, normalisation, panels and styling.
- [API reference](https://jbeirer.github.io/rootfig/api/): full signatures and options.

## Relation to the ecosystem

`rootfig` brings a `TTree::Draw`-like workflow to the Scientific Python HEP
stack, building on familiar libraries:

| Task | Library | What rootfig adds |
| --- | --- | --- |
| Reading ROOT files | [uproot](https://github.com/scikit-hep/uproot5) | file globs, tree auto-detection, reading only the required branches |
| Jagged arrays | [Awkward Array](https://github.com/scikit-hep/awkward) | the per-event/per-object rules for cuts and weights |
| Histograms | [hist](https://github.com/scikit-hep/hist) / boost-histogram | shared binning, automatic ranges, normalisation, ratios |
| Drawing | [mplhep](https://github.com/scikit-hep/mplhep) + matplotlib | overlays, stacks, ratio panels, labels and legends with good defaults |

If you already have `hist.Hist` objects, `rf.plot_histograms` draws them with
the same options. If you want the arrays, `rf.load` returns them. See
[the ecosystem guide](https://jbeirer.github.io/rootfig/ecosystem/) for details.

## Development

```bash
git clone https://github.com/jbeirer/rootfig
cd rootfig
uv sync --all-groups
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT. See [LICENSE](LICENSE).
