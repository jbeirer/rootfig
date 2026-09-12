# Gallery

Every figure on this page is made by
[`examples/gallery`](https://github.com/jbeirer/rootfig/blob/main/examples/gallery/__init__.py).
The script writes a toy dataset (three simulated processes and one "observed"
sample with muons, jets and event-level quantities, as `TTree`s) into a
directory and runs all examples there in a few seconds:

```bash
python examples/gallery               # everything ends up in examples/out/
```

The code next to each figure is the source of that example, and the image is
the reference picture the test suite compares against, so what you see is what
the current release draws. File names such as `signal.root` are those toy
files, relative to that directory, and every example assumes

```python
import rootfig as rf
```

<!-- gallery: quick -->

## Setup

Plain strings and `(bins, low, high)` tuples are accepted everywhere, as the
one-liner shows. Once you draw more than one plot it pays to name the pieces:
`Sample`, `Variable`, `Cut` and `Style` are small frozen dataclasses (see
[Samples, variables, cuts and styles](composable.md)). The rest of this page
shares four samples, three variables and one style, defined once:

<!-- gallery-setup -->

Anything a single example needs is defined inside that example, so every block
below is complete given the names above (plus `matplotlib.pyplot as plt` and
`numpy as np` where they appear).

<!-- gallery -->

## Beyond figures

The same inputs feed tables and arrays:

```python
# entries, mean, std, sem, skewness, min, max per sample and variable
print(rf.summarize(mc, ["MET", "Muon_pt"], selection="nMuon > 0"))

# a cut flow: yields, raw counts and step efficiencies per sample
print(
    rf.cutflow(
        mc,
        ["nMuon >= 2", rf.Cut("MET > 50", label="MET > 50 GeV"), "any(Jet_btag > 0.8)"],
    )
)

# evaluated expressions as an Awkward record array
events = rf.load(signal, ["MET", "count(Muon_pt)", "first(Muon_pt)"], selection="nJet >= 2")
events["MET"]

# a plain hist.Hist to feed into your own code
h = rf.histogram(signal, "MET", bins=(40, 0, 400), selection="nJet >= 2")
```

Cuts compose with `&`, `|` and `~`, carry optional labels, and combine with
selections given to `plot()`:

```python
base = rf.Cut("nMuon >= 2", label="2 muons")
signal_region = base & "abs(Muon_eta) < 2.4" & ~rf.Cut("any(Jet_btag > 0.8)")
control_region = base & rf.Cut("any(Jet_btag > 0.8)") | "MET > 200"
```

See [Plotting options](plotting.md) for every keyword and
[Expressions and selections](expressions.md) for the per-event/per-object rules.
