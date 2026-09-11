# Gallery

Every figure on this page is made by
[`examples/gallery.py`](https://github.com/jbeirer/rootfig/blob/main/examples/gallery.py).
The script writes a toy dataset (three simulated processes and one "observed"
sample with muons, jets and event-level quantities, as `TTree`s) and runs all
examples in a few seconds:

```bash
python examples/gallery.py            # figures end up in examples/out/
```

The code next to each figure is the source of that example, and the image is
the reference picture the test suite compares against, so what you see is what
the current release draws.

```python
import matplotlib.pyplot as plt
import numpy as np

import rootfig as rf
```

## Setup

The examples share a few definitions, made once (`out` is the directory with
the toy files). Everything below is optional: plain strings work everywhere.

<!-- gallery-setup -->

<!-- gallery -->

## Beyond figures

The same inputs feed tables and arrays:

```python
# entries, mean, std, sem, skewness, min, max per sample and variable
print(rf.summarize(mc, ["MET", "Muon_pt"], selection="nMuon > 0"))

# a cut flow: yields at a luminosity, raw counts and step efficiencies per sample
print(
    rf.cutflow(
        [zh, ww, zz],
        ["nMuon >= 2", rf.Cut("MET > 50", label="MET > 50 GeV"), "any(Jet_btag > 0.8)"],
        lumi="10.8 ab^-1",
    )
)

# evaluated expressions as an Awkward record array
events = rf.load(sig, ["MET", "count(Muon_pt)", "first(Muon_pt)"], selection="nJet >= 2")
events["MET"]

# a plain hist.Hist to feed into your own code
h = rf.histogram(sig, "MET", bins=(40, 0, 400), selection="nJet >= 2")
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
