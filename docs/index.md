# rootfig

**Publication-quality figures straight from ROOT trees, without ROOT.**

```python
import rootfig as rf

rf.plot("events.root", "Muon_pt", tree="events", selection="Muon_pt > 20", bins=50)
```

rootfig is the `TTree::Draw` workflow for the Scientific Python HEP stack. It
reads with uproot, computes with Awkward Array, fills hist histograms and
draws with mplhep, and adds the glue those libraries leave to you: predictable
per-event/per-object selection semantics, weights, shared binning across
samples, normalisation, ratio panels and good defaults.

![Stacked simulation with data and a ratio panel](images/gallery/stack_data.png){ width="60%" }

- [Quick start](quickstart.md)
- [Gallery](gallery.md): every feature as a figure next to its code
- [Expressions and selections](expressions.md)
- [Samples, variables, cuts and styles](composable.md)
- [Plotting options](plotting.md)
- [Relation to the ecosystem](ecosystem.md)
- [API reference](api.md)
