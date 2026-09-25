# How rootfig relates to uproot, Awkward, hist, mplhep, matplotlib and SciPy

rootfig is a thin, opinionated layer over the Scientific Python HEP stack. It
owns no file format, array type, histogram type or drawing primitive of its
own; every object you get back belongs to one of the libraries below.

## uproot

[uproot](https://uproot.readthedocs.io) reads and writes ROOT files in pure
Python. rootfig uses it for all I/O: opening files, detecting the tree,
listing branches, and reading the branches an expression needs from one or
many files (`TTree` and `RNTuple`). uproot also has its own expression
language for `TTree.arrays(expressions, cut=...)`; rootfig implements a
similar one so that it can validate names before reading, give precise
errors, support in-memory data, and define the per-object/per-event rules
that `arrays(cut=...)` leaves to the caller.

## Awkward Array

[Awkward Array](https://awkward-array.org) is the array library for jagged
data. Everything rootfig computes is an Awkward array; the selection layer is
a small set of rules (broadcast, mask, flatten) expressed with `ak.num`,
`ak.broadcast_arrays`, boolean indexing and `ak.flatten`. `rf.load` returns
an Awkward record array, and `rf.evaluate` runs one expression on arrays you
already have.

## hist and boost-histogram

[hist](https://hist.readthedocs.io) provides the histogram objects. rootfig
fills `hist.Hist` with `Weight` storage (sum of weights and sum of squared
weights per bin), including under/overflow bins, and returns them unchanged:
`rf.histogram` gives you the `hist.Hist`, `Plot.hists` lists them. Rebinning,
projecting, slicing, saving to ROOT files with uproot: all of that is hist
functionality and works directly on the returned objects. Histograms stored
in ROOT files (`TH1`, `TH2`) are converted by uproot's `to_hist()`; rootfig
adds the summing over files, the scaling and the drawing.

## mplhep

[mplhep](https://mplhep.readthedocs.io) draws histograms with matplotlib and
ships the style sheets and label helpers of the LHC experiments. rootfig
draws every histogram through `mplhep.histplot`/`hist2dplot` and uses
mplhep's per-experiment label functions when a `Style(experiment=...)` is
given. The experiment-neutral default style and the lower panel (ratios,
differences, asymmetries, pulls and significances of histograms; ratios,
differences, asymmetries and pulls of efficiencies and profiles) are rootfig's,
and so are the Poisson intervals of data: mplhep's (and `hist.intervals`) give
an empty bin of scaled counts the scale of a neighbour, where rootfig keeps the
bin's own (see [SciPy](#scipy)).

## matplotlib

Every figure is a plain `matplotlib.figure.Figure` with plain `Axes`. rootfig
applies its style only inside a `plt.style.context` while drawing, so it does
not change global rcParams (unless you call `rf.use_style`). Anything you
would do to a matplotlib figure, you can do to `Plot.fig` and `Plot.ax`.

## SciPy

[SciPy](https://scipy.org) supplies the quantiles behind rootfig's intervals, from
`scipy.special`: the inverse incomplete gamma functions for the Garwood interval
of data (ROOT's `TH1::kPoisson`) and the inverse incomplete beta function for the
Clopper–Pearson and Bayesian intervals of efficiencies (`TEfficiency`'s default
and its Beta priors); `scipy.optimize.brentq` finds the bounds that have no
closed form (mid-P, the shortest Bayesian interval). rootfig decides what the
quantiles are taken of (the counts behind scaled or normalised contents, which
interval ROOT would choose) and imports SciPy only when it computes one.

## Trees and histogram files

rootfig reads trees and stored `TH1`/`TH2` histograms without ROOT, independently
of the framework that wrote them. A `Sample` identifies each input process,
a `Group` combines processes into a category, and `stack=` selects the categories
to stack. Files already scaled to expected yields are read as they are; trees
can be scaled with `Sample(xsec=..., ngen=...)` and `lumi=`.

The same `Variable` describes a stored histogram and the branch it was filled
from: its bins and range crop and merge existing bins, moving the rest into
flow bins. `PlotBook` prepares each variable once for its drawing variants and
can discover shared histograms with `rf.ALL`. See the
[complete histogram-file example](batch.md#a-complete-example-histogram-files).

## When to use something else

- You have flat NumPy columns rather than ROOT files or histograms, or want a
  comparison rootfig's lower panel does not offer (a goodness-of-fit panel):
  [plothist](https://plothist.readthedocs.io).
- You need a full columnar analysis framework with lazy, distributed
  processing: [coffea](https://coffeateam.github.io/coffea/). rootfig reads
  the branches a plot needs a chunk of entries at a time, in threads, and holds
  the values it fills (see [large inputs](plotting.md#large-inputs)), all in one
  process: it suits quick looks and ntuples that one machine reads in minutes,
  not multi-terabyte datasets.
- You want to build fit templates and workspaces: [cabinetry](https://cabinetry.readthedocs.io).
- You want a quick terminal look at a branch: [histoprint](https://github.com/scikit-hep/histoprint).
