# Expressions and selections

Variables, selections and weights are all *expressions*: strings in Python
syntax evaluated with NumPy/Awkward semantics over the branches of the tree.

## Syntax

| Feature | Example |
| --- | --- |
| Branch names | `Muon_pt`, `MET` |
| Names that are not identifiers | `` `jet1_b-tag` > 0.5 `` (backticks) |
| Arithmetic | `sqrt(px**2 + py**2)`, `MET / 1000`, `-eta` |
| Comparisons | `pt > 20`, `charge != 0`, `20 < pt < 100` (chained) |
| Boolean logic | `pt > 20 and abs(eta) < 2.5`, `not isTight`, or `&`, `\|`, `~` |
| Indexing | `Muon_pt[:, 0]`, `Muon_pt[Muon_pt > 20]` |
| Constants | `pi`, `e`, `inf`, `nan`, `True`, `False` |

`and`, `or`, `not` and chained comparisons are rewritten to element-wise
operations, so they work on arrays. Attribute access, lambdas,
comprehensions, string literals and calls to anything but the functions
below are rejected at parse time with an [`ExpressionError`][rootfig.ExpressionError].
An unknown branch raises [`MissingBranchError`][rootfig.MissingBranchError]
with close-match suggestions.

## Functions

Element-wise (NumPy): `abs`, `sqrt`, `cbrt`, `exp`, `expm1`, `log`, `log10`,
`log2`, `log1p`, `power`, `hypot`, `sin`, `cos`, `tan`, `arcsin`, `arccos`,
`arctan`, `arctan2`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`,
`arctanh`, `deg2rad`, `rad2deg`, `floor`, `ceil`, `round`, `trunc`, `sign`,
`minimum`, `maximum`, `clip`, `isnan`, `isinf`, `isfinite`, `where`.

Per-event reductions over jagged branches (they reduce the innermost list,
i.e. over the objects of each event, like ROOT's `Length$`, `Sum$`, `Max$`):

| Function | Meaning |
| --- | --- |
| `count(x)`, `len(x)` | number of objects |
| `sum(x)`, `prod(x)` | sum / product over objects |
| `min(x)`, `max(x)` | extreme value (`None` for empty events) |
| `mean(x)`, `std(x)` | mean / standard deviation over objects |
| `any(x)`, `all(x)` | boolean reductions |
| `argmin(x)`, `argmax(x)` | index of the extreme object |
| `first(x)` | the leading object (`None` for empty events) |

Kinematics from Cartesian components, as stored by EDM4hep (`momentum.x/y/z`,
`energy`) and many flat ntuples (`px`, `py`, `pz`, `E`):

| Function | Meaning |
| --- | --- |
| `pt(px, py)`, `p(px, py, pz)` | transverse and total momentum |
| `theta(px, py, pz)`, `costheta(px, py, pz)` | polar angle and its cosine |
| `eta(px, py, pz)`, `phi(px, py)` | pseudorapidity and azimuth |
| `mass(E, px, py, pz)` | invariant mass (0 for space-like input) |

Reductions applied to a flat (per-event) branch raise an error.

## Sub-branches of object collections

podio/EDM4hep and other files with split object branches list their leaves
as `Collection.field.component`. Write them as they appear, dots included:

```python
rf.plot(
    f,
    "pt(ReconstructedParticles.momentum.x, ReconstructedParticles.momentum.y)",
    selection="abs(costheta(ReconstructedParticles.momentum.x, "
    "ReconstructedParticles.momentum.y, ReconstructedParticles.momentum.z)) < 0.9",
)
```

Names with other unusual characters (spaces, `-`) go in backticks: `` `jet1_b-tag` > 0.5 ``.

## Per-event versus per-object

A branch is *per-event* (flat, depth 1) or *per-object* (jagged, depth 2 or
more). Every expression inherits the structure of the branches it uses:
`Muon_pt > 20` is per-object, `count(Muon_pt) >= 2` is per-event, and
`Muon_pt > 20 and MET > 50` is per-object (the per-event part broadcasts).

The rules for combining a variable with a selection:

| Variable | Selection | Result |
| --- | --- | --- |
| per-event | per-event | events passing the selection |
| per-object | per-event | all objects of the passing events, flattened |
| per-object | per-object | objects passing the selection, flattened |
| per-object | per-object of another collection | [`SelectionError`][rootfig.SelectionError] |
| per-event | per-object | [`SelectionError`][rootfig.SelectionError]: reduce with `any()`, `all()` or `count()` |

The last row is where `TTree::Draw` silently fills the per-event value once
per passing object; rootfig refuses and tells you how to be explicit:

```python
rf.plot(f, "MET", selection="any(Muon_pt > 20)")  # events with a hard muon
rf.plot(f, "MET", selection="sum(Muon_pt > 20) >= 2")  # at least two hard muons
```

(`count(x)` is the number of objects, `sum(mask)` the number of objects
passing; `count(Muon_pt > 20)` would count all muons.) Fixed-size branches
(`float x[3]`) and two-dimensional NumPy arrays are per-object like
variable-length lists.

## Weights

| Variable | Weight | Result |
| --- | --- | --- |
| per-event | per-event | one weight per event |
| per-object | per-event | the event weight is broadcast to each object |
| per-object | per-object (same collection) | one weight per object |
| per-event | per-object | [`IncompatibleWeightError`][rootfig.IncompatibleWeightError] |

Weights defined on a [`Sample`][rootfig.Sample] and passed to `plot()` are
multiplied; `Sample(scale=...)` multiplies a constant in as well. A blank
weight string means no weight. Bin variances are always the sums of squared
weights (`hist` `Weight` storage).

## Missing and non-finite values

`None` values (e.g. `max(Muon_pt)` of an event with no muons) never enter a
histogram; in a selection they count as `False`. `nan` and `inf` values are
dropped with a [`RootfigWarning`][rootfig.RootfigWarning] telling you how
many, or raise with `nonfinite="error"`. Both counts are recorded in the
histogram's `stats`.

## Selections in `load()`

`rf.load` returns arrays for events, so its `selection` must be per-event.
Apply object-level cuts inside the expressions instead:

```python
rf.load(f, {"hard_muons": "Muon_pt[Muon_pt > 20]"}, selection="nMuon > 0")
```

## Composing cuts

[`Cut`][rootfig.Cut] objects combine with `&`, `|` and `~` and keep an
optional label:

```python
base = rf.Cut("nMuon >= 2", label="2 muons")
signal_region = base & "abs(Muon_eta) < 2.4" & ~rf.Cut("has_bjet")
```

Plain strings are accepted wherever a `Cut` is.
