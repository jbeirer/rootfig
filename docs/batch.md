# Batch plotting

An analysis rarely needs one plot. It needs every variable, under each
selection, drawn linear and logarithmic. `rf.PlotBook` describes that set once
and runs [`rf.plot`][rootfig.plot] for each member: it is repeated
`rf.plot(...)`, not a second plotting engine.

```python
import rootfig as rf

ww = rf.Sample("ww.root", tree="events", label="WW", weight="mc_weight")
zz = rf.Sample("zz.root", tree="events", label="ZZ", weight="mc_weight")
signal = rf.Sample("signal.root", tree="events", label="Signal", weight="mc_weight")

book = rf.PlotBook(
    [rf.Group([ww, zz], label="VV"), signal],
    [
        rf.Variable("mass", bins=(50, 100, 150), unit="GeV"),
        rf.Variable("pt", bins=(50, 0, 200), unit="GeV"),
    ],
    selections={
        "baseline": rf.Cut("nMuon >= 2"),
        "sr": rf.Cut("nMuon >= 2") & "recoil_mass > 120",
    },
    variants={
        "lin": {},
        "log": {"logy": True},
    },
    plot_kwargs={"stack": True, "style": rf.Style(experiment="ATLAS", status="Internal")},
)

book.save("plots", formats=["pdf", "png"])
```

This writes eight plots in two formats, `plots/mass__baseline__lin.pdf`,
`plots/mass__baseline__lin.png`, ... `plots/pt__sr__log.png`.

## The three axes

The tasks of a book are the Cartesian product

```text
variables × selections × variants
```

in insertion order, variables outermost and variants innermost. Two variables,
three selections and two variants give twelve tasks, and `book.tasks()` lists
them without reading or drawing anything.

**Variables** are names, expressions or [`Variable`](composable.md#variable)
objects; one may be given bare (`rf.PlotBook(data, "mass")`).
`Variable.safe_name` (the `name=`, else
[`safe_file_stem`][rootfig.model.safe_file_stem] of the expression) identifies a
variable and names its files, so
two variables of one book must not share it: `Variable("Muon_pt", name="pt")`
next to `Variable("Electron_pt", name="pt")` is rejected.

**Selections** are a mapping from a name to a cut (a string, a
[`Cut`](composable.md#cut) or `None` for "no cut"). Names are the file name
component, never derived from the expression or the cut's label.

**Variants** are a mapping from a name to `plot()` keywords. A variant's
keywords override the common `plot_kwargs` for its tasks, so

```python
book = rf.PlotBook(
    data,
    ["mass"],
    plot_kwargs={"stack": True, "logy": False},
    variants={"lin": {}, "log": {"logy": True}},
)
```

draws `lin` with `stack=True, logy=False` and `log` with `stack=True, logy=True`.

Every other keyword of `rf.plot` (`observed=`, `normalize=`, `ratio=`,
`systematics=`, `style=`, ...) goes into `plot_kwargs` or a variant. The keyword
names are checked against `rf.plot`'s signature when the book is built, so a
misspelt `log_y` is reported, with `logy` as the suggestion, before anything is
drawn rather than after the first tasks have written their files; the values are
validated by `rf.plot` itself. Five keywords are the book's own and are rejected:
`data`, `variable` and `selection` come from the task, `save` from
`PlotBook.save()`, and `ax` because every task draws its own figure.

A reserved keyword, an empty `selections=` or `variants=`, an unusable name and a
pair of tasks whose files would collide all raise `ValueError` when the book is
built. A `selections=`, `variants=` or `plot_kwargs=` that is not a mapping, a
name that is not a string, and a keyword `rf.plot` does not take, raise
`TypeError`.

Selection and variant names become file name components and are checked, like
`Variable.name`, for every platform: more than dots and spaces, no trailing dot
or space, no slash, backslash, control character or `<>:"|?*`, and not a Windows
device name (`CON`, `NUL`, `COM1`, ...). The check is
[`check_file_stem`][rootfig.model.check_file_stem]; its message offers a
spelling that works.

## Automatic variable discovery

`rf.ALL` in place of the variable list asks the book to find the variables
itself:

```python
book = rf.PlotBook(
    samples,
    variables=rf.ALL,
)
```

Discovery reads metadata only. For a `TTree` or `RNTuple` it is the schema:
every branch (or nested field, `Muon.pt`) whose values are numbers or booleans,
lists and fixed-size arrays of them included, becomes a variable; strings,
records and other objects are left out. For a ROOT file read without a tree it
is the object list: every stored `TH1` becomes a variable; `TH2`, `TProfile`,
`TParameter` and the like do not, and a histogram inside a directory is named by
its path (`sel/mz`). A file holding both is treated as `rf.plot` treats it: the
branches of its one tree plus the stored `TH1`s no branch shadows; an explicit
`tree=` means branches only, and several trees without `tree=` raise, as they
do for any plot. In-memory arrays contribute their numeric and boolean columns.
No event array and no bin content is read, and the first file of a sample
stands for all of them, as everywhere in rootfig.

With several samples, groups (through their leaf samples) and `observed=`, a
variable must be present in every one of them, the same way: a name that is a
branch in one sample and a stored histogram in another is not a common
variable. Variants that change how the histograms are prepared (`tree`,
`observed`, `weight`, ...) must all be able to plot it too, so the discovered
set is the intersection over the effective configurations. The files or arrays
a `Systematic.samples` variation fills or reads from are surveyed like a sample
of their own and take part in the intersection, in the mode of the sample they
vary; a variation that cannot be built or surveyed is left to the task, which
reports it whatever the variable. Stored histograms
are left out whenever the book is bound to refuse them: a selection, a
`weight`, `nonfinite="error"`, a `stats` box, a `range` without `bins`, or a
systematic varying the weight or branches that applies to a sample (the plot's
unless the sample's own source of that name replaces it, none for observed
data), in the book's keywords or in any variant's. A file of stored histograms
only then fails when the book is built, saying why, rather than at its first
task; the message lists what each sample holds, what was left out and why, and
which names the samples do not share.

`include=` and `exclude=` narrow the set with case-sensitive shell patterns
(`*`, `?`, `[...]`, as `fnmatch` reads them), one or a sequence:

```python
book = rf.PlotBook(
    samples,
    variables=rf.ALL,
    include=["Muon_*", "Electron_*", "MET*"],
    exclude=["*_cov", "*Index"],
)
```

A variable is kept when it matches one `include` pattern (all do when `include`
is not given) and no `exclude` pattern. Patterns match the source name, `sel/mz`
or `jet1_b-tag`, not the file name component `sel_mz` made from it. They are
only valid with `rf.ALL`: an explicit list is used as it is, and `include=` or
`exclude=` next to one raises. So does a filter that leaves nothing; a pattern
that matches nothing is fine as long as others do.

The result is an ordinary tuple of `Variable` objects, sorted by source name,
each addressing exactly its branch or histogram (`` `jet1_b-tag` `` and
`` `sel/mz` `` in backticks). `book.variables` shows exactly what was
discovered, and from there on nothing distinguishes the book from one built
with that list: the same file names (two names that sanitise to one component,
`a-b` and `a_b`, are rejected as for an explicit list; exclude one or name
them), the same `tasks()`, `select()` (which keeps the discovered variables
rather than discovering again) and the same batched execution described below.

`rf.discover_variables(data, ...)` runs the same discovery without building a
book, under the same `selections=`, `variants=`, `plot_kwargs=`, `include=` and
`exclude=` keywords, and returns the variables. It does not check output file
names, so two names that sanitise to one component are both returned; tell them
apart with `Variable.replace(name=...)` and pass the list to `PlotBook`:

```python
variables = rf.discover_variables(samples, include="jet*")
renamed = [v.replace(name="jet1_btag") if v.expression == "`jet1_b-tag`" else v for v in variables]
book = rf.PlotBook(samples, renamed)
```

## Data passes through unchanged

`data` is stored as given and handed to `rf.plot` as is, never copied or
flattened; with an explicit variable list it is not inspected either, and
`rf.ALL` reads its metadata only. Every form of `data` that `rf.plot` accepts
can therefore be used in a book: file paths and globs, `Sample` objects,
`Group` objects (drawn as one histogram), a `Group` as `observed=`, variables
that name a `TH1` stored in the files, in-memory arrays and `hist.Hist` or
`Histogram` objects. The rules of `rf.plot` apply unchanged: a book always
names at least one variable (`rf.ALL` needs inputs with names to discover, so
histogram objects take an explicit one), and a `selection` or `weight` raises
for a ready-made histogram, which is drawn as it is.

The book copies the mappings it is configured with (the variable list,
`selections`, `variants` and each keyword mapping) into read-only copies, so
adding to or replacing entries of those dictionaries afterwards does not change
the book. The values inside them are shared, not copied: a `Style`, a `Group` or
a `systematics=` mapping given in `plot_kwargs` is the caller's object, and
changing it changes what the book draws.

## File names

`book.save(directory, formats=("pdf",), **savefig_kwargs)` creates the directory
if needed and writes one file per task and format, named

```text
<variable>[__<selection>][__<variant>].<format>
```

A component appears whenever its axis was given explicitly, whatever the name:

| Book | File |
| --- | --- |
| `rf.PlotBook(data, ["mass"])`                       | `mass.pdf`         |
| `rf.PlotBook(data, ["mass"], selections={"all": cut})` | `mass__all.pdf` |
| `rf.PlotBook(data, ["mass"], variants={"default": {"logy": True}})` | `mass__default.pdf` |
| both explicit                                       | `mass__sr__log.pdf` |

One format may be given bare, `formats="png"`, a leading dot is accepted and
duplicates are dropped. A format matplotlib cannot write raises `ValueError`
before the first figure is drawn, rather than part way through the batch. The
remaining `savefig_kwargs` go to [`Plot.save`][rootfig.Plot.save]; `format` and
`fname` are refused, because the file names come from `formats` and the task.

Two tasks that would share a name are rejected when the book is built, not
when the second file overwrites the first. Names are compared ignoring case and
Unicode normalisation, since `lin` and `LIN` are one file on the case-insensitive
file systems of macOS and Windows; the message lists the spellings that clash.
`save()` returns the written paths in task order, then format order, and closes
every figure after writing it, also when writing fails, so memory stays bounded
however large the book is.

## Running plots yourself

`book.plots()` is a lazy iterator of `(task, plot)` pairs: one figure per
step, nothing drawn ahead of time. The plots are ordinary
[`Plot`][rootfig.Plot] objects, so this is the place to adjust a figure
or keep it open:

```python
for task, p in book.plots():
    print(task.variable.safe_name, task.selection_name, task.variant_name)
    p.ax.axvline(91.2, color="gray", ls="--")
    p.save(f"plots/{task.stem}.pdf")
    p.close()
```

A `PlotTask` carries the `variable`, the `selection` (`Cut` or `None`), the
`selection_name` and `variant_name` (`None` for an axis the book was built
without), the merged `kwargs` and the file `stem`. Tasks compare and hash by
their `stem`, so they work as set members and dictionary keys whatever the
keyword values hold. Each plot is what

```python
rf.plot(book.data, task.variable, selection=task.selection, **task.kwargs)
```

returns: the same histograms, binning, systematics, labels and errors.

The book only reads less often than that call would. It runs the tasks in
batches of a few dozen variables: the branches those variables and every
selection need are read once per sample for the batch, the files a
`Systematic.samples` variation fills from included, and variants that only
change the drawing (`logy`, `normalize`, `ratio`, `style`, ...) are drawn from
one set of prepared histograms, each figure from its own copy. A variant that
changes how the histograms are prepared (`bins`, `weight`, `observed`,
`systematics`, ...) is prepared on its own, still from the batch's read.
Variables that name histograms stored in the files are read with one pass over
each file per batch. A book with an explicit variable list inspects no input
when it is built; one built with `rf.ALL` inspects the metadata of its inputs
then, but never reads event arrays or histogram contents. Neither reads
anything for `tasks()`; the first batch is read when the first plot is
requested and the next one when the iteration reaches it.

Errors stop the book at the failing task. They keep their type and gain a note
naming the task, so a traceback for a typo in one expression reads

```text
rootfig.errors.MissingBranchError: unknown name 'recoil_mas' ...
while running PlotBook task variable='recoil_mas', selection='sr', variant='log'
```

The words `all` and `default` in such messages stand for an axis that was not
given; they are never used to decide a file name.

## Subsets

`book.select(variables=..., selections=..., variants=...)` returns a new book
restricted to the named variables (by `safe_name`), selections and variants
(by key). Each argument left as `None` keeps its axis, the book's order is kept
rather than the filter's, and an unknown name raises with the available
choices:

```python
book.select(variables=["mass"], variants=["log"]).save("plots/debug")

book.select(variables=["foo"])
# ValueError: unknown variable 'foo'; available: ['mass', 'pt']
```

A filter that names nothing, `select(variables=[])`, raises as well, rather than
quietly producing an empty book.

An axis the book was built without has the single choice `"all"` (selections)
or `"default"` (variants); selecting it is a no-op and the axis stays implicit.

## What a book does not do

A book runs its tasks one after another in the calling process and keeps the
arrays of one batch of variables at a time. It has no filename template,
pattern matching, parallel execution or report generation, and it drives the
1D `rf.plot` only: for `plot2d`, `efficiency` and `profile` you write the loop
yourself.
