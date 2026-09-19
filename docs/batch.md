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
`Variable.safe_name` (the `name=`, else the expression with anything but
letters, digits and `_` replaced) identifies a variable and names its files, so
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

## Data passes through unchanged

`data` is stored as given and handed to `rf.plot` as is, never copied,
flattened or inspected. Every form of `data` that `rf.plot` accepts can therefore
be used in a book: file paths and globs, `Sample` objects, `Group` objects (drawn
as one histogram), a `Group` as `observed=`, variables that name a `TH1` stored
in the files, in-memory arrays and `hist.Hist` or `Histogram` objects. The rules
of `rf.plot` apply unchanged: a book always names at least one variable, and a
`selection` or `weight` raises for a ready-made histogram, which is drawn as it
is.

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

`book.plots()` is a lazy iterator of `(task, plot)` pairs: one `rf.plot` call
per step, nothing drawn ahead of time. The plots are ordinary
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
keyword values hold. Each task is exactly

```python
rf.plot(book.data, task.variable, selection=task.selection, **task.kwargs)
```

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

A book runs its tasks one after another in the calling process and reads the
files again for every task. It has no filename template, pattern matching,
parallel execution or report generation, and it drives the 1D `rf.plot` only:
for `plot2d`, `efficiency` and `profile` you write the loop yourself.
