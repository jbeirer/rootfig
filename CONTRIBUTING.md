# Contributing to rootfig

Thanks for helping. Issues and pull requests are welcome at
<https://github.com/jbeirer/rootfig>.

## Development setup

rootfig is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jbeirer/rootfig
cd rootfig
uv sync --all-groups          # runtime, dev and docs dependencies
uv run pre-commit install     # optional: run the linters on every commit
```

## Checks

```bash
uv run pytest                          # tests (ROOT files are generated on the fly)
uv run pytest --cov                    # with coverage
uv run ruff check . && uv run ruff format --check .
uv run mypy                            # strict type checking of src/
uv run mkdocs serve                    # documentation preview
uv build && uvx twine check dist/*     # packaging
```

CI runs all of these on Python 3.12, 3.13 and 3.14 (Linux), plus macOS
on 3.13, and once more with every direct dependency at the minimum version
declared in `pyproject.toml` (`uv sync --resolution lowest-direct`). Tests
that use ROOT's tutorial files run only when `root-config` is available
locally and are skipped otherwise; do not add
tests that require ROOT or network access.

`key4hep.yml` runs the tests on the Key4hep nightlies (LCG `devkey-head`) and the
latest Key4hep release, weekly as well, since the stacks change without a commit
here. `.github/scripts/key4hep-test.sh` installs rootfig into a
[cvmfs-venv](https://github.com/jbeirer/cvmfs-venv) with `--no-index`, so every
runtime dependency must come from the stack: the floors in `pyproject.toml` stay
at or below what the nightlies ship. Locally: `.github/scripts/key4hep-test.sh
/cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh`.

## Layout

```
src/rootfig/
  api/            plot(), histogram(), load(), ...  (orchestration only)
  errors.py       exception hierarchy
  expressions/    parse, validate and evaluate expression strings
  io/             file and in-memory data sources
  model/          Sample, Variable, Cut, Style, binning
  selection/      per-event / per-object semantics -> flat columns
  histograms/     filling, normalisation, ratios, statistics, pipeline
  plotting/       matplotlib/mplhep rendering, styles, annotations
tests/            one module per layer plus end-to-end API tests
docs/             MkDocs sources
```

Keep the layers independent: `plotting` must not read files, `selection`
must not know about matplotlib, and so on. Public functions carry NumPy-style
docstrings and full type hints.

## Test data

Almost all tests generate their ROOT files with uproot on the fly. The one
committed file, `tests/data/split_collection.root`, has a split
`std::vector<struct>` branch and a `TParameter` the way podio/EDM4hep and
FCCAnalyses write them. uproot cannot write such a file, so it was produced
with PyROOT (ROOT 6.40) by this script; rerun it only if the layout has to
change, and keep the file small:

```python
import ROOT

ROOT.gInterpreter.Declare("""
struct Vec3 { float x; float y; float z; };
struct Particle { Vec3 momentum; float energy; int charge; };
""")
f = ROOT.TFile("split_collection.root", "RECREATE")
t = ROOT.TTree("events", "events")
v = ROOT.std.vector("Particle")()
t.Branch("ReconstructedParticles", v, 32000, 99)  # split level 99 -> dotted sub-branches
r = ROOT.TRandom3(1)
for _ in range(200):
    v.clear()
    for _ in range(r.Poisson(3)):
        p = ROOT.Particle()
        p.momentum.x, p.momentum.y, p.momentum.z = r.Gaus(0, 20), r.Gaus(0, 20), r.Gaus(0, 20)
        p.energy = (p.momentum.x**2 + p.momentum.y**2 + p.momentum.z**2) ** 0.5 + 0.1
        p.charge = -1 if r.Uniform() < 0.5 else 1
        v.push_back(p)
    t.Fill()
ROOT.TParameter("int")("eventsProcessed", 200).Write()
f.Write()
f.Close()
```

## Figures and the gallery

The `examples/gallery` package is both the showcase and the image-regression
suite: `__init__.py` holds the shared `define()` block, the `STYLES` and the
examples, `data.py` writes the toy files and `registry.py` extracts the source
shown in the docs. Each example is a small function returning a `Plot`. The
MkDocs hook `docs/hooks/gallery.py` turns them into the gallery: an overview
(`docs/gallery/index.md`, a card per example, section by section) and a
generated page per example with its figure and complete code, one tab per
style. `tests/test_gallery.py` renders all of them and, with `--mpl`, compares
them pixel-wise (pytest-mpl, RMS tolerance 2) against `docs/images/gallery/`.
An example whose function takes a `style` is rendered in every entry of
`STYLES` (the neutral default, ATLAS, CMS, LHCb, ALICE, DUNE) as
`<name>-<style>.png`, `<name>.png` for the default; every rendering is made
twice, as shown and inside `rf.dark_theme()` (`-dark`), and the docs pick one
per palette. Those PNGs are therefore the documentation images *and* the
baselines. The experiment styles are drawn only when images are compared or
generated, and `-n auto` runs that on every core.

```bash
MPLBACKEND=Agg uv run python examples/gallery               # look at examples/out/*.png
MPLBACKEND=Agg uv run python examples/gallery --style CMS   # ... in another style
uv run pytest tests/test_gallery.py --mpl -n auto           # compare against the baselines
uv run pytest tests/test_gallery.py -n auto --mpl-generate-path=docs/images/gallery   # accept changes
```

After any visual change: regenerate the baselines, open the PNGs and check
them by eye, and commit them with the code. CI compares on Linux only (fonts
differ elsewhere) and, when a comparison fails, uploads an HTML report with
baseline, result and difference images as the `mpl-results-*` artifact. To add
an example, register a function with `@example(name, title, section=...)` and
give it a docstring; the test suite fails until its baseline images exist. Its
parameters are attribute names of `Dataset`, plus `style` when the figure should
be shown in every style (pass it on as `style=style`; leave it out when a style
is the point of the example or the figure goes into axes of your own). It runs
inside the directory holding the toy files (so name them `"signal.root"`, never
through a variable), and the hook prints only the body (blank lines and
comments included) — write it as a user would. Put an object into `define()` —
the *Setup* block of the example pages — only when several examples use it;
anything a single example needs belongs in that example.

## Pull requests

- Add tests for behaviour changes; unit tests assert histogram contents and
  matplotlib structure. Rendered output is covered by the gallery (below).
- Run the checks above before pushing; `pre-commit run --all-files` does most
  of it.

## Releasing

1. Update the version in `src/rootfig/__init__.py`.
2. Commit, tag `vX.Y.Z`, push the tag.
3. The `release.yml` workflow builds the distribution, refuses a tag that does
   not match `rootfig.__version__`, and publishes it to PyPI via Trusted
   Publishing (configure the publisher on PyPI first: repository
   `jbeirer/rootfig`, workflow `release.yml`, environment `pypi`).

The documentation is published by the `docs` job of `ci.yml` on every push to
`main` (`mkdocs gh-deploy` to the `gh-pages` branch). Once, in the repository
settings, set GitHub Pages to serve from that branch.
