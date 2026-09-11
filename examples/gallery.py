"""The rootfig gallery: a toy dataset and one small example per feature.

Run ``uv run python examples/gallery.py`` (or plain ``python`` where rootfig is
installed). Toy ROOT files and one PNG per example are written to
``examples/out/`` in a few seconds. The same functions feed
``docs/gallery.md`` (code shown next to each figure) and
``tests/test_gallery.py`` (pixel comparison against ``docs/images/gallery/``),
so the pictures in the documentation are always made by the code shown.

Each example is a plain function decorated with :func:`example`. Its
parameters are names of :class:`Dataset` attributes (files, samples,
variables) so that the body reads exactly like user code.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

import rootfig as rf

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "out"
N_EVENTS = 20_000

# --------------------------------------------------------------------------------------
# Toy data: three simulated processes and one "observed" sample
# --------------------------------------------------------------------------------------

Kind = Literal["signal", "zjets", "diboson"]


def make_events(kind: Kind, n: int, seed: int) -> dict[str, Any]:
    """Generate ``n`` events of one process as a dict of flat and jagged columns."""
    rng = np.random.default_rng(seed)
    p = {
        "signal": {"mu": 2.0, "pt": 60.0, "tight": 0.85, "nj": 3.0, "jpt": 70.0, "met": 60.0},
        "zjets": {"mu": 1.9, "pt": 28.0, "tight": 0.75, "nj": 1.5, "jpt": 40.0, "met": 22.0},
        "diboson": {"mu": 1.6, "pt": 40.0, "tight": 0.75, "nj": 2.2, "jpt": 45.0, "met": 35.0},
    }[kind]

    n_muon = rng.poisson(p["mu"], size=n)
    n_mu_total = int(n_muon.sum())
    n_jet = rng.poisson(p["nj"], size=n)
    n_jet_total = int(n_jet.sum())
    jet_pt = ak.unflatten(rng.exponential(p["jpt"], n_jet_total) + 20.0, n_jet)
    ht = ak.to_numpy(ak.fill_none(ak.sum(jet_pt, axis=1), 0.0))

    if kind == "signal":
        m_ll = rng.normal(220.0, 5.0, n)
        btag = rng.beta(4.0, 1.5, n_jet_total)
    elif kind == "zjets":
        m_ll = np.where(
            rng.random(n) < 0.85,
            rng.normal(91.2, 3.5, n),
            50.0 + rng.exponential(45.0, n),
        )
        btag = rng.beta(1.0, 6.0, n_jet_total)
    else:
        m_ll = np.where(
            rng.random(n) < 0.35,
            rng.normal(91.2, 4.0, n),
            50.0 + rng.exponential(70.0, n),
        )
        btag = rng.beta(1.5, 4.0, n_jet_total)

    return {
        "event": np.arange(n, dtype=np.int64),
        "nMuon": n_muon.astype(np.int32),
        "Muon_pt": ak.unflatten(rng.exponential(p["pt"], n_mu_total) + 5.0, n_muon),
        "Muon_eta": ak.unflatten(rng.uniform(-2.7, 2.7, n_mu_total), n_muon),
        "Muon_phi": ak.unflatten(rng.uniform(-np.pi, np.pi, n_mu_total), n_muon),
        "Muon_charge": ak.unflatten(rng.choice([-1, 1], n_mu_total).astype(np.int32), n_muon),
        "Muon_isTight": ak.unflatten(rng.random(n_mu_total) < p["tight"], n_muon),
        "nJet": n_jet.astype(np.int32),
        "Jet_pt": jet_pt,
        "Jet_eta": ak.unflatten(rng.uniform(-4.5, 4.5, n_jet_total), n_jet),
        "Jet_btag": ak.unflatten(btag, n_jet),
        "MET": rng.exponential(p["met"], n) + rng.normal(0.25, 0.1, n).clip(0.0) * ht,
        "m_ll": m_ll,
        # an isolation-like variable with a -999 sentinel for "not computed"
        "lep_iso": np.where(rng.random(n) < 0.03, -999.0, rng.exponential(0.08, n)),
        "weight": rng.normal(1.0, 0.1, n),
    }


def concatenate(*parts: dict[str, Any]) -> dict[str, Any]:
    """Concatenate event dicts column by column."""
    return {k: ak.concatenate([ak.Array(part[k]) for part in parts]) for k in parts[0]}


def write_tree(path: Path, columns: dict[str, Any], tree: str = "events") -> None:
    """Write ``columns`` as a TTree (uproot writes RNTuple on assignment, so use mktree)."""
    arrays = {k: v if isinstance(v, ak.Array) else ak.Array(v) for k, v in columns.items()}
    types = {
        k: ak.to_numpy(v).dtype
        if v.layout.purelist_depth == 1
        else f"var * {ak.to_numpy(ak.flatten(v)).dtype}"
        for k, v in arrays.items()
    }
    with uproot.recreate(path) as file:
        out = file.mktree(tree, types)
        out.extend(arrays)


@dataclass
class Dataset:
    """Everything the examples refer to: files, samples, variables and an output directory."""

    signal_file: Path
    background_file: Path
    diboson_file: Path
    data_file: Path
    out: Path
    sig: rf.Sample
    bkg: rf.Sample
    dib: rf.Sample
    data: rf.Sample
    mc: list[rf.Sample]
    pt: rf.Variable
    mll: rf.Variable
    met: rf.Variable
    atlas: rf.Style
    tight: rf.Cut
    zh: rf.Sample
    ww: rf.Sample
    zz: rf.Sample
    fcc: rf.Style


def write_dataset(out: Path) -> None:
    """Write signal.root, background.root, diboson.root and data.root into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    write_tree(out / "signal.root", make_events("signal", N_EVENTS, seed=1))
    write_tree(out / "background.root", make_events("zjets", N_EVENTS, seed=2))
    write_tree(out / "diboson.root", make_events("diboson", N_EVENTS, seed=3))
    # "data" is a Z+jets-like sample plus a diboson-like admixture, unweighted
    observed = concatenate(
        make_events("zjets", N_EVENTS, seed=4), make_events("diboson", N_EVENTS * 3 // 20, seed=5)
    )
    observed["weight"] = np.ones(len(observed["event"]))
    write_tree(out / "data.root", observed)


def define(out: Path) -> Dataset:
    """Describe the samples, variables, cuts and style once; every example reuses them."""
    signal_file = out / "signal.root"
    background_file = out / "background.root"
    diboson_file = out / "diboson.root"
    data_file = out / "data.root"

    sig = rf.Sample(signal_file, tree="events", label="Signal", weight="weight", scale=0.03)
    bkg = rf.Sample(background_file, tree="events", label="Z + jets", weight="weight")
    dib = rf.Sample(diboson_file, tree="events", label="Diboson", weight="weight", scale=0.15)
    data = rf.Sample(data_file, tree="events", label="Data", is_data=True)
    mc = [bkg, dib, sig]  # stacked bottom to top

    pt = rf.Variable("Muon_pt", bins=(30, 0, 300), label=r"$p_T^{\mu}$", unit="GeV")
    mll = rf.Variable("m_ll", bins=(70, 50, 260), label=r"$m_{\ell\ell}$", unit="GeV")
    met = rf.Variable("MET", bins=(40, 0, 400), label=r"$E_T^{miss}$", unit="GeV")

    tight = rf.Cut("Muon_isTight", label="tight") & "abs(Muon_eta) < 2.5"
    atlas = rf.Style(experiment="ATLAS", status="Internal", lumi=140, com=13.6)

    # the same files read as e+e- processes with cross sections, to be scaled to a luminosity
    zh = rf.Sample(signal_file, tree="events", label="ZH", weight="weight", xsec="0.2 pb")
    ww = rf.Sample(background_file, tree="events", label="WW", weight="weight", xsec="16.4 pb")
    zz = rf.Sample(diboson_file, tree="events", label="ZZ", weight="weight", xsec="1.4 pb")
    fcc = rf.Style(experiment="FCC-ee", status="Simulation", com="240 GeV")
    return Dataset(
        signal_file=signal_file,
        background_file=background_file,
        diboson_file=diboson_file,
        data_file=data_file,
        out=out,
        sig=sig,
        bkg=bkg,
        dib=dib,
        data=data,
        mc=mc,
        pt=pt,
        mll=mll,
        met=met,
        atlas=atlas,
        tight=tight,
        zh=zh,
        ww=ww,
        zz=zz,
        fcc=fcc,
    )


def make_dataset(out: Path) -> Dataset:
    """Write the toy files into ``out`` and return the :class:`Dataset` describing them."""
    write_dataset(out)
    return define(out)


# --------------------------------------------------------------------------------------
# Example registry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """One gallery entry: a name (file stem), a title and the function producing the plot."""

    name: str
    title: str
    func: Callable[..., rf.Plot]

    @property
    def description(self) -> str:
        """The function's docstring, dedented."""
        return inspect.cleandoc(self.func.__doc__ or "")

    def run(self, dataset: Dataset) -> rf.Plot:
        """Call the example with the dataset attributes it asks for."""
        names = inspect.signature(self.func).parameters
        return self.func(**{name: getattr(dataset, name) for name in names})


EXAMPLES: list[Example] = []


def example(name: str, title: str) -> Callable[[Callable[..., rf.Plot]], Callable[..., rf.Plot]]:
    """Register a gallery example."""

    def register(func: Callable[..., rf.Plot]) -> Callable[..., rf.Plot]:
        EXAMPLES.append(Example(name, title, func))
        return func

    return register


def body_source(func: Callable[..., Any], *, returns: Literal["strip", "omit"] = "strip") -> str:
    """Return the body of ``func`` as user-facing code: no signature, docstring or ``return``.

    ``returns="strip"`` keeps the returned expression as a bare statement
    (``return rf.plot(...)`` becomes ``rf.plot(...)``); ``"omit"`` drops the
    return statement entirely. A ``return name`` is always dropped.
    """
    source = textwrap.dedent(inspect.getsource(func))
    node = ast.parse(source).body[0]
    if not isinstance(node, ast.FunctionDef):
        msg = f"{func!r} is not a plain function"
        raise TypeError(msg)
    statements = list(node.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
    ):
        statements = statements[1:]  # docstring
    lines = source.splitlines()
    chunks: list[str] = []
    for statement in statements:
        text = "\n".join(lines[statement.lineno - 1 : statement.end_lineno])
        if isinstance(statement, ast.Return):
            if (
                returns == "omit"
                or statement.value is None
                or isinstance(statement.value, ast.Name)
            ):
                continue
            text = text.replace("return ", "", 1)
        chunks.append(text)
    return textwrap.dedent("\n".join(chunks)).rstrip() + "\n"


# --------------------------------------------------------------------------------------
# Examples. Parameters are Dataset attributes; bodies are what a user would write.
# --------------------------------------------------------------------------------------


@example("quick", "The one-liner")
def quick(signal_file: Path) -> rf.Plot:
    """A file, a branch, a selection and a binning. rootfig reads only the branches it
    needs, applies the cut to each muon and draws the result with sensible defaults.
    ``xlabel`` and ``unit`` dress the axes (a ``Variable`` does the same, reusably)."""
    return rf.plot(
        signal_file,
        "Muon_pt",
        tree="events",
        selection="Muon_pt > 20",
        bins=(50, 0, 300),
        xlabel=r"$p_T^{\mu}$",
        unit="GeV",
    )


@example("overlay_ratio", "Several samples, normalised, with a ratio panel")
def overlay_ratio(signal_file: Path, background_file: Path) -> rf.Plot:
    """A ``{label: file}`` mapping gives one histogram per sample with a binning shared by
    all. ``normalize=True`` scales each to unit area and ``ratio=True`` adds a panel with
    every sample divided by the first, uncertainties propagated."""
    return rf.plot(
        {"Signal": signal_file, "Z + jets": background_file},
        "Muon_pt",
        tree="events",
        selection="Muon_isTight and abs(Muon_eta) < 2.5",
        weight="weight",
        bins=(40, 0, 400),
        unit="GeV",
        normalize=True,
        ratio=True,
    )


@example("stack_data", "Stacked simulation with data and an experiment label")
def stack_data(
    mc: list[rf.Sample], data: rf.Sample, pt: rf.Variable, tight: rf.Cut, atlas: rf.Style
) -> rf.Plot:
    """``Sample``, ``Variable``, ``Cut`` and ``Style`` objects are defined once (see *Setup*)
    and reused. Simulation is stacked bottom to top in the given order with a hatched
    statistical-uncertainty band, data is drawn as points, and the ratio panel shows data
    over the total prediction. The returned ``Plot`` holds plain matplotlib objects, so any
    further customisation is ordinary matplotlib code."""
    p = rf.plot(
        mc, pt, observed=data, selection=tight, stack=True, ratio=True, logy=True, style=atlas
    )
    p.ax.axvline(100, color="gray", linestyle="--", linewidth=1)
    return p


@example("fill_stats", "Filled histograms, a statistics box and free text")
def fill_stats(sig: rf.Sample, bkg: rf.Sample, met: rf.Variable) -> rf.Plot:
    """``histtype="fill"`` draws translucent areas. ``stats=True`` lists entries, mean and
    standard deviation per sample below the legend, and ``text`` adds lines under the label.
    The selection uses ``count()`` to reduce a per-muon flag to a per-event requirement."""
    return rf.plot(
        [sig, bkg],
        met,
        selection="count(Muon_isTight) >= 1",
        histtype="fill",
        stats=True,
        text=[r"$\geq 1$ tight muon", "toy simulation"],
    )


@example("variable_bins", "Variable bin widths, per-width normalisation and overflow bins")
def variable_bins(mc: list[rf.Sample], data: rf.Sample) -> rf.Plot:
    """Bin edges may be any increasing sequence. ``normalize="width"`` divides by the bin
    width so the y axis reads *Events / GeV*, and ``flow="show"`` appends the underflow and
    overflow as extra bins instead of the default arrow hints."""
    edges = [0, 20, 40, 60, 80, 100, 130, 160, 200, 250, 320, 400]
    return rf.plot(
        mc,
        rf.Variable("MET", bins=edges, label=r"$E_T^{miss}$", unit="GeV"),
        observed=data,
        stack=True,
        ratio=True,
        normalize="width",
        flow="show",
        logy=True,
    )


@example("log_axes", "Logarithmic axes with log-spaced bins")
def log_axes(sig: rf.Sample, bkg: rf.Sample, dib: rf.Sample) -> rf.Plot:
    """``rf.log_bins`` builds logarithmically spaced edges; ``logx`` and ``logy`` set the
    scales and ``errorbars=True`` draws statistical uncertainties on the outlines. The
    variable is an expression: the scalar sum of the jet transverse momenta per event."""
    return rf.plot(
        [bkg, dib, sig],
        rf.Variable("sum(Jet_pt)", bins=rf.log_bins(30, 20, 2000), label=r"$H_T$", unit="GeV"),
        selection="nJet >= 1",
        logx=True,
        logy=True,
        errorbars=True,
        normalize=True,
    )


@example("robust_range", "Automatic ranges: full versus robust")
def robust_range(sig: rf.Sample, bkg: rf.Sample) -> rf.Plot:
    """Sentinels such as ``-999`` wreck an automatic range. ``range="robust"`` ignores far
    outliers when choosing the range (nothing is removed from the data, they end up in the
    underflow). Passing ``ax=`` draws into your own axes, so two rootfig plots share one
    figure."""
    _, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    rf.plot([sig, bkg], "lep_iso", bins=40, range="auto", ax=left, title='range="auto"')
    return rf.plot([sig, bkg], "lep_iso", bins=40, range="robust", ax=right, title='range="robust"')


@example("xbreak_ratio", "A broken x axis: peak and far tail without the empty middle")
def xbreak_ratio(
    mc: list[rf.Sample], data: rf.Sample, mll: rf.Variable, atlas: rf.Style
) -> rf.Plot:
    """``xbreak=(a, b)`` removes the range between ``a`` and ``b`` from the x axis and draws
    the two segments side by side with break marks; the ratio panel follows. Here the Z peak
    and a high-mass resonance share one figure."""
    return rf.plot(
        mc,
        mll,
        observed=data,
        stack=True,
        ratio=True,
        logy=True,
        xbreak=(125, 195),
        style=atlas,
    )


@example("object_vs_event", "Per-object versus per-event selections")
def object_vs_event(sig: rf.Sample, pt: rf.Variable) -> rf.Plot:
    """``Muon_pt`` is a list per event. A per-object cut such as ``Muon_pt > 100`` masks
    individual muons, while ``any(Muon_pt > 100)`` is per event: it keeps whole events, with all
    their muons, soft ones included. ``Sample.with_`` derives variants of a sample."""
    return rf.plot(
        [
            sig.with_(label="All muons"),
            sig.with_(label="Muon_pt > 100", selection="Muon_pt > 100"),
            sig.with_(label="any(Muon_pt > 100)", selection="any(Muon_pt > 100)"),
        ],
        pt,
        logy=True,
    )


@example("expressions", "Expressions and pre-filled histograms")
def expressions(sig: rf.Sample, pt: rf.Variable) -> rf.Plot:
    """Variables are expressions with NumPy functions and per-event reductions:
    ``first(Muon_pt)`` is the leading muon, ``Muon_pt * cosh(Muon_eta)`` the muon momentum.
    ``rf.histogram`` returns a plain ``hist.Hist``; ``rf.plot_histograms`` draws any collection
    of them with the usual options."""
    all_muons = rf.histogram(sig, pt)
    leading = rf.histogram(sig, pt.with_(expression="first(Muon_pt)"))
    momentum = rf.histogram(sig, pt.with_(expression="Muon_pt * cosh(Muon_eta)"))
    return rf.plot_histograms(
        [all_muons, leading, momentum],
        labels=["All muons", "Leading muon", r"Muon $|\vec{p}|$"],
        variable=pt.with_(label=r"$p_T^{\mu}$ or $|\vec{p}^{\,\mu}|$"),
        logy=True,
    )


@example("ratio_reference", "Ratio to a chosen sample, per-sample drawing styles")
def ratio_reference(sig: rf.Sample, bkg: rf.Sample, dib: rf.Sample) -> rf.Plot:
    """``ratio="Z + jets"`` picks the reference sample by label; ``ratio_ylim`` and
    ``ratio_label`` override the automatic range and label. A ``Sample`` can carry its own
    ``color`` and ``histtype``."""
    return rf.plot(
        [
            bkg.with_(color="black", histtype="errorbar"),
            dib.with_(color="#d95f02"),
            sig.with_(color="#1b9e77", histtype="fill"),
        ],
        rf.Variable("nJet", bins=(9, -0.5, 8.5), label="Jet multiplicity"),
        normalize=True,
        ratio="Z + jets",
        ratio_ylim=(0, 3),
        ratio_label="Ratio to Z + jets",
    )


@example("style_colors", "A custom style: colours, legend position, figure size, rcParams")
def style_colors(sig: rf.Sample, bkg: rf.Sample) -> rf.Plot:
    """``Style`` bundles appearance: a colour cycle, the legend location, the figure size and
    any matplotlib rcParams. It applies only while the figure is drawn; global matplotlib
    state is untouched unless you call ``rf.use_style``."""
    style = rf.Style(
        colors=["#7570b3", "#e7298a"],
        legend="upper left",
        figsize=(6.5, 5),
        rc={"font.size": 14, "axes.grid": True, "grid.alpha": 0.3},
    )
    return rf.plot(
        [sig, bkg],
        rf.Variable("Jet_btag", bins=(25, 0, 1), label="Jet b-tag score"),
        selection="Jet_pt > 30",
        normalize=True,
        histtype="fill",
        style=style,
    )


@example("cms_density", "Another experiment style, density normalisation and summed overflow")
def cms_density(sig: rf.Sample, bkg: rf.Sample, dib: rf.Sample) -> rf.Plot:
    """A bare experiment name selects mplhep's style sheet and label. ``normalize="density"``
    makes the integral one and ``flow="sum"`` adds the under- and overflow to the edge bins,
    which is why the last bin sticks out."""
    return rf.plot(
        [bkg, dib, sig],
        rf.Variable("Jet_pt", bins=(30, 20, 320), label=r"$p_T^{jet}$", unit="GeV"),
        selection="abs(Jet_eta) < 2.5",
        normalize="density",
        flow="sum",
        style="CMS",
    )


@example("arrays", "In-memory arrays instead of files")
def arrays() -> rf.Plot:
    """Any mapping of NumPy or Awkward arrays is a valid source, so rootfig works just as well
    on arrays you already have in memory."""
    rng = np.random.default_rng(7)
    events = {"x": rng.normal(0.0, 1.0, 50_000), "w": rng.uniform(0.5, 1.5, 50_000)}
    return rf.plot(
        events, "x", weight="w", bins=(60, -4, 4), label="Gaussian", xlabel="$x$", errorbars=True
    )


@example("hist2d", "A two-dimensional histogram")
def hist2d(sig: rf.Sample, met: rf.Variable) -> rf.Plot:
    """``rf.plot2d`` fills a 2D histogram of one sample; both variables must have the same
    structure (both per-event here). ``logz`` and ``cmap`` control the colour scale."""
    return rf.plot2d(
        sig,
        rf.Variable("sum(Jet_pt)", bins=(40, 0, 800), label=r"$H_T$", unit="GeV"),
        met,
        selection="nJet >= 2",
        logz=True,
        cmap="magma",
        zlabel="Events",
    )


@example("correlation", "A correlation matrix")
def correlation(sig: rf.Sample) -> rf.Plot:
    """``rf.correlation`` computes the (weighted) linear correlation of several per-event
    quantities and draws it as an annotated matrix; ``percent=True`` labels cells in percent."""
    return rf.correlation(
        sig,
        ["MET", "sum(Jet_pt)", "nJet", "nMuon", "m_ll"],
        labels=[r"$E_T^{miss}$", r"$H_T$", r"$N_{jet}$", r"$N_{\mu}$", r"$m_{\ell\ell}$"],
        percent=True,
    )


@example("luminosity", "Cross sections and a luminosity instead of hand-made scale factors")
def luminosity(
    ww: rf.Sample, zz: rf.Sample, zh: rf.Sample, mll: rf.Variable, fcc: rf.Style
) -> rf.Plot:
    """Samples carrying ``xsec`` (and ``ngen``, here the number of entries) are scaled to
    expected yields with ``lumi=``: weights are multiplied by ``xsec * lumi / ngen``. Units
    may be given in the strings; the luminosity also lands in the label. Any experiment name
    works in a ``Style``, with GeV and ab^-1 where a lepton collider needs them."""
    return rf.plot(
        [ww, zz, zh],
        mll,
        lumi="10.8 ab^-1",
        stack=True,
        logy=True,
        ratio="significance",
        style=fcc,
    )


@example("efficiency", "Efficiency versus a variable with binomial intervals")
def efficiency(sig: rf.Sample, bkg: rf.Sample, pt: rf.Variable) -> rf.Plot:
    """``rf.efficiency`` fills the entries passing ``selection`` (all here) and those also
    passing ``passed`` with one binning, and draws their ratio with Wilson score intervals.
    The muon identification efficiency versus transverse momentum, for two samples."""
    return rf.efficiency([sig, bkg], pt, passed="Muon_isTight", ylim=(0.5, None))


@example("profile", "Profiles: a statistic of one variable in bins of another")
def profile(sig: rf.Sample, bkg: rf.Sample, met: rf.Variable) -> rf.Plot:
    """``rf.profile`` draws the weighted mean of ``y`` per bin of ``x`` with its standard
    error (ROOT's TProfile); ``statistic="std"`` gives the standard deviation instead, the
    usual resolution-versus-variable plot when ``y`` is a residual."""
    return rf.profile(
        [sig, bkg],
        rf.Variable("sum(Jet_pt)", bins=(16, 0, 800), label=r"$H_T$", unit="GeV"),
        met,
        selection="nJet >= 1",
    )


@example("many_plots", "Many plots in a loop, saved by variable name")
def many_plots(mc: list[rf.Sample], data: rf.Sample, atlas: rf.Style, out: Path) -> rf.Plot:
    """The typical analysis script: a list of variables, one call each, saved to a directory.
    ``Plot.save`` names the file after the variable and can write several formats at once."""
    selection = rf.Cut("count(Muon_isTight) >= 1") & "nJet >= 1"
    variables = [
        rf.Variable("MET", bins=(40, 0, 400), label=r"$E_T^{miss}$", unit="GeV"),
        rf.Variable("nMuon", bins=(7, -0.5, 6.5), label=r"$N_{\mu}$"),
        rf.Variable("Muon_phi", bins=(32, -3.2, 3.2), label=r"$\phi^{\mu}$", unit="rad"),
    ]
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    for variable in variables:
        p = rf.plot(
            mc, variable, observed=data, selection=selection, stack=True, ratio=True, style=atlas
        )
        p.save(plots, formats=["pdf", "png"])  # plots/MET.pdf, plots/MET.png, plots/nMuon.pdf ...
    return p


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def main(out: Path = DEFAULT_OUT) -> None:
    """Write the toy dataset and every example figure into ``out``."""
    dataset = make_dataset(out)
    for number, ex in enumerate(EXAMPLES, start=1):
        plot = ex.run(dataset)
        plot.save(out / f"{number:02d}_{ex.name}.png", dpi=150)
        plot.close()
        print(f"{number:02d}_{ex.name}.png  {ex.title}")

    # things that are not figures
    print()
    print(rf.summarize(dataset.mc, ["MET", "Muon_pt"], selection="nMuon > 0"))
    print()
    print(
        rf.cutflow(
            [dataset.zh, dataset.ww, dataset.zz],
            ["nMuon >= 2", rf.Cut("MET > 50", label="MET > 50 GeV"), "any(Jet_btag > 0.8)"],
            lumi="10.8 ab^-1",
        )
    )
    events = rf.load(dataset.sig, ["MET", "count(Muon_pt)"], selection="nJet >= 2")
    print(f"\n{len(events)} signal events with >= 2 jets; fields {events.fields}")
    print(f"\nfigures written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    main(parser.parse_args().out)
