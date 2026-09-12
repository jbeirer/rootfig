"""The rootfig gallery: one small example per feature, on a toy dataset.

Run ``uv run python examples/gallery`` (or plain ``python`` where rootfig is
installed). Toy ROOT files and one PNG per example are written to
``examples/out/`` in a few seconds. The same functions feed
``docs/gallery.md`` (code shown next to each figure) and
``tests/test_gallery.py`` (pixel comparison against ``docs/images/gallery/``),
so the pictures in the documentation are always made by the code shown.

This module holds what a reader of the gallery is looking for: the shared
:func:`define` block and the examples. The toy data lives in :mod:`.data`, the
registry and the source extraction in :mod:`.registry`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rootfig as rf

from .data import write_dataset
from .registry import EXAMPLES, Example, body_source, example

__all__ = [
    "DEFAULT_OUT",
    "EXAMPLES",
    "Dataset",
    "Example",
    "body_source",
    "define",
    "example",
    "main",
    "make_dataset",
]

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "out"


@dataclass
class Dataset:
    """The objects shared by the examples; the toy files are named by relative path."""

    signal: rf.Sample
    zjets: rf.Sample
    diboson: rf.Sample
    data: rf.Sample
    mc: list[rf.Sample]
    pt: rf.Variable
    mll: rf.Variable
    met: rf.Variable
    atlas: rf.Style


def define() -> Dataset:
    """Describe the samples, variables and style that more than one example needs.

    The body of this function is the *Setup* section of ``docs/gallery.md``, so it
    stays deliberately small: anything only one example uses is defined in that
    example instead. Examples run inside the directory holding the toy files
    (see :meth:`Example.run`), hence the bare file names.
    """
    signal = rf.Sample("signal.root", tree="events", label="Signal", weight="weight", scale=0.03)
    zjets = rf.Sample("background.root", tree="events", label="Z + jets", weight="weight")
    diboson = rf.Sample("diboson.root", tree="events", label="Diboson", weight="weight", scale=0.15)
    data = rf.Sample("data.root", tree="events", label="Data", is_data=True)
    mc = [zjets, diboson, signal]  # stacked bottom to top

    pt = rf.Variable("Muon_pt", bins=(30, 0, 300), label=r"$p_T^{\mu}$", unit="GeV")
    mll = rf.Variable("m_ll", bins=(70, 50, 260), label=r"$m_{\ell\ell}$", unit="GeV")
    met = rf.Variable("MET", bins=(40, 0, 400), label=r"$E_T^{miss}$", unit="GeV")

    atlas = rf.Style(experiment="ATLAS", status="Internal", lumi=140, com=13.6)

    return Dataset(
        signal=signal,
        zjets=zjets,
        diboson=diboson,
        data=data,
        mc=mc,
        pt=pt,
        mll=mll,
        met=met,
        atlas=atlas,
    )


def make_dataset(out: Path) -> Dataset:
    """Write the toy files into ``out`` and return the shared :class:`Dataset`.

    ``define()`` names the files by bare relative path and a ``Sample`` checks that
    its files exist, so it runs inside ``out`` — as the examples do later.
    """
    write_dataset(out)
    with contextlib.chdir(out):
        return define()


# --------------------------------------------------------------------------------------
# Examples. Parameters are Dataset attributes; bodies are what a user would write in the
# directory holding the toy files.
# --------------------------------------------------------------------------------------


@example("quick", "The one-liner")
def quick() -> rf.Plot:
    """A file, a branch, a selection and a binning. rootfig reads only the branches it
    needs, applies the cut to each muon and draws the result with sensible defaults.
    ``xlabel`` and ``unit`` dress the axes (a ``Variable`` does the same, reusably)."""
    return rf.plot(
        "signal.root",
        "Muon_pt",
        tree="events",
        selection="Muon_pt > 20",
        bins=(50, 0, 300),
        xlabel=r"$p_T^{\mu}$",
        unit="GeV",
    )


@example("overlay_ratio", "Several samples, normalised, with a ratio panel")
def overlay_ratio() -> rf.Plot:
    """A ``{label: file}`` mapping gives one histogram per sample with a binning shared by
    all. ``normalize=True`` scales each to unit area and ``ratio=True`` adds a panel with
    every sample divided by the first, uncertainties propagated."""
    return rf.plot(
        {"Signal": "signal.root", "Z + jets": "background.root"},
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
def stack_data(mc: list[rf.Sample], data: rf.Sample, pt: rf.Variable, atlas: rf.Style) -> rf.Plot:
    """A ``Cut`` combines selection strings with ``&`` and carries a label for the plot.
    Simulation is stacked bottom to top in the given order with a hatched
    statistical-uncertainty band, data is drawn as points, and the ratio panel shows data
    over the total prediction. The returned ``Plot`` holds plain matplotlib objects, so any
    further customisation is ordinary matplotlib code."""
    tight = rf.Cut("Muon_isTight", label="tight") & "abs(Muon_eta) < 2.5"
    p = rf.plot(
        mc, pt, observed=data, selection=tight, stack=True, ratio=True, logy=True, style=atlas
    )
    p.ax.axvline(100, color="gray", linestyle="--", linewidth=1)
    return p


@example("fill_stats", "Filled histograms, a statistics box and free text")
def fill_stats(signal: rf.Sample, zjets: rf.Sample, met: rf.Variable) -> rf.Plot:
    """``histtype="fill"`` draws translucent areas. ``stats=True`` lists entries, mean and
    standard deviation per sample below the legend, and ``text`` adds lines under the label.
    The selection uses ``count()`` to reduce a per-muon flag to a per-event requirement."""
    return rf.plot(
        [signal, zjets],
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
def log_axes(signal: rf.Sample, zjets: rf.Sample, diboson: rf.Sample) -> rf.Plot:
    """``rf.log_bins`` builds logarithmically spaced edges; ``logx`` and ``logy`` set the
    scales and ``errorbars=True`` draws statistical uncertainties on the outlines. The
    variable is an expression: the scalar sum of the jet transverse momenta per event."""
    return rf.plot(
        [zjets, diboson, signal],
        rf.Variable("sum(Jet_pt)", bins=rf.log_bins(30, 20, 2000), label=r"$H_T$", unit="GeV"),
        selection="nJet >= 1",
        logx=True,
        logy=True,
        errorbars=True,
        normalize=True,
    )


@example("robust_range", "Automatic ranges: full versus robust")
def robust_range(signal: rf.Sample, zjets: rf.Sample) -> rf.Plot:
    """Sentinels such as ``-999`` wreck an automatic range. ``range="robust"`` ignores far
    outliers when choosing the range (nothing is removed from the data, they end up in the
    underflow). Passing ``ax=`` draws into your own axes, so two rootfig plots share one
    figure."""
    _, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    rf.plot([signal, zjets], "lep_iso", bins=40, range="auto", ax=left, title='range="auto"')
    return rf.plot(
        [signal, zjets], "lep_iso", bins=40, range="robust", ax=right, title='range="robust"'
    )


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
        figsize=(7, 5.6),
    )


@example("object_vs_event", "Per-object versus per-event selections")
def object_vs_event(signal: rf.Sample, pt: rf.Variable) -> rf.Plot:
    """``Muon_pt`` is a list per event. A per-object cut such as ``Muon_pt > 100`` masks
    individual muons, while ``any(Muon_pt > 100)`` is per event: it keeps whole events, with all
    their muons, soft ones included. ``Sample.with_`` derives variants of a sample."""
    return rf.plot(
        [
            signal.with_(label="All muons"),
            signal.with_(label="Muon_pt > 100", selection="Muon_pt > 100"),
            signal.with_(label="any(Muon_pt > 100)", selection="any(Muon_pt > 100)"),
        ],
        pt,
        logy=True,
    )


@example("expressions", "Expressions and pre-filled histograms")
def expressions(signal: rf.Sample, pt: rf.Variable) -> rf.Plot:
    """Variables are expressions with NumPy functions and per-event reductions:
    ``first(Muon_pt)`` is the leading muon, ``Muon_pt * cosh(Muon_eta)`` the muon momentum.
    ``rf.histogram`` returns a plain ``hist.Hist``; ``rf.plot_histograms`` draws any collection
    of them with the usual options."""
    all_muons = rf.histogram(signal, pt)
    leading = rf.histogram(signal, pt.with_(expression="first(Muon_pt)"))
    momentum = rf.histogram(signal, pt.with_(expression="Muon_pt * cosh(Muon_eta)"))
    return rf.plot_histograms(
        [all_muons, leading, momentum],
        labels=["All muons", "Leading muon", r"Muon $|\vec{p}|$"],
        variable=pt.with_(label=r"$p_T^{\mu}$ or $|\vec{p}^{\,\mu}|$"),
        logy=True,
    )


@example("ratio_reference", "Ratio to a chosen sample, per-sample drawing styles")
def ratio_reference(signal: rf.Sample, zjets: rf.Sample, diboson: rf.Sample) -> rf.Plot:
    """``ratio="Z + jets"`` picks the reference sample by label; ``ratio_ylim`` and
    ``ratio_label`` override the automatic range and label. A ``Sample`` can carry its own
    ``color`` and ``histtype``."""
    return rf.plot(
        [
            zjets.with_(color="black", histtype="errorbar"),
            diboson.with_(color="#d95f02"),
            signal.with_(color="#1b9e77", histtype="fill"),
        ],
        rf.Variable("nJet", bins=(9, -0.5, 8.5), label="Jet multiplicity"),
        normalize=True,
        ratio="Z + jets",
        ratio_ylim=(0, 3),
        ratio_label="Ratio to Z + jets",
    )


@example("style_colors", "A custom style: colours, legend position, figure size, rcParams")
def style_colors(signal: rf.Sample, zjets: rf.Sample) -> rf.Plot:
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
        [signal, zjets],
        rf.Variable("Jet_btag", bins=(25, 0, 1), label="Jet b-tag score"),
        selection="Jet_pt > 30",
        normalize=True,
        histtype="fill",
        style=style,
    )


@example("cms_density", "Another experiment style, density normalisation and summed overflow")
def cms_density(signal: rf.Sample, zjets: rf.Sample, diboson: rf.Sample) -> rf.Plot:
    """A bare experiment name selects mplhep's style sheet and label. ``normalize="density"``
    makes the integral one and ``flow="sum"`` adds the under- and overflow to the edge bins,
    which is why the last bin sticks out."""
    return rf.plot(
        [zjets, diboson, signal],
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
def hist2d(signal: rf.Sample, met: rf.Variable) -> rf.Plot:
    """``rf.plot2d`` fills a 2D histogram of one sample; both variables must have the same
    structure (both per-event here). ``logz`` and ``cmap`` control the colour scale."""
    return rf.plot2d(
        signal,
        rf.Variable("sum(Jet_pt)", bins=(40, 0, 800), label=r"$H_T$", unit="GeV"),
        met,
        selection="nJet >= 2",
        logz=True,
        cmap="magma",
        zlabel="Events",
    )


@example("correlation", "A correlation matrix")
def correlation(signal: rf.Sample) -> rf.Plot:
    """``rf.correlation`` computes the (weighted) linear correlation of several per-event
    quantities and draws it as an annotated matrix; ``percent=True`` labels cells in percent."""
    return rf.correlation(
        signal,
        ["MET", "sum(Jet_pt)", "nJet", "nMuon", "m_ll"],
        labels=[r"$E_T^{miss}$", r"$H_T$", r"$N_{jet}$", r"$N_{\mu}$", r"$m_{\ell\ell}$"],
        percent=True,
        figsize=(7, 5.6),
    )


@example("luminosity", "Cross sections and a luminosity instead of hand-made scale factors")
def luminosity(mll: rf.Variable) -> rf.Plot:
    """Samples carrying ``xsec`` (and ``ngen``, here the number of entries) are scaled to
    expected yields with ``lumi=``: weights are multiplied by ``xsec * lumi / ngen``. Units
    may be given in the strings; the luminosity also lands in the label. Any experiment name
    works in a ``Style``, with GeV and ab^-1 where a lepton collider needs them."""
    # the same three toy files, this time as e+e- processes with a cross section each
    zh = rf.Sample("signal.root", tree="events", label="ZH", weight="weight", xsec="0.2 pb")
    ww = rf.Sample("background.root", tree="events", label="WW", weight="weight", xsec="16.4 pb")
    zz = rf.Sample("diboson.root", tree="events", label="ZZ", weight="weight", xsec="1.4 pb")
    fcc = rf.Style(experiment="FCC-ee", status="Simulation", com="240 GeV")

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
def efficiency(signal: rf.Sample, zjets: rf.Sample, pt: rf.Variable) -> rf.Plot:
    """``rf.efficiency`` fills the entries passing ``selection`` (all here) and those also
    passing ``passed`` with one binning, and draws their ratio with Wilson score intervals.
    The muon identification efficiency versus transverse momentum, for two samples."""
    return rf.efficiency([signal, zjets], pt, passed="Muon_isTight", ylim=(0.5, None))


@example("profile", "Profiles: a statistic of one variable in bins of another")
def profile(signal: rf.Sample, zjets: rf.Sample, met: rf.Variable) -> rf.Plot:
    """``rf.profile`` draws the weighted mean of ``y`` per bin of ``x`` with its standard
    error (ROOT's TProfile); ``statistic="std"`` gives the standard deviation instead, the
    usual resolution-versus-variable plot when ``y`` is a residual."""
    return rf.profile(
        [signal, zjets],
        rf.Variable("sum(Jet_pt)", bins=(16, 0, 800), label=r"$H_T$", unit="GeV"),
        met,
        selection="nJet >= 1",
    )


@example("many_plots", "Many plots in a loop, saved by variable name")
def many_plots(mc: list[rf.Sample], data: rf.Sample, atlas: rf.Style) -> rf.Plot:
    """The typical analysis script: a list of variables, one call each, saved to a directory.
    Given a directory (created if needed), ``Plot.save`` names the file after the variable;
    ``formats`` writes several at once."""
    selection = rf.Cut("count(Muon_isTight) >= 1") & "nJet >= 1"
    variables = [
        rf.Variable("MET", bins=(40, 0, 400), label=r"$E_T^{miss}$", unit="GeV"),
        rf.Variable("nMuon", bins=(7, -0.5, 6.5), label=r"$N_{\mu}$"),
        rf.Variable("Muon_phi", bins=(32, -3.2, 3.2), label=r"$\phi^{\mu}$", unit="rad"),
    ]
    for variable in variables:
        p = rf.plot(
            mc, variable, observed=data, selection=selection, stack=True, ratio=True, style=atlas
        )
        p.save(
            "plots/", formats=["pdf", "png"]
        )  # plots/MET.pdf, plots/MET.png, plots/nMuon.pdf ...
    return p


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def main(out: Path = DEFAULT_OUT) -> None:
    """Write the toy dataset and every example figure into ``out``."""
    out = out.resolve()  # the examples run inside it; keep saving here after they return
    dataset = make_dataset(out)
    for number, ex in enumerate(EXAMPLES, start=1):
        plot = ex.run(dataset, cwd=out)
        plot.save(out / f"{number:02d}_{ex.name}.png", dpi=150)
        plot.close()
        print(f"{number:02d}_{ex.name}.png  {ex.title}")

    # things that are not figures; like the examples, they read the toy files by bare name
    with contextlib.chdir(out):
        print()
        print(rf.summarize(dataset.mc, ["MET", "Muon_pt"], selection="nMuon > 0"))
        print()
        print(
            rf.cutflow(
                dataset.mc,
                ["nMuon >= 2", rf.Cut("MET > 50", label="MET > 50 GeV"), "any(Jet_btag > 0.8)"],
            )
        )
        events = rf.load(dataset.signal, ["MET", "count(Muon_pt)"], selection="nJet >= 2")
    print(f"\n{len(events)} signal events with >= 2 jets; fields {events.fields}")
    print(f"\nfigures written to {out}")
