"""End-to-end tests of the public API against generated ROOT files."""

from __future__ import annotations

import importlib
import io
import re
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pytest
import uproot
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import ScaledTranslation

import rootfig as rf
from rootfig.api import plots1d
from rootfig.api.plots2d import _split_bins
from rootfig.errors import (
    BinningError,
    IncompatibleWeightError,
    MissingBranchError,
    RootfigWarning,
    SelectionError,
    SourceError,
)
from rootfig.histograms import poisson_interval
from rootfig.histograms.binomial import normal_interval
from rootfig.model.style import EXPERIMENT_STYLES
from rootfig.plotting import (
    add_experiment_label,
    raise_ylim_above,
    style_context,
)
from rootfig.plotting.style import LABEL_MIN_SCALE, align_experiment_label


def panel_ylabel(plot: Any) -> str:
    """The lower panel's y label, ignoring the wrapping that makes it fit the panel."""
    return plot.panel_ax.get_ylabel().replace("\n", " ")


class TestLoad:
    def test_branches_and_expressions(
        self, signal_file_any_format: Path, signal_columns: dict[str, Any]
    ) -> None:
        arrays = rf.load(
            signal_file_any_format, ["MET", "Muon_pt", "count(Muon_pt)"], tree="events"
        )
        assert arrays.fields == ["MET", "Muon_pt", "count(Muon_pt)"]
        assert len(arrays) == 2000
        assert arrays["count(Muon_pt)"].tolist() == signal_columns["nMuon"].tolist()

    def test_named_expressions_and_selection(
        self, signal_file: Path, signal_columns: dict[str, Any]
    ) -> None:
        arrays = rf.load(
            signal_file,
            {"met_gev": "MET / 1000", "n": "nMuon"},
            tree="events",
            selection="nMuon >= 2",
        )
        assert arrays.fields == ["met_gev", "n"]
        assert len(arrays) == int((signal_columns["nMuon"] >= 2).sum())
        assert ak.all(arrays["n"] >= 2)

    def test_all_branches(self, signal_file: Path) -> None:
        arrays = rf.load(signal_file, tree="events")
        assert "jet1_b-tag" in arrays.fields
        assert "Muon_pt" in arrays.fields

    def test_single_string_and_sample(self, signal_file: Path) -> None:
        sample = rf.Sample(signal_file, tree="events", selection="MET > 50", entry_stop=500)
        arrays = rf.load(sample, "MET")
        assert ak.all(arrays["MET"] > 50)
        assert len(arrays) < 500

    def test_entry_range(self, signal_file: Path) -> None:
        arrays = rf.load(signal_file, "event", tree="events", entry_start=10, entry_stop=15)
        assert arrays["event"].tolist() == [10, 11, 12, 13, 14]

    def test_object_selection_rejected(self, signal_file: Path) -> None:
        with pytest.raises(SelectionError, match="per-object"):
            rf.load(signal_file, "MET", tree="events", selection="Muon_pt > 20")

    def test_missing_branch(self, signal_file: Path) -> None:
        with pytest.raises(MissingBranchError, match="Muon_Pt"):
            rf.load(signal_file, "Muon_Pt", tree="events")

    def test_errors(self, signal_file: Path, background_file: Path) -> None:
        with pytest.raises(SourceError, match="single sample"):
            rf.load([signal_file, background_file], "MET", tree="events")
        with pytest.raises(SourceError, match="no expressions"):
            rf.load(signal_file, [], tree="events")


class TestHistogram:
    def test_matches_numpy(
        self, signal_file_any_format: Path, signal_columns: dict[str, Any]
    ) -> None:
        h = rf.histogram(
            signal_file_any_format,
            "MET",
            tree="events",
            selection="nMuon > 0",
            weight="weight",
            bins=(25, 0, 250),
        )
        assert isinstance(h, hist.Hist)
        mask = signal_columns["nMuon"] > 0
        expected, _ = np.histogram(
            signal_columns["MET"][mask],
            bins=25,
            range=(0, 250),
            weights=signal_columns["weight"][mask],
        )
        assert h.values().tolist() == pytest.approx(expected.tolist())
        expected_var, _ = np.histogram(
            signal_columns["MET"][mask],
            bins=25,
            range=(0, 250),
            weights=signal_columns["weight"][mask] ** 2,
        )
        assert h.variances().tolist() == pytest.approx(expected_var.tolist())

    def test_jagged_object_selection(
        self, signal_file: Path, signal_columns: dict[str, Any]
    ) -> None:
        h = rf.histogram(
            signal_file,
            "Muon_pt",
            tree="events",
            selection="Muon_pt > 30 and abs(Muon_eta) < 2.0",
            bins=(10, 0, 500),
        )
        pt, eta = signal_columns["Muon_pt"], signal_columns["Muon_eta"]
        selected = ak.flatten(pt[(pt > 30) & (abs(eta) < 2.0)])
        assert h.sum(flow=True).value == pytest.approx(len(selected))
        assert h.values().sum() == pytest.approx(int(ak.sum(selected < 500)))

    def test_derived_and_backtick(self, signal_file: Path, signal_columns: dict[str, Any]) -> None:
        h = rf.histogram(
            signal_file, "sqrt(MET) * `jet1_b-tag`", tree="events", bins=20, range=(0, 20)
        )
        assert h.sum(flow=True).value == 2000

    def test_auto_and_robust_range(self, signal_file: Path) -> None:
        auto = rf.histogram(signal_file, "sentinel", tree="events", bins=20, range="auto")
        robust = rf.histogram(signal_file, "sentinel", tree="events", bins=20, range="robust")
        default = rf.histogram(signal_file, "sentinel", tree="events", bins=20)
        assert auto.axes[0].edges[0] <= -999
        assert robust.axes[0].edges[0] > -10
        assert robust.values(flow=True)[0] > 0  # sentinels went to underflow
        # "robust" is the default, so an unspecified range bins like it
        np.testing.assert_allclose(default.axes[0].edges, robust.axes[0].edges)

    def test_default_range_keeps_a_positive_axis_for_a_log_scale(self, signal_file: Path) -> None:
        """The robust pad must not push the lower edge of a positive variable below zero."""
        h = rf.histogram(signal_file, "MET", tree="events", bins=30)
        assert h.axes[0].edges[0] >= 0.0
        p = rf.plot(signal_file, "MET", tree="events", bins=30, logx=True)
        assert p.ax.get_xlim()[0] > 0.0

    def test_default_range_leaves_room_for_xbreak(self, signal_file: Path) -> None:
        """``xbreak`` is validated against the inferred axis, which must cover the data."""
        h = rf.histogram(signal_file, "MET", tree="events", bins=40)
        low, high = float(h.axes[0].edges[0]), float(h.axes[0].edges[-1])
        p = rf.plot(
            signal_file,
            "MET",
            tree="events",
            bins=40,
            xbreak=(low + 0.3 * (high - low), low + 0.6 * (high - low)),
        )
        assert p.ax_right is not None

    def test_edges_and_log_bins(self, signal_file: Path) -> None:
        h = rf.histogram(signal_file, "Muon_pt", tree="events", bins=rf.log_bins(10, 5, 500))
        assert isinstance(h.axes[0], hist.axis.Variable)
        assert h.axes[0].edges[0] == pytest.approx(5.0)

    def test_normalize(self, signal_file: Path) -> None:
        h = rf.histogram(signal_file, "MET", tree="events", bins=(10, 0, 100), normalize=True)
        assert h.values().sum() == pytest.approx(1.0)

    def test_multiple_files(self, data_dir: Path, background_file: Path) -> None:
        split = rf.histogram(
            str(data_dir / "bkg_part*.root"), "MET", tree="events", bins=(10, 0, 100)
        )
        whole = rf.histogram(background_file, "MET", tree="events", bins=(10, 0, 100))
        assert split.values().tolist() == whole.values().tolist()

    def test_empty_selection(self, signal_file: Path) -> None:
        h = rf.histogram(
            signal_file, "MET", tree="events", selection="MET > 1e6", bins=(10, 0, 100)
        )
        assert h.values().sum() == 0

    def test_empty_selection_auto_range(self, signal_file: Path) -> None:
        h = rf.histogram(signal_file, "MET", tree="events", selection="MET > 1e6", bins=10)
        assert h.axes[0].edges[0] == 0.0
        assert h.axes[0].edges[-1] == 1.0

    def test_nonfinite(self, signal_file: Path) -> None:
        with pytest.warns(RootfigWarning, match="dropped 8 non-finite"):
            h = rf.histogram(signal_file, "with_nan", tree="events", bins=(10, 0, 200))
        assert h.sum(flow=True).value == 2000 - 8
        with pytest.raises(SelectionError, match="non-finite"):
            rf.histogram(signal_file, "with_nan", tree="events", bins=10, nonfinite="error")

    def test_in_memory_arrays(self, signal_arrays: dict[str, ak.Array]) -> None:
        h = rf.histogram(signal_arrays, "nMuon", bins=(10, -0.5, 9.5))
        assert h.sum().value == 2000

    def test_errors(self, signal_file: Path, background_file: Path) -> None:
        with pytest.raises(SourceError, match="single sample"):
            rf.histogram([signal_file, background_file], "MET", tree="events")
        with pytest.raises(MissingBranchError):
            rf.histogram(signal_file, "nope", tree="events")
        with pytest.raises(SelectionError):
            rf.histogram(signal_file, "MET", tree="events", selection="Muon_pt > 20")
        with pytest.raises(IncompatibleWeightError):
            rf.histogram(signal_file, "MET", tree="events", weight="Muon_pt")
        with pytest.raises(BinningError):
            rf.histogram(signal_file, "MET", tree="events", bins=(10, 5, 1))
        with pytest.raises(SourceError):
            rf.histogram(signal_file, "MET", tree="nope")

    def test_histograms_multiple(self, signal_file: Path, background_file: Path) -> None:
        hists = rf.histograms(
            [signal_file, background_file],
            "MET",
            tree="events",
            bins=20,
            label=["S", "B"],
            normalize="density",
        )
        assert [h.label for h in hists] == ["S", "B"]
        assert hists[0].edges.tolist() == hists[1].edges.tolist()
        assert hists[0].normalization == "Density"


class TestPlot:
    def test_quick(self, signal_file_any_format: Path) -> None:
        p = rf.plot(
            signal_file_any_format, "Muon_pt", tree="events", selection="Muon_pt > 20", bins=50
        )
        assert isinstance(p, rf.Plot)
        assert isinstance(p.fig, plt.Figure)
        assert p.panel_ax is None
        assert len(p.histograms) == 1
        assert p.histograms[0].label == "signal" or p.histograms[0].label == "signal_rntuple"
        assert p.ax.get_xlabel() == "Muon_pt"
        assert p.ax.get_ylabel() == "Entries"
        assert p.ax.get_xlim() == pytest.approx(
            (p.histograms[0].edges[0], p.histograms[0].edges[-1])
        )
        assert p.ax.get_legend() is not None

    def test_overlay_normalized_ratio(self, signal_file: Path, background_file: Path) -> None:
        p = rf.plot(
            [signal_file, background_file],
            "Muon_pt",
            tree="events",
            selection="abs(Muon_eta) < 2.5",
            weight="weight",
            bins=(50, 0, 200),
            normalize=True,
            panel="ratio",
            label=["Signal", "Background"],
            unit="GeV",
        )
        assert p.panel_ax is not None
        assert len(p.comparisons) == 1
        assert p.ax.get_ylabel() == "Normalised to unity"
        assert p.panel_ax.get_xlabel() == "Muon_pt [GeV]"
        assert panel_ylabel(p) == "Ratio to Signal"
        for h in p.histograms:
            assert h.integral == pytest.approx(1.0)
        ratio = p.comparisons[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = p.histograms[1].values() / p.histograms[0].values()
        ok = np.isfinite(ratio.values)
        assert ratio.values[ok].tolist() == pytest.approx(expected[ok].tolist())

    def test_ratio_reference_by_label(self, signal_file: Path, background_file: Path) -> None:
        p = rf.plot(
            [signal_file, background_file],
            "MET",
            tree="events",
            bins=10,
            panel="ratio",
            reference="background",
        )
        assert p.panel_ax is not None
        assert panel_ylabel(p) == "Ratio to background"
        with pytest.raises(ValueError, match="not the label of a drawn histogram"):
            rf.plot(
                [signal_file, background_file],
                "MET",
                tree="events",
                bins=10,
                panel="ratio",
                reference="nope",
            )

    def test_stack_with_observed_data(self, signal_file: Path, background_file: Path) -> None:
        mc = [
            rf.Sample(signal_file, tree="events", label="Signal", weight="weight", scale=0.5),
            rf.Sample(background_file, tree="events", label="Background", weight="weight"),
        ]
        observed = rf.Sample(background_file, tree="events", label="Data", entry_stop=1000)
        p = rf.plot(
            mc,
            "MET",
            observed=observed,
            bins=(20, 0, 200),
            stack=True,
            panel="ratio",
            logy=True,
            style="ATLAS",
        )
        assert [h.label for h in p.histograms] == ["Signal", "Background", "Data"]
        assert p.histograms[2].is_data
        assert p.ax.get_yscale() == "log"
        assert p.panel_ax is not None
        assert panel_ylabel(p) == "Data / MC"
        legend_texts = [t.get_text() for t in p.ax.get_legend().get_texts()]
        assert legend_texts == ["Data", "Background", "Signal", "Stat. unc."]
        total = p.histograms[0].values() + p.histograms[1].values()
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.where(total > 0, p.histograms[2].values() / total, np.nan)
        ok = np.isfinite(expected)
        assert p.comparisons[0].values[ok].tolist() == pytest.approx(expected[ok].tolist())

    @staticmethod
    def _marker_colors(p: rf.Plot) -> set[tuple[float, ...]]:
        lines = [*p.ax.lines, *(p.panel_ax.lines if p.panel_ax else [])]
        return {to_rgba(line.get_color()) for line in lines if line.get_marker() == "o"}

    def test_dark_theme_overrides_experiment_style(
        self, signal_file: Path, background_file: Path, tmp_path: Path
    ) -> None:
        mc = [rf.Sample(signal_file, tree="events", label="Signal")]
        observed = rf.Sample(background_file, tree="events", label="Data", entry_stop=1000)
        kwargs: dict[str, Any] = {"observed": observed, "bins": (20, 0, 200), "panel": "ratio"}
        with rf.dark_theme():
            dark = rf.plot(mc, "MET", style="ATLAS", stack=True, **kwargs)
        ink = to_rgba(rf.plotting.DARK_THEME["text.color"])
        assert self._marker_colors(dark) == {ink}
        assert dark.fig.get_facecolor()[3] == 0.0  # transparent despite ATLAS's white
        assert to_rgba(dark.ax.xaxis.label.get_color()) == ink
        with plt.rc_context({"savefig.facecolor": "white"}):  # global state at save time
            (saved,) = dark.save(tmp_path / "dark.png")
        assert plt.imread(saved)[0, :, 3].max() == 0.0  # the top row is background only
        light = rf.plot(mc, "MET", style="ATLAS", stack=True, **kwargs)
        assert self._marker_colors(light) == {to_rgba("black")}
        assert light.fig.get_facecolor() == to_rgba("white")

    def test_data_follows_rgba_tuple_text_color(
        self, signal_file: Path, background_file: Path
    ) -> None:
        grey = (0.8, 0.8, 0.8, 1.0)
        p = rf.plot(
            [rf.Sample(signal_file, tree="events", label="Signal")],
            "MET",
            observed=rf.Sample(background_file, tree="events", label="Data", entry_stop=1000),
            bins=(20, 0, 200),
            stack=True,
            panel="ratio",
            style=rf.Style(rc={"text.color": grey}),
        )
        assert self._marker_colors(p) == {grey}

    def test_dark_theme_covers_plain_matplotlib_in_the_block(
        self, signal_file: Path, background_file: Path, tmp_path: Path
    ) -> None:
        mc = [rf.Sample(signal_file, tree="events", label="Signal")]
        observed = rf.Sample(background_file, tree="events", label="Data", entry_stop=1000)
        with plt.rc_context({"savefig.facecolor": "white"}):  # an opaque global save setting
            before = dict(plt.rcParams)
            with rf.dark_theme():
                fig, ax = plt.subplots()
                p = rf.plot(mc, "MET", observed=observed, bins=(20, 0, 200), ax=ax)
                note = p.ax.text(0.5, 0.5, "hello")
                fig.savefig(tmp_path / "plain.png")
            assert dict(plt.rcParams) == before  # nothing leaks out of the block
        assert plt.imread(tmp_path / "plain.png")[0, :, 3].max() == 0.0  # background only
        ink = to_rgba(rf.plotting.DARK_THEME["text.color"])
        assert self._marker_colors(p) == {ink}
        assert to_rgba(note.get_color()) == ink
        assert to_rgba(ax.spines["left"].get_edgecolor()) == ink
        assert fig.get_facecolor()[3] == ax.get_facecolor()[3] == 0.0
        plt.close(fig)

    def test_stack_ratio_requires_data(self, signal_file: Path, background_file: Path) -> None:
        with pytest.raises(ValueError, match="observed"):
            rf.plot(
                [signal_file, background_file],
                "MET",
                tree="events",
                bins=10,
                stack=True,
                panel="ratio",
            )

    def test_ratio_needs_two(self, signal_file: Path) -> None:
        with pytest.raises(ValueError, match="at least two"):
            rf.plot(signal_file, "MET", tree="events", bins=10, panel="ratio")

    def test_label_mapping_and_variable_object(
        self, signal_file: Path, background_file: Path
    ) -> None:
        var = rf.Variable(
            "MET / 1000", bins=(20, 0, 0.2), label=r"$E_T^{miss}$", unit="TeV", name="met"
        )
        p = rf.plot(
            {"S": signal_file, "B": background_file},
            var,
            tree="events",
            histtype="fill",
            stats=True,
            text="test",
        )
        assert [h.label for h in p.histograms] == ["S", "B"]
        assert p.ax.get_xlabel() == r"$E_T^{miss}$ [TeV]"
        assert p.ax.get_ylabel() == "Events / 0.01 TeV"
        assert p.variable is not None
        assert p.variable.safe_name == "met"
        assert len(p.ax.texts) >= 2  # stats blocks
        assert p.ax.get_legend()._loc == 1  # type: ignore[attr-defined]  # upper right when stats shown

    def test_cut_objects_and_sample_selection(
        self, signal_file: Path, signal_columns: dict[str, Any]
    ) -> None:
        sample = rf.Sample(signal_file, tree="events", selection=rf.Cut("nMuon >= 1"))
        cut = rf.Cut("MET > 20") & "MET < 100"
        p = rf.plot(sample, "MET", selection=cut, bins=(8, 20, 100))
        met, n = signal_columns["MET"], signal_columns["nMuon"]
        expected = int(((n >= 1) & (met > 20) & (met < 100)).sum())
        assert p.histograms[0].integral == expected

    def test_flow_show_and_xlim(self, signal_file: Path) -> None:
        p = rf.plot(
            signal_file,
            "MET",
            tree="events",
            bins=(10, 0, 50),
            flow="show",
            xlim=(0, 60),
            ylim=(0, 500),
        )
        assert p.ax.get_xlim() == (0, 60)
        assert p.ax.get_ylim() == (0, 500)
        p = rf.plot(signal_file, "MET", tree="events", bins=(10, 0, 50), flow="none", logx=True)
        assert p.ax.get_xscale() == "log"

    def test_flow_show_stack_with_mixed_overflow(
        self, signal_file: Path, background_file: Path
    ) -> None:
        """Signal overflows the range, background does not: one common extra bin for all."""
        sig = rf.Sample(signal_file, tree="events", label="S")
        bkg = rf.Sample(background_file, tree="events", label="B", selection="MET < 100")
        p = rf.plot(
            [bkg, sig],
            "MET",
            observed=sig,
            bins=(10, 0, 100),
            stack=True,
            panel="ratio",
            flow="show",
        )
        edges = p.histograms[0].edges
        assert len(edges) == 12  # 10 bins plus the overflow bin
        assert all(len(h.edges) == 12 for h in p.histograms)
        assert p.ax.get_xlim()[1] == pytest.approx(edges[-1])
        assert p.comparisons[0].edges[-1] == pytest.approx(edges[-1])
        assert p.panel_ax is not None
        assert ">100" in [t.get_text() for t in p.panel_ax.get_xticklabels()]
        assert p.ax.get_ylabel().startswith("Events / 10")  # width of the bins as filled
        with pytest.raises(ValueError, match="flow='show'"):
            rf.plot(
                signal_file, "MET", tree="events", bins=(10, 0, 100), xbreak=(20, 80), flow="show"
            )

    def test_existing_axes(self, signal_file: Path, background_file: Path) -> None:
        fig, axes = plt.subplots(1, 2)
        p = rf.plot(signal_file, "MET", tree="events", bins=10, ax=axes[1], legend=False, title="t")
        assert p.ax is axes[1]
        assert p.fig is fig
        assert p.ax.get_legend() is None
        assert p.ax.get_title() == "t"
        fig, (main, lower) = plt.subplots(2)
        p = rf.plot(
            [signal_file, background_file],
            "MET",
            tree="events",
            bins=10,
            ax=(main, lower),
            panel="ratio",
            reference="signal",
        )
        assert p.panel_ax is lower
        with pytest.raises(ValueError, match="at least two histograms"):
            rf.plot(signal_file, "MET", tree="events", panel="ratio", reference="signal")

    def test_style_object_and_figsize(self, signal_file: Path) -> None:
        style = rf.Style(
            experiment="CMS",
            status="Preliminary",
            lumi=138,
            com=13,
            legend="upper left",
            colors=["red"],
        )
        with warnings.catch_warnings():
            # the wide status and luminosity texts must not make constrained layout collapse
            warnings.simplefilter("error", UserWarning)
            p = rf.plot(signal_file, "MET", tree="events", bins=10, style=style, figsize=(4, 3))
        assert tuple(p.fig.get_size_inches()) == (4, 3)
        assert p.histograms[0].hist is not None
        assert any("CMS" in t.get_text() for t in p.ax.texts)
        assert p.ax.get_position().width > 0.5  # the axes still fills most of the figure

    def test_no_global_state_change(self, signal_file: Path) -> None:
        before = dict(plt.rcParams)
        rf.plot(signal_file, "MET", tree="events", bins=10, style="ATLAS")
        after = dict(plt.rcParams)
        assert before["font.size"] == after["font.size"]
        assert before["figure.figsize"] == after["figure.figsize"]

    def test_save(self, signal_file: Path, tmp_path: Path) -> None:
        p = rf.plot(signal_file, "MET", tree="events", bins=10, save=str(tmp_path / "met.png"))
        assert (tmp_path / "met.png").exists()
        p.save(tmp_path, formats=["pdf"])
        assert (tmp_path / "MET.pdf").exists()

    def test_normalize_variants(self, signal_file: Path) -> None:
        for spec in ("density", "width", 100.0):
            p = rf.plot(signal_file, "MET", tree="events", bins=(10, 0, 100), normalize=spec)
            assert p.histograms[0].normalization is not None
        with pytest.raises(BinningError):
            rf.plot(signal_file, "MET", tree="events", bins=10, normalize="bogus")

    def test_errorbars_and_histtype(self, signal_file: Path, background_file: Path) -> None:
        p = rf.plot(
            [signal_file, background_file],
            "MET",
            tree="events",
            bins=10,
            histtype="errorbar",
            errorbars=True,
        )
        assert len(p.ax.containers) >= 2

    def test_nonfinite_warning_context(self, signal_file: Path) -> None:
        with pytest.warns(RootfigWarning, match="^signal: dropped 8"):
            rf.plot(signal_file, "with_nan", tree="events", bins=10)

    def test_empty_plot(self, signal_file: Path) -> None:
        p = rf.plot(signal_file, "MET", tree="events", bins=10, selection="MET < 0")
        assert p.histograms[0].integral == 0
        assert p.ax.get_ylim()[1] > 0


class TestRatioReference:
    def test_overlaid_data_divides_by_first_non_data_sample(self) -> None:
        # documented: with data and overlaid samples the ratio is data / first non-data sample
        axis = hist.axis.Regular(2, 0, 2, name="x", label="x")

        def make(value: float, label: str, *, is_data: bool = False) -> rf.Histogram:
            h = hist.Hist(axis, storage=hist.storage.Weight())
            h.fill([0.5, 1.5], weight=[value, value])
            return rf.Histogram(h, label=label, is_data=is_data)

        data = make(20.0, "Data", is_data=True)
        p = rf.plot([data, make(10.0, "A"), make(30.0, "B")], panel="ratio")
        assert len(p.comparisons) == 1  # only the data appears in the panel
        # data / A, not data / total
        np.testing.assert_allclose(p.comparisons[0].values, [2.0, 2.0])
        assert p.panel_ax is not None
        assert panel_ylabel(p) == "Data / A"
        stacked = rf.plot([data, make(10.0, "A"), make(30.0, "B")], panel="ratio", stack=True)
        np.testing.assert_allclose(stacked.comparisons[0].values, [0.5, 0.5])  # data / total
        partial_stack = rf.plot(
            [data, make(10.0, "A"), make(30.0, "B")], panel="ratio", stack=["A"]
        )
        np.testing.assert_allclose(partial_stack.comparisons[0].values, [2.0, 2.0])
        without_data = rf.plot([make(10.0, "A"), make(30.0, "B")], panel="ratio", stack=["A"])
        np.testing.assert_allclose(without_data.comparisons[0].values, [3.0, 3.0])
        for side in without_data.comparisons[0].errors:
            np.testing.assert_allclose(side, np.sqrt([18.0, 18.0]))


class TestPanelRoles:
    """Who is compared with what in the lower panel, and what a bad request raises."""

    CONTENTS: ClassVar[dict[str, float]] = {
        "A": 10.0,
        "B": 30.0,
        "C": 5.0,
        "Data": 20.0,
        "Background": 40.0,  # the sum of the others for C: A + B
    }

    def _hists(
        self, *, data: bool, labels: tuple[str, ...] = ("A", "B", "C"), start: float = 0.0
    ) -> list[Any]:
        axis = hist.axis.Regular(2, start, start + 2, name="x", label="x")
        hists = []
        for label in (*labels, "Data") if data else labels:
            h = hist.Hist(axis, storage=hist.storage.Weight())
            h.view().value = self.CONTENTS[label]
            h.view().variance = self.CONTENTS[label]
            hists.append(rf.Histogram(h, label=label, is_data=label == "Data"))
        return hists

    # (stack, data) -> {kind: (numerators, reference)}; None: the roles cannot be filled
    ROLES: ClassVar[list[Any]] = [
        (["A", "B"], True, {"ratio": (["Data"], "Total"), "s/sqrt(b)": (["C"], "Total")}),
        (True, True, {"ratio": (["Data"], "Total"), "s/sqrt(b)": (["C"], "Background")}),
        (["A", "B"], False, {"ratio": (["C"], "Total"), "s/sqrt(b)": (["C"], "Total")}),
        (True, False, {"ratio": None, "s/sqrt(b)": (["C"], "Background")}),
        (False, True, {"ratio": (["Data"], "A"), "s/sqrt(b)": (["C"], "Background")}),
        (False, False, {"ratio": (["B", "C"], "A"), "s/sqrt(b)": (["C"], "Background")}),
    ]

    @pytest.mark.parametrize("kind", ["ratio", "s/sqrt(b)"])
    @pytest.mark.parametrize(("stack", "data", "expected"), ROLES)
    def test_automatic_roles(self, kind: Any, stack: Any, data: bool, expected: Any) -> None:
        before = plt.get_fignums()
        roles = expected[kind]
        if roles is None:
            with pytest.raises(ValueError, match="observed"):
                rf.plot(self._hists(data=data), stack=stack, panel=kind)
            assert plt.get_fignums() == before
            return
        p = rf.plot(self._hists(data=data), stack=stack, panel=kind)
        numerators, reference = roles
        assert [c.label for c in p.comparisons] == numerators
        assert {c.reference for c in p.comparisons} == {reference}
        assert {c.kind for c in p.comparisons} == {kind}
        if reference == "Total":  # the panel compares with the very stack drawn
            assert p.stack is not None
            d = float(p.stack.values()[0])
            stacked = ["A", "B", "C"] if stack is True else stack
            assert d == sum(self.CONTENTS[label] for label in stacked)
        else:
            d = self.CONTENTS[reference]
        for comparison in p.comparisons:
            n = self.CONTENTS[comparison.label]
            np.testing.assert_allclose(
                comparison.values, n / d if kind == "ratio" else n / np.sqrt(d)
            )
        p.close()

    def test_a_named_reference(self) -> None:
        p = rf.plot(self._hists(data=True), panel="ratio", reference="B")
        assert [c.label for c in p.comparisons] == ["A", "C", "Data"]  # data included
        np.testing.assert_allclose(p.comparisons[0].values, 10 / 30)
        assert panel_ylabel(p) == "Ratio to B"  # A / B and C / B are drawn too, not only data
        stacked = rf.plot(self._hists(data=True), stack=True, panel="difference", reference="A")
        assert [c.label for c in stacked.comparisons] == ["B", "C", "Data"]
        np.testing.assert_allclose(stacked.comparisons[0].values, 20.0)
        assert panel_ylabel(stacked) == "Difference to A"
        only_data = rf.plot(self._hists(data=True, labels=("A",)), panel="ratio", reference="A")
        assert [c.label for c in only_data.comparisons] == ["Data"]
        assert panel_ylabel(only_data) == "Data / A"
        only_data.close()
        significance = rf.plot(self._hists(data=True), panel="s/sqrt(s+b)", reference="B")
        assert [c.label for c in significance.comparisons] == ["A", "C"]  # signals: no data
        np.testing.assert_allclose(significance.comparisons[0].values, 10 / np.sqrt(40))
        for plot in (p, stacked, significance):
            plot.close()

    @pytest.mark.parametrize("stack", [False, True])
    def test_observed_data_alone(self, stack: bool) -> None:
        # nothing to stack and no simulation: every further data histogram over the first
        hists = self._hists(data=False, labels=("A", "B"))
        hists = [rf.Histogram(h.hist, label=h.label, is_data=True) for h in hists]
        p = rf.plot(hists, panel="ratio", stack=stack)
        assert [(c.label, c.reference) for c in p.comparisons] == [("B", "A")]
        np.testing.assert_allclose(p.comparisons[0].values, 3.0)
        assert p.comparisons[0].uncertainty == "propagate"  # a data reference is no band
        assert panel_ylabel(p) == "Ratio to A"
        assert p.panel_ax is not None
        assert not [c for c in p.panel_ax.collections if isinstance(c, PolyCollection)]
        with pytest.raises(ValueError, match="at least two non-data"):
            rf.plot(hists, panel="s/sqrt(b)", stack=stack)
        p.close()

    def test_every_kind_through_plot(self) -> None:
        for kind, expected in (
            ("ratio", 0.5),
            ("relative_difference", -0.5),
            ("difference", -20.0),
            ("pull", -20.0 / np.sqrt(60.0)),
        ):
            # data over the stack of A and B, 40: C is overlaid and not compared
            p = rf.plot(
                self._hists(data=True),
                stack=["A", "B"],
                panel=kind,  # type: ignore[arg-type]
            )
            (comparison,) = p.comparisons
            np.testing.assert_allclose(comparison.values, expected)
            p.close()

    @pytest.mark.parametrize(
        ("kind", "over_stack", "mc_over_mc"),
        [
            ("ratio", "Data / MC", "Ratio to A"),
            ("relative_difference", "(Data \N{MINUS SIGN} MC) / MC", "Rel. difference to A"),
            ("difference", "Data \N{MINUS SIGN} MC", "Difference to A"),
            ("pull", "Pull", "Pull"),
        ],
    )
    def test_default_labels(self, kind: Any, over_stack: str, mc_over_mc: str) -> None:
        p = rf.plot(self._hists(data=True), stack=True, panel=kind)
        assert panel_ylabel(p) == over_stack
        simulated = rf.plot(self._hists(data=False), panel=kind)
        assert panel_ylabel(simulated) == mc_over_mc
        custom = rf.plot(self._hists(data=False), panel=kind, panel_label="Custom")
        assert panel_ylabel(custom) == "Custom"
        for plot in (p, simulated, custom):
            plot.close()

    def test_uncertainty_modes(self) -> None:
        p = rf.plot(self._hists(data=True), panel="difference", reference="A")
        by_label = {c.label: c for c in p.comparisons}
        # data over simulation: the reference is the band; simulation: both propagated
        np.testing.assert_allclose(by_label["Data"].errors, np.sqrt(20.0))
        np.testing.assert_allclose(by_label["B"].errors, np.sqrt(40.0))
        assert p.panel_ax is not None
        assert len([c for c in p.panel_ax.collections if isinstance(c, PolyCollection)]) == 1
        both = rf.plot(
            self._hists(data=True),
            panel="difference",
            reference="A",
            panel_uncertainty="propagate",
        )
        np.testing.assert_allclose(both.comparisons[-1].errors, np.sqrt(30.0))
        assert both.panel_ax is not None
        assert not [c for c in both.panel_ax.collections if isinstance(c, PolyCollection)]
        for plot in (p, both):
            plot.close()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"panel": "ratio", "reference": "nope"}, "is not the label of a drawn histogram"),
            ({"panel": "s/sqrt(b)", "reference": "Data"}, "observed data"),
            ({"reference": "A"}, "choose the panel too"),
            ({"panel": True}, r"is not one of .* panel='ratio'"),
            ({"panel": False}, "is not one of"),
            ({"panel": "significance"}, r"is not one of .*panel='s/sqrt\(b\)'"),
            ({"panel": "bogus"}, "is not one of"),
            ({"panel": "pull", "panel_uncertainty": "numerator"}, "does not have"),
            ({"panel": "s/sqrt(b)", "panel_uncertainty": "propagate"}, "does not have"),
            ({"panel": "ratio", "panel_uncertainty": "both"}, "'numerator' or 'poisson-ratio'"),
            ({"panel": "difference", "panel_uncertainty": "poisson-ratio"}, "interval of a ratio"),
        ],
    )
    def test_bad_requests_raise_before_a_figure_exists(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        before = plt.get_fignums()
        with pytest.raises(ValueError, match=match):
            rf.plot(self._hists(data=True), stack=["A"], **kwargs)
        assert plt.get_fignums() == before

    @pytest.mark.parametrize("panel", [None, "ratio"])
    def test_a_stack_that_cannot_be_summed_raises_before_a_figure_exists(self, panel: Any) -> None:
        # the panel compares with the stack total, which must exist: every component counts,
        # not only the first, and the check runs whether or not a panel is drawn
        before = plt.get_fignums()
        coarse = hist.Hist(hist.axis.Regular(1, 0, 2), storage=hist.storage.Weight())
        coarse.view().value, coarse.view().variance = 5.0, 5.0
        hists = [
            *self._hists(data=False, labels=("A",)),
            rf.Histogram(coarse, label="B"),
            *[h for h in self._hists(data=True, labels=()) if h.is_data],
        ]
        with pytest.raises(BinningError, match="a stack needs histograms with identical bin"):
            rf.plot(hists, stack=True, panel=panel)
        # summing also needs the same flow bins, which drawing alone does not
        axis = hist.axis.Regular(2, 0, 2, underflow=False)
        without_flow = hist.Hist(axis, storage=hist.storage.Weight())
        without_flow.view().value, without_flow.view().variance = 1.0, 1.0
        with pytest.raises(BinningError, match="the same flow bins"):
            rf.plot(
                [*self._hists(data=False, labels=("A",)), rf.Histogram(without_flow, label="B")],
                stack=True,
                panel=panel,
            )
        assert plt.get_fignums() == before

    @pytest.mark.parametrize("kwargs", [{}, {"reference": "A"}, {"stack": ["A"]}])
    def test_mismatched_binning_raises_before_a_figure_exists(self, kwargs: Any) -> None:
        before = plt.get_fignums()
        coarse = hist.Hist(hist.axis.Regular(1, 0, 2), storage=hist.storage.Weight())
        coarse.view().value, coarse.view().variance = 5.0, 5.0
        hists = [*self._hists(data=False, labels=("A",)), rf.Histogram(coarse, label="B")]
        with pytest.raises(BinningError, match="do not share one binning"):
            rf.plot(hists, panel="ratio", **kwargs)
        assert plt.get_fignums() == before

    def test_a_reference_is_one_histogram(self) -> None:
        before = plt.get_fignums()
        hists = self._hists(data=False, labels=("A", "B", "A"))
        with pytest.raises(ValueError, match="label of 2 drawn histograms"):
            rf.plot(hists, panel="ratio", reference="A")
        with pytest.raises(ValueError, match="at least two non-data"):
            rf.plot(self._hists(data=True, labels=("A",)), panel="s/sqrt(b)")
        with pytest.raises(ValueError, match="besides the background"):
            rf.plot(self._hists(data=True, labels=("A",)), panel="s/sqrt(b)", reference="A")
        assert plt.get_fignums() == before

    def test_the_panel_follows_a_log_x_axis(self) -> None:
        p = rf.plot(self._hists(data=False, start=1.0), panel="pull", logx=True)
        assert p.panel_ax is not None
        assert p.panel_ax.get_xscale() == "log"
        assert p.panel_ax.get_xlim() == p.ax.get_xlim()
        p.close()


class TestBrokenAxis:
    def test_two_segments(self, signal_file: Path, background_file: Path) -> None:
        p = rf.plot(
            [signal_file, background_file],
            "MET",
            tree="events",
            bins=(50, 0, 250),
            unit="GeV",
            xbreak=(60, 200),
            stats=True,
        )
        assert p.ax_right is not None
        assert p.panel_ax is None
        assert p.axes == (p.ax, p.ax_right)
        assert p.ax.get_xlim() == (0.0, 60.0)
        assert p.ax_right.get_xlim() == (200.0, 250.0)
        assert p.ax.get_ylim() == p.ax_right.get_ylim()
        assert p.ax.get_ylabel() == "Events / 5 GeV"
        assert p.ax_right.get_ylabel() == ""
        assert p.ax.get_xlabel() == ""
        assert p.ax_right.get_xlabel() == "MET [GeV]"
        assert p.ax.get_legend() is None
        assert p.ax_right.get_legend() is not None
        assert not p.ax.spines["right"].get_visible()
        # both segments carry the histograms
        assert len(p.ax.get_legend_handles_labels()[1]) == 2
        assert len(p.ax_right.get_legend_handles_labels()[1]) == 2

    def test_with_ratio_and_xlim(self, signal_file: Path, background_file: Path) -> None:
        p = rf.plot(
            [signal_file, background_file],
            "MET",
            tree="events",
            bins=(50, 0, 250),
            xlim=(10, 240),
            xbreak=(50, 150),
            panel="ratio",
            logy=True,
        )
        assert p.panel_ax is not None
        assert p.panel_ax_right is not None
        assert len(p.axes) == 4
        assert p.panel_ax.get_xlim() == (10.0, 50.0)
        assert p.panel_ax_right.get_xlim() == (150.0, 240.0)
        assert panel_ylabel(p) == "Ratio to signal"
        assert p.panel_ax_right.get_ylabel() == ""
        assert p.panel_ax_right.get_xlabel() == "MET"
        assert p.ax.get_xlabel() == p.ax_right.get_xlabel() == ""  # type: ignore[union-attr]
        assert p.ax.get_yscale() == "log"
        assert len(p.comparisons) == 1

    def test_invalid(self, signal_file: Path) -> None:
        with pytest.raises(ValueError, match="xbreak must satisfy"):
            rf.plot(signal_file, "MET", tree="events", bins=(10, 0, 100), xbreak=(80, 20))
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="existing axes"):
            rf.plot(signal_file, "MET", tree="events", bins=(10, 0, 100), xbreak=(20, 80), ax=ax)
        with pytest.raises(ValueError, match="flow='show'"):
            rf.plot(
                signal_file, "MET", tree="events", bins=(10, 0, 100), xbreak=(20, 80), flow="show"
            )

    def test_save(self, signal_file: Path, tmp_path: Path) -> None:
        p = rf.plot(signal_file, "MET", tree="events", bins=(10, 0, 100), xbreak=(20, 80))
        assert p.save(tmp_path / "broken.png")[0].exists()


class TestPlotHistograms:
    def test_lost_variances_need_variances_from_contents(self) -> None:
        weighted = hist.Hist(hist.axis.Regular(2, 0, 2)).fill([0.5, 1.5], weight=[2.0, -1.0])
        with pytest.raises(ValueError, match="variances_from_contents=True"):
            rf.plot([weighted])
        with pytest.warns(RootfigWarning, match="absolute bin contents as variances"):
            p = rf.plot([weighted], variances_from_contents=True)
        np.testing.assert_allclose(p.histograms[0].variances(), [2.0, 1.0])

    def test_skipped_normalisation_keeps_events_label(self) -> None:
        empty = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        with pytest.warns(RootfigWarning, match="no entries"):
            p = rf.plot([empty], normalize=True)
        assert p.ax.get_ylabel() == "Events"  # the label does not claim a normalisation
        assert p.histograms[0].normalization is None
        cancelling = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        cancelling.fill([0.5, 1.5], weight=[2.0, -2.0])
        with pytest.warns(RootfigWarning, match="sum to zero"):
            p = rf.plot([cancelling], normalize="density")
        assert p.ax.get_ylabel() == "Events"
        np.testing.assert_allclose(p.histograms[0].values(), [2.0, -2.0])

    def test_raw_hists(self) -> None:
        h1 = hist.Hist(
            hist.axis.Regular(5, 0, 5, name="x", label="x"), storage=hist.storage.Weight()
        )
        h1.fill([0.5, 1.5, 2.5])
        h2 = h1 * 2
        p = rf.plot([h1, h2], label=["one", "two"], panel="ratio", normalize=True)
        assert [h.label for h in p.histograms] == ["one", "two"]
        assert p.ax.get_xlabel() == ""
        assert p.panel_ax is not None
        assert p.panel_ax.get_xlabel() == "x"
        assert p.comparisons[0].values[:3].tolist() == pytest.approx([1.0, 1.0, 1.0])
        p = rf.plot([h1], xlabel="custom")
        assert p.ax.get_xlabel() == "custom"
        assert p.histograms[0].label == "x"

    def test_double_normalisation_warns(self) -> None:
        h = rf.Histogram(
            hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight()),
            label="h",
            normalization="Density",
        )
        h.hist.fill([0.5])
        with pytest.warns(RootfigWarning, match="already normalised"):
            rf.plot([h], normalize=True)

    def test_errors(self) -> None:
        h1 = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        with pytest.raises(ValueError, match="no histograms"):
            rf.plot([])
        with pytest.raises(ValueError, match="labels"):
            rf.plot([h1], label=["a", "b"])
        h2 = hist.Hist(hist.axis.Regular(2, 0, 2), hist.axis.Regular(2, 0, 2))
        with pytest.raises(ValueError, match="one-dimensional"):
            rf.plot([h2])


class TestPlot2D:
    @pytest.mark.parametrize(
        ("bins", "expected"),
        [
            (20, (20, 20)),
            ((20, 0, 100), ((20, 0, 100), (20, 0, 100))),
            ([0, 1, 2], ([0, 1, 2], [0, 1, 2])),
            ((20, 30), (20, 30)),  # two integers are two bin counts
            (((20, 0, 100), (30, -3, 3)), ((20, 0, 100), (30, -3, 3))),
            (None, (None, None)),
        ],
    )
    def test_split_bins_forms(self, bins: Any, expected: Any) -> None:
        assert _split_bins(bins) == expected

    def test_split_bins_arrays(self) -> None:
        x, y = np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])
        split_x, split_y = _split_bins((x, y))
        assert split_x is x
        assert split_y is y
        both = _split_bins(x)
        assert both[0] is x
        assert both[1] is x

    @pytest.mark.parametrize("bins", [(0.0, 1.0), (np.float64(0), 1.0), (0, 1.5)])
    def test_split_bins_rejects_range_pair(self, bins: Any) -> None:
        with pytest.raises(BinningError, match=r"\(x_bins, y_bins\)"):
            _split_bins(bins)
        with pytest.raises(BinningError, match="x_bins"):
            rf.plot2d({"x": [0.5], "y": [0.5]}, "x", "y", bins=bins)

    def test_basic(self, signal_file: Path, signal_columns: dict[str, Any]) -> None:
        p = rf.plot2d(
            signal_file,
            "Muon_pt",
            "Muon_eta",
            tree="events",
            selection="Muon_pt > 10",
            bins=(20, 10),
            logz=True,
        )
        assert p.histograms[0].ndim == 2
        assert p.histograms[0].hist.axes[0].size == 20
        assert p.histograms[0].hist.axes[1].size == 10
        pt = signal_columns["Muon_pt"]
        assert p.histograms[0].hist.sum(flow=True).value == pytest.approx(int(ak.sum(pt > 10)))
        assert len(p.fig.axes) == 2

    def test_default_range_survives_a_sentinel(self, signal_file: Path) -> None:
        """Both 2D axes infer their range robustly, so one sentinel column does not
        collapse the plot into a single populated pixel."""
        p = rf.plot2d(signal_file, "sentinel", "MET", tree="events", bins=(20, 20))
        h = p.histograms[0].hist
        assert h.axes[0].edges[0] > -10  # not dragged down to -999
        filled = int((h.values() > 0).sum())
        assert filled > 20

    def test_variables_and_shared_bins(self, signal_file: Path, tmp_path: Path) -> None:
        x = rf.Variable("MET", bins=(10, 0, 100), label="MET", unit="GeV")
        p = rf.plot2d(
            signal_file,
            x,
            rf.Variable("nMuon", bins=(6, -0.5, 5.5)),
            tree="events",
            weight="weight",
            normalize=True,
            title="2d",
            save=str(tmp_path / "h2.png"),
        )
        assert p.histograms[0].values().sum() == pytest.approx(1.0)
        assert p.ax.get_xlabel() == "MET [GeV]"
        assert (tmp_path / "h2.png").exists()
        p = rf.plot2d(
            signal_file,
            "MET",
            "nMuon",
            tree="events",
            bins=(10, 0, 100),
            colorbar=False,
            logx=True,
            logy=True,
        )
        assert p.histograms[0].hist.axes[1].size == 10

    def test_structure_mismatch(self, signal_file: Path) -> None:
        with pytest.raises(SelectionError, match="different structures"):
            rf.plot2d(signal_file, "MET", "Muon_pt", tree="events")

    def test_same_canvas_as_1d(self, signal_file: Path) -> None:
        # The colour bar must not widen the figure (rootfig's factor or mplhep's cbarextend).
        p1 = rf.plot(signal_file, "MET", tree="events")
        p2 = rf.plot2d(signal_file, "MET", "nMuon", tree="events")
        np.testing.assert_allclose(p2.fig.get_size_inches(), p1.fig.get_size_inches())
        p2.fig.canvas.draw()  # the colour bar's space is only taken from the axes when drawn
        assert p2.ax.get_position().width < p1.ax.get_position().width
        # figsize wins over the style's size
        wide = rf.Style(figsize=(9, 3))
        p3 = rf.plot2d(signal_file, "MET", "nMuon", tree="events", style=wide, figsize=(4, 5))
        np.testing.assert_allclose(p3.fig.get_size_inches(), (4, 5))


class TestNotebookDisplay:
    """Contracts that keep ``rf.plot`` displaying in a Jupyter kernel."""

    def test_figure_is_registered_with_pyplot(self, signal_file: Path) -> None:
        # The inline backend flushes pyplot-managed figures after each cell; a figure
        # built with Figure() instead of plt.figure() would never be displayed.
        import matplotlib._pylab_helpers as pylab_helpers

        plot = rf.plot(signal_file, "MET", tree="events")
        try:
            assert plot.fig.number in pylab_helpers.Gcf.figs
        finally:
            plot.close()

    def test_plot_does_not_change_interactive_mode(self, signal_file: Path) -> None:
        import matplotlib

        before = matplotlib.is_interactive()
        rf.plot(signal_file, "MET", tree="events").close()
        assert matplotlib.is_interactive() is before


class TestFontLogging:
    """Drawing and saving must not spam matplotlib's findfont warnings."""

    @pytest.mark.parametrize("experiment", [None, "ATLAS", "CMS", "LHCb"])
    def test_no_findfont_warnings_on_draw_and_save(
        self,
        signal_file: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        experiment: str | None,
    ) -> None:
        import logging

        plot = rf.plot(signal_file, "MET", tree="events", style=rf.Style(experiment=experiment))
        # the warnings are emitted at draw time, i.e. after the style context has ended
        with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
            plot.fig.canvas.draw()
            plot.save(tmp_path / "f.png")
            plot.save(tmp_path / "f.pdf")  # a different findfont caller than Agg
        plot.close()
        missing = [
            record
            for record in caplog.records
            if record.msg == "findfont: Font family %r not found."
        ]
        assert missing == []


class TestFigureShape:
    """Every plot type fits its canvas and is saved at exactly ``figsize`` (no cropping)."""

    @staticmethod
    def _content_within_canvas(fig: Any) -> None:
        fig.canvas.draw()
        width, height = fig.get_size_inches()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        tolerance = 0.01  # inches
        assert box.x0 >= -tolerance
        assert box.y0 >= -tolerance
        assert box.x1 <= width + tolerance
        assert box.y1 <= height + tolerance

    def test_content_fits(self, signal_file: Path, background_file: Path) -> None:
        plots = [
            rf.plot(signal_file, "MET", tree="events", text=["a line", "another"]),
            rf.plot([signal_file, background_file], "MET", tree="events", panel="ratio"),
            rf.plot(
                [signal_file, background_file],
                "MET",
                tree="events",
                panel="ratio",
                xbreak=(40, 60),
                style=rf.Style(experiment="ATLAS", status="Internal", lumi="140 fb^-1"),
            ),
            rf.plot2d(signal_file, "MET", "nMuon", tree="events", zlabel="Events / bin"),
            rf.correlation(signal_file, ["MET", "nMuon", "event"], tree="events", percent=True),
        ]
        for p in plots:
            assert p.fig.get_layout_engine() is not None
            self._content_within_canvas(p.fig)

    def test_saved_size_is_figsize(self, signal_file: Path, tmp_path: Path) -> None:
        from PIL import Image

        for maker, name in [(rf.plot, "one"), (lambda *a, **k: rf.plot2d(*a, "nMuon", **k), "two")]:
            p = maker(signal_file, "MET", tree="events", figsize=(6, 4))
            [path] = p.save(tmp_path / f"{name}.png", dpi=50)
            assert Image.open(path).size == (300, 200)

    def test_experiment_label_anchored_after_layout(
        self, signal_file: Path, background_file: Path
    ) -> None:
        # Above-axes label on a log axis: mplhep's dodge for the (transient) offset text
        # must be undone, and the status word must follow the name at a fixed gap.
        fcc = rf.Style(experiment="FCC-ee", status="Simulation", com="240 GeV")
        p = rf.plot([signal_file, background_file], "MET", tree="events", logy=True, style=fcc)
        self._check_label(p, flush=True)
        # Inside label (ATLAS style), with a ratio panel that changes the layout afterwards.
        atlas = rf.Style(experiment="ATLAS", status="Internal")
        p = rf.plot(
            [signal_file, background_file], "MET", tree="events", panel="ratio", style=atlas
        )
        self._check_label(p, flush=False)

    @staticmethod
    def _check_label(p: rf.Plot, *, flush: bool) -> None:
        import mplhep as hep

        from rootfig.plotting.style import LABEL_WORD_GAP_EM

        [name] = [t for t in p.ax.texts if isinstance(t, hep.label.ExpLabel)]
        [status] = [t for t in p.ax.texts if isinstance(t, hep.label.ExpText)]
        assert status.get_text()
        width, height = p.fig.get_size_inches()
        # The gap is set in points, so it must depend neither on the dpi nor on the axes size.
        for dpi, size in (
            (100, (width, height)),
            (250, (width, height)),
            (100, (1.6 * width, 0.8 * height)),
        ):
            p.fig.set_dpi(dpi)
            p.fig.set_size_inches(*size)
            p.fig.canvas.draw()
            renderer = p.fig.canvas.get_renderer()  # type: ignore[attr-defined]
            exp_box = name.get_window_extent(renderer)
            status_box = status.get_window_extent(renderer)
            if flush:
                axes_left = p.ax.get_window_extent(renderer).x0
                assert exp_box.x0 == pytest.approx(axes_left, abs=1.0)
            em = status.get_fontproperties().get_size_in_points() / 72 * dpi
            assert (status_box.x0 - exp_box.x1) / em == pytest.approx(LABEL_WORD_GAP_EM, abs=0.1)
            # Matplotlib's text layout reports the first line's baseline relative to
            # its anchor. Check the baseline that is painted, including after resizing.
            baselines = [
                text.get_transform().transform(text.get_position())[1]
                + text._get_layout(renderer)[1][0][2][1]
                for text in (name, status)
            ]
            assert baselines[0] == pytest.approx(baselines[1], abs=1.0)

    def test_name_above_the_frame_clears_the_y_offset_text_at_any_size(self) -> None:
        # the name starts after the y axis' offset text (1e7) by a gap in points
        rng = np.random.default_rng(0)
        data = {"x": rng.normal(0, 1, 10_000), "w": np.full(10_000, 1e4)}
        cms = rf.Style(experiment="CMS", status="Preliminary")
        p = rf.plot(data, "x", weight="w", bins=20, style=cms, figsize=(5, 4))
        name = next(t for t in p.ax.texts if isinstance(t, hep.label.ExpLabel))
        [status] = [t for t in p.ax.texts if isinstance(t, hep.label.ExpText)]
        gaps = []
        for width, dpi in [(5.0, 100), (3.0, 100), (8.0, 100), (5.0, 200)]:
            p.fig.set_size_inches(width, 4.0)
            p.fig.set_dpi(dpi)
            p.fig.canvas.draw()
            renderer = p.fig.canvas.get_renderer()
            offset = p.ax.yaxis.get_offset_text()
            assert offset.get_text()
            name_box = name.get_window_extent(renderer)
            gaps.append((name_box.x0 - offset.get_window_extent(renderer).x1) / dpi * 72)
            assert status.get_window_extent(renderer).x0 > name_box.x1  # the status follows
        assert min(gaps) > 0
        # the same at every width; another dpi only changes the text's own metrics
        np.testing.assert_allclose(gaps[:3], gaps[0], atol=0.1)
        plt.close(p.fig)

    @pytest.mark.parametrize("loc", [2, 3])
    def test_explicit_multiline_status_stays_below_the_name(self, loc: int) -> None:
        p = rf.plot(
            {"x": np.linspace(0, 1, 100)},
            "x",
            style=rf.Style(experiment="ATLAS", status="Internal", label_loc=loc),
        )
        p.fig.canvas.draw()
        renderer = p.fig.canvas.get_renderer()
        name = next(t for t in p.ax.texts if isinstance(t, hep.label.ExpLabel))
        status = next(t for t in p.ax.texts if isinstance(t, hep.label.ExpText))
        assert status.get_window_extent(renderer).y1 <= name.get_window_extent(renderer).y0

    def test_fonts_are_pinned(self, signal_file: Path, background_file: Path) -> None:
        # Text must render in the style's font after the style context has ended, or the
        # layout computed inside it no longer matches what is painted (clipped labels).
        atlas = rf.Style(experiment="ATLAS", status="Internal")
        p = rf.plot(
            [signal_file, background_file], "MET", tree="events", panel="ratio", style=atlas
        )
        p.fig.canvas.draw()
        assert "sans-serif" not in p.ax.yaxis.label.get_fontfamily()
        assert p.ax.yaxis.label.get_fontname() != "DejaVu Sans"
        tick = (
            p.ax.get_xticklabels()[0] if p.ax.get_xticklabels() else p.ax.yaxis.get_ticklabels()[0]
        )
        assert tick.get_fontname() == p.ax.yaxis.label.get_fontname()
        legend = p.ax.get_legend()
        assert legend is not None
        assert legend.get_texts()[0].get_fontname() == p.ax.yaxis.label.get_fontname()

    @pytest.mark.parametrize("experiment", EXPERIMENT_STYLES)
    def test_experiment_labels_fit_in_every_style(
        self, signal_file: Path, background_file: Path, experiment: str
    ) -> None:
        # A full label on the plots that squeeze it: a colour bar narrows the axes, a matrix
        # leaves no room inside the frame, a broken axis splits it.
        style = rf.Style(experiment=experiment, status="Preliminary", lumi=138, com=13.6)
        samples = [signal_file, background_file]
        plots = [
            rf.plot2d(signal_file, "MET", "nMuon", tree="events", style=style),
            rf.correlation(signal_file, ["MET", "nMuon", "event"], tree="events", style=style),
            rf.plot(samples, "MET", tree="events", panel="ratio", xbreak=(40, 60), style=style),
        ]
        for p in plots:
            p.fig.canvas.draw()
            renderer = p.fig.canvas.get_renderer()  # type: ignore[attr-defined]
            texts = self._label_texts(p)
            assert texts
            boxes = [t.get_window_extent(renderer) for t in texts]
            canvas = p.fig.bbox.padded(1)
            for box in boxes:
                assert canvas.contains(box.x0, box.y0), experiment
                assert canvas.contains(box.x1, box.y1), experiment
            for index, box in enumerate(boxes):
                for other in boxes[index + 1 :]:
                    shared = min(box.y1, other.y1) - max(box.y0, other.y0)
                    if shared > 0.5 * min(box.height, other.height):  # one line: side by side
                        assert box.x1 <= other.x0 or other.x1 <= box.x0, experiment
        name = next(t for t in plots[0].ax.texts if isinstance(t, hep.label.ExpLabel))
        frame = plots[0].ax.get_window_extent(renderer)
        assert name.get_window_extent(renderer).y0 >= frame.y1  # 2D: above the bins

    @staticmethod
    def _label_texts(p: rf.Plot) -> list[Any]:
        axes = [p.ax] if p.ax_right is None else [p.ax, p.ax_right]
        kinds = (hep.label.ExpLabel, hep.label.ExpText, hep.label.LumiText)
        return [t for a in axes for t in a.texts if isinstance(t, kinds) and t.get_text()]

    def test_label_line_above_a_narrow_frame_shrinks(self, signal_file: Path) -> None:
        cms = rf.Style(experiment="CMS", status="Preliminary", lumi=138, com=13.6)
        wide = rf.plot(signal_file, "MET", tree="events", style=cms)
        narrow = rf.plot2d(signal_file, "MET", "nMuon", tree="events", style=cms)
        [wide_name] = [t for t in wide.ax.texts if isinstance(t, hep.label.ExpLabel)]
        [narrow_name] = [t for t in narrow.ax.texts if isinstance(t, hep.label.ExpLabel)]
        assert narrow_name.get_fontsize() < wide_name.get_fontsize()
        # an explicit position inside the frame is kept for 2D plots too
        inside = rf.plot2d(
            signal_file, "MET", "nMuon", tree="events", style=cms.replace(label_loc=1)
        )
        inside.fig.canvas.draw()
        renderer = inside.fig.canvas.get_renderer()  # type: ignore[attr-defined]
        [name] = [t for t in inside.ax.texts if isinstance(t, hep.label.ExpLabel)]
        assert name.get_window_extent(renderer).y1 <= inside.ax.get_window_extent(renderer).y1

    def test_broken_axis_luminosity_ends_the_right_segment(
        self, signal_file: Path, background_file: Path
    ) -> None:
        samples = [signal_file, background_file]
        for experiment in ("CMS", "LHCb"):  # label above the frame, and inside it
            style = rf.Style(experiment=experiment, status="Preliminary", lumi=9, com=13.6)
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)  # "constrained_layout not applied"
                p = rf.plot(samples, "MET", tree="events", xbreak=(40, 60), style=style)
                p.fig.canvas.draw()
            assert p.ax_right is not None
            [lumi] = [t for t in p.ax_right.texts if isinstance(t, hep.label.LumiText)]
            renderer = p.fig.canvas.get_renderer()  # type: ignore[attr-defined]
            right_end = p.ax_right.get_window_extent(renderer).x1
            assert lumi.get_window_extent(renderer).x1 == pytest.approx(right_end, abs=2)

    def test_correlation_title_yields_to_an_experiment_label(self, signal_file: Path) -> None:
        variables = ["MET", "nMuon"]
        plain = rf.correlation(signal_file, variables, tree="events")
        assert plain.ax.get_title().endswith(": correlation")
        cms = rf.correlation(signal_file, variables, tree="events", style="CMS")
        assert cms.ax.get_title() == ""
        assert any(isinstance(t, hep.label.ExpLabel) for t in cms.ax.texts)
        titled = rf.correlation(signal_file, variables, tree="events", style="CMS", title="Mine")
        assert titled.ax.get_title() == "Mine"

    @staticmethod
    def _shown_boxes(p: rf.Plot) -> dict[str, Any]:
        """Boxes of the label texts and the title in the figure as saved."""
        p.fig.savefig(io.BytesIO(), format="png")
        p.fig.canvas.draw()
        renderer = p.fig.canvas.get_renderer()  # type: ignore[attr-defined]
        texts = {f"{type(t).__name__}{i}": t for i, t in enumerate(TestFigureShape._label_texts(p))}
        if p.ax.get_title():
            texts["title"] = p.ax.title
        return {name: text.get_window_extent(renderer) for name, text in texts.items()}

    @staticmethod
    def _assert_apart_on_canvas(p: rf.Plot, boxes: dict[str, Any], *, pairs: str = "all") -> None:
        canvas = p.fig.bbox.padded(1)
        for name, box in boxes.items():
            assert canvas.contains(box.x0, box.y0), name
            assert canvas.contains(box.x1, box.y1), name
        names = list(boxes)
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                if pairs == "title" and "title" not in (name, other):
                    continue
                assert not boxes[name].padded(-1).overlaps(boxes[other].padded(-1)), (name, other)

    @pytest.mark.parametrize("title", ["Mine", "A longer explicit title"])
    @pytest.mark.parametrize(
        "style",
        [
            rf.Style(experiment="CMS", status="Preliminary", lumi=138, com=13.6),
            rf.Style(experiment="ATLAS", status="Internal", lumi=140, com=13.6),
            rf.Style(experiment="ATLAS", status="Internal", lumi=140, com=13.6, label_loc=4),
        ],
        ids=["CMS", "ATLAS", "ATLAS-inside"],
    )
    def test_titles_clear_experiment_labels(
        self, signal_columns: dict[str, Any], style: rf.Style, title: str
    ) -> None:
        # 2D plots and matrices put the label above the frame, where the title goes too
        plots = [
            rf.plot2d(signal_columns, "MET", "nMuon", style=style, title=title),
            rf.correlation(signal_columns, ["MET", "nMuon", "event"], style=style, title=title),
        ]
        with style_context(style):
            title_size = FontProperties(size=plt.rcParams["axes.titlesize"]).get_size_in_points()
        for p in plots:
            assert p.ax.get_title() == title
            assert p.ax.title.get_fontsize() == pytest.approx(title_size)  # lifted, not restyled
            # mplhep stacks the lines of a label inside the frame its own way: check the title
            pairs = "title" if style.label_loc is not None else "all"
            self._assert_apart_on_canvas(p, self._shown_boxes(p), pairs=pairs)

    @pytest.mark.parametrize("width", [4, 6, 8])
    def test_label_fits_narrow_frames_and_aligning_again_keeps_it(
        self, signal_columns: dict[str, Any], width: float
    ) -> None:
        cms = rf.Style(experiment="CMS", status="Preliminary", lumi=138, com=13.6)
        with style_context(cms):  # the sizes mplhep gives the label where it has room
            fig, ax = plt.subplots(figsize=(20, 6))
            add_experiment_label(ax, cms, has_data=False)
            full = {type(t): t.get_fontsize() for t in ax.texts}
            plt.close(fig)
        p = rf.plot2d(signal_columns, "MET", "nMuon", style=cms, figsize=(width, 4))
        boxes = self._shown_boxes(p)
        self._assert_apart_on_canvas(p, boxes)
        texts = self._label_texts(p)
        sizes = [text.get_fontsize() for text in texts]
        for text, size in zip(texts, sizes, strict=True):
            assert size >= LABEL_MIN_SCALE * full[type(text)] - 0.01, type(text).__name__
        align_experiment_label(p.ax)  # as a second finishing pass does
        assert [text.get_fontsize() for text in texts] == pytest.approx(sizes, rel=0.01)
        again = self._shown_boxes(p)
        for name, box in boxes.items():
            np.testing.assert_allclose(again[name].extents, box.extents, atol=2, err_msg=name)

    @pytest.mark.parametrize("location", ["left", "center", "right"])
    def test_title_location_and_padding_survive_outer_rcparams(
        self, signal_columns: dict[str, Any], location: str
    ) -> None:
        cms = rf.Style(
            experiment="CMS",
            status="Preliminary",
            lumi=138,
            com=13.6,
            rc={
                "axes.titlelocation": location,
                "axes.titlepad": 18,
                "axes.titlesize": 24,
                "axes.titley": 1.0,
                "axes.titlecolor": "purple",
                "axes.titleweight": "bold",
            },
        )
        with plt.rc_context({"axes.titlelocation": "left", "axes.titlepad": 2, "axes.titley": 1.5}):
            p = rf.correlation(signal_columns, ["MET", "nMuon", "event"], style=cms, title="Mine")
            for _ in range(2):
                p.fig.savefig(io.BytesIO(), format="png")
                p.fig.canvas.draw()
                renderer = p.fig.canvas.get_renderer()
                titles = [p.ax._left_title, p.ax.title, p.ax._right_title]
                [title] = [t for t in titles if t.get_text()]
                assert p.ax.get_title(loc=location) == "Mine"
                assert title.get_fontsize() == 24
                assert title.get_position()[1] == 1.0
                assert title.get_color() == "purple"
                assert title.get_fontweight() == "bold"
                label_top = max(t.get_window_extent(renderer).y1 for t in self._label_texts(p))
                assert title.get_window_extent(renderer).y0 >= label_top + 18 / 72 * p.fig.dpi - 1
                align_experiment_label(p.ax)

    def test_user_axes_are_cropped_tight(self, signal_file: Path, tmp_path: Path) -> None:
        # A figure the user made has no layout engine; keep the tight bounding box there.
        _, ax = plt.subplots(figsize=(6, 4))
        p = rf.plot(signal_file, "MET", tree="events", ax=ax)
        [path] = p.save(tmp_path / "user.png", dpi=50)
        from PIL import Image

        assert Image.open(path).size != (300, 200)


class TestSummaryAndCorrelation:
    def test_summarize_reads_each_sample_once(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rootfig.histograms import load_columns
        from rootfig.io import FileSource

        calls: list[list[str]] = []
        original = FileSource.iterate

        def counting(self: FileSource, branches: Any, *args: Any) -> Any:
            calls.append(list(branches))
            return original(self, branches, *args)

        monkeypatch.setattr(FileSource, "iterate", counting)
        variables = ["MET", "Muon_pt", "nMuon * 2"]
        table = rf.summarize(signal_file, variables, tree="events", selection="nMuon > 0")
        assert len(calls) == 1  # one read for all variables, the selection and the weight
        assert set(calls[0]) == {"MET", "Muon_pt", "nMuon"}
        # identical to reading each variable on its own
        sample = rf.Sample(signal_file, tree="events")
        for var in variables:
            single = load_columns(sample, [var], selection="nMuon > 0")
            row = table.get(var)
            assert row.entries == single.n_entries
            assert row.n_selected_events == single.n_selected_events
            assert row.mean == pytest.approx(float(np.mean(single.values)))

    def test_summarize(
        self, signal_file: Path, background_file: Path, signal_columns: dict[str, Any]
    ) -> None:
        table = rf.summarize(
            [signal_file, background_file], ["MET", "Muon_pt"], tree="events", selection="nMuon > 0"
        )
        assert table.samples == ["signal", "background"]
        assert table.variables == ["MET", "Muon_pt"]
        s = table.get("MET", "signal")
        mask = signal_columns["nMuon"] > 0
        assert s.mean == pytest.approx(signal_columns["MET"][mask].mean())
        assert s.entries == int(mask.sum())
        text = str(table)
        assert "signal: MET" in text
        assert "entries" in text
        with pytest.raises(KeyError, match="several"):
            table.get("MET")
        with pytest.raises(KeyError, match="no summary"):
            table.get("nope")

    def test_summarize_single(self, signal_file: Path) -> None:
        table = rf.summarize(signal_file, "MET", tree="events", weight="weight", label="S")
        assert table.samples == ["S"]
        assert "S:" not in str(table)
        assert table.get("MET").effective_entries < 2000

    def test_correlation(self, signal_file: Path, signal_columns: dict[str, Any]) -> None:
        p = rf.correlation(
            signal_file,
            ["MET", "nMuon", "weight"],
            tree="events",
            selection="nMuon > 0",
            labels=["a", "b", "c"],
            percent=True,
        )
        assert p.matrix is not None
        assert p.matrix.shape == (3, 3)
        np.testing.assert_allclose(np.diag(p.matrix), 1.0)
        mask = signal_columns["nMuon"] > 0
        expected = np.corrcoef(signal_columns["MET"][mask], signal_columns["nMuon"][mask])[0, 1]
        assert p.matrix[0, 1] == pytest.approx(expected)
        assert [t.get_text() for t in p.ax.get_xticklabels()] == ["a", "b", "c"]
        assert p.ax.get_title() == "signal: correlation"

    def test_correlation_errors(self, signal_file: Path) -> None:
        with pytest.raises(SelectionError, match="two variables"):
            rf.correlation(signal_file, ["MET"], tree="events")
        with pytest.raises(SelectionError, match="different structures"):
            rf.correlation(signal_file, ["MET", "Muon_pt"], tree="events")


class TestPublicSurface:
    def test_version_and_exports(self) -> None:
        assert rf.__version__
        for name in rf.__all__:
            assert hasattr(rf, name), name
            # rf.histograms is the function: a subpackage of the same name must not shadow it
            assert not isinstance(getattr(rf, name), ModuleType), name

    def test_documented_lower_layer_exports(self) -> None:
        reference = (Path(__file__).resolve().parents[1] / "docs" / "api.md").read_text()
        lower_layers = reference.split("## Lower layers\n", 1)[1].split("\n## ", 1)[0]
        entries = []
        for entry in lower_layers.split("\n::: ")[1:]:
            path, _, options = entry.partition("\n")
            entries.append((path, options))
        assert [path for path, _ in entries] == [
            "rootfig.io",
            "rootfig.expressions",
            "rootfig.selection",
            "rootfig.histograms",
            "rootfig.histograms.normalize.normalize",
            "rootfig.plotting",
        ]
        for path, options in entries:
            if "      members:\n" in options:
                module_name = path
                members = re.findall(r"^        - (\w+)$", options, re.MULTILINE)
                assert members, f"{path}: list the documented members"
            else:
                package, _, name = path.rpartition(".")
                parent = importlib.import_module(package)
                assert not isinstance(getattr(parent, name, None), ModuleType), (
                    f"{path} documents a module: give it a members list of its own"
                )
                # A defining-function path still documents its package's re-export.
                module_name = parent.__package__ or package
                members = [name]
            module = importlib.import_module(module_name)
            for member in members:
                assert hasattr(module, member), f"{module_name}.{member}"
                assert member in module.__all__, f"{module_name}.{member}"

    def test_evaluate_reexport(self) -> None:
        assert rf.evaluate("a + 1", {"a": ak.Array([1, 2])}).tolist() == [2, 3]

    def test_comparison_names(self) -> None:
        # rf.histograms is the function; the subpackages are imported by name
        histograms = importlib.import_module("rootfig.histograms")
        plotting = importlib.import_module("rootfig.plotting")

        assert {"Comparison", "compare"} <= set(rf.__all__)
        assert not {"Ratio", "ratio", "significance"} & set(rf.__all__)
        assert {"COMPARISON_KINDS", "Comparison", "ComparisonKind", "UncertaintyMode"} <= set(
            histograms.__all__
        )
        assert histograms.COMPARISON_KINDS == (
            "ratio",
            "difference",
            "relative_difference",
            "pull",
            "asymmetry",
            "s/sqrt(b)",
            "s/sqrt(s+b)",
        )
        removed = {"SIGNIFICANCE_KINDS", "Ratio", "RatioUncertainty", "SignificanceKind", "ratio"}
        assert not removed & set(histograms.__all__)
        assert {"draw_panel", "panel_ylim"} <= set(plotting.__all__)
        assert not {"draw_ratio_panel", "draw_significance_panel", "ratio_ylim"} & set(
            plotting.__all__
        )

    def test_compare_reexport(self) -> None:
        h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        h.fill([0.5, 0.5, 1.5])
        r = rf.compare(h, h, kind="ratio")
        assert isinstance(r, rf.Comparison)
        assert r.values.tolist() == [1.0, 1.0]

    def test_use_style_reexport(self) -> None:
        before = dict(plt.rcParams)
        try:
            assert rf.use_style().legend is True
        finally:
            plt.rcParams.update(before)

    def test_warning_free_default_plot(self, signal_file: Path) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RootfigWarning)
            rf.plot(signal_file, "MET", tree="events", bins=(10, 0, 100))


class TestLuminosityAndYields:
    def test_plot_scales_to_luminosity(self, signal_file: Path, background_file: Path) -> None:
        sig = rf.Sample(signal_file, tree="events", label="S", xsec="0.2 pb", ngen=2000)
        bkg = rf.Sample(background_file, tree="events", label="B", xsec=16.0)  # ngen: 2000 entries
        p = rf.plot([bkg, sig], "MET", bins=(10, 0, 1000), lumi="1 ab^-1", stack=True)
        yields = {h.label: h.hist.sum(flow=True).value for h in p.histograms}
        assert yields["S"] == pytest.approx(0.2e-12 * 1e18)  # xsec x lumi in events
        assert yields["B"] == pytest.approx(16e-12 * 1e18)
        assert any("ab^{-1}" in t.get_text() for t in p.ax.texts) or any(
            "ab^{-1}" in a.txt.get_text() for a in p.ax.artists if hasattr(a, "txt")
        )
        # the style's own luminosity wins over the scaling one
        p = rf.plot([sig], "MET", bins=(10, 0, 1000), lumi=1.0, style=rf.Style(lumi="2 fb"))
        assert any("2 $" in a.txt.get_text() for a in p.ax.artists if hasattr(a, "txt"))
        with pytest.raises(rf.LuminosityError):
            rf.plot(sig, "MET", bins=10)
        table = rf.summarize(sig, "MET", lumi=1.0)
        assert table.get("MET").sum_weights == pytest.approx(200.0)
        assert rf.histogram(sig, "MET", bins=(1, 0, 1e4), lumi=1.0).sum().value == pytest.approx(
            200.0
        )
        assert rf.plot2d(sig, "MET", "nMuon", bins=(2, 0, 1e4), lumi=1.0).histograms[
            0
        ].hist.sum().value == pytest.approx(200.0)
        rf.correlation(sig, ["MET", "nMuon"], lumi=1.0)

    def test_cutflow(self, signal_file: Path, signal_columns: dict[str, Any]) -> None:
        table = rf.cutflow(
            [signal_file],
            ["nMuon >= 1", rf.Cut("MET > 50", label="MET"), "Muon_pt > 100"],
            tree="events",
        )
        flow = table.rows[0]
        n, met = signal_columns["nMuon"], signal_columns["MET"]
        assert flow.events[0] == 2000
        assert flow.events[1] == int((n >= 1).sum())
        assert flow.events[2] == int(((n >= 1) & (met > 50)).sum())
        assert flow.labels == ["All", "nMuon >= 1", "MET", "Muon_pt > 100"]
        assert "MET" in str(table)
        weighted = rf.cutflow(
            signal_file, ["nMuon >= 1"], tree="events", weight="weight", lumi=None
        )
        assert weighted.rows[0].yields[0] == pytest.approx(float(signal_columns["weight"].sum()))


class TestEfficiencyProfileSignificance:
    def test_efficiency(self, signal_file: Path, background_file: Path) -> None:
        p = rf.efficiency(
            [signal_file, background_file],
            "Muon_pt",
            passed="Muon_isTight",
            tree="events",
            bins=(5, 0, 200),
            unit="GeV",
            label=["S", "B"],
        )
        assert len(p.efficiencies) == 2
        eff = p.efficiencies[0]
        assert eff.label == "S"
        finite = eff.values[np.isfinite(eff.values)]
        assert np.all((finite >= 0) & (finite <= 1))
        assert np.all(eff.lower[np.isfinite(eff.lower)] <= finite)
        assert p.ax.get_ylabel() == "Efficiency"
        assert p.ax.get_xlabel() == "Muon_pt [GeV]"
        assert p.ax.get_ylim()[1] >= 1.0
        assert len(p.histograms) == 2  # the numerators
        # per-event denominator selection combines with the pass criterion
        p2 = rf.efficiency(
            signal_file,
            "MET",
            passed="nMuon >= 2",
            selection="MET > 20",
            tree="events",
            bins=(4, 0, 200),
            ylim=(0, 1),
            ax=plt.subplots()[1],
            legend=False,
        )
        assert p2.ax.get_ylim() == (0.0, 1.0)
        assert p2.ax.get_legend() is None
        with pytest.raises(ValueError, match=r"empty|passed"):
            rf.efficiency(signal_file, "MET", passed="", tree="events")

    def test_profile(self, signal_file: Path, signal_columns: dict[str, Any]) -> None:
        p = rf.profile(signal_file, "nMuon", "MET", tree="events", bins=(3, -0.5, 2.5))
        prof = p.profiles[0]
        met, n = signal_columns["MET"], signal_columns["nMuon"]
        assert prof.values[1] == pytest.approx(met[n == 1].mean())
        assert prof.counts[1] == pytest.approx((n == 1).sum())
        assert p.ax.get_ylabel() == "MET"
        p = rf.profile(
            signal_file,
            "nMuon",
            rf.Variable("MET", label="E", unit="GeV"),
            tree="events",
            statistic="std",
            bins=(3, -0.5, 2.5),
            label="sig",
            logy=True,
            xlim=(0, 2),
        )
        assert p.profiles[0].values[1] == pytest.approx(met[n == 1].std())
        assert p.ax.get_ylabel() == "Std. dev. of E [GeV]"
        assert p.ax.get_xlim() == (0.0, 2.0)

    @pytest.mark.parametrize("scale", [1.0, 0.03, 2.5, -2.5, 0.0])
    def test_efficiency_leaves_the_sample_scale_out(self, scale: float) -> None:
        # a sample scale (or cross section and luminosity) cancels in an efficiency, also
        # when it is zero or negative: the counts keep Clopper-Pearson, as TEfficiency
        # filled without that factor, and weighted entries their normal interval
        x = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
        ok = np.array([1, 1, 1, 0, 1, 0, 1])
        w = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 3.0, 2.0])
        counts = rf.Sample({"x": x, "ok": ok}, scale=scale)
        weighted = rf.Sample({"x": x, "ok": ok, "w": w}, scale=scale, weight="w")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RootfigWarning)
            p = rf.efficiency([counts, weighted], "x", passed="ok == 1", bins=(2, 0, 2))
        plain, heavy = p.efficiencies
        np.testing.assert_allclose(
            [plain.lower[0], plain.upper[0]], [0.38159757449607973, 0.9577308936963108]
        )
        np.testing.assert_allclose(plain.values, [0.75, 2 / 3])
        np.testing.assert_allclose(heavy.values, [4.0 / 4.5, 3.0 / 6.0])
        lower, upper = normal_interval([4.0, 3.0], [4.5, 6.0], [6.0, 5.0], [6.25, 14.0])
        np.testing.assert_allclose(heavy.lower, lower)
        np.testing.assert_allclose(heavy.upper, upper)
        # the numerators in Plot.histograms are yields: they keep the factor
        np.testing.assert_allclose(p.histograms[0].values(), scale * np.array([3.0, 2.0]))
        np.testing.assert_allclose(p.histograms[1].values(), scale * np.array([4.0, 3.0]))
        p.close()

    def test_efficiency_panel(self, signal_file: Path, background_file: Path) -> None:
        mc = rf.Sample(signal_file, tree="events", label="MC")
        data = rf.Sample(background_file, tree="events", label="Data", is_data=True)
        options: dict[str, Any] = {"passed": "Muon_isTight", "bins": (5, 0, 200), "unit": "GeV"}
        p = rf.efficiency([data, mc], "Muon_pt", panel="ratio", **options)
        assert p.panel_ax is not None
        (scale_factor,) = p.comparisons  # data over simulation, whatever the order
        assert (scale_factor.label, scale_factor.reference) == ("Data", "MC")
        assert panel_ylabel(p) == "Data / MC"
        eff_data, eff_mc = p.efficiencies
        np.testing.assert_allclose(scale_factor.values, eff_data.values / eff_mc.values)
        assert isinstance(scale_factor.errors, tuple)
        assert p.panel_ax.get_xlabel() == "Muon_pt [GeV]"
        assert p.ax.get_xlabel() == ""
        named = rf.efficiency(
            [data, mc], "Muon_pt", panel="difference", reference="Data", panel_label="D", **options
        )
        assert (named.comparisons[0].label, named.comparisons[0].reference) == ("MC", "Data")
        assert panel_ylabel(named) == "D"
        fig, (main, lower) = plt.subplots(2)
        into = rf.efficiency([mc, data], "Muon_pt", panel="pull", ax=(main, lower), **options)
        assert (into.ax, into.panel_ax) == (main, lower)
        assert lower.get_ylabel() == "Pull"
        options["bins"] = (5, 1, 200)
        logx = rf.efficiency([mc, data], "Muon_pt", panel="ratio", logx=True, **options)
        assert logx.panel_ax is not None
        assert logx.panel_ax.get_xscale() == "log"
        assert logx.panel_ax.get_xlim() == pytest.approx((1.0, 200.0))

    def test_profile_panel(self, signal_file: Path, background_file: Path) -> None:
        a = rf.Sample(signal_file, tree="events", label="A")
        b = rf.Sample(background_file, tree="events", label="B")
        p = rf.profile(
            [a, b], "nMuon", "MET", bins=(3, -0.5, 2.5), panel="difference", panel_ylim=(-9, 9)
        )
        assert p.panel_ax is not None
        assert p.panel_ax.get_ylim() == (-9.0, 9.0)
        (difference,) = p.comparisons
        assert (difference.label, difference.reference) == ("B", "A")
        assert panel_ylabel(p) == "Difference to A"
        first, second = p.profiles
        np.testing.assert_allclose(difference.values, second.values - first.values)
        assert rf.profile([a, b], "nMuon", "MET", bins=(3, -0.5, 2.5)).panel_ax is None

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"panel": "s/sqrt(b)"}, "count events"),
            ({"panel": "bogus"}, "is not one of"),
            ({"panel": "ratio", "reference": "X"}, "is not the label"),
            ({"reference": "A"}, "choose the panel too"),
        ],
    )
    def test_efficiency_and_profile_panels_refuse_before_drawing(
        self, signal_file: Path, background_file: Path, options: dict[str, Any], match: str
    ) -> None:
        samples = [
            rf.Sample(signal_file, tree="events", label="A"),
            rf.Sample(background_file, tree="events", label="B"),
        ]
        before = plt.get_fignums()
        with pytest.raises(ValueError, match=match):
            rf.efficiency(samples, "MET", passed="nMuon >= 1", bins=(4, 0, 200), **options)
        with pytest.raises(ValueError, match=match):
            rf.profile(samples, "nMuon", "MET", bins=(3, -0.5, 2.5), **options)
        with pytest.raises(ValueError, match="at least two"):
            rf.profile(samples[:1], "nMuon", "MET", bins=(3, -0.5, 2.5), panel="ratio")
        assert plt.get_fignums() == before

    def test_significance_panel(self, signal_file: Path, background_file: Path) -> None:
        sig = rf.Sample(signal_file, tree="events", label="S", scale=0.1)
        bkg = rf.Sample(background_file, tree="events", label="B")
        p = rf.plot([bkg, sig], "MET", bins=(10, 0, 200), stack=True, panel="s/sqrt(b)")
        assert p.panel_ax is not None
        assert panel_ylabel(p) == r"$S/\sqrt{B}$"
        assert len(p.comparisons) == 1
        result = p.comparisons[0]
        s = p.histograms[1].values()
        b = p.histograms[0].values()
        ok = b > 0
        np.testing.assert_allclose(result.values[ok], s[ok] / np.sqrt(b[ok]))
        p = rf.plot(
            [sig, bkg],
            "MET",
            bins=(10, 0, 200),
            panel="s/sqrt(s+b)",
            reference="B",
            panel_label="Z",
        )
        assert p.panel_ax is not None
        assert panel_ylabel(p) == "Z"
        (named,) = p.comparisons
        assert (named.kind, named.label, named.reference) == ("s/sqrt(s+b)", "S", "B")
        s, b = p.histograms[0].values(), p.histograms[1].values()
        np.testing.assert_allclose(named.values[ok], s[ok] / np.sqrt(s[ok] + b[ok]))
        # the signal inside the stack or drawn over it: the same panel
        overlaid = rf.plot([bkg, sig], "MET", bins=(10, 0, 200), stack="B", panel="s/sqrt(b)")
        np.testing.assert_allclose(overlaid.comparisons[0].values, result.values)
        np.testing.assert_allclose(overlaid.comparisons[0].errors, result.errors)
        with pytest.raises(ValueError, match="at least two"):
            rf.plot([sig], "MET", bins=(10, 0, 200), panel="s/sqrt(b)")
        with pytest.raises(ValueError, match="is not the label of a drawn histogram"):
            rf.plot([sig, bkg], "MET", bins=(10, 0, 200), panel="s/sqrt(b)", reference="X")
        with pytest.raises(ValueError, match="is not one of"):
            rf.plot([sig, bkg], "MET", bins=(10, 0, 200), panel="bogus")  # type: ignore[arg-type]

    def test_split_collection_end_to_end(self) -> None:
        path = Path(__file__).parent / "data" / "split_collection.root"
        p = rf.plot(
            path,
            "pt(ReconstructedParticles.momentum.x, ReconstructedParticles.momentum.y)",
            tree="events",
            selection="ReconstructedParticles.charge > 0",
            bins=(10, 0, 100),
            xlabel=r"$p_T$",
        )
        assert p.histograms[0].entries > 0
        arrays = rf.load(
            path,
            {"e": "ReconstructedParticles.energy", "n": "count(ReconstructedParticles.energy)"},
            tree="events",
        )
        assert arrays.fields == ["e", "n"]
        assert len(arrays) == 200


class TestReviewRegressions:
    """End-to-end regression checks for previously fixed issues."""

    @staticmethod
    def _hist(values: list[float], weights: list[float] | None = None) -> Any:
        return hist.Hist(hist.axis.Variable([0, 1, 2]), storage=hist.storage.Weight()).fill(
            values, weight=weights
        )

    def test_load_rejects_numeric_selection(self) -> None:
        with pytest.raises(SelectionError, match="must be boolean"):
            rf.load({"x": np.array([10, 20, 30]), "cut": np.array([0, 1, 0])}, "x", selection="cut")
        loaded = rf.load(
            {"x": np.array([10, 20, 30]), "cut": np.array([0, 1, 0])}, "x", selection="cut != 0"
        )
        assert loaded.x.tolist() == [20]
        with pytest.raises(SelectionError, match="per-object"):
            rf.load({"x": ak.Array([[1.0], [2.0]])}, "x", selection="x > 1")
        assert rf.load({"x": np.arange(5.0)}, "x", entry_start=1, entry_stop=3).x.tolist() == [
            1.0,
            2.0,
        ]

    def test_fixed_size_branches_end_to_end(self, tmp_path: Path) -> None:
        import uproot

        path = tmp_path / "fixed.root"
        with uproot.recreate(path) as file:
            tree = file.mktree("events", {"x": np.dtype(("float64", (3,))), "w": "float64"})
            tree.extend({"x": np.arange(6.0).reshape(2, 3), "w": np.array([1.0, 2.0])})
        h = rf.histogram(path, "x", tree="events", weight="w", bins=(6, 0, 6))
        np.testing.assert_allclose(h.values(), [1, 1, 1, 2, 2, 2])
        h = rf.histogram(path, "x", tree="events", selection="x >= 4", bins=(6, 0, 6))
        np.testing.assert_allclose(h.values(), [0, 0, 0, 0, 1, 1])
        h = rf.histogram(path, "max(x)", tree="events", selection="w > 1", bins=(6, 0, 6))
        np.testing.assert_allclose(h.values(), [0, 0, 0, 0, 0, 1])

    def test_signed_weights_fill_and_draw(self) -> None:
        h = rf.histogram(
            {"x": np.array([0.0, 1.0]), "w": np.array([1.0, -0.5])}, "x", weight="w", bins=(2, 0, 2)
        )
        np.testing.assert_allclose(h.values(), [1.0, -0.5])
        p = rf.plot(
            {"x": np.array([0.0, 1.0]), "w": np.array([1.0, -0.5])},
            "x",
            weight="w",
            bins=(2, 0, 2),
            stats=True,
        )
        assert p.histograms[0].stats is not None
        assert np.isnan(p.histograms[0].stats.std)

    @pytest.mark.parametrize(("n", "passed"), [(6, "x > 0"), (9, "x < 0"), (1, "x > 0")])
    def test_efficiency_endpoints_draw(self, n: int, passed: str) -> None:
        p = rf.efficiency({"x": np.full(n, 0.5)}, "x", passed=passed, bins=(1, 0, 1))
        low, high = p.efficiencies[0].errors
        assert (low >= 0).all()
        assert (high >= 0).all()

    def test_flow_sum_is_consistent_everywhere(self) -> None:
        ref = self._hist([0.5, 1.5])
        num = self._hist([0.5, 1.5] + [3.0] * 20)
        p = rf.plot([ref, num], label=["ref", "num"], flow="sum", panel="ratio")
        np.testing.assert_allclose(p.comparisons[0].values, [1.0, 21.0])
        assert p.ax.get_ylim()[1] > 21
        np.testing.assert_allclose(p.histograms[1].values(), [1.0, 21.0])
        stacked = rf.plot(
            [ref, num], label=["ref", "num"], flow="sum", stack=True, panel="s/sqrt(b)"
        )
        np.testing.assert_allclose(stacked.comparisons[0].values, [1.0, 21.0])

    def test_profile_offset_and_unit(self) -> None:
        p = rf.profile(
            {"x": np.array([0.5, 0.5]), "y": np.array([1e9, 1e9 + 1.0])},
            "x",
            "y",
            bins=(1, 0, 1),
            statistic="std",
            unit="GeV",
        )
        np.testing.assert_allclose(p.profiles[0].values, [0.5])
        assert p.ax.get_xlabel() == "x [GeV]"

    def test_variable_log_flag(self) -> None:
        data = {"x": np.array([1.0, 2.0]), "y": np.array([1.0, 2.0])}
        var = rf.Variable("x", bins=(2, 1, 3), log=True)
        assert rf.plot(data, var).ax.get_xscale() == "log"
        assert rf.plot(data, var, logx=False).ax.get_xscale() == "linear"
        assert rf.plot(data, "x", bins=(2, 1, 3)).ax.get_xscale() == "linear"
        p2 = rf.plot2d(data, var, rf.Variable("y", bins=(2, 1, 3), log=True))
        assert (p2.ax.get_xscale(), p2.ax.get_yscale()) == ("log", "log")
        assert rf.efficiency(data, var, passed="y > 1").ax.get_xscale() == "log"
        prof = rf.profile(data, var, rf.Variable("y", log=True))
        assert (prof.ax.get_xscale(), prof.ax.get_yscale()) == ("log", "log")

    def test_plain_hist_storage_and_overlay_binning(self) -> None:
        plain = hist.Hist(hist.axis.Regular(2, 0, 4)).fill([1.0, 3.0])
        p = rf.plot([plain], normalize="width")
        np.testing.assert_allclose(p.hists[0].values(), [0.5, 0.5])
        overlay = rf.plot(
            [
                self._hist([0.5]),
                hist.Hist(
                    hist.axis.Variable([0, 0.5, 1, 1.5, 2]), storage=hist.storage.Weight()
                ).fill([0.5]),
            ]
        )
        assert len(overlay.hists) == 2
        assert overlay.ax.get_ylim()[1] > 1.0
        with pytest.raises(BinningError, match="do not share one binning"):
            rf.plot(overlay.hists, panel="ratio")

    def test_ratio_colors_follow_the_main_panel(self) -> None:
        from matplotlib.colors import to_rgba
        from matplotlib.container import ErrorbarContainer

        hists = [self._hist([0.5]), self._hist([0.5] * 2), self._hist([0.5] * 3)]
        p = rf.plot(hists, label=["A", "B", "C"], panel="ratio")
        assert p.panel_ax is not None
        ratio_colors = [
            to_rgba(c.lines[0].get_color())
            for c in p.panel_ax.containers
            if isinstance(c, ErrorbarContainer)
        ]
        main_colors = [to_rgba(c) for c in rf.plotting.color_cycle(3, rf.Style())[1:]]
        assert ratio_colors == main_colors
        named = rf.plot(hists, label=["A", "B", "C"], panel="ratio", reference="C")
        assert named.panel_ax is not None
        named_colors = [
            to_rgba(c.lines[0].get_color())
            for c in named.panel_ax.containers
            if isinstance(c, ErrorbarContainer)
        ]
        assert named_colors == [to_rgba(c) for c in rf.plotting.color_cycle(2, rf.Style())]

    def test_flow_show_edge_cases(self) -> None:
        no_flow = hist.Hist(
            hist.axis.Regular(2, 0, 4, underflow=False, overflow=False),
            storage=hist.storage.Weight(),
        ).fill([1.0, 3.0])
        np.testing.assert_allclose(rf.plot([no_flow], flow="show").hists[0].values(), [1.0, 1.0])
        negative = self._hist([0.5, 3.0], [1.0, -2.0])
        np.testing.assert_allclose(
            rf.plot([negative], flow="show").hists[0].values(), [1.0, 0.0, -2.0]
        )

    def test_linear_2d_keeps_negative_bins(self) -> None:
        p = rf.plot2d(
            {
                "x": np.array([0.5, 1.5, 2.5]),
                "y": np.array([0.5, 1.5, 2.5]),
                "w": np.array([2.0, -1.0, 2.0]),
            },
            "x",
            "y",
            weight="w",
            bins=(3, 0, 3),
            colorbar=False,
        )
        drawn = p.ax.collections[0].get_array()
        assert not np.ma.getmaskarray(drawn)[1, 1]
        assert drawn[1, 1] == -1.0

    def test_nested_rntuple_and_input_dispatch(self, tmp_path: Path) -> None:
        import uproot

        path = tmp_path / "nested.root"
        with uproot.recreate(path) as file:
            file["events"] = {
                "Muon": ak.zip({"pt": [[1.0, 2.0], [3.0]], "eta": [[0.1, 0.2], [0.3]]})
            }
        assert rf.histogram(path, "Muon.pt", tree="events", bins=(4, 0, 4)).sum().value == 3
        assert rf.histogram({"x": [1.0, 2.0, 3.0]}, "x", bins=(4, 0, 4)).sum().value == 3
        [h] = rf.histograms(
            {"A": rf.Sample({"x": np.arange(3.0)}, weight="2")}, "x", bins=(3, 0, 3)
        )
        assert (h.label, h.integral) == ("A", 6.0)
        labelled = rf.histograms({"A": {"x": np.arange(3.0)}, "B": {"x": np.arange(2.0)}}, "x")
        assert [h.label for h in labelled] == ["A", "B"]

    def test_constants_and_cutflow_policy(self) -> None:
        assert rf.histogram({"x": np.arange(3.0)}, "1", bins=(1, 0, 2)).sum().value == 3
        table = rf.cutflow({"x": np.arange(3.0)}, [], weight="2")
        np.testing.assert_allclose(table.rows[0].yields, [6.0])
        sample = rf.Sample({"x": np.arange(3.0), "w": np.array([1.0, np.inf, 1.0])})
        with pytest.raises(SelectionError, match="non-finite"):
            rf.cutflow(sample, ["x > 0"], weight="w", nonfinite="error")
        with pytest.raises(ValueError, match="scale must be"):
            rf.Sample({"x": np.array([1.0])}, scale=np.inf)


class TestHeadroomIsMeasured:
    """The y limit is a small fixed margin, raised only as far as artists need.

    A peaked distribution leaves a corner free, so the static margin stands. An
    axis-filling one has nowhere to put a legend, a statistics box, a text line
    or an experiment label, so ``raise_ylim_above`` lifts the limit further.
    """

    @staticmethod
    def _peaked() -> dict[str, np.ndarray]:
        return {"x": np.random.default_rng(0).normal(0.0, 1.0, 20_000)}

    @staticmethod
    def _flat() -> dict[str, np.ndarray]:
        return {"x": np.random.default_rng(1).uniform(0.0, 10.0, 20_000)}

    @staticmethod
    def _ratio(p: rf.Plot) -> float:
        tallest = max(float(h.hist.values().max()) for h in p.histograms)
        return p.ax.get_ylim()[1] / tallest

    def test_peaked_plot_keeps_the_static_margin(self) -> None:
        p = rf.plot(self._peaked(), "x", bins=50, legend="upper right")
        assert self._ratio(p) == pytest.approx(1.20)
        plt.close(p.fig)

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({"stats": True}, id="stats"),
            pytest.param({"text": ["a line", "another line"]}, id="text"),
            pytest.param(
                {"style": rf.Style(experiment="ATLAS", lumi=140, com=13)}, id="experiment-label"
            ),
            pytest.param({"legend": "upper right"}, id="anchored-legend"),
            pytest.param({"xbreak": (3.0, 7.0), "legend": "upper right"}, id="xbreak"),
        ],
    )
    def test_axis_filling_plot_is_lifted(self, options: dict[str, Any]) -> None:
        p = rf.plot(self._flat(), "x", bins=50, **options)
        assert self._ratio(p) > 1.20
        plt.close(p.fig)

    @pytest.mark.parametrize("experiment", ["ATLAS", "CMS", "LHCb"])
    def test_label_is_measured_where_it_ends_up(
        self, monkeypatch: pytest.MonkeyPatch, experiment: str
    ) -> None:
        # the status word moves when the label is aligned; the headroom must see it there
        measured: dict[int, np.ndarray] = {}

        def recording(axes: Any, obstacles: Any, **kwargs: Any) -> None:
            renderer = axes[0].figure.canvas.get_renderer()
            for artist in obstacles:
                measured[id(artist)] = artist.get_window_extent(renderer).extents
            raise_ylim_above(axes, obstacles, **kwargs)

        monkeypatch.setattr(plots1d, "raise_ylim_above", recording)
        style = rf.Style(experiment=experiment, status="Internal", lumi=140, com=13.6)
        p = rf.plot(self._peaked(), "x", bins=50, style=style)
        p.fig.canvas.draw()
        renderer = p.fig.canvas.get_renderer()
        label = [t for t in p.ax.texts if isinstance(t, hep.label.ExpText) and t.get_text()]
        assert label
        for text in label:
            np.testing.assert_allclose(
                measured[id(text)], text.get_window_extent(renderer).extents, atol=0.5
            )
        plt.close(p.fig)


class TestOffsetText:
    """The x label stays clear of the axis' offset text, which shares its corner."""

    @staticmethod
    def _boxes(p: rf.Plot) -> tuple[Any, Any]:
        axis = (p.panel_ax or p.ax).xaxis
        p.fig.canvas.draw()
        renderer = p.fig.canvas.get_renderer()
        offset = axis.get_offset_text()
        assert offset.get_text()  # the values need one
        return axis.label.get_window_extent(renderer), offset.get_window_extent(renderer)

    @pytest.mark.parametrize("use_offset", [True, False], ids=["offset", "no-offset"])
    @pytest.mark.parametrize("panel", [None, "ratio"])
    def test_label_and_offset_text_do_not_overlap(self, panel: Any, use_offset: bool) -> None:
        # without an additive offset matplotlib still shows the order of magnitude
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        samples = [
            rf.Sample({"x": values}, label="A"),
            rf.Sample({"x": values * 1.1}, label="B"),
        ]
        style = rf.Style(rc={"axes.formatter.useoffset": use_offset})
        p = rf.plot(samples, "x", bins=20, panel=panel, style=style)
        label, offset = self._boxes(p)
        assert not label.overlaps(offset)
        assert label.x1 <= offset.x0  # beside it, on the same line
        assert label.y0 < offset.y1
        assert offset.y0 < label.y1
        plt.close(p.fig)

    def test_plot2d_label_and_offset_text_do_not_overlap(self) -> None:
        rng = np.random.default_rng(0)
        data = {"x": rng.normal(2e-6, 1e-6, 5_000), "y": rng.normal(0, 1, 5_000)}
        p = rf.plot2d(data, "x", "y", bins=20)
        label, offset = self._boxes(p)
        assert not label.overlaps(offset)
        plt.close(p.fig)

    @pytest.mark.parametrize("experiment", ["LHCb", "ALICE"])
    def test_offset_clearance_uses_the_finished_axes_width(self, experiment: str) -> None:
        values = np.random.default_rng(0).uniform(0, 1e-6, 10_000)
        p = rf.plot(
            {"x": values},
            "x",
            bins=20,
            style=rf.Style(experiment=experiment, lumi=138, com=13.6, figsize=(5, 3)),
            title="Some plot title",
        )
        label, offset = self._boxes(p)
        assert label.x1 <= offset.x0

    @pytest.mark.parametrize("xlabel", ["x", "x" * 30], ids=["beside", "below"])
    def test_clearance_holds_when_the_figure_is_resized(self, xlabel: str) -> None:
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        p = rf.plot({"x": values}, "x", bins=20, xlabel=xlabel, figsize=(5, 4))
        for size, dpi in [((3.5, 3.0), 100.0), ((12.0, 8.0), 100.0), ((5.0, 4.0), 200.0)]:
            p.fig.set_size_inches(size)
            p.fig.set_dpi(dpi)
            label, offset = self._boxes(p)
            assert not label.overlaps(offset), (size, dpi)
        plt.close(p.fig)

    def test_label_changes_line_as_the_figure_is_resized(self) -> None:
        # beside the offset text where the axes leave room, below it where they do not
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        p = rf.plot({"x": values}, "x", bins=20, xlabel="x" * 20, figsize=(5, 4))
        for width, below in [(5.0, False), (3.5, True), (5.0, False)]:
            p.fig.set_size_inches(width, 4.0)
            label, offset = self._boxes(p)
            assert not label.overlaps(offset), width
            assert bool(label.y1 <= offset.y0) == below, width
            assert label.x0 >= p.ax.get_window_extent().x0, width
        plt.close(p.fig)

    def test_label_without_room_beside_goes_below(self) -> None:
        # constrained layout reserves an x label's height, never its width, so a label
        # moved past the left end of the axes could leave the canvas
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        p = rf.plot({"x": values}, "x", bins=20, xlabel="x" * 30, figsize=(5, 4))
        label, offset = self._boxes(p)
        assert label.y1 <= offset.y0
        assert label.x0 >= p.ax.get_window_extent().x0
        assert label.y0 >= 0
        plt.close(p.fig)

    def test_headroom_holds_with_the_label_below(self) -> None:
        # the label below the offset text takes height from the axes; the headroom must
        # be measured with it there
        rng = np.random.default_rng(0)
        samples = [rf.Sample({"x": rng.uniform(0, 1e-6, 20_000)}, label=f"{i}") for i in range(5)]
        p = rf.plot(samples, "x", bins=20, xlabel="x" * 30, legend="upper right", figsize=(5, 4))
        label, offset = self._boxes(p)
        assert label.y1 <= offset.y0
        legend = p.ax.get_legend()
        (x0, y0), (x1, _) = p.ax.transData.inverted().transform(
            legend.get_window_extent().get_points()
        )
        edges = p.histograms[0].edges
        under = (edges[1:] > x0) & (edges[:-1] < x1)
        tallest = max(float(h.hist.values()[under].max()) for h in p.histograms)
        assert y0 >= 1.08 * tallest * (1 - 1e-6)
        plt.close(p.fig)

    def test_changes_made_after_plotting_are_kept(self) -> None:
        # the axes stay the caller's: a pad or transform set later survives every draw
        values = np.random.default_rng(0).normal(0, 1, 5_000)  # no offset text
        p = rf.plot({"x": values}, "x", bins=20)
        p.fig.canvas.draw()
        label = p.ax.xaxis.label
        lowered = label.get_transform() + ScaledTranslation(0, -0.1, p.fig.dpi_scale_trans)
        label.set_transform(lowered)
        p.ax.xaxis.labelpad = 25
        for _ in range(2):
            p.fig.canvas.draw()
            assert p.ax.xaxis.labelpad == 25
            assert label.get_transform() is lowered
        plt.close(p.fig)

    def test_a_pad_set_after_plotting_counts_against_the_offset_text(self) -> None:
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        p = rf.plot({"x": values}, "x", bins=20, xlabel="x" * 30, figsize=(5, 4))
        label, offset = self._boxes(p)
        assert label.y1 <= offset.y0  # below, on rootfig's own pad
        p.ax.xaxis.labelpad = 30  # clears the offset text on its own
        label, offset = self._boxes(p)
        assert p.ax.xaxis.labelpad == 30
        assert label.y1 <= offset.y0
        p.ax.xaxis.labelpad = 2  # back onto the offset text's line: rootfig moves it again
        label, offset = self._boxes(p)
        assert not label.overlaps(offset)
        plt.close(p.fig)

    def test_a_pad_changed_while_below_is_laid_out_at_once(self) -> None:
        # the label stays below: the layout must still reserve the new pad on this draw
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        p = rf.plot({"x": values}, "x", bins=20, xlabel="x" * 30, figsize=(5, 4))
        p.ax.xaxis.labelpad = 6
        self._boxes(p)
        p.ax.xaxis.labelpad = 4
        label, offset = self._boxes(p)
        assert label.y1 <= offset.y0
        assert label.y0 >= 0
        again, _ = self._boxes(p)
        np.testing.assert_allclose(again.extents, label.extents, atol=0.5)
        plt.close(p.fig)

    def test_label_on_axes_the_caller_made_holds_when_resized(self) -> None:
        # rootfig does not lay out a figure it did not make: the label goes below the
        # offset text, which fits at any width the label itself fits
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        fig, ax = plt.subplots(figsize=(5, 4))
        p = rf.plot({"x": values}, "x", bins=20, xlabel="x" * 20, ax=ax)
        for width in (5.0, 3.5, 8.0):
            fig.set_size_inches(width, 4.0)
            label, offset = self._boxes(p)
            assert not label.overlaps(offset), width
            assert label.y1 <= offset.y0, width
            assert label.x0 >= ax.get_window_extent().x0, width
        plt.close(fig)


class TestWeightedRangeInference:
    def test_high_weight_entries_stay_on_the_axis(self) -> None:
        """An inferred range accounts for the content a cut would remove, not just entries."""
        rng = np.random.default_rng(0)
        values = np.r_[rng.normal(0, 1, 20_000), rng.normal(8, 0.2, 50)]
        weights = np.r_[np.ones(20_000), np.full(50, 1000.0)]
        p = rf.plot({"x": values, "w": weights}, "x", weight="w", bins=50)
        low, high = p.ax.get_xlim()
        assert high > 8.0
        kept = (values >= low) & (values < high)
        assert weights[kept].sum() / weights.sum() > 0.99
        plt.close(p.fig)
        # the same values unweighted are a thin tail and are cut
        unweighted = rf.plot({"x": values}, "x", bins=50)
        assert unweighted.ax.get_xlim()[1] < 8.0
        plt.close(unweighted.fig)


class TestSystematics:
    @pytest.mark.filterwarnings("ignore::rootfig.errors.RootfigWarning")
    def test_stack_with_data_and_ratio(self, signal_file: Path, background_file: Path) -> None:
        bkg = rf.Sample(
            background_file,
            tree="events",
            label="Background",
            weight="weight",
            systematics={
                "weight": ("weight * 1.1", "weight * 0.95"),
                "met": {"MET": ("with_nan", "sentinel")},
            },
        )
        sig = rf.Sample(signal_file, tree="events", label="Signal", systematics={"xsec": 0.2})
        data = rf.Sample(background_file, tree="events", label="Data")
        p = rf.plot(
            [bkg, sig],
            "MET",
            observed=data,
            bins=(20, 0, 200),
            stack=True,
            panel="ratio",
            systematics={"lumi": 0.02},
        )
        legend_texts = [t.get_text() for t in p.ax.get_legend().get_texts()]
        assert legend_texts == ["Data", "Signal", "Background", "Stat. + syst. unc."]
        total = p.uncertainty()
        assert list(total.components) == ["lumi", "weight", "met", "xsec"]
        np.testing.assert_allclose(
            total.components["lumi"][0],
            0.02 * (p.histograms[0].values() + p.histograms[1].values()),
        )
        assert np.all(total.total_up >= total.stat_up)
        assert p.comparisons[0].syst_band is not None
        filled = total.nominal > 0
        np.testing.assert_allclose(
            p.comparisons[0].syst_band[1][filled], total.syst_up[filled] / total.nominal[filled]
        )
        signal = p.uncertainty("Signal")
        assert set(signal.components) == {"lumi", "xsec"}
        assert not p.uncertainty("Data").has_systematics
        with pytest.raises(KeyError, match="no histogram labelled"):
            p.uncertainty("nope")

    def test_replaced_branches_move_the_selection(self, signal_arrays: dict[str, Any]) -> None:
        sample = rf.Sample(signal_arrays, systematics={"s": {"MET": "weight"}})
        (h,) = rf.histograms(sample, "MET", selection="MET > 0.9", bins=(10, 0, 2))
        swapped = {**signal_arrays, "MET": signal_arrays["weight"]}
        (expected,) = rf.histograms(swapped, "MET", selection="MET > 0.9", bins=(10, 0, 2))
        np.testing.assert_allclose(h.variations["s"][0].values(), expected.values())

    def test_varied_files_inherit_the_tree(self, signal_file: Path, background_file: Path) -> None:
        sample = rf.Sample(
            signal_file,
            tree="events",
            label="S",
            systematics={"generator": rf.Systematic.samples(str(background_file))},
        )
        (h,) = rf.histograms(sample, "MET", bins=(10, 0, 100), systematics={"lumi": 0.1})
        (other,) = rf.histograms(background_file, "MET", tree="events", bins=(10, 0, 100))
        np.testing.assert_allclose(h.variations["generator"][0].values(), other.values())
        assert set(h.variations) == {"lumi", "generator"}

    def test_prefilled_histograms_and_empty_plot(self) -> None:
        nominal = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
        nominal.fill([0.5, 1.5, 1.5, 2.5])
        h = rf.Histogram(nominal, label="MC", variations={"s": (nominal * 1.1, nominal * 0.8)})
        p = rf.plot([h], panel=None)
        np.testing.assert_allclose(p.uncertainty().syst_down, 0.2 * nominal.values())
        data = rf.Histogram(nominal, label="Data", is_data=True)
        with pytest.raises(ValueError, match="no non-data histograms"):
            rf.plot([data]).uncertainty()

    def test_named_reference_chooses_uncertainty_per_numerator(self) -> None:
        rng = np.random.default_rng(3)
        reference = rf.Sample(
            {"x": rng.uniform(0, 2, 500)}, label="Reference", systematics={"lumi": 0.10}
        )
        other = rf.Sample({"x": rng.uniform(0, 2, 500)}, label="Other", systematics={"lumi": 0.10})
        data = rf.Sample({"x": rng.uniform(0, 2, 500)}, label="Data", is_data=True)
        p = rf.plot(
            [reference, other, data], "x", bins=(2, 0, 2), panel="ratio", reference="Reference"
        )
        other_ratio, data_ratio = p.comparisons
        assert other_ratio.syst_errors is not None  # simulation / simulation: lumi cancels
        np.testing.assert_allclose(other_ratio.syst_errors, 0.0, atol=1e-12)
        for side in other_ratio.errors:
            np.testing.assert_allclose(side**2, _propagated_variance(p))
        assert data_ratio.syst_errors is None  # data / simulation: the reference is the band
        assert data_ratio.syst_band is not None
        np.testing.assert_allclose(data_ratio.syst_band, 0.1)
        explicit = rf.plot(
            [reference, other, data],
            "x",
            bins=(2, 0, 2),
            panel="ratio",
            reference="Reference",
            panel_uncertainty="numerator",
        )
        assert explicit.comparisons[0].syst_errors is not None
        for side in explicit.comparisons[
            0
        ].syst_errors:  # its own lumi against the nominal reference
            np.testing.assert_allclose(side, 0.1 * explicit.comparisons[0].values)

    def test_efficiency_numerators_remember_their_weights(self) -> None:
        # nothing passes: the numerator's (0, 0) looks like counts, but weights filled it
        sample = rf.Sample(
            {"x": np.array([0.5, 1.5]), "w": np.array([0.5, 2.0]), "ok": np.zeros(2)},
            label="Data",
            is_data=True,
            weight="w",
        )
        p = rf.efficiency(sample, "x", passed="ok == 1", bins=(2, 0, 2))
        numerator = p.histograms[0]
        np.testing.assert_array_equal(numerator.values(), 0.0)
        with pytest.raises(ValueError, match="filled with weights or scaled"):
            numerator.replace(poisson=True)
        again = rf.plot(
            [numerator.replace(is_data=False)], observed=[numerator], data_errors="auto"
        )
        assert not again.histograms[-1].poisson
        plt.close("all")

    def test_efficiency_intervals_need_non_negative_weights(self) -> None:
        # bin 0: {1, 1, -0.1} pass and {1} fails, a ratio in [0, 1] whose sums hide the sign
        x = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
        w = np.array([1.0, 1.0, -0.1, 1.0, 1.0, 2.0, 1.0])
        passes = np.array([1, 1, 1, 0, 1, 0, 1])
        sample = rf.Sample({"x": x, "w": w, "ok": passes}, label="nlo", weight="w")
        with pytest.warns(RootfigWarning, match="nlo: 1 bin"):
            p = rf.efficiency(
                sample, "x", passed="ok == 1", bins=(2, 0, 2), interval="wilson-effective"
            )
        (eff,) = p.efficiencies
        np.testing.assert_allclose(eff.values, [1.9 / 2.9, 2.0 / 4.0])
        assert np.isnan([eff.lower[0], eff.upper[0]]).all()
        assert np.isfinite([eff.lower[1], eff.upper[1]]).all()  # positive weights only
        p.close()
        # the default, ROOT's normal approximation for weighted entries, holds for signed ones
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p = rf.efficiency(sample, "x", passed="ok == 1", bins=(2, 0, 2))
        assert np.isfinite(p.efficiencies[0].lower).all()
        p.close()

    def test_efficiency_ignores_systematics(self) -> None:
        sample = rf.Sample({"x": [0.5, 1.5]}, systematics={"unused": "missing_weight"})
        p = rf.efficiency(sample, "x", passed="x > 1", bins=(2, 0, 2))
        np.testing.assert_allclose(p.efficiencies[0].values, [0.0, 1.0])

    def test_errors_name_the_variation(self) -> None:
        sample = rf.Sample({"x": [0.5, 1.5]}, label="MC", systematics={"pileup": ("x", "w_down")})
        with pytest.raises(MissingBranchError) as caught:
            rf.histograms(sample, "x", bins=(2, 0, 2))
        assert any("MC [pileup down]" in note for note in caught.value.__notes__)

    def test_varied_arrays_keep_the_nominal_ngen_from_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nominal.root"
        with uproot.recreate(path) as file:
            file.mktree("events", {"x": "float64"}).extend({"x": np.array([0.5, 1.5, 2.5])})
            file["eventsProcessed"] = np.histogram([0.5] * 6, bins=1)
        sample = rf.Sample(
            str(path),
            tree="events",
            xsec=1.0,
            ngen="eventsProcessed",
            systematics={"generator": rf.Systematic.samples({"x": [0.5, 0.5, 2.5]})},
        )
        (h,) = rf.histograms(sample, "x", bins=(3, 0, 3), lumi=1)
        scale = 1e3 / 6  # 1 pb x 1 fb^-1 / 6 generated events, for nominal and variation alike
        np.testing.assert_allclose(h.values(), [scale, scale, scale])
        np.testing.assert_allclose(h.variations["generator"][0].values(), [2 * scale, 0, scale])

    def test_varied_data_warnings_name_the_variation(self) -> None:
        sample = rf.Sample(
            {"x": [0.5, 1.5]},
            label="MC",
            systematics={"gen": rf.Systematic.samples({"x": [np.nan, 1.5]})},
        )
        with pytest.warns(RootfigWarning, match=r"^MC \[gen up\]: dropped 1 non-finite"):
            rf.histograms(sample, "x", bins=(2, 0, 2))
        replaced = rf.Sample({"x": [0.5]}, label="MC", systematics={"jes": {"x": "x_up"}})
        with pytest.raises(MissingBranchError, match=r"MC \[jes up\]: replacing 'x'"):
            rf.histograms(replaced, "x", bins=(2, 0, 2))

    @pytest.mark.parametrize("bins", [2, 3])
    def test_uncertainty_of_an_overlay_needs_a_label(self, bins: int) -> None:
        a = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        b = hist.Hist(hist.axis.Regular(bins, 0, 2), storage=hist.storage.Weight())
        p = rf.plot([rf.Histogram(a, "a"), rf.Histogram(b, "b")])
        with pytest.raises(ValueError, match=r"pass the label of one, e.g. uncertainty\('a'\)"):
            p.uncertainty()
        assert p.uncertainty("b").nominal.size == bins

    def test_exports(self) -> None:
        assert rf.Systematic.samples("alt.root").kind == "samples"
        assert issubclass(rf.SystematicError, rf.RootfigError)
        from rootfig.histograms import Uncertainty

        assert rf.Uncertainty is Uncertainty


def _propagated_variance(p: rf.Plot) -> Any:
    reference, other = p.histograms[0], p.histograms[1]
    n, d = other.values(), reference.values()
    return other.variances() / d**2 + n**2 * reference.variances() / d**4


def test_plain_histogram_ignores_sample_systematics() -> None:
    sample = rf.Sample(
        {"x": [0.5, 1.5]},
        systematics={
            "missing": "missing_weight",
            "files": rf.Systematic.samples("does-not-exist.root"),
        },
    )
    result = rf.histogram(sample, "x", bins=(2, 0, 2))
    np.testing.assert_allclose(result.values(), [1, 1])


def test_varied_files_inherit_auto_detected_tree_and_range(tmp_path: Path) -> None:
    nominal = tmp_path / "nominal.root"
    alternate = tmp_path / "alternate.root"
    with uproot.recreate(nominal) as f:
        f["events"] = {"x": np.array([0.5, 1.5, 2.5, 3.5])}
    with uproot.recreate(alternate) as f:
        f["events"] = {"x": np.array([0.5, 1.5, 2.5, 3.5])}
        f["other"] = {"x": np.array([3.5, 2.5, 1.5, 0.5])}
    sample = rf.Sample(
        nominal,
        entry_start=0,
        entry_stop=2,
        systematics={
            "inherit": rf.Systematic.samples(alternate),
            "explicit": rf.Systematic.samples(f"{alternate}:other"),
            "arrays": rf.Systematic.samples({"x": [0.5, 1.5, 2.5, 3.5]}),
        },
    )
    (h,) = rf.histograms(sample, "x", bins=(4, 0, 4))
    np.testing.assert_allclose(h.variations["inherit"][0].values(), [1, 1, 0, 0])
    np.testing.assert_allclose(h.variations["explicit"][0].values(), [0, 0, 1, 1])
    np.testing.assert_allclose(h.variations["arrays"][0].values(), [1, 1, 0, 0])


class TestStoredHistogramPlots:
    """`plot`, `histogram(s)` and `plot2d` on histograms stored in files or given as objects."""

    def test_plot_from_files(self, stored_dir: Path) -> None:
        samples = {
            "ZH": rf.Sample(stored_dir / "ZH_sel0_histo.root", color="C3", scale=2.0),
            "VV": [stored_dir / "WW_sel0_histo.root", stored_dir / "ZZ_sel0_histo.root"],
        }
        p = rf.plot(samples, "mz", bins=50, stack=True, logy=True)
        assert [h.label for h in p.histograms] == ["ZH", "VV"]
        assert p.histograms[0].axis.size == 50
        assert p.histograms[1].sum_weights == pytest.approx(0.5 * 3000)
        assert p.ax.get_xlabel() == "m_{Z} [GeV]"  # the stored title
        assert p.ax.get_ylabel() == "Events / 5 GeV"  # its unit feeds the bin-width label
        assert p.ax.get_yscale() == "log"
        p = rf.plot(samples, "mz", xlabel="$m_{Z}$", unit="GeV")
        assert p.ax.get_xlabel() == "$m_{Z}$ [GeV]"

    def test_histogram_functions(self, stored_dir: Path) -> None:
        vv = rf.Sample([stored_dir / "WW_sel0_histo.root", stored_dir / "ZZ_sel0_histo.root"])
        h = rf.histogram(vv, "mz")
        assert isinstance(h, hist.Hist)
        assert h.values(flow=True).sum() == pytest.approx(0.5 * 3000)
        (only,) = rf.histograms(stored_dir / "ZH_sel0_histo.root", "mz", label="ZH", normalize=True)
        assert only.label == "ZH"
        assert only.integral == pytest.approx(1.0)
        # the Variable the histogram was filled with describes it, also once normalised
        mz = rf.Variable("mz", bins=only.edges)
        assert rf.plot(only, mz).histograms[0] is only
        assert rf.plot(only, rf.Variable("mz", bins=only.axis.size)).histograms[0] is only
        with pytest.raises(rf.BinningError, match="rebin before normalising"):
            rf.plot(only, rf.Variable("mz", bins=only.axis.size // 2))

    def test_plot2d_stored_and_object(self, stored_dir: Path) -> None:
        p = rf.plot2d(stored_dir / "ZH_sel0_histo.root", "mz_recoil_2D", bins=(5, 6), logz=True)
        # plot2d ignores systematics, for a stored histogram as for a tree
        varied = rf.Sample(stored_dir / "ZH_sel0_histo.root", systematics={"w": ("w_up", "w_dn")})
        assert rf.plot2d(varied, "mz_recoil_2D").histograms[0].variations == {}
        assert [a.size for a in p.histograms[0].hist.axes] == [5, 6]
        assert p.ax.get_xlabel() == "m_{Z} [GeV]"
        assert p.ax.get_ylabel() == "recoil [GeV]"
        again = rf.plot2d(p.histograms[0], cmap="magma", title="again")
        assert again.histograms[0] is p.histograms[0]
        assert again.ax.get_title() == "again"
        merged = rf.plot2d(p.histograms[0], bins=(1, 2))
        assert [a.size for a in merged.histograms[0].hist.axes] == [1, 2]
        # a single Variable describes the stored histogram and its x axis only
        described = rf.plot2d(
            stored_dir / "ZH_sel0_histo.root",
            rf.Variable("mz_recoil_2D", label="Mass", unit="GeV", log=True),
        )
        assert described.ax.get_xlabel() == "Mass [GeV]"
        assert described.ax.get_ylabel() == "recoil [GeV]"
        assert described.ax.get_xscale() == "log"
        assert described.ax.get_yscale() == "linear"
        # a stored TH2 without axis titles: the y axis does not take the object name
        untitled = rf.plot2d(stored_dir / "untitled_2D.root", "mz_recoil_2D")
        assert untitled.ax.get_xlabel() == "mz_recoil_2D"
        assert untitled.ax.get_ylabel() == "mz_recoil_2D_y"
        titled = rf.plot2d(
            stored_dir / "untitled_2D.root",
            "mz_recoil_2D",
            rf.Variable("mz_recoil_2D", label="Recoil", unit="GeV"),
        )
        assert titled.ax.get_ylabel() == "Recoil [GeV]"
        # an explicit name on x names the y axis too, as build_histograms_2d does
        named = rf.plot2d(
            stored_dir / "untitled_2D.root",
            rf.Variable("mz_recoil_2D", name="mass", label="Mass", unit="GeV"),
        )
        assert [a.name for a in named.histograms[0].hist.axes] == ["mass", "mass_y"]
        assert named.ax.get_xlabel() == "Mass [GeV]"
        assert named.ax.get_ylabel() == "mass_y"
        from rootfig.histograms import build_histograms_2d

        (low_level,) = build_histograms_2d(
            [rf.Sample(stored_dir / "untitled_2D.root")], rf.Variable("mz_recoil_2D", name="mass")
        )
        assert [a.name for a in low_level.hist.axes] == ["mass", "mass_y"]
        # an object rootfig cannot plot is reported as such through the public functions
        with pytest.raises(rf.SourceError, match=r"'prof' .* is a TProfile, which rootfig cannot"):
            rf.plot(stored_dir / "unsupported.root", "prof")
        with pytest.raises(rf.SourceError, match=r"'h3' .* is a TH3D"):
            rf.plot2d(stored_dir / "unsupported_with_tree.root", "h3")

        with pytest.raises(TypeError, match="needs the x and y variables"):
            rf.plot2d(stored_dir / "ZH_sel0_histo.root")
        with pytest.raises(ValueError, match="two-dimensional histograms"):
            rf.plot2d(p.histograms[0].hist[:, :: hist.sum])
        # the dimensionality is checked before any rebinning
        with pytest.raises(ValueError, match="two-dimensional histograms, got 1D"):
            rf.plot2d(p.histograms[0].hist[:, :: hist.sum], bins=2)
        with pytest.raises(ValueError, match=r"one-dimensional histograms, got 2D.*use plot2d"):
            rf.plot(p.histograms[0], bins=5)
        # neither function draws a 3D histogram, so none is suggested
        cube = hist.Hist(*[hist.axis.Regular(2, 0, 1, name=n) for n in "abc"])
        with pytest.raises(ValueError, match=r"got 3D \(.*\)$"):
            rf.plot2d(cube)
        with pytest.raises(ValueError, match=r"got 3D \(.*\)$"):
            rf.plot(cube)
        with pytest.raises(ValueError, match="selection, weight apply when filling"):
            rf.plot2d(p.histograms[0], selection="x > 1", weight="w")

    def test_plot2d_object_described_by_variables(self) -> None:
        h2 = hist.Hist(
            hist.axis.Regular(10, 0.0, 100.0, name="x", label="X"),
            hist.axis.Regular(6, 0.0, 3.0, name="y"),
            storage=hist.storage.Weight(),
        )
        h2.fill([5.0, 15.0, 95.0], [0.5, 1.5, 2.5], weight=[1.0, 2.0, 3.0])
        before = h2.copy()
        plain = rf.plot2d(h2)
        assert (plain.ax.get_xlabel(), plain.ax.get_ylabel()) == ("X", "y")
        described = rf.plot2d(
            h2,
            rf.Variable("mass", label="Mass", unit="GeV", log=True),
            rf.Variable("recoil", label="Recoil", bins=3),
        )
        assert described.ax.get_xlabel() == "Mass [GeV]"
        assert described.ax.get_ylabel() == "Recoil"
        assert (described.ax.get_xscale(), described.ax.get_yscale()) == ("log", "linear")
        assert [a.name for a in described.histograms[0].hist.axes] == ["mass", "recoil"]
        assert [a.size for a in described.histograms[0].hist.axes] == [10, 3]
        assert described.variable is not None
        assert described.variable.expression == "mass"
        np.testing.assert_allclose(described.histograms[0].values().sum(), 6.0)
        # a unit alone is appended to the existing title; x alone leaves y as it is
        unit_only = rf.plot2d(h2, rf.Variable("mass", unit="GeV"))
        assert (unit_only.ax.get_xlabel(), unit_only.ax.get_ylabel()) == ("X [GeV]", "y")
        # explicit arguments override the variables, as in plot()
        overridden = rf.plot2d(
            h2,
            rf.Variable("mass", bins=2, log=True),
            rf.Variable("recoil", bins=3),
            bins=(5, 6),
            logx=False,
        )
        assert [a.size for a in overridden.histograms[0].hist.axes] == [5, 6]
        assert overridden.ax.get_xscale() == "linear"
        # only merges of the existing bins are possible
        with pytest.raises(rf.BinningError, match="can be merged into"):
            rf.plot2d(h2, rf.Variable("mass", bins=7))
        for variable in (
            rf.Variable("mass", bins=(5, 0.0, 50.0)),
            rf.Variable("mass", range=(0.0, 50.0)),
        ):
            cropped = rf.plot2d(h2, variable).histograms[0]
            assert cropped.hist.axes[0].size == 5
            assert cropped.values().sum() == 3
            assert cropped.values(flow=True)[-1].sum() == 3
        with pytest.raises(ValueError, match="tree applies when filling"):
            rf.plot2d(h2, rf.Variable("mass"), tree="events")
        # the histogram given is untouched by all of this
        assert [(a.name, a.label) for a in h2.axes] == [("x", "X"), ("y", "y")]
        np.testing.assert_array_equal(h2.values(), before.values())
        np.testing.assert_array_equal(h2.variances(), before.variances())

    def test_explicit_intent_wins(self, stored_dir: Path) -> None:
        # tree= means a branch even when a histogram of that name exists
        with pytest.raises(MissingBranchError, match="unknown name 'mz'"):
            rf.plot(stored_dir / "tree_without_branch.root", "mz", tree="events")
        with pytest.raises(SourceError, match="no TTree or RNTuple"):
            rf.plot(rf.Sample(stored_dir / "ZH_sel0_histo.root", entry_stop=5), "mz")
        # a branch of the same name is filled, not read
        p = rf.plot(stored_dir / "branch_and_histogram.root", "mz", bins=(10, 0, 250))
        assert p.histograms[0].stats is not None
        assert p.histograms[0].stats.entries == 300
        # a file with several trees and the histogram is ambiguous
        with pytest.raises(SourceError, match="several trees"):
            rf.plot(stored_dir / "two_trees.root", "mz")
        # a stored histogram next to a tree lacking the branch is read
        p = rf.plot(stored_dir / "tree_without_branch.root", "mz")
        assert p.histograms[0].stats is None
        assert p.histograms[0].sum_weights == 500

    def test_category_axes_need_the_same_categories(self) -> None:
        def cutflow(categories: list[str]) -> hist.Hist:
            h = hist.Hist(
                hist.axis.StrCategory(categories, name="cut"), storage=hist.storage.Weight()
            )
            h.fill(categories)
            return h

        a, b = (
            cutflow(["all", "preselection", "final"]),
            cutflow(["all", "preselection", "control"]),
        )
        p = rf.plot([a, cutflow(["all", "preselection", "final"])], label=["A", "B"], stack=True)
        assert [t.get_text() for t in p.ax.get_xticklabels()][:3] == [
            "all",
            "preselection",
            "final",
        ]
        with pytest.raises(BinningError, match="a stack"):
            rf.plot([a, b], label=["A", "B"], stack=True)
        with pytest.raises(BinningError):
            rf.plot([a, b], label=["A", "B"], panel="ratio")

    def test_significance_sums_backgrounds_whatever_their_axis_labels(
        self, stored_dir: Path
    ) -> None:
        # the stored titles differ between files (a placeholder title gives way to the name)
        samples = {
            "ZH": stored_dir / "ZH_sel0_histo.root",
            "M": stored_dir / "mixed_storage.root",
            "S": stored_dir / "WW_sel0_histo.root",
        }
        p = rf.plot(samples, "mz", stack=True, panel="s/sqrt(b)")
        assert p.panel_ax is not None
        assert np.nansum(p.comparisons[0].values) > 0
        axis = hist.axis.Regular(4, 0, 4, name="x", label="A")
        other = hist.axis.Regular(4, 0, 4, name="y", label="B")
        a = hist.Hist(axis, storage=hist.storage.Weight()).fill([0.5, 1.5])
        b = hist.Hist(other, storage=hist.storage.Weight()).fill([1.5, 2.5])
        s = hist.Hist(axis, storage=hist.storage.Weight()).fill([2.5, 3.5])
        # the last non-data histogram over the sum of the others, whose axes differ in name
        p = rf.plot([a, b, s], label=["A", "B", "S"], panel="s/sqrt(b)")
        assert (p.comparisons[0].label, p.comparisons[0].reference) == ("S", "Background")
        np.testing.assert_allclose(p.comparisons[0].values, [0.0, 0.0, 1.0, np.nan])

    def test_category_axes_and_flow(self) -> None:
        axis = hist.axis.StrCategory(["all", "sel0"], name="cut")
        listed = hist.Hist(axis, storage=hist.storage.Weight()).fill(["all", "all", "sel0"])
        unlisted = hist.Hist(axis, storage=hist.storage.Weight()).fill(["all", "other"])
        assert unlisted.values(flow=True)[-1] == 1.0
        # no entries outside the categories: nothing to show or fold
        for flow in ("hint", "show", "sum", "none"):
            assert rf.plot(listed, flow=flow).histograms[0].axis.size == 2  # type: ignore[arg-type]
        rf.plot(unlisted, flow="hint")  # the arrow marks the unlisted entries
        for flow in ("show", "sum"):
            with pytest.raises(BinningError, match="categories it does not list"):
                rf.plot(unlisted, flow=flow)  # type: ignore[arg-type]

    def test_luminosity_needs_ngen_without_a_tree(self, stored_dir: Path) -> None:
        with pytest.raises(rf.LuminosityError, match="ngen=<number>"):
            rf.plot(rf.Sample(stored_dir / "ZH_sel0_histo.root", xsec=1.2), "mz", lumi=10)
        (h,) = rf.histograms(
            rf.Sample(stored_dir / "ZH_sel0_histo.root", xsec=1.2, ngen=4000), "mz", lumi=10
        )
        assert h.sum_weights == pytest.approx(0.5 * 4000 * 1.2 * 1e4 / 4000)

    def test_stats_needs_filled_statistics(self, stored_dir: Path) -> None:
        with pytest.raises(ValueError, match="stats= needs the unbinned statistics"):
            rf.plot(stored_dir / "ZH_sel0_histo.root", "mz", stats=True)
        h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight()).fill([0.5])
        with pytest.raises(ValueError, match="stats= needs the unbinned statistics"):
            rf.plot(h, stats=True)

    def test_histogram_objects(self) -> None:
        axis = hist.axis.Regular(4, 0, 4, name="x", label="$x$ [cm]")
        mc = hist.Hist(axis, storage=hist.storage.Weight()).fill([0.5, 1.5, 2.5], weight=2.0)
        data = hist.Hist(axis, storage=hist.storage.Weight()).fill([0.5, 1.5, 1.5, 3.5])
        p = rf.plot(mc, label="MC", observed=data, panel="ratio")
        assert [h.label for h in p.histograms] == ["MC", "x"]
        assert p.histograms[1].is_data
        assert p.ax.get_ylabel() == "Events / 1 cm"
        assert p.panel_ax is not None
        p = rf.plot([mc], rf.Variable("x", label="$x$", unit="cm", log=True))
        assert p.ax.get_xscale() == "log"
        assert p.panel_ax is None
        assert p.ax.get_xlabel() == "$x$ [cm]"
        with pytest.raises(ValueError, match="selection applies when filling"):
            rf.plot([mc], selection="x > 1")
        # bins= merges bins, as for stored histograms: a count, or edges coinciding with its own
        assert rf.plot(mc, bins=2).histograms[0].axis.size == 2
        assert rf.plot(mc, rf.Variable("x", bins=2)).histograms[0].axis.size == 2
        assert rf.plot(mc, bins=(2, 0, 4)).histograms[0].axis.size == 2
        assert rf.plot(mc, bins=[0, 1, 4]).histograms[0].edges.tolist() == [0.0, 1.0, 4.0]
        # the Variable a histogram was filled with can be passed along with it
        described = rf.Variable("x", bins=(4, 0, 4), label="$x$", unit="cm")
        filled = rf.Histogram(mc, label="MC")
        p = rf.plot(filled, described)
        assert p.histograms[0] is filled
        assert p.ax.get_xlabel() == "$x$ [cm]"
        with pytest.raises(BinningError, match=r"has 4 bins.*not 3"):
            rf.plot(mc, bins=3)
        with pytest.raises(BinningError, match=r"no bin edge at 1\.333"):
            rf.plot(mc, bins=(3, 0, 4))
        with pytest.raises(BinningError, match="no bin edge at 50"):
            rf.plot(mc, rf.Variable("x", bins=2, range=(0, 100)))
        cropped = rf.plot(mc, range=(0, 2)).histograms[0]
        np.testing.assert_allclose(cropped.edges, [0, 1, 2])
        # a unit given for histogram objects reaches both axes, once
        p = rf.plot(mc, unit="cm")
        assert p.ax.get_xlabel() == "$x$ [cm]"
        assert p.ax.get_ylabel() == "Events / 1 cm"
        bare = hist.Hist(
            hist.axis.Regular(4, 0, 4, name="p", label="$p$"), storage=hist.storage.Weight()
        )
        p = rf.plot(bare, unit="GeV")
        assert p.ax.get_xlabel() == "$p$ [GeV]"
        assert p.ax.get_ylabel() == "Events / 1 GeV"
        p = rf.plot(bare, rf.Variable("momentum", unit="GeV"))
        assert p.ax.get_xlabel() == "$p$ [GeV]"
        assert p.ax.get_ylabel() == "Events / 1 GeV"
        p = rf.plot(bare, xlabel="$|p|$ [MeV]")
        assert p.ax.get_xlabel() == "$|p|$ [MeV]"
        assert p.ax.get_ylabel() == "Events / 1 MeV"
        with pytest.raises(TypeError, match="observed= must be histogram objects"):
            rf.plot([mc], observed="data.root")
        with pytest.raises(TypeError, match="plot\\(\\) needs a variable"):
            rf.plot("data.root")

    def test_public_surface_has_no_separate_histogram_plotter(self) -> None:
        assert "plot_histograms" not in rf.__all__
        assert not hasattr(rf, "plot_histograms")


class TestGroups:
    @staticmethod
    def _sample(mean: float, n: int, seed: int, **kwargs: Any) -> rf.Sample:
        rng = np.random.default_rng(seed)
        return rf.Sample({"x": rng.normal(mean, 1.0, n)}, **kwargs)

    def test_histograms_and_histogram(self) -> None:
        a = self._sample(0.0, 200, 1, label="A", systematics={"lumi": 0.1})
        b = self._sample(0.5, 100, 2, label="B")
        s = self._sample(3.0, 100, 3, label="S")
        group = rf.Group([a, b], label="AB")
        (h,) = rf.histograms(group, "x", bins=(10, -4, 6))
        assert (h.label, h.sum_weights) == ("AB", 300)
        assert [h.label for h in rf.histograms([group, s], "x", bins=10)] == ["AB", "S"]
        labels = [h.label for h in rf.histograms({"VV": group, "Signal": s}, "x", bins=10)]
        assert labels == ["VV", "Signal"]
        assert group.label == "AB"
        plain = rf.histogram(group, "x", bins=(10, -4, 6))
        assert isinstance(plain, hist.Hist)
        np.testing.assert_allclose(plain.values(), h.values())
        with pytest.raises(SourceError, match="single sample or group"):
            rf.histogram([group, s], "x")

    def test_plot_treats_a_group_as_one_histogram(self) -> None:
        ww = self._sample(0.0, 400, 1, label="WW", xsec="2 pb", ngen=100_000)
        zz = self._sample(0.5, 200, 2, label="ZZ", xsec="8 pb", ngen=500_000)
        qq = self._sample(1.0, 300, 3, label="qq")
        tautau = self._sample(1.5, 100, 4, label="tautau")
        signal = self._sample(3.0, 300, 5, label="Signal")
        vv = rf.Group([ww, zz], label="VV", color="C0")
        background = rf.Group([vv, rf.Group([qq, tautau], label="Other")], label="Background")
        p = rf.plot(
            [background, signal],
            "x",
            bins=(20, -4, 6),
            lumi="5 ab^-1",
            stack=True,
            panel="s/sqrt(b)",
            reference="Background",
        )
        assert [h.label for h in p.histograms] == ["Background", "Signal"]
        legend = [t.get_text() for t in p.ax.get_legend().get_texts()]
        assert legend[:2] == ["Signal", "Background"]
        assert len(p.comparisons) == 1
        partial_stack = rf.plot(
            [background, signal],
            "x",
            bins=(20, -4, 6),
            lumi="5 ab^-1",
            stack=["Background"],
            panel="s/sqrt(b)",
        )
        assert len(partial_stack.comparisons) == 1
        assert partial_stack.stack is not None
        np.testing.assert_allclose(partial_stack.stack.values(), p.histograms[0].values())
        np.testing.assert_allclose(partial_stack.comparisons[0].values, p.comparisons[0].values)
        with pytest.raises(ValueError, match=r"labels: \['Background', 'Signal'\].*Group"):
            rf.plot([background, signal], "x", bins=10, lumi=1.0, stack=["VV"])
        assert len(rf.plot(background.components, "x", bins=10, lumi=1.0).histograms) == 2
        assert len(rf.plot(background.samples, "x", bins=10, lumi=1.0).histograms) == 4
        p = rf.plot([vv, signal], "x", bins=(20, -4, 6), lumi=1.0, panel="ratio", reference="VV")
        assert len(p.comparisons) == 1
        assert "VV" in panel_ylabel(p)

    def test_observed_group_and_normalisation(self) -> None:
        mc = rf.Group(
            [
                self._sample(0.0, 300, 1, label="A", systematics={"lumi": 0.1}),
                self._sample(1.0, 100, 2, label="B"),
            ],
            label="MC",
        )
        data = rf.Group(
            [
                self._sample(0.0, 200, 3, label="D1", is_data=True),
                self._sample(1.0, 100, 4, label="D2", is_data=True),
            ],
            label="Data",
        )
        p = rf.plot(mc, "x", bins=(10, -4, 5), observed=data, stack=True, panel="ratio")
        assert [(h.label, h.is_data) for h in p.histograms] == [("MC", False), ("Data", True)]
        assert p.histograms[1].sum_weights == 300
        assert p.uncertainty("MC").has_systematics
        assert not p.uncertainty("Data").has_systematics
        # simulation given as observed is marked as data, group and all
        observed = rf.Group([self._sample(0.0, 50, 5, label="X")], label="Obs")
        assert rf.plot(mc, "x", bins=10, observed=observed).histograms[1].is_data
        # normalised after summing: the components' totals weight the sum
        a = rf.Sample({"x": [0.25] * 3}, label="A")
        b = rf.Sample({"x": [0.75]}, label="B")
        (h,) = rf.histograms(rf.Group([a, b], label="AB"), "x", bins=(2, 0, 1), normalize=True)
        np.testing.assert_allclose(h.values(), [0.75, 0.25])
        p = rf.plot(rf.Group([a, b], label="AB"), "x", bins=(2, 0, 1), normalize=True)
        np.testing.assert_allclose(p.histograms[0].values(), [0.75, 0.25])

    @pytest.mark.parametrize("grouped", [False, True])
    def test_observed_conversion_clears_simulation_systematics(self, grouped: bool) -> None:
        sample = rf.Sample(
            {"x": [0.25, 0.75], "w": [2.0, 3.0]},
            label="A",
            selection="x > 0.5",
            weight="w",
            scale=2.0,
            systematics={"shape": "missing_weight", "lumi": 0.1},
        )
        observed = (
            rf.Group([rf.Group([sample], label="inner"), sample.replace(label="B")], label="Obs")
            if grouped
            else sample
        )
        p = rf.plot({"x": [0.5]}, "x", bins=(2, 0, 1), observed=observed)
        result = p.histograms[1]
        assert result.is_data
        assert not result.variations
        np.testing.assert_allclose(result.values(), [0.0, 12.0 if grouped else 6.0])
        assert not sample.is_data
        assert set(sample.systematics) == {"shape", "lumi"}

    def test_observed_conversion_preserves_data_samples(self) -> None:
        from rootfig.api._common import as_observed

        data = rf.Sample({"x": [0.5]}, label="Data", is_data=True)
        assert as_observed(data) is data
        group = rf.Group([rf.Group([data], label="inner")], label="outer")
        converted = as_observed(group)
        assert isinstance(converted, rf.Group)
        assert converted.samples[0] is data

    def test_stats_box_and_functions_that_take_samples_only(self) -> None:
        a = self._sample(0.0, 50, 1, label="A")
        group = rf.Group([a, self._sample(1.0, 50, 2, label="B")], label="AB")
        with pytest.raises(ValueError, match="groups have none"):
            rf.plot(group, "x", bins=10, stats=True)
        p = rf.plot([group, a], "x", bins=10, stats=True)
        boxes = [t.get_text() for t in p.ax.texts if "N = " in t.get_text()]
        assert len(boxes) == 1
        assert boxes[0].startswith("A\n")
        many: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = [
            (rf.summarize, ("x",), {}),
            (rf.cutflow, (["x > 0"],), {}),
            (rf.efficiency, ("x",), {"passed": "x > 0"}),
            (rf.profile, ("x", "x"), {}),
        ]
        one: list[tuple[Any, tuple[Any, ...]]] = [
            (rf.load, ("x",)),
            (rf.plot2d, ("x", "x")),
            (rf.correlation, (["x", "x"],)),
        ]
        refused = r"groups are not accepted here \(\['AB'\]\).*Pass group.samples"
        single = r"takes one sample and 'AB' is a group of 2; pass one of group.samples"
        for function, args, kwargs in many:
            with pytest.raises(TypeError, match=refused):
                function(group, *args, **kwargs)
            with pytest.raises(TypeError, match=refused):
                function([group], *args, **kwargs)
        for function, args in one:
            for data in (group, [group], {"AB": group}):
                with pytest.raises(TypeError, match=single):
                    function(data, *args)

    def test_per_object_label_survives_grouping(self) -> None:
        jagged = rf.Sample({"pt": ak.Array([[1.0, 2.0], [3.0], []])}, label="A")
        alone = rf.plot(jagged, "pt", bins=(4, 0, 4))
        group = rf.Group([jagged, jagged.replace(label="B")], label="AB")
        grouped = rf.plot(group, "pt", bins=(4, 0, 4))
        assert alone.ax.get_ylabel() == grouped.ax.get_ylabel() == "Entries"
        assert grouped.histograms[0].per_object
        assert not rf.plot(rf.Sample({"x": [1.0]}), "x", bins=(1, 0, 2)).histograms[0].per_object
        # a sum made by hand keeps the flag too
        from rootfig.histograms import sum_histograms

        parts = rf.histograms(group.samples, "pt", bins=(4, 0, 4))
        assert rf.plot(sum_histograms(parts), "pt").ax.get_ylabel() == "Entries"


class TestPreparedPlot:
    """plot() is prepare_plot() followed by draw_plot(); one preparation draws several ways."""

    def test_plot_is_prepare_then_draw(self, signal_file: Path) -> None:
        from rootfig.api.plots1d import PreparedPlot, draw_plot, prepare_plot

        prepared = prepare_plot(signal_file, "MET", tree="events", bins=(10, 0, 100), unit="GeV")
        assert isinstance(prepared, PreparedPlot)
        assert prepared.variable == rf.Variable("MET", bins=(10, 0, 100), unit="GeV")
        assert (prepared.xlabel, prepared.unit, prepared.lumi) == (None, "GeV", None)
        p = draw_plot(prepared, logy=True)
        assert p.histograms[0] is prepared.histograms[0]
        direct = rf.plot(
            signal_file, "MET", tree="events", bins=(10, 0, 100), unit="GeV", logy=True
        )
        np.testing.assert_array_equal(
            p.histograms[0].values(flow=True), direct.histograms[0].values(flow=True)
        )
        assert p.ax.get_xlabel() == direct.ax.get_xlabel() == "MET [GeV]"
        assert p.ax.get_ylabel() == direct.ax.get_ylabel()
        assert p.ax.get_yscale() == "log"
        # drawing again from the same preparation leaves it untouched
        again = draw_plot(prepared, normalize=True)
        assert again.histograms[0].normalization == "Normalised to unity"
        assert prepared.histograms[0].normalization is None
        assert prepared.histograms[0].integral == direct.histograms[0].integral

    def test_histogram_objects_and_missing_variable(self) -> None:
        from rootfig.api.plots1d import draw_plot, prepare_plot

        h = hist.Hist(hist.axis.Regular(4, 0, 1, name="x"), storage=hist.storage.Weight())
        h.fill([0.1, 0.5, 0.6])
        prepared = prepare_plot(h, label="Ready", xlabel="x [cm]")
        assert prepared.variable is None
        assert [item.label for item in prepared.histograms] == ["Ready"]
        assert draw_plot(prepared).ax.get_xlabel() == "x [cm]"
        with pytest.raises(TypeError, match="needs a variable"):
            prepare_plot({"x": [1.0, 2.0]})


class TestSelectiveStacking:
    @staticmethod
    def _hist(value: float, label: str, **kwargs: Any) -> rf.Histogram:
        h = hist.Hist(hist.axis.Regular(2, 0, 4), storage=hist.storage.Weight())
        h.view().value[:] = value
        h.view().variance[:] = value
        return rf.Histogram(h, label=label, **kwargs)

    def _signals(self, observed: bool) -> list[rf.Histogram]:
        hists = [
            self._hist(4, "B"),
            self._hist(6, "S1", color="red"),
            self._hist(10, "S2", color="blue"),
        ]
        if observed:
            hists.append(self._hist(100, "Data", is_data=True))
        return hists

    @staticmethod
    def _panel_colors(p: rf.Plot) -> list[str]:
        from matplotlib.container import ErrorbarContainer

        assert p.panel_ax is not None
        return [
            c.lines[0].get_color()
            for c in p.panel_ax.containers
            if isinstance(c, ErrorbarContainer)
        ]

    @pytest.mark.parametrize("observed", [False, True])
    def test_multiple_signals(self, observed: bool) -> None:
        p = rf.plot(self._signals(observed), stack=["B"], panel="s/sqrt(b)")
        assert len(p.comparisons) == 2
        np.testing.assert_allclose(p.comparisons[0].values, [3, 3])
        np.testing.assert_allclose(p.comparisons[1].values, [5, 5])
        assert self._panel_colors(p) == ["red", "blue"]

    @pytest.mark.parametrize("stack", [False, True])
    @pytest.mark.parametrize("observed", [False, True])
    def test_without_overlays_on_a_stack_the_last_is_the_signal(
        self, stack: bool, observed: bool
    ) -> None:
        hists = self._signals(observed)
        p = rf.plot(hists, stack=stack, panel="s/sqrt(b)")
        # the same roles spelled out: S2 overlaid on a stack of the others
        named = rf.plot(hists, stack=["B", "S1"], panel="s/sqrt(b)")
        assert len(p.comparisons) == 1
        np.testing.assert_allclose(p.comparisons[0].values, np.sqrt([10, 10]))  # 10 / sqrt(4 + 6)
        np.testing.assert_allclose(p.comparisons[0].values, named.comparisons[0].values)
        np.testing.assert_allclose(p.comparisons[0].errors, named.comparisons[0].errors)
        assert self._panel_colors(p) == ["blue"]

    def test_stack_uncertainty(self) -> None:
        from rootfig.histograms import sum_histograms, uncertainty

        samples = [
            rf.Sample({"x": [0.5, 1.5]}, label=label, scale=scale)
            for label, scale in [("A", 1), ("B", 2), ("S", 10)]
        ]
        p = rf.plot(samples, "x", bins=(2, 0, 2), stack=["B", "A"], systematics={"lumi": 0.02})
        assert p.stack is not None
        assert p.stack.label == "Total"
        expected = uncertainty(sum_histograms(p.histograms[:2]))
        np.testing.assert_allclose(p.stack.values(), expected.nominal)
        np.testing.assert_allclose(p.uncertainty().total_up, expected.total_up)
        np.testing.assert_allclose(p.uncertainty().components["lumi"], expected.components["lumi"])
        np.testing.assert_allclose(p.uncertainty("S").components["lumi"][0], [0.2, 0.2])
        assert rf.plot(p.histograms[0]).stack is None
        np.testing.assert_allclose(rf.plot(p.histograms[0]).uncertainty().nominal, [1, 1])

    @pytest.mark.parametrize("normalize", [True, "unity", "density", 10, 0.5, 0])
    def test_normalization_refused(self, normalize: Any) -> None:
        before = plt.get_fignums()
        with pytest.raises(ValueError, match="would normalise") as exc:
            rf.plot([self._hist(4, "A"), self._hist(6, "B")], stack=["A"], normalize=normalize)
        assert str(exc.value) == (
            f"normalize={normalize!r} would normalise every stacked histogram on its own, "
            "so the stack would not add up to a normalised total; compare shapes with "
            "stack=False, draw a combination as one histogram with rf.Group, "
            "or use normalize='width'"
        )
        assert plt.get_fignums() == before

    @pytest.mark.parametrize("normalize", [None, False, "width"])
    def test_normalization_allowed(self, normalize: Any) -> None:
        p = rf.plot([self._hist(4, "A"), self._hist(6, "B")], stack="A", normalize=normalize)
        assert p.stack is not None
        np.testing.assert_allclose(p.stack.values(), [2, 2] if normalize == "width" else [4, 4])

    @pytest.mark.parametrize("stack", ["missing", "Data", None, 1, [1], {"A"}])
    def test_invalid_stack_leaves_no_figure(self, stack: Any) -> None:
        before = plt.get_fignums()
        with pytest.raises((ValueError, TypeError)):
            rf.plot([self._hist(4, "A"), self._hist(6, "Data", is_data=True)], stack=stack)
        assert plt.get_fignums() == before

    def test_only_stack_requires_matching_bins(self) -> None:
        a = self._hist(4, "A")
        b = hist.Hist(hist.axis.Regular(3, 0, 4), storage=hist.storage.Weight()).fill([1, 2])
        p = rf.plot([a, rf.Histogram(b, "B")], stack="A")
        assert p.stack is not None
        np.testing.assert_allclose(p.stack.values(), a.values())
        with pytest.raises(BinningError, match="a stack"):
            rf.plot(p.histograms, stack=True)


@pytest.mark.parametrize("kind", ["overlay", "stack", "data", "systematics"])
@pytest.mark.parametrize("logy", [False, True])
@pytest.mark.parametrize("broken", [False, True])
def test_y_limits_follow_visible_bins(kind: str, logy: bool, broken: bool) -> None:
    h = hist.Hist(hist.axis.Regular(6, 0, 6), storage=hist.storage.Weight())
    h.view().value = [0.001, 1000, 10, 10, 1000, 0.001]
    h.view().variance = 0
    wrapped = rf.Histogram(
        h,
        label="h",
        is_data=kind == "data",
        variations={"s": (h * 1.1, h * 0.9)} if kind == "systematics" else None,
    )
    window = {"xlim": (2, 4), "xbreak": (2.5, 3.5)} if broken else {"xlim": (2, 4)}
    p = rf.plot(wrapped, stack=kind == "stack", logy=logy, style=rf.Style(legend=False), **window)
    low, high = p.ax.get_ylim()
    assert high == pytest.approx((11 if kind == "systematics" else 10) * (12 if logy else 1.2))
    assert low == pytest.approx(5 if logy else 0)
    p.close()


@pytest.mark.parametrize("panel", ["ratio", "s/sqrt(b)"])
@pytest.mark.parametrize("broken", [False, True])
def test_lower_panel_limits_follow_visible_bins(panel: Any, broken: bool) -> None:
    background = hist.Hist(hist.axis.Regular(6, 0, 6), storage=hist.storage.Weight())
    background.view().value = 1
    background.view().variance = 0
    signal = background.copy()
    signal.view().value = [100, 100, 1, 1, 100, 100]
    window = {"xlim": (0, 6), "xbreak": (1, 5)} if broken else {"xlim": (2, 4)}
    if broken:
        signal.view().value = [1, 100, 100, 100, 100, 1]
    p = rf.plot([background, signal], panel=panel, style=rf.Style(legend=False), **window)
    assert p.panel_ax is not None
    expected = (0.5, 1.5) if panel == "ratio" else (0, 1.25)
    np.testing.assert_allclose(p.panel_ax.get_ylim(), expected)
    p.close()


def test_book_crops_cached_histogram_independently(stored_dir: Path) -> None:
    source = stored_dir / "ZH_sel0_histo.root"
    variables = [
        rf.Variable("mz", bins=(10, 0, 100), name="low"),
        rf.Variable("mz", range=(100, 250), name="high"),
    ]
    for (_, plot), variable in zip(rf.PlotBook(source, variables).plots(), variables, strict=True):
        separate = rf.plot(source, variable)
        np.testing.assert_allclose(plot.histograms[0].edges, separate.histograms[0].edges)
        np.testing.assert_allclose(
            plot.histograms[0].values(flow=True), separate.histograms[0].values(flow=True)
        )
        np.testing.assert_allclose(
            plot.histograms[0].variances(flow=True), separate.histograms[0].variances(flow=True)
        )
        plot.close()
        separate.close()


@pytest.mark.parametrize("logy", [False, True])
def test_xbreak_excludes_hidden_extrema(logy: bool) -> None:
    h = hist.Hist(hist.axis.Regular(6, 0, 6), storage=hist.storage.Weight())
    h.view().value = [10, 10, 1000, 0.001, 10, 10]
    h.view().variance = 0
    p = rf.plot(h, xbreak=(2, 4), logy=logy, style=rf.Style(legend=False))
    np.testing.assert_allclose(p.ax.get_ylim(), (5, 120) if logy else (0, 12))
    p.close()


def _vertical_bars(ax: Any) -> list[tuple[float, float]]:
    """``(low, high)`` of the vertical error bars of the one errorbar container of ``ax``."""
    (container,) = ax.containers
    for collection in container.lines[2]:
        segments = collection.get_segments()
        if segments and all(np.isclose(seg[0][0], seg[1][0]) for seg in segments):
            return [(float(seg[0][1]), float(seg[1][1])) for seg in segments]
    return []


class TestDataErrors:
    """``data_errors``: the Poisson interval of observed counts, in the main and lower panel."""

    COUNTS = np.array([1.0, 4.0, 0.0, 9.0])

    def _mc(self) -> rf.Sample:
        rng = np.random.default_rng(5)
        return rf.Sample({"x": rng.uniform(0, 4, 800)}, label="MC", scale=0.01)

    def _data(self, weights: Any = None, **options: Any) -> rf.Sample:
        values = np.repeat([0.5, 1.5, 2.5, 3.5], self.COUNTS.astype(int))
        columns = {"x": values}
        if weights is not None:
            columns["w"] = np.broadcast_to(np.asarray(weights, dtype=float), values.shape)
            options["weight"] = "w"
        return rf.Sample(columns, label="Data", is_data=True, **options)

    @pytest.mark.parametrize("mode", [None, "sumw2"])
    def test_the_default_is_roots_square_root_of_the_counts(self, mode: Any) -> None:
        p = rf.plot(
            self._mc(), "x", bins=(4, 0, 4), observed=self._data(), panel="ratio", data_errors=mode
        )
        assert not p.histograms[-1].poisson  # TH1's kNormal: sqrt(N), 0 +- 0 when empty
        (ratio,) = p.comparisons
        for side in ratio.errors:
            np.testing.assert_allclose(side, np.sqrt(self.COUNTS) / p.histograms[0].values())
        assert ratio.errors[1][2] == 0.0
        p.close()

    @pytest.mark.parametrize("mode", ["poisson", "auto"])
    def test_poisson_draws_the_interval_of_counts_in_both_panels(self, mode: Any) -> None:
        p = rf.plot(
            self._mc(), "x", bins=(4, 0, 4), observed=self._data(), panel="ratio", data_errors=mode
        )
        data = p.histograms[-1]
        assert data.poisson
        low, high = poisson_interval(self.COUNTS)
        np.testing.assert_allclose(_vertical_bars(p.ax), np.c_[low, high])
        (ratio,) = p.comparisons
        mc = p.histograms[0].values()
        np.testing.assert_allclose(ratio.errors[0], (self.COUNTS - low) / mc)
        np.testing.assert_allclose(ratio.errors[1], (high - self.COUNTS) / mc)
        assert ratio.values[2] == 0.0
        assert ratio.errors[1][2] == pytest.approx(1.8410216450 / mc[2])  # empty: not 0 +- 0
        assert p.panel_ax is not None
        drawn = np.array(_vertical_bars(p.panel_ax))
        np.testing.assert_allclose(drawn[:, 0], ratio.values - ratio.errors[0])
        np.testing.assert_allclose(drawn[:, 1], ratio.values + ratio.errors[1])
        u = p.uncertainty("Data")
        np.testing.assert_allclose(u.stat_up, high - self.COUNTS)
        p.close()

    @pytest.mark.parametrize("options", [{"weight": "w"}, {"scale": 2.0}])
    def test_an_empty_selection_of_weighted_data_is_not_counts(self, options: Any) -> None:
        # nothing passes: the contents (0, 0) look like counts, but the weights filled them
        columns = {"x": np.arange(0.5, 4.0), "w": np.full(4, 0.5)}
        empty = rf.Sample(columns, label="Data", is_data=True, selection="x > 10", **options)
        auto = rf.plot(self._mc(), "x", bins=(4, 0, 4), observed=empty, data_errors="auto")
        assert not auto.histograms[-1].poisson
        np.testing.assert_array_equal(auto.histograms[-1].errors()[1], 0.0)  # not 0 +1.84
        auto.close()
        with pytest.raises(ValueError, match="'Data' was filled with weights or scaled"):
            rf.plot(self._mc(), "x", bins=(4, 0, 4), observed=empty, data_errors="poisson")
        # the same selection without weights holds no counts: 0 +1.84
        unweighted = rf.Sample(columns, label="Data", is_data=True, selection="x > 10")
        counts = rf.plot(self._mc(), "x", bins=(4, 0, 4), observed=unweighted, data_errors="auto")
        assert counts.histograms[-1].poisson
        np.testing.assert_allclose(counts.histograms[-1].errors()[1], 1.8410216450)
        counts.close()

    def test_an_empty_stored_histogram_is_judged_like_root(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.root"
        with uproot.recreate(path) as file:
            file["h"] = (np.zeros(3), np.array([0.0, 1.0, 2.0, 3.0]))
        # as ROOT, whose sums of weights and of their squares agree (both zero)
        stored = rf.plot(path, "h", observed=path, data_errors="auto")
        assert stored.histograms[-1].poisson
        scaled = rf.Sample(path, label="Data", is_data=True, scale=2.0)
        known = rf.plot(path, "h", observed=scaled, data_errors="auto")
        assert not known.histograms[-1].poisson  # read with a scale: not counts
        plt.close("all")

    @pytest.mark.parametrize("weights", [0.5, "varying"])
    def test_weighted_data_is_not_counts(self, weights: Any) -> None:
        # one weight for every event sums like unequal weights with the same totals: only
        # unit-weight counts are known to be counts
        if weights == "varying":
            weights = np.linspace(0.6, 1.4, int(self.COUNTS.sum()))
        weighted = self._data(weights=weights)
        auto = rf.plot(self._mc(), "x", bins=(4, 0, 4), observed=weighted, data_errors="auto")
        assert not auto.histograms[-1].poisson
        auto.close()
        with pytest.raises(ValueError, match=r"'Data' is weighted or scaled.*data_errors='sumw2'"):
            rf.plot(self._mc(), "x", bins=(4, 0, 4), observed=weighted, data_errors="poisson")

    def test_signed_weights_are_refused(self) -> None:
        signed = self._data(weights=np.where(np.arange(int(self.COUNTS.sum())) < 1, -1.0, 1.0))
        figures = plt.get_fignums()
        with pytest.raises(ValueError, match="negative contents"):
            rf.plot(self._mc(), "x", bins=(4, 0, 4), observed=signed, data_errors="poisson")
        assert plt.get_fignums() == figures  # refused before a figure exists
        auto = rf.plot(self._mc(), "x", bins=(4, 0, 4), observed=signed, data_errors="auto")
        assert not auto.histograms[-1].poisson
        auto.close()

    def test_the_model_is_decided_before_normalising(self) -> None:
        p = rf.plot(
            self._mc(),
            "x",
            bins=(4, 0, 4),
            observed=self._data(),
            normalize=True,
            data_errors="poisson",
        )
        data = p.histograms[-1]
        assert data.poisson
        _, high = poisson_interval(self.COUNTS)
        total = self.COUNTS.sum()
        np.testing.assert_allclose(data.errors()[1], (high - self.COUNTS) / total, rtol=1e-12)
        p.close()

    def test_stored_histograms(self, stored_dir: Path, tmp_path: Path) -> None:
        mc = stored_dir / "ZZ_sel0_histo.root"
        observed = stored_dir / "ZH_sel0_histo.root"
        # whole numbers without Sumw2: unweighted counts, as ROOT itself reads them
        raw = rf.plot(mc, "mz_raw", observed=observed, data_errors="auto")
        assert raw.histograms[-1].poisson
        assert not rf.plot(mc, "mz_raw", observed=observed).histograms[-1].poisson
        # Sumw2 with weights of 0.5: not unit-weight counts
        weighted = rf.plot(mc, "mz", observed=observed, data_errors="auto")
        assert not weighted.histograms[-1].poisson
        # contents that are not whole numbers, without Sumw2: sqrt(content), as ROOT draws them
        path = tmp_path / "fractional.root"
        with uproot.recreate(path) as file:
            file["h"] = (np.array([0.5, 2.5, 3.0]), np.array([0.0, 1.0, 2.0, 3.0]))
        fractional = rf.plot(path, "h", observed=path, data_errors="auto")
        assert not fractional.histograms[-1].poisson
        with pytest.raises(ValueError, match="not whole numbers"):
            rf.plot(path, "h", observed=path, data_errors="poisson")
        plt.close("all")

    def test_histogram_objects(self) -> None:
        counts = hist.Hist(hist.axis.Regular(4, 0, 4))  # a plain count storage
        counts.fill(np.repeat([0.5, 1.5, 3.5], [1, 4, 9]))
        expected = rf.Histogram(counts.copy(), label="MC")
        assert not rf.plot([expected], observed=[counts]).histograms[-1].poisson
        assert rf.plot([expected], observed=[counts], data_errors="auto").histograms[-1].poisson
        # a histogram carrying the model keeps it, scaled counts included, unless "sumw2"
        flagged = rf.Histogram(counts, label="Data", is_data=True, poisson=True).scaled(2.0)
        for mode in (None, "auto", "poisson"):
            kept = rf.plot([expected.scaled(2.0)], observed=[flagged], data_errors=mode)
            assert kept.histograms[-1].poisson
        forced = rf.plot([expected.scaled(2.0)], observed=[flagged], data_errors="sumw2")
        assert not forced.histograms[-1].poisson
        np.testing.assert_allclose(
            forced.histograms[-1].errors()[1], 2.0 * np.sqrt([1.0, 4.0, 0.0, 9.0])
        )
        with pytest.raises(ValueError, match="data_errors must be"):
            rf.plot([expected], observed=[counts], data_errors="garwood")  # type: ignore[arg-type]
        plt.close("all")


ERROR_OPTIONS = Path(__file__).parent / "data" / "error_options.root"


class TestStatisticalOptions:
    """Confidence levels, ROOT's saved error options, shapes and Poisson ratios through the api."""

    @staticmethod
    def _counts(label: str = "Data", **options: Any) -> rf.Sample:
        values = np.repeat([0.5, 1.5, 2.5, 3.5], [1, 4, 0, 9])
        return rf.Sample({"x": values}, label=label, **options)

    def test_data_errors_at_a_confidence_level(self) -> None:
        mc = self._counts("MC")
        p = rf.plot(mc, "x", bins=(4, 0, 4), observed=self._counts(is_data=True), data_errors=0.95)
        data = p.histograms[-1]
        assert data.poisson == 0.95
        low, high = poisson_interval([1.0, 4.0, 0.0, 9.0], 0.95)
        np.testing.assert_allclose(data.errors()[1], high - [1.0, 4.0, 0.0, 9.0])
        p.close()
        with pytest.raises(ValueError, match="confidence level between 0 and 1"):
            rf.plot(mc, "x", bins=(4, 0, 4), observed=self._counts(is_data=True), data_errors=2.0)

    def test_saved_error_options_are_the_default(self) -> None:
        stored = rf.plot(ERROR_OPTIONS, "normal", observed=rf.Sample(ERROR_OPTIONS, label="D"))
        assert not stored.histograms[-1].poisson  # kNormal
        poisson2 = rf.Sample(ERROR_OPTIONS, label="Data")
        # None and "auto" keep what was saved; an explicit choice replaces it
        for mode, level in ((None, 0.95), ("auto", 0.95), ("poisson", True), (0.9, 0.9)):
            p = rf.plot(ERROR_OPTIONS, "poisson2", observed=poisson2, data_errors=mode)
            assert p.histograms[-1].poisson == level, mode
            assert p.histograms[-1].poisson is not False
        forced = rf.plot(ERROR_OPTIONS, "poisson2", observed=poisson2, data_errors="sumw2")
        assert not forced.histograms[-1].poisson
        for side in forced.histograms[-1].errors():
            np.testing.assert_array_equal(side, [0.0, 1.0, 2.0])  # sqrt(N) on both sides
        # a histogram rootfig read from the file keeps it when given as an object
        read = rf.io.FileSource(ERROR_OPTIONS).read_histogram("poisson")
        assert rf.plot([read]).histograms[0].poisson is True
        plt.close("all")

    def test_shape_uncertainty(self) -> None:
        p = rf.plot(
            self._counts(), "x", bins=(4, 0, 4), normalize=True, normalize_uncertainty="shape"
        )
        fraction = np.array([1.0, 4.0, 0.0, 9.0]) / 14
        np.testing.assert_allclose(p.histograms[0].variances(), fraction * (1 - fraction) / 14)
        p.close()
        with pytest.raises(ValueError, match="own total"):
            rf.plot(self._counts(), "x", bins=(4, 0, 4), normalize_uncertainty="shape")
        (shape,) = rf.histograms(
            self._counts(), "x", bins=(4, 0, 4), normalize=True, normalize_uncertainty="shape"
        )
        np.testing.assert_allclose(shape.variances(), fraction * (1 - fraction) / 14)
        # a stored kPoisson TH1 (counts 0, 1, 4): the plain Hist returned has the shape's
        # variances, not those of a constant factor ([0, 0.04, 0.16])
        stored = rf.histogram(
            ERROR_OPTIONS, "poisson", normalize=True, normalize_uncertainty="shape"
        )
        np.testing.assert_allclose(stored.variances(), [0.0, 0.032, 0.032])

    def test_a_shape_keeps_its_flow_bins_apart(self) -> None:
        figures = plt.get_fignums()
        with pytest.raises(ValueError, match="flow='sum'"):
            rf.plot(
                self._counts(),
                "x",
                bins=(3, 0, 3),
                normalize=True,
                normalize_uncertainty="shape",
                flow="sum",
            )
        assert plt.get_fignums() == figures  # refused before a figure exists
        shown = rf.plot(
            self._counts(), "x", bins=(3, 0, 3), normalize=True, normalize_uncertainty="shape",
            flow="show",
        )  # fmt: skip
        shown.close()

    def test_poisson_ratio_panel(self) -> None:
        data = self._counts(is_data=True)
        mc = rf.Sample({"x": np.repeat([0.5, 1.5, 2.5, 3.5], [2, 3, 1, 7])}, label="MC")
        p = rf.plot(
            mc, "x", bins=(4, 0, 4), observed=data, panel="ratio", panel_uncertainty="poisson-ratio"
        )
        (ratio,) = p.comparisons
        assert ratio.uncertainty == "poisson-ratio"
        expected = rf.compare(p.histograms[-1], p.histograms[0], uncertainty="poisson-ratio")
        for got, want in zip(ratio.errors, expected.errors, strict=True):
            np.testing.assert_allclose(got, want)
        p.close()
        figures = plt.get_fignums()
        weighted = mc.replace(scale=0.5)
        with pytest.raises(ValueError, match="holds no known counts"):
            rf.plot(
                weighted,
                "x",
                bins=(4, 0, 4),
                observed=data,
                panel="ratio",
                panel_uncertainty="poisson-ratio",
            )
        assert plt.get_fignums() == figures  # refused before a figure exists

    def test_efficiency_methods_levels_and_empty_bins(self) -> None:
        x = np.array([0.5, 0.5, 0.5, 0.5])
        sample = rf.Sample({"x": x, "ok": np.array([1, 1, 1, 0])}, label="S")
        p = rf.efficiency(
            sample,
            "x",
            passed="ok == 1",
            bins=(2, 0, 2),
            interval=rf.Bayesian(1, 1),
            cl=0.95,
            show_empty=True,
        )
        (eff,) = p.efficiencies
        assert eff.values[0] == pytest.approx(4 / 6)  # posterior mean of 3 of 4, Beta(1, 1)
        assert eff.values[1] == 0.5  # the empty bin shows the prior
        from scipy.special import betaincinv

        np.testing.assert_allclose(eff.lower[0], betaincinv(4, 2, 0.025))
        p.close()
        table = rf.cutflow(sample, ["ok == 1"], interval="wilson", cl=0.95)
        assert table.get("S").cl == 0.95


def test_weights_of_zero_and_one_fill_counts() -> None:
    # as ROOT, whose TH1 counts as unweighted when its sum of weights equals the sum of their
    # squares: weight="1" or a 0/1 flag gives counts, a common factor does not
    values = np.repeat([0.5, 1.5, 2.5], [1, 4, 2])
    columns = {"x": values, "flag": (np.arange(values.size) % 3 != 0).astype(float)}
    mc = rf.Sample({"x": np.linspace(0.1, 2.9, 40)}, label="MC")
    for options, poisson in (
        ({}, True),
        ({"weight": "1"}, True),
        ({"weight": "flag"}, True),
        ({"weight": "flag", "scale": 2.0}, False),
        ({"weight": "0.5 * flag"}, False),
        ({"weight": "flag", "selection": "x > 10"}, True),  # empty, but its weights were 0 or 1
    ):
        data = rf.Sample(columns, label="Data", is_data=True, **options)
        p = rf.plot(mc, "x", bins=(3, 0, 3), observed=data, data_errors="auto")
        assert p.histograms[-1].poisson is poisson, options
        p.close()


class TestDataErrorPrecedence:
    """``None`` and ``"auto"`` keep a model of the histogram's own; explicit choices replace it."""

    COUNTS = [1.0, 4.0, 0.0]
    GIVEN = ([0.5, 1.0, 0.0], [1.0, 2.0, 1.0])

    def _drawn(self, observed: rf.Histogram, mode: Any) -> rf.Histogram:
        mc = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
        mc.fill([0.5, 1.5, 1.5, 2.5], weight=0.5)
        p = rf.plot([mc], observed=[observed], data_errors=mode)
        p.close()
        return p.histograms[-1]

    def _counts(self, values: list[float] | None = None) -> hist.Hist:
        h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
        view: Any = h.view()
        view.value = view.variance = self.COUNTS if values is None else values
        return h

    def test_errors_of_its_own(self) -> None:
        fit = rf.Histogram(self._counts(), label="Data", is_data=True, stat_errors=self.GIVEN)
        for mode in (None, "auto"):
            np.testing.assert_array_equal(self._drawn(fit, mode).errors(), self.GIVEN)
        for side in self._drawn(fit, "sumw2").errors():
            np.testing.assert_array_equal(side, np.sqrt(self.COUNTS))
        low, high = poisson_interval(self.COUNTS)
        for mode, level in (("poisson", True), (0.9, 0.9)):
            drawn = self._drawn(fit, mode)  # the contents are counts: the Poisson interval
            assert drawn.poisson == level
            assert drawn._errors is None
        np.testing.assert_allclose(self._drawn(fit, "poisson").errors()[1], high - self.COUNTS)

    def test_errors_of_its_own_on_contents_that_are_not_counts(self) -> None:
        weighted = self._counts([1.5, 4.0, 0.0])
        fit = rf.Histogram(weighted, label="Data", is_data=True, stat_errors=self.GIVEN)
        with pytest.raises(ValueError, match=r"not whole numbers.*data_errors='sumw2'"):
            self._drawn(fit, "poisson")
        assert not self._drawn(fit, "sumw2").poisson
        np.testing.assert_array_equal(self._drawn(fit, "auto").errors(), self.GIVEN)

    def test_scaled_counts_keep_what_they_are(self) -> None:
        # counts scaled by 2 and drawn with sqrt(sum w^2): still known counts, so the Poisson
        # interval can be asked for again; "auto" takes only unit-weight counts
        counts = rf.Histogram(self._counts(), label="Data", is_data=True, poisson=True)
        plain = counts.scaled(2.0).replace(poisson=False)
        assert not self._drawn(plain, "auto").poisson
        np.testing.assert_allclose(
            self._drawn(plain, "poisson").errors(), counts.scaled(2.0).errors()
        )


def test_a_confidence_level_replaces_a_saved_one() -> None:
    kept = rf.Sample(ERROR_OPTIONS, label="Data")
    p = rf.plot(ERROR_OPTIONS, "poisson", observed=kept, data_errors=0.9)
    assert p.histograms[-1].poisson == 0.9  # the counts are known: any level
    p.close()
