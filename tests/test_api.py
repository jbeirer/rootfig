"""End-to-end tests of the public API against generated ROOT files."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pytest
import uproot
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties

import rootfig as rf
from rootfig.api.plots2d import _split_bins
from rootfig.errors import (
    BinningError,
    IncompatibleWeightError,
    MissingBranchError,
    RootfigWarning,
    SelectionError,
    SourceError,
)
from rootfig.model.style import EXPERIMENT_STYLES
from rootfig.plotting import add_experiment_label, align_experiment_label, style_context
from rootfig.plotting.style import LABEL_MIN_SCALE


def ratio_ylabel(plot: Any) -> str:
    """The ratio panel's y label, ignoring the wrapping that makes it fit the panel."""
    return plot.ratio_ax.get_ylabel().replace("\n", " ")


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
        assert p.ratio_ax is None
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
            ratio=True,
            label=["Signal", "Background"],
            unit="GeV",
        )
        assert p.ratio_ax is not None
        assert len(p.ratios) == 1
        assert p.ax.get_ylabel() == "Normalised to unity"
        assert p.ratio_ax.get_xlabel() == "Muon_pt [GeV]"
        assert ratio_ylabel(p) == "Ratio to Signal"
        for h in p.histograms:
            assert h.integral == pytest.approx(1.0)
        ratio = p.ratios[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = p.histograms[1].values() / p.histograms[0].values()
        ok = np.isfinite(ratio.values)
        assert ratio.values[ok].tolist() == pytest.approx(expected[ok].tolist())

    def test_ratio_reference_by_label(self, signal_file: Path, background_file: Path) -> None:
        p = rf.plot(
            [signal_file, background_file], "MET", tree="events", bins=10, ratio="background"
        )
        assert p.ratio_ax is not None
        assert ratio_ylabel(p) == "Ratio to background"
        with pytest.raises(ValueError, match="not one of"):
            rf.plot([signal_file, background_file], "MET", tree="events", bins=10, ratio="nope")

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
            ratio=True,
            logy=True,
            style="ATLAS",
        )
        assert [h.label for h in p.histograms] == ["Signal", "Background", "Data"]
        assert p.histograms[2].is_data
        assert p.ax.get_yscale() == "log"
        assert p.ratio_ax is not None
        assert ratio_ylabel(p) == "Data / MC"
        legend_texts = [t.get_text() for t in p.ax.get_legend().get_texts()]
        assert legend_texts == ["Data", "Background", "Signal", "Stat. unc."]
        total = p.histograms[0].values() + p.histograms[1].values()
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.where(total > 0, p.histograms[2].values() / total, np.nan)
        ok = np.isfinite(expected)
        assert p.ratios[0].values[ok].tolist() == pytest.approx(expected[ok].tolist())

    @staticmethod
    def _marker_colors(p: rf.Plot) -> set[tuple[float, ...]]:
        lines = [*p.ax.lines, *(p.ratio_ax.lines if p.ratio_ax else [])]
        return {to_rgba(line.get_color()) for line in lines if line.get_marker() == "o"}

    def test_dark_theme_overrides_experiment_style(
        self, signal_file: Path, background_file: Path, tmp_path: Path
    ) -> None:
        mc = [rf.Sample(signal_file, tree="events", label="Signal")]
        observed = rf.Sample(background_file, tree="events", label="Data", entry_stop=1000)
        kwargs: dict[str, Any] = {"observed": observed, "bins": (20, 0, 200), "ratio": True}
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
            ratio=True,
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
                ratio=True,
            )

    def test_ratio_needs_two(self, signal_file: Path) -> None:
        with pytest.raises(ValueError, match="at least two"):
            rf.plot(signal_file, "MET", tree="events", bins=10, ratio=True)

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
            [bkg, sig], "MET", observed=sig, bins=(10, 0, 100), stack=True, ratio=True, flow="show"
        )
        edges = p.histograms[0].edges
        assert len(edges) == 12  # 10 bins plus the overflow bin
        assert all(len(h.edges) == 12 for h in p.histograms)
        assert p.ax.get_xlim()[1] == pytest.approx(edges[-1])
        assert p.ratios[0].edges[-1] == pytest.approx(edges[-1])
        assert p.ratio_ax is not None
        assert ">100" in [t.get_text() for t in p.ratio_ax.get_xticklabels()]
        assert p.ax.get_ylabel().startswith("Events / 10")  # width of the bins as filled
        with pytest.raises(ValueError, match="flow='show'"):
            rf.plot(
                signal_file, "MET", tree="events", bins=(10, 0, 100), xbreak=(20, 80), flow="show"
            )

    def test_existing_axes(self, signal_file: Path) -> None:
        fig, axes = plt.subplots(1, 2)
        p = rf.plot(signal_file, "MET", tree="events", bins=10, ax=axes[1], legend=False, title="t")
        assert p.ax is axes[1]
        assert p.fig is fig
        assert p.ax.get_legend() is None
        assert p.ax.get_title() == "t"
        fig, (main, lower) = plt.subplots(2)
        p = rf.plot(signal_file, "MET", tree="events", bins=10, ax=(main, lower), ratio="signal")
        assert p.ratio_ax is lower

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
        p = rf.plot([data, make(10.0, "A"), make(30.0, "B")], ratio=True)
        assert len(p.ratios) == 1  # only the data appears in the panel
        np.testing.assert_allclose(p.ratios[0].values, [2.0, 2.0])  # data / A, not data / total
        assert p.ratio_ax is not None
        assert ratio_ylabel(p) == "Data / A"
        stacked = rf.plot([data, make(10.0, "A"), make(30.0, "B")], ratio=True, stack=True)
        np.testing.assert_allclose(stacked.ratios[0].values, [0.5, 0.5])  # data / total


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
        assert p.ratio_ax is None
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
            ratio=True,
            logy=True,
        )
        assert p.ratio_ax is not None
        assert p.ratio_ax_right is not None
        assert len(p.axes) == 4
        assert p.ratio_ax.get_xlim() == (10.0, 50.0)
        assert p.ratio_ax_right.get_xlim() == (150.0, 240.0)
        assert ratio_ylabel(p) == "Ratio to signal"
        assert p.ratio_ax_right.get_ylabel() == ""
        assert p.ratio_ax_right.get_xlabel() == "MET"
        assert p.ax.get_xlabel() == p.ax_right.get_xlabel() == ""  # type: ignore[union-attr]
        assert p.ax.get_yscale() == "log"
        assert len(p.ratios) == 1

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
    def test_lost_variances_need_assume_poisson(self) -> None:
        weighted = hist.Hist(hist.axis.Regular(2, 0, 2)).fill([0.5, 1.5], weight=[2.0, -1.0])
        with pytest.raises(ValueError, match="assume_poisson=True"):
            rf.plot([weighted])
        with pytest.warns(RootfigWarning, match="Poisson guess"):
            p = rf.plot([weighted], assume_poisson=True)
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
        p = rf.plot([h1, h2], label=["one", "two"], ratio=True, normalize=True)
        assert [h.label for h in p.histograms] == ["one", "two"]
        assert p.ax.get_xlabel() == ""
        assert p.ratio_ax is not None
        assert p.ratio_ax.get_xlabel() == "x"
        assert p.ratios[0].values[:3].tolist() == pytest.approx([1.0, 1.0, 1.0])
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
            rf.plot([signal_file, background_file], "MET", tree="events", ratio=True),
            rf.plot(
                [signal_file, background_file],
                "MET",
                tree="events",
                ratio=True,
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
        p = rf.plot([signal_file, background_file], "MET", tree="events", ratio=True, style=atlas)
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
        p = rf.plot([signal_file, background_file], "MET", tree="events", ratio=True, style=atlas)
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
            rf.plot(samples, "MET", tree="events", ratio=True, xbreak=(40, 60), style=style),
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
        align_experiment_label(p.ax)  # as a user may, or another finalisation
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
        original = FileSource.arrays

        def counting(self: FileSource, branches: Any) -> Any:
            calls.append(list(branches))
            return original(self, branches)

        monkeypatch.setattr(FileSource, "arrays", counting)
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

    def test_evaluate_reexport(self) -> None:
        assert rf.evaluate("a + 1", {"a": ak.Array([1, 2])}).tolist() == [2, 3]

    def test_ratio_reexport(self) -> None:
        h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        h.fill([0.5, 0.5, 1.5])
        r = rf.ratio(h, h)
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

    def test_significance_panel(self, signal_file: Path, background_file: Path) -> None:
        sig = rf.Sample(signal_file, tree="events", label="S", scale=0.1)
        bkg = rf.Sample(background_file, tree="events", label="B")
        p = rf.plot([bkg, sig], "MET", bins=(10, 0, 200), stack=True, ratio="significance")
        assert p.ratio_ax is not None
        assert ratio_ylabel(p) == r"$S/\sqrt{B}$"
        assert len(p.ratios) == 1
        result = p.ratios[0]
        s = p.histograms[1].values()
        b = p.histograms[0].values()
        ok = b > 0
        np.testing.assert_allclose(result.values[ok], s[ok] / np.sqrt(b[ok]))
        p = rf.plot(
            [sig, bkg], "MET", bins=(10, 0, 200), ratio=("s/sqrt(s+b)", "S"), ratio_label="Z"
        )
        assert p.ratio_ax is not None
        assert ratio_ylabel(p) == "Z"
        with pytest.raises(ValueError, match="at least two"):
            rf.plot([sig], "MET", bins=(10, 0, 200), ratio="s/sqrt(b)")
        with pytest.raises(ValueError, match="is not one of"):
            rf.plot([sig, bkg], "MET", bins=(10, 0, 200), ratio=("s/sqrt(b)", "X"))
        with pytest.raises(ValueError, match="must use one of"):
            rf.plot([sig, bkg], "MET", bins=(10, 0, 200), ratio=("bogus", "S"))

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
        p = rf.plot([ref, num], label=["ref", "num"], flow="sum", ratio=True)
        np.testing.assert_allclose(p.ratios[0].values, [1.0, 21.0])
        assert p.ax.get_ylim()[1] > 21
        np.testing.assert_allclose(p.histograms[1].values(), [1.0, 21.0])
        stacked = rf.plot(
            [ref, num], label=["ref", "num"], flow="sum", stack=True, ratio="s/sqrt(b)"
        )
        np.testing.assert_allclose(stacked.ratios[0].values, [1.0, 21.0])

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
        with pytest.raises(BinningError, match="identical"):
            rf.plot(overlay.hists, ratio=True)

    def test_ratio_colors_follow_the_main_panel(self) -> None:
        from matplotlib.colors import to_rgba
        from matplotlib.container import ErrorbarContainer

        hists = [self._hist([0.5]), self._hist([0.5] * 2), self._hist([0.5] * 3)]
        p = rf.plot(hists, label=["A", "B", "C"], ratio=True)
        assert p.ratio_ax is not None
        ratio_colors = [
            to_rgba(c.lines[0].get_color())
            for c in p.ratio_ax.containers
            if isinstance(c, ErrorbarContainer)
        ]
        main_colors = [to_rgba(c) for c in rf.plotting.color_cycle(3, rf.Style())[1:]]
        assert ratio_colors == main_colors
        named = rf.plot(hists, label=["A", "B", "C"], ratio="C")
        assert named.ratio_ax is not None
        named_colors = [
            to_rgba(c.lines[0].get_color())
            for c in named.ratio_ax.containers
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


class TestOffsetText:
    """The x label stays clear of the axis' offset text, which shares its corner."""

    @staticmethod
    def _boxes(p: rf.Plot) -> tuple[Any, Any]:
        axis = (p.ratio_ax or p.ax).xaxis
        p.fig.canvas.draw()
        renderer = p.fig.canvas.get_renderer()
        offset = axis.get_offset_text()
        assert offset.get_text()  # the values need one
        return axis.label.get_window_extent(renderer), offset.get_window_extent(renderer)

    @pytest.mark.parametrize("ratio", [False, True])
    def test_label_and_offset_text_do_not_overlap(self, ratio: bool) -> None:
        values = np.random.default_rng(0).normal(2e-6, 1e-6, 5_000)
        samples = [
            rf.Sample({"x": values}, label="A"),
            rf.Sample({"x": values * 1.1}, label="B"),
        ]
        p = rf.plot(samples, "x", bins=20, ratio=ratio)
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
            ratio=True,
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
        assert np.all(total.total_up >= total.stat)
        assert p.ratios[0].syst_band is not None
        filled = total.nominal > 0
        np.testing.assert_allclose(
            p.ratios[0].syst_band[1][filled], total.syst_up[filled] / total.nominal[filled]
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
        p = rf.plot([h], ratio=False)
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
        p = rf.plot([reference, other, data], "x", bins=(2, 0, 2), ratio="Reference")
        other_ratio, data_ratio = p.ratios
        assert other_ratio.syst_errors is not None  # simulation / simulation: lumi cancels
        np.testing.assert_allclose(other_ratio.syst_errors, 0.0, atol=1e-12)
        np.testing.assert_allclose(other_ratio.errors**2, _propagated_variance(p))
        assert data_ratio.syst_errors is None  # data / simulation: the reference is the band
        assert data_ratio.syst_band is not None
        np.testing.assert_allclose(data_ratio.syst_band, 0.1)
        explicit = rf.plot(
            [reference, other, data],
            "x",
            bins=(2, 0, 2),
            ratio="Reference",
            ratio_uncertainty="numerator",
        )
        assert explicit.ratios[0].syst_errors is not None
        for side in explicit.ratios[0].syst_errors:  # its own lumi against the nominal reference
            np.testing.assert_allclose(side, 0.1 * explicit.ratios[0].values)

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

    def test_uncertainty_of_an_overlay_with_different_binnings_needs_a_label(self) -> None:
        a = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        b = hist.Hist(hist.axis.Regular(3, 0, 2), storage=hist.storage.Weight())
        p = rf.plot([rf.Histogram(a, "a"), rf.Histogram(b, "b")])
        with pytest.raises(BinningError, match=r"pass the label of one, e.g. uncertainty\('a'\)"):
            p.uncertainty()
        assert p.uncertainty("b").nominal.size == 3

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
        with pytest.raises(rf.BinningError, match="range of a histogram that already exists"):
            rf.plot2d(h2, rf.Variable("mass", bins=(5, 0.0, 50.0)))
        with pytest.raises(rf.BinningError, match="cannot be applied to a histogram"):
            rf.plot2d(h2, rf.Variable("mass", range=(0.0, 50.0)))
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
            rf.plot([a, b], label=["A", "B"], ratio=True)

    def test_significance_sums_backgrounds_whatever_their_axis_labels(
        self, stored_dir: Path
    ) -> None:
        # the stored titles differ between files (a placeholder title gives way to the name)
        samples = {
            "ZH": stored_dir / "ZH_sel0_histo.root",
            "M": stored_dir / "mixed_storage.root",
            "S": stored_dir / "WW_sel0_histo.root",
        }
        p = rf.plot(samples, "mz", stack=True, ratio="s/sqrt(b)")
        assert p.ratio_ax is not None
        assert np.nansum(p.ratios[0].values) > 0
        axis = hist.axis.Regular(4, 0, 4, name="x", label="A")
        other = hist.axis.Regular(4, 0, 4, name="y", label="B")
        a = hist.Hist(axis, storage=hist.storage.Weight()).fill([0.5, 1.5])
        b = hist.Hist(other, storage=hist.storage.Weight()).fill([1.5, 2.5])
        s = hist.Hist(axis, storage=hist.storage.Weight()).fill([2.5, 3.5])
        p = rf.plot([a, b, s], label=["A", "B", "S"], ratio=("s/sqrt(b)", "S"))
        np.testing.assert_allclose(p.ratios[0].values, [0.0, 0.0, 1.0, np.nan])

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
        p = rf.plot(mc, label="MC", observed=data, ratio=True)
        assert [h.label for h in p.histograms] == ["MC", "x"]
        assert p.histograms[1].is_data
        assert p.ax.get_ylabel() == "Events / 1 cm"
        assert p.ratio_ax is not None
        p = rf.plot([mc], rf.Variable("x", label="$x$", unit="cm", log=True))
        assert p.ax.get_xscale() == "log"
        assert p.ratio_ax is None
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
        with pytest.raises(BinningError, match="range of a histogram that already exists is fixed"):
            rf.plot(mc, rf.Variable("x", bins=2, range=(0, 100)))
        with pytest.raises(BinningError, match="axis range is fixed"):
            rf.plot(mc, range=(0, 2))
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
            ratio=("s/sqrt(b)", "Signal"),
        )
        assert [h.label for h in p.histograms] == ["Background", "Signal"]
        legend = [t.get_text() for t in p.ax.get_legend().get_texts()]
        assert legend[:2] == ["Signal", "Background"]
        assert len(p.ratios) == 1
        assert len(rf.plot(background.components, "x", bins=10, lumi=1.0).histograms) == 2
        assert len(rf.plot(background.samples, "x", bins=10, lumi=1.0).histograms) == 4
        p = rf.plot([vv, signal], "x", bins=(20, -4, 6), lumi=1.0, ratio="VV")
        assert len(p.ratios) == 1
        assert "VV" in ratio_ylabel(p)

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
        p = rf.plot(mc, "x", bins=(10, -4, 5), observed=data, stack=True, ratio=True)
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
