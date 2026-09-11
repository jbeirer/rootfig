"""End-to-end tests of the public API against generated ROOT files."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
import pytest

import rootfig as rf
from rootfig.errors import (
    BinningError,
    IncompatibleWeightError,
    MissingBranchError,
    RootfigWarning,
    SelectionError,
    SourceError,
)


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
        auto = rf.histogram(signal_file, "sentinel", tree="events", bins=20)
        robust = rf.histogram(signal_file, "sentinel", tree="events", bins=20, range="robust")
        assert auto.axes[0].edges[0] <= -999
        assert robust.axes[0].edges[0] > -10
        assert robust.values(flow=True)[0] > 0  # sentinels went to underflow

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
        assert p.ratio_ax.get_ylabel() == "Ratio to Signal"
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
        assert p.ratio_ax.get_ylabel() == "Ratio to background"
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
        assert p.ratio_ax.get_ylabel() == "Data / MC"
        legend_texts = [t.get_text() for t in p.ax.get_legend().get_texts()]
        assert legend_texts == ["Data", "Background", "Signal", "Stat. unc."]
        total = p.histograms[0].values() + p.histograms[1].values()
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.where(total > 0, p.histograms[2].values() / total, np.nan)
        ok = np.isfinite(expected)
        assert p.ratios[0].values[ok].tolist() == pytest.approx(expected[ok].tolist())

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
        p = rf.plot(signal_file, "MET", tree="events", bins=10, style=style, figsize=(4, 3))
        assert tuple(p.fig.get_size_inches()) == (4, 3)
        assert p.histograms[0].hist is not None
        assert any("CMS" in t.get_text() for t in p.ax.texts)

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
        assert p.ratio_ax.get_ylabel() == "Ratio to signal"
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
    def test_raw_hists(self) -> None:
        h1 = hist.Hist(
            hist.axis.Regular(5, 0, 5, name="x", label="x"), storage=hist.storage.Weight()
        )
        h1.fill([0.5, 1.5, 2.5])
        h2 = h1 * 2
        p = rf.plot_histograms([h1, h2], labels=["one", "two"], ratio=True, normalize=True)
        assert [h.label for h in p.histograms] == ["one", "two"]
        assert p.ax.get_xlabel() == ""
        assert p.ratio_ax is not None
        assert p.ratio_ax.get_xlabel() == "x"
        assert p.ratios[0].values[:3].tolist() == pytest.approx([1.0, 1.0, 1.0])
        p = rf.plot_histograms([h1], xlabel="custom")
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
            rf.plot_histograms([h], normalize=True)

    def test_errors(self) -> None:
        h1 = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        with pytest.raises(ValueError, match="no histograms"):
            rf.plot_histograms([])
        with pytest.raises(ValueError, match="labels"):
            rf.plot_histograms([h1], labels=["a", "b"])
        h2 = hist.Hist(hist.axis.Regular(2, 0, 2), hist.axis.Regular(2, 0, 2))
        with pytest.raises(ValueError, match="one-dimensional"):
            rf.plot_histograms([h2])


class TestPlot2D:
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


class TestSummaryAndCorrelation:
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
        assert p.ratio_ax.get_ylabel() == r"$S/\sqrt{B}$"
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
        assert p.ratio_ax.get_ylabel() == "Z"
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
            tree = file.mktree("events", {"x": ("float64", (3,)), "w": "float64"})
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
        p = rf.plot_histograms([ref, num], labels=["ref", "num"], flow="sum", ratio=True)
        np.testing.assert_allclose(p.ratios[0].values, [1.0, 21.0])
        assert p.ax.get_ylim()[1] > 21
        np.testing.assert_allclose(p.histograms[1].values(), [1.0, 21.0])
        stacked = rf.plot_histograms(
            [ref, num], labels=["ref", "num"], flow="sum", stack=True, ratio="s/sqrt(b)"
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
        p = rf.plot_histograms([plain], normalize="width")
        np.testing.assert_allclose(p.hists[0].values(), [0.5, 0.5])
        overlay = rf.plot_histograms(
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
            rf.plot_histograms(overlay.hists, ratio=True)

    def test_ratio_colors_follow_the_main_panel(self) -> None:
        from matplotlib.colors import to_rgba
        from matplotlib.container import ErrorbarContainer

        hists = [self._hist([0.5]), self._hist([0.5] * 2), self._hist([0.5] * 3)]
        p = rf.plot_histograms(hists, labels=["A", "B", "C"], ratio=True)
        assert p.ratio_ax is not None
        ratio_colors = [
            to_rgba(c.lines[0].get_color())
            for c in p.ratio_ax.containers
            if isinstance(c, ErrorbarContainer)
        ]
        main_colors = [to_rgba(c) for c in rf.plotting.color_cycle(3, rf.Style())[1:]]
        assert ratio_colors == main_colors
        named = rf.plot_histograms(hists, labels=["A", "B", "C"], ratio="C")
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
        np.testing.assert_allclose(
            rf.plot_histograms([no_flow], flow="show").hists[0].values(), [1.0, 1.0]
        )
        negative = self._hist([0.5, 3.0], [1.0, -2.0])
        np.testing.assert_allclose(
            rf.plot_histograms([negative], flow="show").hists[0].values(), [1.0, 0.0, -2.0]
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
