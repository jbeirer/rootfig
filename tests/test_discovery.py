"""PlotBook variable discovery: ``rf.ALL`` with ``include=``/``exclude=``.

Discovery reads metadata only (branch types, object classes, array types) and
hands the book an ordinary tuple of variables, which then runs through the same
batched execution as an explicit list.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import awkward as ak
import hist
import numpy as np
import pytest
import uproot

import rootfig as rf
from rootfig.api import batch
from rootfig.io import ArraySource, FileSource

DATA = Path(__file__).parent / "data"


def _columns() -> dict[str, Any]:
    """Columns of every plottable kind, a string and an oddly named branch."""
    return {
        "Muon_pt": ak.Array([[10.0, 20.0], [], [30.0]]),
        "Muon_charge": ak.Array([[1, -1], [], [1]]),
        "Muon_isTight": ak.Array([[True, False], [], [True]]),
        "Electron_pt": ak.Array([[5.0], [7.0, 8.0], []]),
        "MET": np.array([12.0, 30.0, 7.5]),
        "MET_cov": np.array([1.0, 2.0, 3.0]),
        "nMuon": np.array([2, 0, 1], dtype=np.int32),
        "run": np.array([1, 1, 2], dtype=np.uint32),
        "passTrigger": np.array([True, False, True]),
        "jetIndex": np.array([0, 1, 0], dtype=np.int64),
        "jet1_b-tag": np.array([0.1, 0.9, 0.5]),
        "name": ak.Array(["a", "bb", "ccc"]),
    }


PLOTTABLE = sorted(name for name in _columns() if name != "name")
"""The columns of :func:`_columns` a histogram can be filled from, sorted by name."""


def _write_tree(path: Path, columns: dict[str, Any], *, rntuple: bool) -> None:
    arrays = {k: v if isinstance(v, ak.Array) else ak.Array(v) for k, v in columns.items()}
    with uproot.recreate(path) as file:
        if rntuple:
            file["events"] = arrays
        else:
            file.mktree("events", arrays)  # the branch types are inferred from the arrays


@pytest.fixture(scope="module")
def ttree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("discovery") / "ttree.root"
    _write_tree(path, _columns(), rntuple=False)
    return path


@pytest.fixture(scope="module")
def rntuple(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("discovery") / "rntuple.root"
    _write_tree(path, _columns(), rntuple=True)
    return path


@pytest.fixture(scope="module")
def nested_rntuple(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An RNTuple with a collection of records and a plain record next to flat fields."""
    path = tmp_path_factory.mktemp("discovery") / "nested.root"
    with uproot.recreate(path) as file:
        file["events"] = {
            "Muon": ak.zip({"pt": [[1.0, 2.0], [3.0]], "q": [[1, -1], [1]]}),
            "Vertex": ak.Array([{"x": 1.0, "tag": "a"}, {"x": 2.0, "tag": "b"}]),
            "met": np.array([1.0, 2.0]),
            "label": ak.Array(["x", "y"]),
        }
    return path


@pytest.fixture(scope="module")
def dotted_rntuple(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An RNTuple whose field names hold dots: a plain field, a record and a record's field."""
    path = tmp_path_factory.mktemp("discovery") / "dotted.root"
    with uproot.recreate(path) as file:
        file["events"] = {
            "a.b": np.array([1.0, 2.0, 3.0]),
            "c.d": ak.zip({"e": [1.0, 2.0, 3.0], "f.g": [1, 2, 3]}),
            "Muon": ak.zip({"pt": [[1.0], [], [2.0]]}),
        }
    return path


@pytest.fixture(scope="module")
def shared_trees(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two trees that share the branch ``x``; ``a`` also holds ``y``, ``b`` also ``z``."""
    path = tmp_path_factory.mktemp("discovery") / "shared.root"
    with uproot.recreate(path) as file:
        file.mktree("a", {"x": np.float64, "y": np.float64}).extend(
            {"x": np.arange(3.0), "y": np.arange(3.0)}
        )
        file.mktree("b", {"x": np.float64, "z": np.float64}).extend(
            {"x": np.arange(3.0), "z": np.arange(3.0)}
        )
    return path


def _names(book: rf.PlotBook) -> list[str]:
    return [variable.expression for variable in book.variables]


def _reads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the branches of every ``FileSource.arrays`` call."""
    calls: list[list[str]] = []
    original = FileSource.arrays

    def counting(self: FileSource, branches: Any) -> Any:
        calls.append(list(branches))
        return original(self, branches)

    monkeypatch.setattr(FileSource, "arrays", counting)
    return calls


def _forbid_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every method that reads entries or bin contents fail."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        msg = "discovery must not read data"
        raise AssertionError(msg)

    for owner, name in (
        (FileSource, "arrays"),
        (FileSource, "read_histogram"),
        (FileSource, "read_histograms"),
        (ArraySource, "arrays"),
        # uproot's own readers, so a direct call bypassing rootfig's wrappers fails too
        (uproot, "concatenate"),
        (uproot, "iterate"),
        (uproot.behaviors.TBranch.HasBranches, "arrays"),
        (uproot.behaviors.TBranch.TBranch, "array"),
        (uproot.behaviors.RNTuple.HasFields, "arrays"),
        (uproot.behaviors.TH1.TH1, "to_hist"),
        (uproot.behaviors.TH1.TH1, "values"),
        (uproot.behaviors.TH2.TH2, "to_hist"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


def _opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the file name of every ``uproot.open`` call made with a path."""
    calls: list[str] = []
    original = uproot.open

    def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, str | os.PathLike):
            calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(uproot, "open", counting)
    return calls


class TestSentinel:
    def test_public_sentinel(self) -> None:
        assert repr(rf.ALL) == "ALL"
        assert rf.ALL is batch.ALL
        assert "ALL" in rf.__all__
        assert rf.ALL != "all"  # a branch or histogram may be called "all"

    def test_positional_and_keyword(self, ttree: Path) -> None:
        positional = rf.PlotBook(ttree, rf.ALL)
        keyword = rf.PlotBook(ttree, variables=rf.ALL)
        assert positional.variables == keyword.variables
        assert all(isinstance(v, rf.Variable) for v in positional.variables)
        assert isinstance(positional.variables, tuple)

    def test_explicit_lists_are_unchanged(self, ttree: Path) -> None:
        assert rf.PlotBook(ttree, "MET").variables == (rf.Variable("MET"),)
        assert rf.PlotBook(ttree, ["MET", "run"]).variables == (
            rf.Variable("MET"),
            rf.Variable("run"),
        )
        # a string is never the sentinel, whatever it spells
        assert rf.PlotBook({"all": [1.0]}, "all").variables == (rf.Variable("all"),)


class TestTrees:
    def test_ttree_plottable_leaves_only(self, ttree: Path) -> None:
        book = rf.PlotBook(ttree, rf.ALL)
        names = _names(book)
        # uproot writes one int32 counter branch per jagged branch; they are real branches
        counters = sorted(set(FileSource(ttree).branches()) - set(_columns()))
        assert counters
        assert all(c.startswith("n") for c in counters)
        assert names == [
            "`jet1_b-tag`" if n == "jet1_b-tag" else n for n in sorted(PLOTTABLE + counters)
        ]
        assert "name" not in names  # a string branch
        assert [v.safe_name for v in book.variables if "b-tag" in v.expression] == ["jet1_b_tag"]

    def test_rntuple_plottable_fields_only(self, rntuple: Path, ttree: Path) -> None:
        names = _names(rf.PlotBook(rntuple, rf.ALL))
        assert names == ["`jet1_b-tag`" if n == "jet1_b-tag" else n for n in PLOTTABLE]
        # the same schema as the TTree, up to the counter branches only a TTree writes
        from_ttree = _names(rf.PlotBook(ttree, rf.ALL, exclude=["n*"]))
        assert [n for n in from_ttree if n != "nMuon"] == [n for n in names if n != "nMuon"]

    def test_rntuple_nested_numeric_leaves(self, nested_rntuple: Path) -> None:
        book = rf.PlotBook(nested_rntuple, rf.ALL)
        assert _names(book) == ["Muon.pt", "Muon.q", "Vertex.x", "met"]
        for task, result in book.plots():
            assert result.hists[0].sum().value == (3.0 if "Muon" in task.stem else 2.0)
            result.close()

    def test_rntuple_fields_named_with_dots(self, dotted_rntuple: Path) -> None:
        # a dot in a field's own name is not nesting: the schema decides, level by level
        book = rf.PlotBook(dotted_rntuple, rf.ALL)
        assert _names(book) == ["Muon.pt", "a.b", "c.d.e", "c.d.f.g"]
        for task, result in book.plots():
            direct = rf.plot(dotted_rntuple, task.variable)
            np.testing.assert_array_equal(
                result.hists[0].values(flow=True), direct.hists[0].values(flow=True)
            )
            assert result.hists[0].sum().value == (2.0 if task.stem == "Muon_pt" else 3.0)
            result.close()
            direct.close()

    @pytest.mark.parametrize("literal_numeric", [True, False])
    def test_discovery_uses_the_type_of_the_resolved_dotted_field(
        self, tmp_path: Path, literal_numeric: bool
    ) -> None:
        literal, nested = (1.0, "text") if literal_numeric else ("text", 1.0)
        path = tmp_path / "collision.root"
        with uproot.recreate(path) as file:
            file["events"] = ak.Array([{"a.b": literal, "a": {"b": nested}, "ok": 2.0}])
        book = rf.PlotBook(path, rf.ALL)
        assert _names(book) == (["a.b", "ok"] if literal_numeric else ["ok"])
        for _, result in book.plots():
            assert result.hists[0].sum(flow=True).value == 1.0
            result.close()

    def test_discovered_nested_field_survives_an_overlapping_scalar(self, tmp_path: Path) -> None:
        path = tmp_path / "prefix.root"
        with uproot.recreate(path) as file:
            file["events"] = ak.Array([{"a": {"b": {"c": 1.0}}, "a.b": 10.0}])
        book = rf.PlotBook(path, rf.ALL)
        assert _names(book) == ["a.b", "a.b.c"]
        for _, result in book.plots():
            assert result.hists[0].sum(flow=True).value == 1.0
            result.close()

    def test_split_collection_ttree_and_tparameter(self) -> None:
        # dotted leaves of a podio-style split branch; the TParameter next to the tree is not one
        book = rf.PlotBook(DATA / "split_collection.root", rf.ALL, plot_kwargs={"tree": "events"})
        assert _names(book) == [
            "ReconstructedParticles.charge",
            "ReconstructedParticles.energy",
            "ReconstructedParticles.momentum.x",
            "ReconstructedParticles.momentum.y",
            "ReconstructedParticles.momentum.z",
        ]
        assert "eventsProcessed" not in _names(rf.PlotBook(DATA / "split_collection.root", rf.ALL))

    def test_sorted_by_source_name(self, ttree: Path) -> None:
        names = _names(rf.PlotBook(ttree, rf.ALL, exclude="n*"))
        originals = [n.strip("`") for n in names]
        assert originals == sorted(originals)
        assert originals[:3] == ["Electron_pt", "MET", "MET_cov"]

    def test_several_trees_need_tree(self, stored_dir: Path) -> None:
        with pytest.raises(rf.SourceError, match=r"several trees found .*Pass tree=..."):
            rf.PlotBook(stored_dir / "two_trees.root", rf.ALL)
        assert _names(
            rf.PlotBook(stored_dir / "two_trees.root", rf.ALL, plot_kwargs={"tree": "a"})
        ) == ["x"]
        assert _names(rf.PlotBook(f"{stored_dir / 'two_trees.root'}:b", rf.ALL)) == ["y"]

    def test_missing_tree_is_reported(self, ttree: Path) -> None:
        with pytest.raises(rf.SourceError, match="tree 'nope' not found"):
            rf.PlotBook(ttree, rf.ALL, plot_kwargs={"tree": "nope"})


class TestNoDataReads:
    def test_branch_schema_is_fetched_once_for_a_wide_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "wide.root"
        with uproot.recreate(path) as file:
            file["events"] = {f"x{i:03d}": np.arange(2.0) for i in range(100)}
        original = FileSource.branch_forms
        calls = 0

        def counted(source: FileSource) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return original(source)

        monkeypatch.setattr(FileSource, "branch_forms", counted)
        _forbid_reads(monkeypatch)
        book = rf.PlotBook(path, rf.ALL, variants={"lin": {}, "log": {"logy": True}})
        assert len(book.variables) == 100
        assert calls == 1

    def test_construction_and_tasks_read_no_data(
        self, monkeypatch: pytest.MonkeyPatch, stored_dir: Path, ttree: Path
    ) -> None:
        _forbid_reads(monkeypatch)
        for data in (
            ttree,
            stored_dir / "ZH_sel0_histo.root",
            stored_dir / "tree_without_branch.root",
            {"x": [1.0, 2.0], "s": ["a", "b"]},
            [rf.Sample(ttree, label="A"), rf.Sample(ttree, label="B", weight="MET")],
        ):
            book = rf.PlotBook(data, rf.ALL, variants={"lin": {}, "log": {"logy": True}})
            assert book.tasks()
            assert book.select(variables=book.variables[0].safe_name).tasks()
        with_cut = rf.PlotBook(ttree, rf.ALL, selections={"all": None, "cut": "MET > 1"})
        assert len(with_cut.tasks()) == 2 * len(with_cut.variables)

    def test_each_file_is_opened_a_bounded_number_of_times(
        self, monkeypatch: pytest.MonkeyPatch, ttree: Path, data_dir: Path
    ) -> None:
        opens = _opens(monkeypatch)
        # the same file for two samples, a group and observed data: one survey
        samples = [rf.Sample(ttree, label="A"), rf.Sample(ttree, label="B")]
        rf.PlotBook(
            [rf.Group(samples, label="AB"), ttree],
            rf.ALL,
            plot_kwargs={"observed": ttree},
            variants={"lin": {}, "obs": {"observed": rf.Sample(ttree, is_data=True)}},
        )
        assert 0 < len(opens) <= 2  # the object listing and the tree schema
        # a multi-file sample is surveyed through its first file only
        opens.clear()
        book = rf.PlotBook(
            rf.Sample([data_dir / "bkg_part1.root", data_dir / "bkg_part2.root"], tree="events"),
            rf.ALL,
        )
        assert set(opens) == {"bkg_part1.root"}
        assert "MET" in _names(book)


class TestStoredHistograms:
    def test_only_1d_histograms(self, stored_dir: Path) -> None:
        # TH1s (a one-bin one and a labelled one included); the TH2 is not a 1D variable
        book = rf.PlotBook(stored_dir / "ZH_sel0_histo.root", rf.ALL)
        assert _names(book) == ["cutflow", "eventsProcessed", "mz", "mz_raw"]
        for task, result in book.plots():
            direct = rf.plot(stored_dir / "ZH_sel0_histo.root", task.variable)
            np.testing.assert_array_equal(
                result.hists[0].values(flow=True), direct.hists[0].values(flow=True)
            )
            result.close()
            direct.close()

    def test_unsupported_objects_are_not_variables(self, stored_dir: Path) -> None:
        # a TProfile and a TH3 only: nothing to plot, and the message says what was found
        with pytest.raises(rf.SourceError, match="found nothing that every task can plot") as info:
            rf.PlotBook(stored_dir / "unsupported.root", rf.ALL)
        assert str(info.value).splitlines()[1] == (
            "'unsupported' (unsupported.root): no plottable branches; "
            "not plottable ['h3 (TH3D)', 'prof (TProfile)']"
        )
        # next to a tree they are ignored; the branch prof takes that name
        book = rf.PlotBook(stored_dir / "unsupported_with_tree.root", rf.ALL)
        assert _names(book) == ["prof", "pt"]

    def test_histogram_in_directory(self, stored_dir: Path) -> None:
        book = rf.PlotBook(stored_dir / "in_directory.root", rf.ALL)
        assert _names(book) == ["`sub/mz`"]
        assert [t.stem for t in book.tasks()] == ["sub_mz"]
        ((_, result),) = list(book.plots())
        assert result.hists[0].sum().value == 500
        result.close()
        # patterns see the original name
        assert _names(rf.PlotBook(stored_dir / "in_directory.root", rf.ALL, include="sub/*")) == [
            "`sub/mz`"
        ]

    def test_tree_and_histograms_in_one_file(self, stored_dir: Path) -> None:
        # a branch pt and a stored mz that no branch shadows: both, each its own way
        book = rf.PlotBook(stored_dir / "tree_without_branch.root", rf.ALL)
        assert _names(book) == ["mz", "pt"]
        results = {task.stem: result for task, result in book.plots()}
        assert results["mz"].hists[0].sum().value == 500  # the stored histogram
        assert results["pt"].hists[0].sum().value == 3  # the tree
        for result in results.values():
            result.close()

    def test_branch_wins_over_same_named_histogram(self, stored_dir: Path) -> None:
        book = rf.PlotBook(stored_dir / "branch_and_histogram.root", rf.ALL)
        assert _names(book) == ["mz"]
        ((_, result),) = list(book.plots())
        direct = rf.plot(stored_dir / "branch_and_histogram.root", "mz")
        assert result.hists[0].sum().value == 300  # the tree's entries, not the 500 stored
        np.testing.assert_array_equal(
            result.hists[0].values(flow=True), direct.hists[0].values(flow=True)
        )
        result.close()
        direct.close()

    def test_explicit_tree_means_branches_only(self, stored_dir: Path) -> None:
        book = rf.PlotBook(
            stored_dir / "tree_without_branch.root", rf.ALL, plot_kwargs={"tree": "events"}
        )
        assert _names(book) == ["pt"]
        as_sample = rf.Sample(stored_dir / "tree_without_branch.root", tree="events")
        assert _names(rf.PlotBook(as_sample, rf.ALL)) == ["pt"]
        ranged = rf.Sample(stored_dir / "tree_without_branch.root", entry_stop=2)
        assert _names(rf.PlotBook(ranged, rf.ALL)) == ["pt"]

    @pytest.mark.parametrize(
        ("options", "reason"),
        [
            ({"selections": {"sr": "pt > 1"}}, "selections= needs event data"),
            ({"plot_kwargs": {"weight": "pt"}}, "weight= needs event data"),
            ({"plot_kwargs": {"nonfinite": "error"}}, "nonfinite='error' needs event data"),
            ({"plot_kwargs": {"systematics": {"w": "pt"}}}, "systematic 'w' .* varies the weight"),
            (
                {"plot_kwargs": {"systematics": {"r": {"pt": ("pt", "pt")}}}},
                "systematic 'r' .* varies branches",
            ),
            ({"variants": {"a": {}, "b": {"weight": "pt"}}}, "weight= needs event data"),
            ({"plot_kwargs": {"stats": True}}, "stats= needs the unbinned statistics"),
            ({"plot_kwargs": {"stats": "upper left"}}, "stats= needs the unbinned statistics"),
            ({"variants": {"a": {}, "b": {"stats": True}}}, "stats= needs the unbinned statistics"),
            (
                {"plot_kwargs": {"range": (0.0, 100.0)}},
                r"range=\(low, high\) without bins= cannot be applied",
            ),
        ],
    )
    def test_event_data_options_rule_stored_histograms_out(
        self, stored_dir: Path, options: dict[str, Any], reason: str
    ) -> None:
        # with a tree next to them only the branches remain, and every task runs ...
        mixed = rf.PlotBook(stored_dir / "tree_without_branch.root", rf.ALL, **options)
        assert _names(mixed) == ["pt"]
        for _, result in mixed.plots():
            result.close()
        # ... and a file of histograms only fails when built, saying why
        with pytest.raises(
            rf.SourceError, match=rf"stored 1D histograms \['cutflow', .*\] left out \({reason}"
        ):
            rf.PlotBook(stored_dir / "ZH_sel0_histo.root", rf.ALL, **options)

    def test_drawing_options_count_per_task(self, stored_dir: Path) -> None:
        path = stored_dir / "tree_without_branch.root"
        # the book asks for a stats box but its only variant turns it off: stored histograms stay
        book = rf.PlotBook(
            path, rf.ALL, plot_kwargs={"stats": True}, variants={"v": {"stats": False}}
        )
        assert _names(book) == ["mz", "pt"]
        # a range with bins merges the stored bins, so it does not rule them out
        assert _names(
            rf.PlotBook(path, rf.ALL, plot_kwargs={"bins": 50, "range": (0.0, 250.0)})
        ) == [
            "mz",
            "pt",
        ]

    def test_effective_systematics_follow_the_plot(self, stored_dir: Path) -> None:
        zh = stored_dir / "ZH_sel0_histo.root"
        ww = stored_dir / "WW_sel0_histo.root"
        varied = {"systematics": {"w": "weight_expr"}}  # varies the weight: event data
        # the sample's own source of that name replaces the plot's, and is a normalisation
        overriding = rf.Sample(zh, label="ZH", systematics={"w": 0.05})
        book = rf.PlotBook(overriding, rf.ALL, include="mz", plot_kwargs=varied)
        assert _names(book) == ["mz"]
        ((_, result),) = list(book.plots())
        assert list(result.histograms[0].variations) == ["w"]
        direct = rf.plot(overriding, "mz", **varied)
        np.testing.assert_array_equal(
            result.histograms[0].variations["w"][0].values(),
            direct.histograms[0].variations["w"][0].values(),
        )
        result.close()
        direct.close()
        # observed data carries no systematics, so the plot's do not rule its histograms out
        book = rf.PlotBook(overriding, rf.ALL, include="mz", plot_kwargs={**varied, "observed": ww})
        assert _names(book) == ["mz"]
        ((_, result),) = list(book.plots())
        assert [h.is_data for h in result.histograms] == [False, True]
        result.close()
        # without the override the plot's systematic applies to the simulation, as in plot()
        with pytest.raises(rf.SourceError, match="systematic 'w' of sample 'ZH' varies the weight"):
            rf.PlotBook(rf.Sample(zh, label="ZH"), rf.ALL, plot_kwargs={**varied, "observed": ww})
        with pytest.raises(rf.SystematicError, match="varies the weight"):
            rf.plot(rf.Sample(zh, label="ZH"), "mz", **varied, observed=ww)

    def test_sample_options_rule_stored_histograms_out(self, stored_dir: Path) -> None:
        path = stored_dir / "tree_without_branch.root"
        assert _names(rf.PlotBook(rf.Sample(path, selection="pt > 1"), rf.ALL)) == ["pt"]
        assert _names(rf.PlotBook(rf.Sample(path, weight="pt"), rf.ALL)) == ["pt"]
        varied = rf.Sample(path, systematics={"w": ("pt", "pt")})
        assert _names(rf.PlotBook(varied, rf.ALL)) == ["pt"]
        with pytest.raises(rf.SourceError, match="has a weight of its own"):
            rf.PlotBook(rf.Sample(stored_dir / "ZH_sel0_histo.root", weight="x"), rf.ALL)

    def test_normalisation_systematics_keep_stored_histograms(self, stored_dir: Path) -> None:
        sample = rf.Sample(stored_dir / "ZH_sel0_histo.root", systematics={"lumi": 0.02})
        book = rf.PlotBook(sample, rf.ALL, include="mz", plot_kwargs={"systematics": {"n": 0.1}})
        assert _names(book) == ["mz"]
        ((_, result),) = list(book.plots())
        assert sorted(result.histograms[0].variations) == ["lumi", "n"]
        result.close()

    def test_stored_group(self, stored_dir: Path) -> None:
        vv = rf.Group(
            [
                rf.Sample(stored_dir / "WW_sel0_histo.root", label="WW"),
                rf.Sample(stored_dir / "ZZ_sel0_histo.root", label="ZZ"),
            ],
            label="VV",
        )
        book = rf.PlotBook(vv, rf.ALL, include="mz*")
        assert _names(book) == ["mz", "mz_raw"]
        results = list(book.plots())
        assert [h.label for _, r in results for h in r.histograms] == ["VV", "VV"]
        assert results[0][1].hists[0].sum().value == pytest.approx(0.5 * 3000)
        for _, result in results:
            result.close()

    def test_malformed_systematics_are_left_to_the_task(self, stored_dir: Path) -> None:
        # not a discovery problem: every task fails alike, so the stored names stay
        book = rf.PlotBook(
            stored_dir / "ZH_sel0_histo.root",
            rf.ALL,
            include="mz",
            plot_kwargs={"systematics": {"bad": [1, 2, 3]}},
        )
        assert _names(book) == ["mz"]
        with pytest.raises(rf.SystematicError):
            list(book.plots())


class TestSamples:
    def test_intersection_of_samples(self) -> None:
        a = rf.Sample({"x": [1.0], "y": [1.0], "z": [1.0]}, label="A")
        b = rf.Sample({"x": [2.0], "y": [2.0]}, label="B")
        c = rf.Sample({"x": [3.0], "y": [3.0], "q": [3.0]}, label="C")
        assert _names(rf.PlotBook([a, b, c], rf.ALL)) == ["x", "y"]
        assert _names(rf.PlotBook({"A": a, "C": c}, rf.ALL)) == ["x", "y"]

    def test_groups_are_expanded_for_discovery_only(self) -> None:
        a = rf.Sample({"x": [1.0], "y": [1.0], "z": [1.0]}, label="A")
        b = rf.Sample({"x": [2.0], "y": [2.0]}, label="B")
        c = rf.Sample({"x": [3.0], "y": [3.0], "q": [3.0]}, label="C")
        book = rf.PlotBook([rf.Group([a, rf.Group([b], label="B")], label="AB"), c], rf.ALL)
        assert _names(book) == ["x", "y"]
        for _, result in book.plots():
            assert [h.label for h in result.histograms] == ["AB", "C"]
            assert result.histograms[0].sample is None
            assert result.hists[0].sum().value == 2.0
            result.close()

    def test_observed_takes_part(self) -> None:
        mc = rf.Sample({"x": [1.0], "y": [1.0], "z": [1.0]}, label="MC")
        data = {"x": [1.0, 2.0], "y": [1.0, 2.0]}
        book = rf.PlotBook(mc, rf.ALL, plot_kwargs={"observed": data, "stack": True, "ratio": True})
        assert _names(book) == ["x", "y"]
        for _, result in book.plots():
            assert [h.is_data for h in result.histograms] == [False, True]
            assert result.ratio_ax is not None
            result.close()
        group = rf.Group([rf.Sample(data, label="D", is_data=True)], label="Data")
        assert _names(rf.PlotBook(mc, rf.ALL, plot_kwargs={"observed": group})) == ["x", "y"]

    def test_stored_and_branch_of_one_name_do_not_mix(self, stored_dir: Path) -> None:
        # mz stored in one sample, a branch in the other: not a common variable
        samples = [
            rf.Sample(stored_dir / "tree_without_branch.root", label="stored"),
            rf.Sample(stored_dir / "branch_and_histogram.root", label="branch"),
        ]
        with pytest.raises(rf.SourceError, match="found nothing that every task can plot"):
            rf.PlotBook(samples, rf.ALL)
        with pytest.raises(rf.SourceError, match="every sample of one plot must provide it"):
            rf.plot(samples, "mz")

    def test_in_memory_columns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _forbid_reads(monkeypatch)
        data = {
            "f": np.array([1.0, 2.0]),
            "i": np.array([1, 2], dtype=np.int8),
            "u": np.array([1, 2], dtype=np.uint64),
            "b": np.array([True, False]),
            "jag": ak.Array([[1.0], []]),
            "opt": ak.Array([1.0, None]),
            "empty": ak.Array([[], []]),
            "s": ["a", "b"],
            "rec": ak.Array([{"a": 1.0}, {"a": 2.0}]),
            "c": np.array([1 + 1j, 2 + 2j]),
        }
        book = rf.PlotBook(data, rf.ALL)
        assert _names(book) == ["b", "empty", "f", "i", "jag", "opt", "u"]
        assert book.data is data
        structured = np.array([(1.0, 2)], dtype=[("x", "f8"), ("n", "i4")])
        assert _names(rf.PlotBook(structured, rf.ALL)) == ["n", "x"]
        record = ak.Array([{"pt": 1.0, "tag": "a"}])
        assert _names(rf.PlotBook(record, rf.ALL)) == ["pt"]

    def test_source_without_types_is_refused(self) -> None:
        class Opaque:
            def branches(self) -> list[str]:
                return ["x"]

            def arrays(self, branches: Sequence[str]) -> dict[str, ak.Array]:
                return {name: ak.Array([1.0]) for name in branches}

            def describe(self) -> str:
                return "opaque"

        with pytest.raises(rf.SourceError, match=r"Opaque does not provide \(no branch_forms"):
            rf.PlotBook(rf.Sample(Opaque()), rf.ALL)
        # an explicit list needs nothing of the sort
        assert rf.PlotBook(rf.Sample(Opaque()), "x").variables == (rf.Variable("x"),)

    def test_histogram_objects_need_an_explicit_variable(self) -> None:
        h = hist.Hist(hist.axis.Regular(4, 0, 1, name="x"), storage=hist.storage.Weight())
        with pytest.raises(TypeError, match=r"histogram objects .* name the variable"):
            rf.PlotBook(h, rf.ALL)
        with pytest.raises(TypeError, match="histogram objects"):
            rf.PlotBook([rf.Histogram(h, label="h")], rf.ALL)


class TestVariants:
    def test_tree_variants_intersect(self, shared_trees: Path) -> None:
        book = rf.PlotBook(shared_trees, rf.ALL, variants={"a": {"tree": "a"}, "b": {"tree": "b"}})
        assert _names(book) == ["x"]
        assert [t.stem for t in book.tasks()] == ["x__a", "x__b"]
        for _, result in book.plots():
            assert result.hists[0].sum().value == 3.0
            result.close()
        assert _names(rf.PlotBook(shared_trees, rf.ALL, plot_kwargs={"tree": "a"})) == ["x", "y"]

    def test_observed_variants_intersect(self) -> None:
        mc = rf.Sample({"x": [1.0], "y": [1.0], "z": [1.0]}, label="MC")
        book = rf.PlotBook(
            mc,
            rf.ALL,
            variants={"plain": {}, "data": {"observed": {"x": [1.0], "z": [2.0]}}},
        )
        assert _names(book) == ["x", "z"]
        results = {task.stem: result for task, result in book.plots()}
        assert [h.is_data for h in results["x__plain"].histograms] == [False]
        assert [h.is_data for h in results["x__data"].histograms] == [False, True]
        for result in results.values():
            result.close()


class TestFilters:
    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            ({"include": "Muon_*"}, ["Muon_charge", "Muon_isTight", "Muon_pt"]),
            (
                {"include": ["Muon_*", "MET*"]},
                ["MET", "MET_cov", "Muon_charge", "Muon_isTight", "Muon_pt"],
            ),
            ({"exclude": "*_cov"}, [n for n in PLOTTABLE if n != "MET_cov"]),
            (
                {"exclude": ["*_cov", "*Index"]},
                [n for n in PLOTTABLE if n not in ("MET_cov", "jetIndex")],
            ),
            (
                {"include": ["Muon_*", "MET*"], "exclude": "*_cov"},
                ["MET", "Muon_charge", "Muon_isTight", "Muon_pt"],
            ),
            ({"include": "jet1_b-*"}, ["jet1_b-tag"]),  # the source name, not jet1_b_tag
            ({"include": "*"}, PLOTTABLE),
        ],
    )
    def test_include_and_exclude(
        self, rntuple: Path, filters: dict[str, Any], expected: list[str]
    ) -> None:
        names = _names(rf.PlotBook(rntuple, rf.ALL, **filters))
        assert names == ["`jet1_b-tag`" if n == "jet1_b-tag" else n for n in expected]

    def test_patterns_are_case_sensitive(self, rntuple: Path) -> None:
        with pytest.raises(ValueError, match=re.escape("remain after include=['muon_*']")):
            rf.PlotBook(rntuple, rf.ALL, include="muon_*")
        assert _names(rf.PlotBook(rntuple, rf.ALL, exclude="muon_*")) == _names(
            rf.PlotBook(rntuple, rf.ALL)
        )

    def test_patterns_see_source_names_not_file_stems(self, rntuple: Path) -> None:
        with pytest.raises(ValueError, match="no plottable variables remain"):
            rf.PlotBook(rntuple, rf.ALL, include="jet1_b_tag")

    def test_nothing_left_names_the_filters_and_the_candidates(self, rntuple: Path) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "no plottable variables remain after include=['Muon_*'] and exclude=['Muon_*']; "
                "discovered: ['Electron_pt', 'MET', 'MET_cov', 'Muon_charge'"
            ),
        ):
            rf.PlotBook(rntuple, rf.ALL, include="Muon_*", exclude="Muon_*")

    def test_a_pattern_matching_nothing_is_fine_when_others_match(self, rntuple: Path) -> None:
        assert _names(rf.PlotBook(rntuple, rf.ALL, include=["nothing*", "MET"])) == ["MET"]

    @pytest.mark.parametrize("value", [3, ["a", 1], b"x", {"a": 1}])
    def test_non_string_patterns_are_rejected(self, rntuple: Path, value: Any) -> None:
        with pytest.raises(TypeError, match="include= must be a shell pattern"):
            rf.PlotBook(rntuple, rf.ALL, include=value)
        with pytest.raises(TypeError, match="exclude= must be a shell pattern"):
            rf.PlotBook(rntuple, rf.ALL, exclude=value)

    def test_empty_pattern_lists_are_rejected(self, rntuple: Path) -> None:
        with pytest.raises(ValueError, match="include= must hold at least one pattern"):
            rf.PlotBook(rntuple, rf.ALL, include=[])
        with pytest.raises(ValueError, match="exclude= must hold at least one pattern"):
            rf.PlotBook(rntuple, rf.ALL, exclude=())

    def test_filters_need_the_sentinel(self) -> None:
        data = {"x": [1.0], "y": [2.0]}
        with pytest.raises(ValueError, match="include= and exclude= are only valid with"):
            rf.PlotBook(data, ["x"], include="x")
        with pytest.raises(ValueError, match=r"only valid with variables=rf\.ALL"):
            rf.PlotBook(data, "x", exclude="y")
        with pytest.raises(ValueError, match=r"only valid with variables=rf\.ALL"):
            rf.PlotBook(data, [rf.Variable("x")], include="*", exclude="y")
        # checked before the patterns themselves: an explicit list is never filtered
        with pytest.raises(ValueError, match=r"only valid with variables=rf\.ALL"):
            rf.PlotBook(data, ["x"], include=3)  # type: ignore[arg-type]


class TestFileNames:
    def test_identifier_collision_is_reported_with_the_source_names(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("variables '`a-b`' and 'a_b' share the identifier 'a_b'")
            + ".*with variables=rf.ALL, exclude= one of them",
        ):
            rf.PlotBook({"a-b": [1.0], "a_b": [2.0]}, rf.ALL)
        assert _names(rf.PlotBook({"a-b": [1.0], "a_b": [2.0]}, rf.ALL, exclude="a-b")) == ["a_b"]

    def test_case_collision_is_reported(self) -> None:
        with pytest.raises(ValueError, match=r"output names collide: 'MET' and 'met'.*rf\.ALL"):
            rf.PlotBook({"MET": [1.0], "met": [2.0]}, rf.ALL)

    def test_unaddressable_names_are_left_out(self) -> None:
        # a name holding a backtick has no expression; one that is a keyword takes backticks
        book = rf.PlotBook({"a`b": [1.0], "if": [2.0], "x": [3.0]}, rf.ALL)
        assert _names(book) == ["`if`", "x"]
        ((_, first), _) = list(book.plots())
        assert first.hists[0].sum().value == 1.0
        first.close()


class TestExecution:
    @pytest.fixture
    def files(self, signal_file: Path, background_file: Path) -> list[rf.Sample]:
        return [
            rf.Sample(signal_file, tree="events", label="Signal", weight="weight"),
            rf.Sample(background_file, tree="events", label="Background", weight="weight"),
        ]

    def test_discovered_variables_share_the_batch_reads(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        discovered = rf.PlotBook(files, rf.ALL, include=["MET", "Muon_pt", "nMuon"])
        assert _names(discovered) == ["MET", "Muon_pt", "nMuon"]
        assert reads == []  # discovery read no branch
        from_book = {task.stem: result for task, result in discovered.plots()}
        assert len(reads) == 2  # one read per sample for the whole batch
        assert all(set(call) == {"MET", "Muon_pt", "nMuon", "weight"} for call in reads)
        reads.clear()
        explicit = {
            task.stem: result
            for task, result in rf.PlotBook(files, ["MET", "Muon_pt", "nMuon"]).plots()
        }
        assert len(reads) == 2
        for stem, result in from_book.items():
            for got, want in zip(result.histograms, explicit[stem].histograms, strict=True):
                np.testing.assert_array_equal(got.values(flow=True), want.values(flow=True))
            result.close()
            explicit[stem].close()

    def test_select_keeps_the_discovered_variables_without_rediscovering(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[Any] = []
        original = batch.discover_variables

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(batch, "discover_variables", counting)
        book = rf.PlotBook(files, rf.ALL, exclude=["with_nan", "sentinel"])
        assert len(calls) == 1
        subset = book.select(variables=["Muon_pt", "MET"])
        assert len(calls) == 1
        assert _names(subset) == ["MET", "Muon_pt"]
        assert subset.variables == tuple(
            v for v in book.variables if v.safe_name in ("MET", "Muon_pt")
        )
        assert "with_nan" not in _names(book)

    def test_tasks_and_names_match_an_explicit_book(self, files: list[rf.Sample]) -> None:
        discovered = rf.PlotBook(
            files,
            rf.ALL,
            include=["MET", "nMuon"],
            selections={"all": None, "hi": "MET > 20"},
            variants={"lin": {}, "log": {"logy": True}},
        )
        explicit = rf.PlotBook(
            files,
            ["MET", "nMuon"],
            selections={"all": None, "hi": "MET > 20"},
            variants={"lin": {}, "log": {"logy": True}},
        )
        assert discovered.tasks() == explicit.tasks()
        assert [t.stem for t in discovered.tasks()][:4] == [
            "MET__all__lin",
            "MET__all__log",
            "MET__hi__lin",
            "MET__hi__log",
        ]
        assert repr(discovered) == repr(explicit)


class TestDiagnostics:
    @pytest.mark.parametrize("second", [{"y": [2.0]}, {"y": [2.0, 3.0]}])
    def test_distinct_sources_with_default_labels_are_all_reported(
        self, second: dict[str, list[float]]
    ) -> None:
        with pytest.raises(rf.SourceError) as info:
            rf.PlotBook([{"x": [1.0]}, second], rf.ALL)
        lines = str(info.value).splitlines()
        assert "'arrays [1]'" in lines[1]
        assert "branches ['x']" in lines[1]
        assert "'arrays [2]'" in lines[2]
        assert "branches ['y']" in lines[2]
        assert lines[3:] == [
            "['x'] missing from ['arrays [2]']",
            "['y'] missing from ['arrays [1]']",
        ]

    def test_same_basename_and_distinct_tree_variants_are_all_reported(
        self, tmp_path: Path
    ) -> None:
        paths = []
        for directory, branches in (("one", ("x", "y")), ("two", ("z", "w"))):
            target = tmp_path / directory
            target.mkdir()
            path = target / "sample.root"
            paths.append(path)
            with uproot.recreate(path) as file:
                for tree, branch in zip(("a", "b"), branches, strict=True):
                    file[tree] = {branch: np.arange(2.0)}
        with pytest.raises(rf.SourceError) as info:
            rf.PlotBook(paths, rf.ALL, variants={"a": {"tree": "a"}, "b": {"tree": "b"}})
        lines = str(info.value).splitlines()
        for index, branch in enumerate(("x", "z", "y", "w"), start=1):
            assert f"'sample [{index}]'" in lines[index]
            assert f"branches ['{branch}']" in lines[index]

    def test_changed_constraints_are_reported_without_repeating_identical_ones(
        self, stored_dir: Path
    ) -> None:
        with pytest.raises(rf.SourceError) as info:
            rf.PlotBook(
                stored_dir / "ZH_sel0_histo.root",
                rf.ALL,
                variants={"plain": {}, "log": {"logy": True}, "stats": {"stats": True}},
            )
        lines = str(info.value).splitlines()
        assert "'ZH_sel0_histo [1]'" in lines[1]
        assert "left out" not in lines[1]
        assert "'ZH_sel0_histo [2]'" in lines[2]
        assert "left out (stats=" in lines[2]
        assert "missing from ['ZH_sel0_histo [2]']" in lines[3]
        assert len(lines) == 4

    def test_numbered_labels_do_not_collide_with_user_labels(self) -> None:
        samples = [
            rf.Sample({"x": [1.0]}, label="A"),
            rf.Sample({"y": [1.0]}, label="A"),
            rf.Sample({"z": [1.0]}, label="A [1]"),
        ]
        with pytest.raises(rf.SourceError) as info:
            rf.PlotBook(samples, rf.ALL)
        lines = str(info.value).splitlines()
        assert "'A [2]'" in lines[1]
        assert "'A [3]'" in lines[2]
        assert "'A [1]'" in lines[3]

    def test_nothing_common_names_what_each_sample_lacks(self) -> None:
        a = rf.Sample({"x": [1.0], "y": [1.0], "name": ["a"]}, label="A")
        b = rf.Sample({"z": [1.0], "y": ["b"]}, label="B")
        with pytest.raises(rf.SourceError) as info:
            rf.PlotBook([a, b], rf.ALL)
        lines = str(info.value).splitlines()
        assert lines[0].startswith("variables=rf.ALL found nothing that every task can plot")
        assert lines[1:] == [
            "'A' (in-memory arrays (1 events, 3 columns)): plottable branches ['x', 'y']; "
            "not plottable ['name (string)']",
            "'B' (in-memory arrays (1 events, 2 columns)): plottable branches ['z']; "
            "not plottable ['y (string)']",
            "['x', 'y'] missing from ['B']",
            "['z'] missing from ['A']",
        ]

    def test_mode_conflicts_and_left_out_histograms_are_named(self, stored_dir: Path) -> None:
        samples = [
            rf.Sample(stored_dir / "tree_without_branch.root", label="cut", selection="pt > 0"),
            rf.Sample(stored_dir / "branch_and_histogram.root", label="branch"),
        ]
        with pytest.raises(rf.SourceError) as info:
            rf.PlotBook(samples, rf.ALL)
        lines = str(info.value).splitlines()
        assert lines[1] == (
            "'cut' (tree_without_branch.root): plottable branches ['pt']; stored 1D "
            "histograms ['mz'] left out (sample 'cut' has a selection of its own, which "
            "needs event data)"
        )
        assert lines[2] == "'branch' (branch_and_histogram.root): plottable branches ['mz']"
        assert lines[3:] == ["['mz'] missing from ['cut']", "['pt'] missing from ['branch']"]
        # the same name provided two ways is reported as such
        plain = [rf.Sample(stored_dir / "tree_without_branch.root", label="stored"), samples[1]]
        with pytest.raises(
            rf.SourceError,
            match=re.escape("'mz' is a branch of ['branch'] but a stored histogram of ['stored']"),
        ):
            rf.PlotBook(plain, rf.ALL)

    def test_long_lists_are_abbreviated(self) -> None:
        a = rf.Sample({f"x{i:02d}": [1.0] for i in range(20)}, label="A")
        b = rf.Sample({"z": [1.0]}, label="B")
        with pytest.raises(rf.SourceError) as info:
            rf.PlotBook([a, b], rf.ALL)
        assert "'x11', ... (8 more)] missing from ['B']" in str(info.value)


class TestPublicFunction:
    def test_matches_the_book(self, rntuple: Path) -> None:
        options: dict[str, Any] = {
            "include": ["Muon_*", "MET*"],
            "exclude": "*_cov",
            "selections": {"all": None},
            "variants": {"lin": {}, "log": {"logy": True}},
            "plot_kwargs": {"stack": True},
        }
        variables = rf.discover_variables(rntuple, **options)
        assert variables == rf.PlotBook(rntuple, rf.ALL, **options).variables
        assert isinstance(variables, tuple)
        assert [v.expression for v in variables] == [
            "MET",
            "Muon_charge",
            "Muon_isTight",
            "Muon_pt",
        ]
        assert rf.discover_variables(rntuple) == rf.PlotBook(rntuple, rf.ALL).variables
        assert "discover_variables" in rf.__all__

    def test_collisions_are_returned_for_renaming(self) -> None:
        data = {"a-b": [1.0], "a_b": [2.0]}
        variables = rf.discover_variables(data)
        assert [v.expression for v in variables] == ["`a-b`", "a_b"]
        with pytest.raises(ValueError, match="share the identifier"):
            rf.PlotBook(data, rf.ALL)
        renamed = [v.replace(name="a_minus_b") if v.expression == "`a-b`" else v for v in variables]
        book = rf.PlotBook(data, renamed)
        assert [t.stem for t in book.tasks()] == ["a_minus_b", "a_b"]
        for _, result in book.plots():
            result.close()

    def test_keywords_are_checked_like_the_book(self, rntuple: Path, stored_dir: Path) -> None:
        with pytest.raises(TypeError, match=r"plot_kwargs names keywords plot\(\) does not have"):
            rf.discover_variables(rntuple, plot_kwargs={"log_y": True})
        with pytest.raises(TypeError, match="include= must be a shell pattern"):
            rf.discover_variables(rntuple, include=3)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="no plottable variables remain"):
            rf.discover_variables(rntuple, include="nothing*")
        with pytest.raises(ValueError, match="selections= must name at least one"):
            rf.discover_variables(rntuple, selections={})
        with pytest.raises(rf.SourceError, match="found nothing that every task can plot"):
            rf.discover_variables(stored_dir / "ZH_sel0_histo.root", selections={"c": "mz > 0"})
        # the drawing options of every variant count, as for the book
        path = stored_dir / "tree_without_branch.root"
        assert [v.expression for v in rf.discover_variables(path)] == ["mz", "pt"]
        stats = rf.discover_variables(path, variants={"a": {}, "b": {"stats": True}})
        assert [v.expression for v in stats] == ["pt"]


class TestSampleVariations:
    """``Systematic.samples`` variations are filled or read like samples, so they take part."""

    def test_branch_missing_from_a_variation(self) -> None:
        sample = rf.Sample(
            {"x": [1.0, 2.0], "y": [2.0, 3.0]},
            label="A",
            systematics={"alt": rf.Systematic.samples({"x": [1.1, 2.1]})},
        )
        book = rf.PlotBook(sample, rf.ALL)
        assert _names(book) == ["x"]
        ((_, result),) = list(book.plots())
        assert list(result.histograms[0].variations) == ["alt"]
        result.close()
        # y would fail when the variation is loaded, as the explicit call does
        with pytest.raises(rf.MissingBranchError):
            rf.plot(sample, "y")
        # the down variation counts too, and the message names the variation
        both = rf.Sample(
            {"x": [1.0], "y": [2.0]},
            label="A",
            systematics={"alt": rf.Systematic.samples({"x": [1.1], "y": [2.0]}, {"x": [0.9]})},
        )
        assert _names(rf.PlotBook(both, rf.ALL)) == ["x"]
        only = rf.Sample(
            {"y": [2.0]}, label="A", systematics={"alt": rf.Systematic.samples({"x": [1.1]})}
        )
        with pytest.raises(rf.SourceError, match=re.escape("['y'] missing from ['A [alt up]']")):
            rf.PlotBook(only, rf.ALL)

    def test_file_variation_lacking_branches(self, signal_file: Path, tmp_path: Path) -> None:
        alt = tmp_path / "alt.root"
        _write_tree(
            alt, {"MET": np.array([1.0, 2.0]), "weight": np.array([1.0, 1.0])}, rntuple=False
        )
        sample = rf.Sample(
            signal_file,
            tree="events",
            label="S",
            weight="weight",
            systematics={"alt": rf.Systematic.samples(str(alt))},
        )
        book = rf.PlotBook(sample, rf.ALL)
        assert _names(book) == ["MET", "weight"]
        ((_, result),) = list(book.select(variables="MET").plots())
        assert list(result.histograms[0].variations) == ["alt"]
        result.close()
        # the plot-level form applies to every simulated sample alike
        plain = rf.Sample(signal_file, tree="events", label="S")
        book = rf.PlotBook(
            plain, rf.ALL, plot_kwargs={"systematics": {"alt": rf.Systematic.samples(str(alt))}}
        )
        assert _names(book) == ["MET", "weight"]

    def test_stored_histogram_missing_from_a_variation(self, stored_dir: Path) -> None:
        zh = stored_dir / "ZH_sel0_histo.root"
        # the variation file holds mz only, next to a tree, which does not matter for a variation
        alt = str(stored_dir / "tree_without_branch.root")
        sample = rf.Sample(zh, label="ZH", systematics={"alt": rf.Systematic.samples(alt)})
        book = rf.PlotBook(sample, rf.ALL)
        assert _names(book) == ["mz"]
        ((_, result),) = list(book.plots())
        assert list(result.histograms[0].variations) == ["alt"]
        result.close()
        with pytest.raises(rf.SourceError, match="not found"):
            rf.plot(sample, "mz_raw")
        # in-memory data holds no stored histograms, so such a variation rules them out
        arrays = rf.Sample(
            zh, label="ZH", systematics={"alt": rf.Systematic.samples({"mz": [1.0]})}
        )
        with pytest.raises(
            rf.SourceError,
            match=re.escape("no stored 1D histograms (in-memory data holds no stored histograms)"),
        ):
            rf.PlotBook(arrays, rf.ALL)
        with pytest.raises(rf.SystematicError):
            rf.plot(arrays, "mz")

    def test_variation_that_cannot_be_built_is_left_to_the_task(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "missing.root")
        sample = rf.Sample(
            {"x": [1.0]}, label="A", systematics={"alt": rf.Systematic.samples(missing)}
        )
        book = rf.PlotBook(sample, rf.ALL)
        assert _names(book) == ["x"]
        with pytest.raises(rf.SystematicError, match="cannot use"):
            list(book.plots())


class TestBranchReplacements:
    """A ``replace`` systematic reads the replacement whenever the replaced branch is used."""

    def test_missing_replacement_drops_the_replaced_branch(self) -> None:
        sample = rf.Sample(
            {"x": [1.0], "y": [2.0], "z": [3.0], "z_up": [3.1], "z_down": [2.9]},
            label="A",
            systematics={"shift": {"x": "x_up"}, "jes": {"z": ("z_up", "z_down")}},
        )
        book = rf.PlotBook(sample, rf.ALL)
        assert _names(book) == ["y", "z", "z_down", "z_up"]  # x: its x_up is missing
        for _, result in book.plots():
            result.close()
        with pytest.raises(rf.MissingBranchError):
            rf.plot(sample, "x")
        # a replacement that no histogram can be filled from counts as missing too
        text = rf.Sample(
            {"x": [1.0], "x_up": ["a"]}, label="A", systematics={"shift": {"x": "x_up"}}
        )
        with pytest.raises(
            rf.SourceError,
            match=re.escape(
                "not plottable ['x_up (string)', \"x (its replacement 'x_up' under systematic "
                "'shift' is not a plottable branch)\"]"
            ),
        ):
            rf.PlotBook(text, rf.ALL)

    def test_effective_replacements_follow_the_plot(self) -> None:
        plot_level: dict[str, Any] = {"systematics": {"shift": {"x": "x_up"}}}
        plain = rf.Sample({"x": [1.0], "y": [2.0]}, label="A")
        assert _names(rf.PlotBook(plain, rf.ALL, plot_kwargs=plot_level)) == ["y"]
        # the sample's own source of that name replaces the plot's; observed data has none
        overriding = rf.Sample({"x": [1.0], "y": [2.0]}, label="A", systematics={"shift": 0.1})
        book = rf.PlotBook(
            overriding,
            rf.ALL,
            plot_kwargs={**plot_level, "observed": {"x": [1.0, 2.0], "y": [2.0, 3.0]}},
        )
        assert _names(book) == ["x", "y"]
        for _, result in book.plots():
            assert [h.is_data for h in result.histograms] == [False, True]
            result.close()
        # a replacement that is present keeps the branch, and the variation is filled
        complete = rf.Sample({"x": [1.0], "x_up": [1.5]}, label="A")
        book = rf.PlotBook(complete, rf.ALL, plot_kwargs=plot_level)
        assert _names(book) == ["x", "x_up"]
        results = {task.stem: result for task, result in book.plots()}
        assert list(results["x"].histograms[0].variations) == ["shift"]
        for result in results.values():
            result.close()
