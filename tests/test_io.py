"""Tests for data sources: files (TTree and RNTuple), globs, multi-file, in-memory arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

from rootfig.errors import SourceError
from rootfig.io import ArraySource, FileSource, Source, as_source, resolve_files


class TestResolveFiles:
    def test_single_path(self, signal_file: Path) -> None:
        paths, tree = resolve_files(signal_file)
        assert paths == (str(signal_file),)
        assert tree is None

    def test_path_with_tree(self, signal_file: Path) -> None:
        paths, tree = resolve_files(f"{signal_file}:events")
        assert paths == (str(signal_file),)
        assert tree == "events"

    def test_glob_is_sorted(self, data_dir: Path) -> None:
        paths, _ = resolve_files(str(data_dir / "bkg_part*.root"))
        assert [Path(p).name for p in paths] == ["bkg_part1.root", "bkg_part2.root"]

    def test_list_mixed(self, data_dir: Path) -> None:
        paths, tree = resolve_files([data_dir / "signal.root", f"{data_dir}/bkg_part*.root:events"])
        assert len(paths) == 3
        assert tree == "events"

    def test_missing_file(self, data_dir: Path) -> None:
        with pytest.raises(SourceError, match="file not found"):
            resolve_files(data_dir / "nope.root")

    def test_glob_without_match(self, data_dir: Path) -> None:
        with pytest.raises(SourceError, match="no files match"):
            resolve_files(str(data_dir / "zzz*.root"))

    def test_conflicting_trees(self, data_dir: Path) -> None:
        with pytest.raises(SourceError, match="conflicting tree names"):
            resolve_files([f"{data_dir}/signal.root:a", f"{data_dir}/background.root:b"])

    def test_empty(self) -> None:
        with pytest.raises(SourceError, match="no input files"):
            resolve_files([])

    def test_remote_url_passthrough(self) -> None:
        paths, tree = resolve_files("root://eos.example.org//store/file.root:events")
        assert paths == ("root://eos.example.org//store/file.root",)
        assert tree == "events"


class TestFileSource:
    def test_branches_any_format(self, signal_file_any_format: Path) -> None:
        source = FileSource(signal_file_any_format, "events")
        names = source.branches()
        assert {"Muon_pt", "MET", "weight", "jet1_b-tag"} <= set(names)

    def test_arrays_any_format(
        self, signal_file_any_format: Path, signal_columns: dict[str, Any]
    ) -> None:
        source = FileSource(signal_file_any_format, "events")
        arrays = source.arrays(["MET", "Muon_pt", "nMuon"])
        assert set(arrays) == {"MET", "Muon_pt", "nMuon"}
        assert arrays["MET"].tolist() == pytest.approx(signal_columns["MET"].tolist())
        assert ak.num(arrays["Muon_pt"]).tolist() == signal_columns["nMuon"].tolist()

    def test_empty_request(self, signal_file: Path) -> None:
        assert FileSource(signal_file, "events").arrays([]) == {}

    def test_tree_autodetect(self, signal_file: Path) -> None:
        source = FileSource(signal_file)
        assert source.resolved_tree() == "events"
        assert "MET" in source.branches()

    def test_tree_from_path(self, signal_file: Path) -> None:
        assert FileSource(f"{signal_file}:events").tree == "events"

    def test_tree_conflict(self, signal_file: Path) -> None:
        with pytest.raises(SourceError, match="conflicts"):
            FileSource(f"{signal_file}:events", tree="other")

    def test_autodetect_ambiguous(self, data_dir: Path) -> None:
        with pytest.raises(SourceError, match=r"several trees.*\['a', 'b'\]"):
            FileSource(data_dir / "multi.root").branches()

    def test_autodetect_none(self, data_dir: Path) -> None:
        with pytest.raises(SourceError, match="no TTree or RNTuple"):
            FileSource(data_dir / "no_tree.root").branches()

    def test_missing_tree(self, signal_file: Path) -> None:
        with pytest.raises(SourceError, match="not found"):
            FileSource(signal_file, "nope").branches()
        with pytest.raises(SourceError):
            FileSource(signal_file, "nope").arrays(["MET"])

    def test_missing_branch_ttree(self, signal_file: Path) -> None:
        with pytest.raises(SourceError):
            FileSource(signal_file, "events").arrays(["MET", "nope"])

    def test_missing_branch_rntuple(self, data_dir: Path) -> None:
        # RNTuple reads silently ignore unknown fields; FileSource must not.
        with pytest.raises(SourceError, match=r"\['nope'\] not found"):
            FileSource(data_dir / "signal_rntuple.root", "events").arrays(["MET", "nope"])

    def test_multi_file_concatenation(
        self, data_dir: Path, background_columns: dict[str, Any]
    ) -> None:
        source = FileSource(str(data_dir / "bkg_part*.root"), "events")
        assert len(source.files) == 2
        arrays = source.arrays(["event", "Muon_pt"])
        assert arrays["event"].tolist() == background_columns["event"].tolist()
        assert ak.flatten(arrays["Muon_pt"]).tolist() == pytest.approx(
            ak.flatten(background_columns["Muon_pt"]).tolist()
        )
        assert source.describe().startswith("2 files")

    def test_entry_range(self, data_dir: Path) -> None:
        source = FileSource(
            str(data_dir / "bkg_part*.root"), "events", entry_start=990, entry_stop=1010
        )
        events = source.arrays(["event"])["event"].tolist()
        assert events == list(range(990, 1010))

    def test_describe_and_label(self, signal_file: Path) -> None:
        source = FileSource(signal_file, "events")
        assert source.describe() == "signal.root"
        assert source.default_label == "signal"

    def test_is_source(self, signal_file: Path) -> None:
        assert isinstance(FileSource(signal_file), Source)


class TestArraySource:
    def test_mapping(self) -> None:
        source = ArraySource({"x": np.arange(3), "y": ak.Array([[1.0], [], [2.0, 3.0]])})
        assert source.branches() == ["x", "y"]
        arrays = source.arrays(["y"])
        assert arrays["y"].tolist() == [[1.0], [], [2.0, 3.0]]
        assert "3 events" in source.describe()
        assert source.default_label == "arrays"

    def test_record_array(self) -> None:
        source = ArraySource(ak.Array({"x": [1, 2], "y": [3, 4]}))
        assert source.branches() == ["x", "y"]

    def test_structured_numpy(self) -> None:
        data = np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")])
        source = ArraySource(data)
        assert source.arrays(["b"])["b"].tolist() == [2.0, 4.0]

    def test_length_mismatch(self) -> None:
        with pytest.raises(SourceError, match="same length"):
            ArraySource({"x": [1, 2], "y": [1]})

    def test_missing_column(self) -> None:
        with pytest.raises(SourceError, match="not found"):
            ArraySource({"x": [1]}).arrays(["z"])

    @pytest.mark.parametrize("bad", [{}, ak.Array([1, 2]), np.arange(3), 42])
    def test_unsupported(self, bad: Any) -> None:
        with pytest.raises(SourceError):
            ArraySource(bad)

    def test_unconvertible_value(self) -> None:
        with pytest.raises(SourceError, match="cannot convert"):
            ArraySource({"x": object()})


class TestAsSource:
    def test_dispatch(self, signal_file: Path) -> None:
        assert isinstance(as_source(signal_file), FileSource)
        assert isinstance(as_source(str(signal_file)), FileSource)
        assert isinstance(as_source([signal_file, signal_file]), FileSource)
        assert isinstance(as_source({"x": [1]}), ArraySource)
        assert isinstance(as_source(ak.Array({"x": [1]})), ArraySource)
        existing = ArraySource({"x": [1]})
        assert as_source(existing) is existing

    def test_existing_source_rejects_overrides(self, signal_file: Path) -> None:
        source = FileSource(signal_file, tree="events")
        assert as_source(source, tree="events") is source  # the same tree is no override
        with pytest.raises(SourceError, match="entry_stop=100 cannot be applied"):
            as_source(source, entry_stop=100)
        with pytest.raises(SourceError, match="tree='other'"):
            as_source(source, tree="other")
        detected = FileSource(signal_file)  # tree auto-detected: compared after detection
        assert as_source(detected, tree="events") is detected
        arrays = ArraySource({"x": [1, 2]})
        with pytest.raises(SourceError, match="entry_start=1"):
            as_source(arrays, entry_start=1)
        assert as_source(arrays, tree="events") is arrays  # tree only applies to files

    def test_custom_source_object(self) -> None:
        class Custom:
            def branches(self) -> list[str]:
                return ["x"]

            def arrays(self, branches: Any) -> dict[str, ak.Array]:
                return {b: ak.Array([1.0]) for b in branches}

            def describe(self) -> str:
                return "custom"

        custom = Custom()
        assert as_source(custom) is custom

    def test_unsupported(self) -> None:
        with pytest.raises(SourceError, match="cannot interpret"):
            as_source(42)


DATA = Path(__file__).parent / "data"


class TestSplitCollections:
    """A file with a split object branch, as podio/EDM4hep write them (see CONTRIBUTING.md)."""

    def test_leaf_names_and_reading(self) -> None:
        source = FileSource(DATA / "split_collection.root", tree="events")
        names = source.branches()
        assert "ReconstructedParticles.momentum.x" in names
        assert "ReconstructedParticles.energy" in names
        assert "ReconstructedParticles" not in names  # the parent holds no array
        assert not any("/" in n for n in names)
        arrays = source.arrays(["ReconstructedParticles.energy", "ReconstructedParticles.charge"])
        assert set(arrays) == {"ReconstructedParticles.energy", "ReconstructedParticles.charge"}
        assert len(arrays["ReconstructedParticles.energy"]) == 200
        assert arrays["ReconstructedParticles.energy"].layout.purelist_depth == 2
        assert source.num_entries() == 200
        assert (
            FileSource(DATA / "split_collection.root", tree="events", entry_stop=50).num_entries()
            == 50
        )

    def test_leaf_names_helper(self) -> None:
        from rootfig.io.sources import _leaf_names

        keys = ["MET", "RP", "RP/RP.energy", "RP/RP.momentum.x", "Jet", "Jet/Jet.pt"]
        assert _leaf_names(keys) == ["MET", "RP.energy", "RP.momentum.x", "Jet.pt"]


class TestScalars:
    def test_read_scalar_and_entries(self, tmp_path: Path, signal_file: Path) -> None:
        import uproot

        path = tmp_path / "meta.root"
        with uproot.recreate(path) as file:
            file.mktree("events", {"x": "float64"}).extend({"x": np.zeros(7)})
            file["sumw"] = np.histogram(np.zeros(30), bins=3)
        source = FileSource([path, path], tree="events")
        assert source.read_scalar("sumw") == 60.0  # both files
        assert source.num_entries() == 14
        with pytest.raises(SourceError, match="not found"):
            source.read_scalar("missing")
        with pytest.raises(SourceError, match="holds no number"):
            source.read_scalar("events")
        split = FileSource(DATA / "split_collection.root", tree="events")
        assert split.read_scalar("eventsProcessed") == 200.0  # a TParameter<int>
        assert (
            FileSource(signal_file, tree="events", entry_start=10, entry_stop=30).num_entries()
            == 20
        )
        assert ArraySource({"x": np.arange(3.0)}).num_entries() == 3


class TestNestedRNTuple:
    @pytest.fixture
    def nested(self, tmp_path: Path) -> Path:
        import uproot

        path = tmp_path / "nested.root"
        muons = ak.zip({"pt": [[1.0, 2.0], [3.0]], "eta": [[0.1, 0.2], [0.3]]})
        with uproot.recreate(path) as file:
            file["events"] = {"Muon": muons, "met": [1.0, 2.0]}
        return path

    def test_dotted_fields_are_listed_and_readable(self, nested: Path) -> None:
        source = FileSource(nested, tree="events")
        assert source.branches() == ["Muon.pt", "Muon.eta", "met"]
        arrays = source.arrays(["Muon.pt", "met"])
        assert arrays["Muon.pt"].tolist() == [[1.0, 2.0], [3.0]]
        assert arrays["met"].tolist() == [1.0, 2.0]
        assert source.arrays(["Muon.eta"])["Muon.eta"].tolist() == [[0.1, 0.2], [0.3]]
        with pytest.raises(SourceError, match="not found"):
            source.arrays(["Muon.phi"])

    def test_multiple_files(self, nested: Path) -> None:
        source = FileSource([nested, nested], tree="events")
        assert len(source.arrays(["Muon.pt"])["Muon.pt"]) == 4
        assert source.num_entries() == 4

    def test_leaf_names_drop_record_parents(self) -> None:
        from rootfig.io.sources import _leaf_names

        assert _leaf_names(["Muon", "Muon.pt", "Muon.eta", "met"]) == ["Muon.pt", "Muon.eta", "met"]


class TestEntryRanges:
    def test_negative_and_clipped_entry_ranges(self, tmp_path: Path) -> None:
        import uproot

        path = tmp_path / "counts.root"
        with uproot.recreate(path) as file:
            file.mktree("events", {"x": np.arange(3.0)})
        assert FileSource(path, tree="events", entry_stop=-1).num_entries() == 2
        assert FileSource(path, tree="events", entry_start=-1).num_entries() == 1
        assert FileSource(path, tree="events", entry_start=1, entry_stop=10).num_entries() == 2
        assert FileSource(path, tree="events", entry_start=5).num_entries() == 0
        source = FileSource(path, tree="events", entry_stop=-1)
        assert len(source.arrays(["x"])["x"]) == source.num_entries()

    def test_array_source_entry_range(self) -> None:
        source = ArraySource({"x": np.arange(5.0)}, entry_start=1, entry_stop=3)
        assert source.arrays(["x"])["x"].tolist() == [1.0, 2.0]
        assert source.num_entries() == 2
        via_dispatch = as_source({"x": np.arange(5.0)}, entry_start=3)
        assert isinstance(via_dispatch, ArraySource)
        assert via_dispatch.num_entries() == 2

    def test_tree_in_directory_shorthand(self, tmp_path: Path) -> None:
        import uproot

        path = tmp_path / "dir.root"
        with uproot.recreate(path) as file:
            file.mktree("dir/events", {"x": np.arange(3.0)})
        source = FileSource(f"{path}:dir/events")
        assert source.tree == "dir/events"
        assert source.arrays(["x"])["x"].tolist() == [0.0, 1.0, 2.0]
        assert resolve_files("root://host:1094//store/file.root")[1] is None
