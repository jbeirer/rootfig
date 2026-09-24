"""Tests for data sources: files (TTree and RNTuple), globs, multi-file, in-memory arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest
import uproot

from rootfig.errors import BinningError, RootfigWarning, SourceError
from rootfig.io import ArraySource, FileSource, ReadCache, Source, as_source, resolve_files


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

    def test_fields_named_with_dots(self, tmp_path: Path) -> None:
        import uproot

        path = tmp_path / "dotted.root"
        with uproot.recreate(path) as file:
            file["events"] = {
                "a.b": np.array([1.0, 2.0]),
                "c.d": ak.zip({"e": [1.0, 2.0], "f.g": [1, 2]}),
                "c": np.array([5.0, 6.0]),
            }
        source = FileSource(path, tree="events")
        assert sorted(source.branches()) == ["a.b", "c", "c.d.e", "c.d.f.g"]
        arrays = source.arrays(["a.b", "c.d.f.g", "c", "c.d.e"])
        assert arrays["a.b"].tolist() == [1.0, 2.0]
        assert arrays["c.d.f.g"].tolist() == [1, 2]
        assert arrays["c"].tolist() == [5.0, 6.0]
        assert arrays["c.d.e"].tolist() == [1.0, 2.0]
        assert {n: str(f.type) for n, f in source.branch_forms().items()} == {
            "a.b": "float64",
            "c": "float64",
            "c.d.e": "float64",
            "c.d.f.g": "int64",
        }
        with pytest.raises(SourceError, match="not found"):
            source.arrays(["c.d.h"])

    @pytest.mark.parametrize("longer_field", [10.0, {"d": 10.0}])
    def test_dotted_prefix_falls_back_to_a_complete_path(
        self, tmp_path: Path, longer_field: Any
    ) -> None:
        path = tmp_path / "prefix.root"
        with uproot.recreate(path) as file:
            file["events"] = ak.Array([{"a": {"b": {"c": 1.0}}, "a.b": longer_field}])
        source = FileSource(path)
        assert "a.b.c" in source.branches()
        assert source.arrays(["a.b.c"])["a.b.c"].tolist() == [1.0]
        with pytest.raises(SourceError, match="not found"):
            source.arrays(["a.b.missing"])

    @pytest.mark.parametrize("literal_numeric", [True, False])
    @pytest.mark.parametrize("literal_first", [True, False])
    @pytest.mark.parametrize("collection", [True, False])
    def test_colliding_paths_keep_the_type_of_the_field_that_is_read(
        self, tmp_path: Path, literal_numeric: bool, literal_first: bool, collection: bool
    ) -> None:
        literal, nested = (1.0, "text") if literal_numeric else ("text", 1.0)
        fields = [("a.b", literal), ("a", {"b": nested})]
        row = dict(fields if literal_first else reversed(fields))
        data = ak.Array([{"items": [row]}] if collection else [row])
        path = tmp_path / "collision.root"
        with uproot.recreate(path) as file:
            file["events"] = data
        source = FileSource(path)
        name = "items.a.b" if collection else "a.b"
        actual = source.arrays([name])[name]
        assert actual.tolist() == ([[literal]] if collection else [literal])
        assert source.branch_forms()[name].type == actual.layout.form.type

    def test_literal_record_shadows_a_nested_numeric_field(self, tmp_path: Path) -> None:
        path = tmp_path / "record.root"
        with uproot.recreate(path) as file:
            file["events"] = ak.Array([{"a.b": {"c": 2.0}, "a": {"b": 1.0}}])
        source = FileSource(path)
        assert source.branches() == ["a.b.c"]
        assert list(source.branch_forms()) == ["a.b.c"]
        assert source.arrays(["a.b.c"])["a.b.c"].tolist() == [2.0]

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


class TestIterate:
    """FileSource.iterate reads what arrays() reads, a chunk of entries at a time."""

    RANGES = ((None, None), (100, 1500), (900, 1100), (-300, None), (None, -1500), (1500, 100))

    @staticmethod
    def _joined(chunks: list[dict[str, ak.Array]]) -> dict[str, ak.Array]:
        return {name: ak.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}

    @pytest.mark.parametrize(
        "names",
        [["signal.root"], ["signal_rntuple.root"], ["bkg_part1.root", "bkg_part2.root"]],
    )
    @pytest.mark.parametrize(("start", "stop"), RANGES)
    def test_chunks_join_to_the_whole_read(
        self, data_dir: Path, names: list[str], start: int | None, stop: int | None
    ) -> None:
        source = FileSource(
            [str(data_dir / name) for name in names],
            tree="events",
            entry_start=start,
            entry_stop=stop,
        )
        branches = ["Muon_pt", "MET", "nMuon"]
        whole = source.arrays(branches)
        chunks = list(source.iterate(branches, chunk_bytes=1_000))
        assert len(chunks) > 1 or len(whole["MET"]) == 0
        assert all(list(chunk) == branches for chunk in chunks)
        joined = self._joined(chunks)
        for name in branches:
            assert ak.array_equal(joined[name], whole[name])

    def test_no_entry_in_range_is_one_empty_chunk(self, signal_file: Path) -> None:
        source = FileSource(signal_file, tree="events", entry_start=5000)
        [chunk] = source.iterate(["Muon_pt", "MET"])
        assert len(chunk["MET"]) == 0
        assert chunk["Muon_pt"].layout.purelist_depth == 2  # the types are known

    def test_small_files_are_joined(self, data_dir: Path) -> None:
        source = FileSource(
            [str(data_dir / "bkg_part1.root"), str(data_dir / "bkg_part2.root")], tree="events"
        )
        [chunk] = source.iterate(["MET"], chunk_bytes=10**8)  # both files: less than a chunk
        assert len(chunk["MET"]) == 2000

    @pytest.mark.parametrize("kind", ["TTree", "RNTuple"])
    def test_chunks_hold_about_the_chunk_size_uncompressed(self, tmp_path: Path, kind: str) -> None:
        # 2.4 MB of arrays that compress about fifty-fold, in baskets of 1 000 entries: a
        # step counted from the compressed sizes would read it all in one piece
        values = np.repeat(np.arange(50.0), 4_000)
        jagged = ak.unflatten(values, np.full(100_000, 2))
        path = tmp_path / "compressible.root"
        with uproot.recreate(path, compression=uproot.ZLIB(9)) as file:
            make = file.mktree if kind == "TTree" else file.mkrntuple
            make("events", {"x": "var * float64"})
            for start in range(0, 100_000, 1_000):
                file["events"].extend({"x": jagged[start : start + 1_000]})
        chunks = list(FileSource(path, tree="events").iterate(["x"], chunk_bytes=400_000))
        sizes = [chunk["x"].nbytes for chunk in chunks]
        assert sum(len(chunk["x"]) for chunk in chunks) == 100_000
        assert len(sizes) >= 5
        assert max(sizes) <= 500_000  # a basket partly in range is held whole

    @staticmethod
    def _clusters(path: Path, kind: str, sizes: list[int]) -> None:
        """Write ``x``, a list of two doubles per event, as one cluster or basket per size."""
        with uproot.recreate(path) as file:
            make = file.mktree if kind == "TTree" else file.mkrntuple
            make("events", {"x": "var * float64"})
            for size in sizes:
                values = np.arange(2.0 * size)
                file["events"].extend({"x": ak.unflatten(values, np.full(size, 2))})

    def test_rntuples_are_read_in_runs_of_whole_clusters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def steps_across_clusters(*args: Any, **kwargs: Any) -> Any:
            msg = "RNTuple.iterate steps without regard to clusters"
            raise AssertionError(msg)

        monkeypatch.setattr(uproot.behaviors.RNTuple.HasFields, "iterate", steps_across_clusters)
        path = tmp_path / "clusters.root"
        self._clusters(path, "RNTuple", [1_000] * 10)  # 24 kB of arrays per cluster
        source = FileSource(path, tree="events", entry_start=1_500, entry_stop=8_500)
        chunks = list(source.iterate(["x"], chunk_bytes=80_000))
        stops = np.cumsum([len(chunk["x"]) for chunk in chunks]) + 1_500
        assert len(stops) > 2
        assert all(stop % 1_000 == 0 for stop in stops[:-1])  # at the ends of clusters
        assert max(chunk["x"].nbytes for chunk in chunks) <= 100_000
        joined = ak.concatenate([chunk["x"] for chunk in chunks])
        assert ak.array_equal(joined, source.arrays(["x"])["x"])

    def test_a_cluster_larger_than_a_chunk_is_read_once(self, tmp_path: Path) -> None:
        path = tmp_path / "cluster.root"
        self._clusters(path, "RNTuple", [100_000])  # 2.4 MB of arrays
        [chunk] = FileSource(path, tree="events").iterate(["x"], chunk_bytes=100_000)
        assert len(chunk["x"]) == 100_000

    def test_a_range_is_sized_by_its_own_events(self, tmp_path: Path) -> None:
        # the second half holds 99 values per event, the first one: a step from the
        # average of the whole tree would take twice as many events as fit
        path = tmp_path / "uneven.root"
        with uproot.recreate(path) as file:
            file.mktree("events", {"x": "var * float64"})
            for counts in [1] * 50 + [99] * 50:
                values = np.arange(1_000.0 * counts)
                file["events"].extend({"x": ak.unflatten(values, np.full(1_000, counts))})
        source = FileSource(path, tree="events", entry_start=60_000, entry_stop=90_000)
        chunks = list(source.iterate(["x"], chunk_bytes=2_000_000))
        assert sum(len(chunk["x"]) for chunk in chunks) == 30_000
        assert max(chunk["x"].nbytes for chunk in chunks) <= 2_500_000

    def test_embedded_baskets_are_sized_too(self) -> None:
        # a tree filled in memory keeps its baskets in its branches, without keys
        source = FileSource(DATA / "embedded_basket.root", tree="events")
        chunks = list(source.iterate(["x", "y"], chunk_bytes=1_000))
        whole = source.arrays(["x", "y"])
        assert len(chunks) > 1
        for name in ("x", "y"):
            assert ak.array_equal(ak.concatenate([chunk[name] for chunk in chunks]), whole[name])

    def test_split_collections_and_missing_branches(self) -> None:
        source = FileSource(DATA / "split_collection.root", tree="events")
        branches = ["ReconstructedParticles.energy", "ReconstructedParticles.momentum.x"]
        joined = self._joined(list(source.iterate(branches, chunk_bytes=500)))
        whole = source.arrays(branches)
        assert all(ak.array_equal(joined[name], whole[name]) for name in branches)
        rntuple = FileSource(DATA / "split_collection.root", tree="events")
        with pytest.raises(SourceError, match="not found"):
            list(rntuple.iterate(["ReconstructedParticles.nosuch"]))

    @pytest.mark.parametrize("entries", [(None, None), (10, None), (None, 5)])
    def test_a_file_without_the_tree_is_reported(
        self, signal_file: Path, tmp_path: Path, entries: tuple[int | None, int | None]
    ) -> None:
        other = tmp_path / "other.root"
        with uproot.recreate(other) as file:
            file.mktree("other", {"MET": np.arange(3.0)})
        start, stop = entries
        source = FileSource(
            [str(signal_file), str(other)], tree="events", entry_start=start, entry_stop=stop
        )
        with pytest.raises(SourceError, match="could not read 'events'"):
            source.arrays(["MET"])
        with pytest.raises(SourceError, match="could not read 'events'"):
            list(source.iterate(["MET"]))
        with pytest.raises(SourceError, match="could not read 'events'"):
            source.num_entries()

    def test_one_thread_starts_no_worker_pool(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading

        import rootfig._threads as threads_module

        pools: list[Any] = []

        def counted(*args: Any, **kwargs: Any) -> Any:
            pools.append(args)
            raise AssertionError("no pool with one thread")

        monkeypatch.setattr(threads_module, "THREADS", 1)
        monkeypatch.setattr(threads_module, "ThreadPoolExecutor", counted)
        before = threading.active_count()
        source = FileSource(signal_file, tree="events")
        with threads_module.worker_pool(lend=True) as pool:
            assert pool is None
            assert threading.active_count() == before
        whole = source.arrays(["MET"])
        [chunk] = source.iterate(["MET"], chunk_bytes=10**8)
        assert ak.array_equal(chunk["MET"], whole["MET"])
        assert pools == []  # uproot fetches the file contents in threads of its own

    def test_a_lent_worker_pool_serves_the_blocks_inside(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rootfig._threads as threads_module

        monkeypatch.setattr(threads_module, "THREADS", 2)
        with threads_module.worker_pool() as own, threads_module.worker_pool() as other:
            assert own is not None
            assert other is not own  # not lent
        with threads_module.worker_pool(lend=True) as lent:
            with threads_module.worker_pool() as inner:
                assert inner is lent
            assert lent is not None
            assert lent.submit(len, "ab").result() == 2  # still open: the inner block took it
        with threads_module.worker_pool() as after:
            assert after is not lent
        with pytest.raises(RuntimeError):
            lent.submit(len, "ab")

    def test_threads_follow_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rootfig._threads import _threads

        monkeypatch.setenv("ROOTFIG_THREADS", "3")
        assert _threads() == 3
        monkeypatch.setenv("ROOTFIG_THREADS", "0")
        assert _threads() == 1
        monkeypatch.setenv("ROOTFIG_THREADS", "many")
        with pytest.raises(ValueError, match="ROOTFIG_THREADS"):
            _threads()
        monkeypatch.delenv("ROOTFIG_THREADS")
        assert 1 <= _threads() <= 8


class TestStoredHistograms:
    def test_listing(self, stored_dir: Path) -> None:
        source = FileSource(stored_dir / "ZH_sel0_histo.root")
        assert source.trees() == []
        assert source.histograms() == ["cutflow", "eventsProcessed", "mz", "mz_raw", "mz_recoil_2D"]
        assert source.objects()["mz_recoil_2D"] == "TH2D"
        mixed = FileSource(stored_dir / "branch_and_histogram.root")
        assert mixed.trees() == ["events"]
        assert mixed.histograms() == ["mz"]
        nested = FileSource(stored_dir / "in_directory.root")
        assert nested.histograms() == ["sub/mz"]
        unsupported = FileSource(stored_dir / "unsupported.root")
        assert unsupported.histograms() == []
        assert unsupported.objects() == {"h3": "TH3D", "prof": "TProfile"}

    def test_read_sums_files_and_keeps_flow(self, stored_dir: Path) -> None:
        one = FileSource(stored_dir / "ZH_sel0_histo.root").read_histogram("mz")
        both = FileSource(
            [stored_dir / "ZH_sel0_histo.root", stored_dir / "WW_sel0_histo.root"]
        ).read_histogram("mz")
        assert both.axes[0].size == 100
        assert both.axes[0].label == "m_{Z} [GeV]"
        assert both.values(flow=True).sum() == pytest.approx(3000.0)  # 0.5 * (4000 + 2000)
        assert both.values(flow=True).sum() > one.values(flow=True).sum()
        assert both.storage_type.__name__ == "Weight"
        plain = FileSource(stored_dir / "ZH_sel0_histo.root").read_histogram("mz_raw")
        assert plain.storage_type.__name__ == "Weight"  # counts as variances
        assert plain.values(flow=True).sum() == 4000
        np.testing.assert_array_equal(plain.variances(flow=True), plain.values(flow=True))
        two_d = FileSource(stored_dir / "ZH_sel0_histo.root").read_histogram("mz_recoil_2D")
        assert two_d.ndim == 2
        assert [a.label for a in two_d.axes] == ["m_{Z} [GeV]", "recoil [GeV]"]
        nested = FileSource(stored_dir / "in_directory.root").read_histogram("sub/mz")
        assert nested.values().sum() == 500

    def test_errors_name_the_file(self, stored_dir: Path) -> None:
        files = [stored_dir / "ZH_sel0_histo.root", stored_dir / "other_binning.root"]
        with pytest.raises(SourceError, match=r"'mz_raw' not found in .*other_binning.*present"):
            FileSource(files).read_histogram("mz_raw")  # the second file lacks it
        with pytest.raises(BinningError, match="different binnings"):
            FileSource(files).read_histogram("mz")
        with pytest.raises(SourceError, match="not a 1D or 2D histogram"):
            FileSource(stored_dir / "branch_and_histogram.root").read_histogram("events")
        with pytest.raises(
            SourceError, match=r"is a \w*Directory, not a 1D or 2D histogram.*sub/mz"
        ):
            FileSource(stored_dir / "in_directory.root").read_histogram("sub")

    def test_sum_across_storages_and_titles(self, stored_dir: Path) -> None:
        # one file with Sumw2 and a title, one without either: added bin by bin
        files = [stored_dir / "ZH_sel0_histo.root", stored_dir / "mixed_storage.root"]
        total = FileSource(files).read_histogram("mz")
        assert total.storage_type.__name__ == "Weight"
        assert total.axes[0].label == "m_{Z} [GeV]"  # the first file's title
        assert total.values(flow=True).sum() == pytest.approx(0.5 * 4000 + 500)
        assert total.variances(flow=True).sum() == pytest.approx(0.25 * 4000 + 500)
        reverse = FileSource(files[::-1]).read_histogram("mz")
        assert reverse.axes[0].label == "xaxis"
        np.testing.assert_allclose(reverse.values(flow=True), total.values(flow=True))

    def test_negative_contents_without_sumw2(self, stored_dir: Path) -> None:
        source = FileSource(stored_dir / "negative.root")
        with pytest.raises(
            SourceError, match=r"negative\.root.*negative bin contents.*assume_poisson"
        ):
            source.read_histogram("mz")
        with pytest.warns(RootfigWarning, match="Poisson guess"):
            h = source.read_histogram("mz", assume_poisson=True)
        assert (h.variances() >= 0).all()
        np.testing.assert_array_equal(h.variances(), np.abs(h.values()))
        # a partner file cannot hide the problem in a sum either
        both = FileSource([stored_dir / "ZH_sel0_histo.root", stored_dir / "negative.root"])
        with pytest.raises(SourceError, match="negative bin contents"):
            both.read_histogram("mz")

    def test_category_axis_round_trip(self, stored_dir: Path) -> None:
        h = FileSource(stored_dir / "ZH_sel0_histo.root").read_histogram("cutflow")
        assert type(h.axes[0]).__name__ == "StrCategory"
        assert list(h.axes[0]) == ["all", "sel0", "sel1"]
        np.testing.assert_allclose(h.values(), [2000.0, 1000.0, 500.0])

    def test_object_map_is_read_once(
        self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rootfig.io import objects

        calls: list[str] = []
        original = objects.object_classes

        def counted(path: str) -> dict[str, str]:
            calls.append(path)
            return original(path)

        monkeypatch.setattr(objects, "object_classes", counted)
        source = FileSource(stored_dir / "tree_without_branch.root")
        assert source.histograms() == ["mz"]  # the stored-histogram lookup
        assert source.resolved_tree() == "events"  # tree detection reuses the same map
        assert source.trees() == ["events"]
        assert len(calls) == 1

    def test_histogram_classes(self) -> None:
        from rootfig.io.objects import is_histogram_class, is_tree_class

        assert is_histogram_class("TH1D")
        assert is_histogram_class("TH2F")
        assert not is_histogram_class("TH3D")
        assert not is_histogram_class("TProfile")
        assert not is_histogram_class("TTree")
        assert is_tree_class("ROOT::RNTuple")


def _opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the file name of every ``uproot.open`` call."""
    calls: list[str] = []
    original = uproot.open

    def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(uproot, "open", counting)
    return calls


class TestBatchReads:
    """Several stored histograms per file open, and the ReadCache in front of the reads."""

    def test_read_histograms_opens_each_file_once(
        self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rootfig.io import objects

        opens = _opens(monkeypatch)
        files = [str(stored_dir / "WW_sel0_histo.root"), str(stored_dir / "ZZ_sel0_histo.root")]
        names = ["mz", "mz_raw", "cutflow"]
        read = objects.read_histograms(files, names)
        assert opens == ["WW_sel0_histo.root", "ZZ_sel0_histo.root"]
        assert list(read) == names
        assert read["mz"].values(flow=True).sum() == pytest.approx(0.5 * 3000)
        for name in names:
            single = objects.read_histogram(files, name)
            np.testing.assert_array_equal(read[name].values(flow=True), single.values(flow=True))
            np.testing.assert_array_equal(
                read[name].variances(flow=True), single.variances(flow=True)
            )
        assert FileSource(files).read_histograms(["mz"])["mz"] == read["mz"]

    def test_read_histograms_errors_match_the_single_read(self, stored_dir: Path) -> None:
        from rootfig.io import objects

        files = [str(stored_dir / "WW_sel0_histo.root"), str(stored_dir / "other_binning.root")]
        with pytest.raises(BinningError) as single:
            objects.read_histogram(files, "mz")
        with pytest.raises(BinningError) as many:
            objects.read_histograms(files, ["mz"])
        assert str(many.value) == str(single.value)
        files = [
            str(stored_dir / "WW_sel0_histo.root"),
            str(stored_dir / "tree_without_branch.root"),
        ]
        missing = r"'mz_raw' not found in .*tree_without_branch"
        with pytest.raises(SourceError, match=missing) as one:
            objects.read_histogram(files, "mz_raw")
        with pytest.raises(SourceError) as several:
            objects.read_histograms(files, ["mz", "mz_raw"])
        assert str(several.value) == str(one.value)
        with pytest.raises(SourceError, match="no files"):
            objects.read_histograms([], ["mz"])

    def test_read_cache_reads_missing_branches_only(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        original = FileSource.arrays

        def counting(self: FileSource, branches: Any) -> Any:
            calls.append(list(branches))
            return original(self, branches)

        monkeypatch.setattr(FileSource, "arrays", counting)
        source = FileSource(signal_file, tree="events")
        cache = ReadCache()
        first = cache.arrays(source, ["MET"])
        assert calls == [["MET"]]
        assert list(first) == ["MET"]
        second = cache.arrays(source, ["nMuon", "MET"])
        assert calls == [["MET"], ["nMuon"]]
        assert list(second) == ["nMuon", "MET"]  # a fresh mapping, in the requested order
        assert second["MET"] is first["MET"]
        assert cache.arrays(source, ["nMuon", "MET"]) is not second
        # keyed by value: another source over the same file and tree shares the arrays
        assert cache.arrays(FileSource(signal_file, tree="events"), ["MET"])["MET"] is first["MET"]
        assert cache.arrays(source, []) == {}
        assert len(calls) == 2
        np.testing.assert_array_equal(
            ak.to_numpy(first["MET"]), ak.to_numpy(source.arrays(["MET"])["MET"])
        )

    def test_read_cache_histograms(self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from rootfig.io import objects

        calls: list[list[str]] = []
        original = objects.read_histograms

        def counting(files: Any, names: Any, **kwargs: Any) -> Any:
            calls.append(list(names))
            return original(files, names, **kwargs)

        monkeypatch.setattr(objects, "read_histograms", counting)
        source = FileSource(stored_dir / "WW_sel0_histo.root")
        cache = ReadCache()
        cache.histograms(source, ["mz", "mz_raw"])
        assert calls == [["mz", "mz_raw"]]
        h = cache.histogram(source, "mz")
        assert h is cache.histogram(source, "mz")
        assert len(calls) == 1
        cache.histograms(source, ["mz", "cutflow"])
        assert calls[-1] == ["cutflow"]
        cache.histograms(source, ["mz", "cutflow"])  # everything held: nothing is read
        assert len(calls) == 2
        cache.histogram(source, "mz", assume_poisson=True)  # another key
        assert len(calls) == 3
        assert h.values(flow=True).sum() == pytest.approx(0.5 * 2000)

    def test_read_cache_hands_out_one_source_per_value(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opens = _opens(monkeypatch)
        cache = ReadCache()
        first = FileSource(signal_file, tree="events")
        assert cache.source(first) is first
        again = FileSource(signal_file, tree="events")
        assert cache.source(again) is first  # equal by value: the first instance is kept
        other = FileSource(signal_file, tree="events", entry_stop=10)
        assert cache.source(other) is other
        assert cache.source(first).branches() == first.branches()
        assert cache.source(again).branches() == first.branches()
        assert opens == ["signal.root"]  # what the first instance learnt serves both

    def test_read_scalar_is_cached(self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        opens = _opens(monkeypatch)
        source = FileSource(stored_dir / "WW_sel0_histo.root")
        assert source.read_scalar("eventsProcessed") == 2000.0
        assert source.read_scalar("eventsProcessed") == 2000.0
        assert opens == ["WW_sel0_histo.root"]


class TestBranchForms:
    """Branch types from metadata: what ``PlotBook(data, rf.ALL)`` classifies."""

    def test_ttree_forms_without_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uproot

        path = tmp_path / "types.root"
        with uproot.recreate(path) as file:
            # the branch types are inferred from the arrays, the same on every uproot version
            file.mktree(
                "events",
                {
                    "f": np.zeros(2),
                    "i": np.zeros(2, dtype=np.int16),
                    "u": np.zeros(2, dtype=np.uint8),
                    "b": np.zeros(2, dtype=bool),
                    "jag": ak.values_astype(ak.Array([[1.0], []]), np.float32),
                    "fixed": np.zeros((2, 3)),
                    "s": ak.Array(["a", "b"]),
                    "odd-name": np.zeros(2, dtype=np.float32),
                },
            )
        source = FileSource(path, tree="events")
        monkeypatch.setattr(FileSource, "arrays", lambda *a, **k: pytest.fail("read data"))
        forms = source.branch_forms()
        assert {name: str(form.type) for name, form in forms.items()} == {
            "f": "float64",
            "i": "int16",
            "u": "uint8",
            "b": "bool",
            "njag": "int32",
            "jag": "var * float32",
            "fixed": "3 * float64",
            "s": "string",
            "odd-name": "float32",
        }
        assert list(forms) == source.branches()
        assert source.branch_forms() is not forms  # a copy each time, the forms shared
        assert source.branch_forms()["f"] is forms["f"]

    def test_rntuple_forms_follow_nested_fields(self, data_dir: Path, tmp_path: Path) -> None:
        import uproot

        path = tmp_path / "nested.root"
        with uproot.recreate(path) as file:
            file["events"] = {
                "Muon": ak.zip({"pt": [[1.0, 2.0], [3.0]], "q": [[1, -1], [1]]}),
                "Vertex": ak.Array([{"x": 1.0}, {"x": 2.0}]),
                "met": np.array([1.0, 2.0]),
                "s": ak.Array(["a", "b"]),
            }
        forms = FileSource(path, tree="events").branch_forms()
        assert {name: str(form.type) for name, form in forms.items()} == {
            "Muon.pt": "var * float64",
            "Muon.q": "var * int64",
            "Vertex.x": "float64",
            "met": "float64",
            "s": "string",
        }
        # the physics fixture as TTree and as RNTuple describe the same branches
        ttree = FileSource(data_dir / "signal.root", tree="events").branch_forms()
        rntuple = FileSource(data_dir / "signal_rntuple.root", tree="events").branch_forms()
        assert {n: str(f.type) for n, f in rntuple.items()} == {
            n: str(f.type) for n, f in ttree.items() if not n.startswith("nMuon_")
        }

    def test_array_source_forms(self) -> None:
        source = ArraySource(
            {"x": np.arange(3.0), "jag": ak.Array([[1], [], [2, 3]]), "s": ["a", "b", "c"]}
        )
        assert {n: str(f.type) for n, f in source.branch_forms().items()} == {
            "x": "float64",
            "jag": "var * int64",
            "s": "string",
        }

    def test_split_collection_forms(self) -> None:
        forms = FileSource(DATA / "split_collection.root", tree="events").branch_forms()
        assert str(forms["ReconstructedParticles.energy"].type) == "var * float32"
        assert str(forms["ReconstructedParticles.charge"].type) == "var * int32"

    def test_histograms_by_dimension(self, stored_dir: Path) -> None:
        from rootfig.io.objects import histogram_dimension, histogram_names, is_histogram_class

        source = FileSource(stored_dir / "ZH_sel0_histo.root")
        assert source.histograms(ndim=1) == ["cutflow", "eventsProcessed", "mz", "mz_raw"]
        assert source.histograms(ndim=2) == ["mz_recoil_2D"]
        assert source.histograms() == source.histograms(1) + source.histograms(2)
        assert histogram_names(str(stored_dir / "in_directory.root"), ndim=1) == ["sub/mz"]
        assert [histogram_dimension(c) for c in ("TH1D", "TH1F", "TH2D", "TH3F", "TProfile")] == [
            1,
            1,
            2,
            None,
            None,
        ]
        assert histogram_dimension("TTree") is None
        assert histogram_dimension("THnSparseD") is None
        assert is_histogram_class("TH2D")
        assert is_histogram_class("TH2D", ndim=2)
        assert not is_histogram_class("TH2D", ndim=1)


class TestSchema:
    """Classifying forms: numeric and boolean leaves under any list or option wrapper."""

    @pytest.mark.parametrize(
        ("array", "dtype"),
        [
            (ak.Array([1.0, 2.0]), "float64"),
            (ak.Array([True]), "bool"),
            (ak.Array(np.array([1], dtype=np.uint16)), "uint16"),
            (ak.Array([[1, 2], []]), "int64"),
            (ak.Array([[[1.5]], []]), "float64"),
            (ak.to_regular(ak.Array([[1.0, 2.0]])), "float64"),
            (ak.Array([1.0, None]), "float64"),
            (ak.Array([[], []]), "float64"),
            (ak.Array([1 + 1j]), "complex128"),
        ],
    )
    def test_leaf_dtype(self, array: ak.Array, dtype: str) -> None:
        from rootfig.io.schema import leaf_dtype

        assert leaf_dtype(ak.to_layout(array).form) == np.dtype(dtype)

    @pytest.mark.parametrize(
        "array",
        [
            ak.Array(["a", "b"]),
            ak.Array([b"a"]),
            ak.Array([["a"], []]),
            ak.Array([{"x": 1.0}]),
            ak.Array([[{"x": 1.0}], []]),
            ak.Array([1, "a"]),
        ],
    )
    def test_text_records_and_unions_have_no_leaf(self, array: ak.Array) -> None:
        from rootfig.io.schema import leaf_dtype, plottable

        form = ak.to_layout(array).form
        assert leaf_dtype(form) is None
        assert not plottable(form)

    def test_plottable_kinds(self) -> None:
        from rootfig.io.schema import plottable, plottable_names

        forms = {
            n: ak.to_layout(a).form
            for n, a in {
                "f": ak.Array([1.0]),
                "b": ak.Array([True]),
                "i": ak.Array([1]),
                "c": ak.Array([1j]),
                "t": ak.Array([np.datetime64("2020-01-01")]),
                "s": ak.Array(["x"]),
            }.items()
        }
        assert plottable_names(forms) == ["f", "b", "i"]
        assert not plottable(forms["t"])

    def test_select_field(self) -> None:
        from rootfig.io.schema import select_field

        collection = ak.to_layout(ak.Array([[{"pt": 1.0, "q": 1}], []])).form
        assert str(select_field(collection, "pt").type) == "var * float64"
        record = ak.to_layout(ak.Array([{"x": 1.0}])).form
        assert str(select_field(record, "x").type) == "float64"
        optional = ak.to_layout(ak.Array([{"x": 1.0}, None])).form
        assert str(select_field(optional, "x").type) == "?float64"
        with pytest.raises(KeyError):
            select_field(record, "y")
        with pytest.raises(KeyError):
            select_field(ak.to_layout(ak.Array([1.0])).form, "x")


class TestBranchFormFailures:
    """Only uproot's own "cannot read this branch" errors leave a branch out; the rest surface."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        import uproot

        path = tmp_path / "plain.root"
        with uproot.recreate(path) as file:
            file.mktree("events", {"x": "float64", "y": "float64"}).extend(
                {"x": np.zeros(2), "y": np.zeros(2)}
            )
        return path

    @staticmethod
    def _failing_interpretation(
        monkeypatch: pytest.MonkeyPatch, branch: str, error: Exception
    ) -> None:
        import uproot

        original = uproot.behaviors.TBranch.TBranch.interpretation

        def interpretation(self: Any) -> Any:
            if self.name == branch:
                raise error
            return original.fget(self)

        monkeypatch.setattr(
            uproot.behaviors.TBranch.TBranch, "interpretation", property(interpretation)
        )

    def test_unknown_interpretation_is_left_out(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from uproot.interpretation.identify import UnknownInterpretation

        self._failing_interpretation(
            monkeypatch, "y", UnknownInterpretation("no streamer", str(path), "events/y")
        )
        assert list(FileSource(path, tree="events").branch_forms()) == ["x"]

    def test_interpretation_without_form_is_left_out(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from uproot.interpretation.objects import CannotBeAwkward

        class Formless:
            def awkward_form(self, *args: Any, **kwargs: Any) -> Any:
                raise CannotBeAwkward("Formless")

        import uproot

        original = uproot.behaviors.TBranch.TBranch.interpretation
        monkeypatch.setattr(
            uproot.behaviors.TBranch.TBranch,
            "interpretation",
            property(lambda self: Formless() if self.name == "x" else original.fget(self)),
        )
        assert list(FileSource(path, tree="events").branch_forms()) == ["y"]

    def test_unexpected_errors_surface(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._failing_interpretation(monkeypatch, "y", RuntimeError("broken streamer"))
        with pytest.raises(RuntimeError, match="broken streamer"):
            FileSource(path, tree="events").branch_forms()
