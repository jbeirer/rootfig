"""Tests for the selection semantics: depth rules, weights, flattening, non-finite handling."""

from __future__ import annotations

import warnings
from itertools import pairwise
from typing import Any, ClassVar

import awkward as ak
import numpy as np
import pytest

from rootfig.errors import IncompatibleWeightError, RootfigWarning, SelectionError
from rootfig.selection import Columns, depth_of, prepare
from rootfig.selection.columns import same_structure


@pytest.fixture
def arrays() -> dict[str, ak.Array]:
    return {
        "met": ak.Array([10.0, 20.0, 30.0, 40.0]),
        "njet": ak.Array([2, 0, 1, 3]),
        "w": ak.Array([0.5, 1.0, 2.0, 4.0]),
        "jet_pt": ak.Array([[10.0, 30.0], [], [50.0], [5.0, 60.0, 70.0]]),
        "jet_eta": ak.Array([[0.5, -1.5], [], [2.0], [0.1, -0.2, 3.0]]),
        "jet_w": ak.Array([[1.0, 2.0], [], [3.0], [1.0, 1.0, 2.0]]),
        "el_pt": ak.Array([[7.0], [8.0, 9.0], [], []]),
        "name": ak.Array(["a", "b", "c", "d"]),
    }


class TestStructure:
    def test_depth(self, arrays: dict[str, ak.Array]) -> None:
        assert depth_of(arrays["met"]) == 1
        assert depth_of(arrays["jet_pt"]) == 2
        assert depth_of(ak.Array([[[1.0]], []])) == 3

    def test_strings_rejected(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match="string"):
            depth_of(arrays["name"])

    def test_same_structure(self, arrays: dict[str, ak.Array]) -> None:
        assert same_structure(arrays["jet_pt"], arrays["jet_eta"])
        assert not same_structure(arrays["jet_pt"], arrays["el_pt"])
        assert not same_structure(arrays["jet_pt"], arrays["met"])
        assert same_structure(arrays["met"], arrays["w"])
        assert not same_structure(arrays["met"], arrays["met"][:2])


class TestEventLevel:
    def test_no_selection(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met")
        assert isinstance(cols, Columns)
        assert cols.values.tolist() == [10.0, 20.0, 30.0, 40.0]
        assert cols.weights is None
        assert cols.n_events == 4
        assert cols.n_selected_events == 4
        assert cols.n_entries == 4
        assert cols.sum_weights == 4.0
        assert cols.effective_weights().tolist() == [1.0] * 4

    def test_event_selection(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met", selection="met > 15 and njet > 0")
        assert cols.values.tolist() == [30.0, 40.0]
        assert cols.n_selected_events == 2

    def test_derived_expression(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "sqrt(met) * 2", selection="njet >= 1")
        assert cols.values.tolist() == pytest.approx(
            [2 * np.sqrt(10.0), 2 * np.sqrt(30.0), 2 * np.sqrt(40.0)]
        )

    def test_event_weight(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met", selection="met >= 20", weight="w")
        assert cols.values.tolist() == [20.0, 30.0, 40.0]
        assert cols.weights is not None
        assert cols.weights.tolist() == [1.0, 2.0, 4.0]
        assert cols.sum_weights == 7.0

    def test_weight_expression_and_scale(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met", weight="w * 2", scale=0.5)
        assert cols.weights is not None
        assert cols.weights.tolist() == [0.5, 1.0, 2.0, 4.0]

    def test_scale_without_weight(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met", scale=3.0)
        assert cols.weights is not None
        assert cols.weights.tolist() == [3.0] * 4

    def test_constant_weight(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met", weight="2.5")
        assert cols.weights is not None
        assert cols.weights.tolist() == [2.5] * 4

    def test_empty_selection(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met", selection="met > 1000", weight="w")
        assert cols.n_entries == 0
        assert cols.n_selected_events == 0
        assert cols.weights is not None
        assert cols.weights.size == 0

    def test_boolean_variable(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met > 25")
        assert cols.values.tolist() == [0.0, 0.0, 1.0, 1.0]

    def test_integer_variable(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "njet")
        assert cols.values.dtype == np.float64
        assert cols.values.tolist() == [2.0, 0.0, 1.0, 3.0]

    def test_non_boolean_selection(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match="must be boolean"):
            prepare(arrays, "met", selection="njet")

    def test_string_variable(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match="string"):
            prepare(arrays, "name")


class TestObjectLevel:
    def test_flatten(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "jet_pt")
        assert cols.values.tolist() == [10.0, 30.0, 50.0, 5.0, 60.0, 70.0]
        assert cols.n_events == 4
        assert cols.n_selected_events == 3  # events with at least one jet

    def test_object_selection(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "jet_pt", selection="jet_pt > 20 and abs(jet_eta) < 2.5")
        assert cols.values.tolist() == [30.0, 50.0, 60.0]
        assert cols.n_selected_events == 3

    def test_event_selection_on_jagged(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "jet_pt", selection="met > 25")
        assert cols.values.tolist() == [50.0, 5.0, 60.0, 70.0]

    def test_mixed_selection_broadcasts(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "jet_pt", selection="jet_pt > 20 and met > 25")
        assert cols.values.tolist() == [50.0, 60.0, 70.0]

    def test_object_selection_on_event_variable_fails(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match=r"any\(jet_pt > 20\)"):
            prepare(arrays, "met", selection="jet_pt > 20")

    def test_reduced_selection_on_event_variable(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "met", selection="any(jet_pt > 20)")
        assert cols.values.tolist() == [10.0, 30.0, 40.0]
        cols = prepare(arrays, "met", selection="count(jet_pt) >= 2")
        assert cols.values.tolist() == [10.0, 40.0]

    def test_mismatched_collections_fail(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match="different structures"):
            prepare(arrays, "jet_pt", selection="el_pt > 5")

    def test_event_weight_broadcast(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "jet_pt", weight="w")
        assert cols.weights is not None
        assert cols.weights.tolist() == [0.5, 0.5, 2.0, 4.0, 4.0, 4.0]

    def test_event_weight_broadcast_with_object_selection(
        self, arrays: dict[str, ak.Array]
    ) -> None:
        cols = prepare(arrays, "jet_pt", selection="jet_pt > 20", weight="w")
        assert cols.values.tolist() == [30.0, 50.0, 60.0, 70.0]
        assert cols.weights is not None
        assert cols.weights.tolist() == [0.5, 2.0, 4.0, 4.0]

    def test_event_weight_with_event_selection_on_jagged(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "jet_pt", selection="met > 25", weight="w")
        assert cols.weights is not None
        assert cols.weights.tolist() == [2.0, 4.0, 4.0, 4.0]

    def test_object_weight(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "jet_pt", selection="jet_pt > 20", weight="jet_w * w")
        assert cols.weights is not None
        assert cols.weights.tolist() == [1.0, 6.0, 4.0, 8.0]

    def test_object_weight_on_event_variable_fails(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(IncompatibleWeightError, match="per-object"):
            prepare(arrays, "met", weight="jet_w")

    def test_object_weight_mismatched_structure_fails(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(IncompatibleWeightError, match="different structure"):
            prepare(arrays, "jet_pt", weight="el_pt")

    def test_per_event_reduction_as_variable(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, "sum(jet_pt)", selection="njet > 0")
        assert cols.values.tolist() == [40.0, 50.0, 135.0]

    def test_leading_object(self, arrays: dict[str, ak.Array]) -> None:
        # first() yields None for empty events; missing values are dropped and counted
        cols = prepare(arrays, "first(jet_pt)")
        assert cols.values.tolist() == [10.0, 50.0, 5.0]
        assert cols.n_missing == 1
        assert cols.n_nonfinite == 0

    def test_leading_object_with_selection_on_missing(self, arrays: dict[str, ak.Array]) -> None:
        # comparison with None is None -> treated as False
        cols = prepare(arrays, "met", selection="max(jet_pt) > 20")
        assert cols.values.tolist() == [10.0, 30.0, 40.0]

    def test_deeper_nesting(self) -> None:
        arrays = {"x": ak.Array([[[1.0, 2.0], [3.0]], [], [[4.0]]]), "w": ak.Array([1.0, 2.0, 3.0])}
        cols = prepare(arrays, "x", selection="x > 1.5", weight="w")
        assert cols.values.tolist() == [2.0, 3.0, 4.0]
        assert cols.weights is not None
        assert cols.weights.tolist() == [1.0, 1.0, 3.0]


class TestTwoVariables:
    def test_aligned_pairs(self, arrays: dict[str, ak.Array]) -> None:
        cols = prepare(arrays, ["jet_pt", "jet_eta"], selection="jet_pt > 20", weight="w")
        assert len(cols.arrays) == 2
        assert cols.arrays[0].tolist() == [30.0, 50.0, 60.0, 70.0]
        assert cols.arrays[1].tolist() == [-1.5, 2.0, -0.2, 3.0]

    def test_structure_mismatch(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match="different structures"):
            prepare(arrays, ["jet_pt", "met"])

    def test_empty_list(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match="at least one"):
            prepare(arrays, [])


class TestNonFinite:
    @pytest.fixture
    def dirty(self) -> dict[str, ak.Array]:
        return {
            "x": ak.Array([1.0, np.nan, 3.0, np.inf, -np.inf, 6.0]),
            "w": ak.Array([1.0, 1.0, np.nan, 1.0, 1.0, 2.0]),
        }

    def test_dropped_with_warning(self, dirty: dict[str, ak.Array]) -> None:
        with pytest.warns(RootfigWarning, match="dropped 3 non-finite"):
            cols = prepare(dirty, "x")
        assert cols.values.tolist() == [1.0, 3.0, 6.0]
        assert cols.n_nonfinite == 3
        assert cols.n_missing == 0

    def test_selected_events_exclude_nonfinite(self, dirty: dict[str, ak.Array]) -> None:
        # n_selected_events counts events that contribute an entry, so dropped values
        # (and dropped weights) do not count
        with pytest.warns(RootfigWarning):
            assert prepare(dirty, "x").n_selected_events == 3
        with pytest.warns(RootfigWarning):
            assert prepare(dirty, "x", weight="w").n_selected_events == 2
        with pytest.warns(RootfigWarning):
            cols = prepare({"j": ak.Array([[1.0, np.nan], [np.nan], [], [2.0]])}, "j")
        assert (cols.n_entries, cols.n_selected_events) == (2, 2)
        with pytest.warns(RootfigWarning):
            cols = prepare(
                {"j": ak.Array([[1.0], [2.0]]), "w": ak.Array([1.0, np.nan])}, "j", weight="w"
            )
        assert cols.n_selected_events == 1

    def test_policy_is_validated(self, dirty: dict[str, ak.Array]) -> None:
        from rootfig.selection import event_weights

        with pytest.raises(SelectionError, match="nonfinite must be"):
            prepare(dirty, "x", nonfinite="raise")  # type: ignore[arg-type]
        with pytest.raises(SelectionError, match="nonfinite must be"):
            event_weights("w", dirty, 6, nonfinite="raise")  # type: ignore[arg-type]

    def test_nonfinite_weights_dropped(self, dirty: dict[str, ak.Array]) -> None:
        with pytest.warns(RootfigWarning, match="dropped 4 non-finite"):
            cols = prepare(dirty, "x", weight="w")
        assert cols.values.tolist() == [1.0, 6.0]
        assert cols.weights is not None
        assert cols.weights.tolist() == [1.0, 2.0]

    def test_error_policy(self, dirty: dict[str, ak.Array]) -> None:
        with pytest.raises(SelectionError, match="non-finite"):
            prepare(dirty, "x", nonfinite="error")

    def test_context_in_warning(self, dirty: dict[str, ak.Array]) -> None:
        with pytest.warns(RootfigWarning, match="^Signal: dropped"):
            prepare(dirty, "x", context="Signal")

    def test_clean_data_no_warning(self, arrays: dict[str, ak.Array]) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prepare(arrays, "met")


class TestRegularArrays:
    """Fixed-size collections (``float x[3]``, 2D NumPy arrays) behave like jagged lists."""

    @pytest.fixture
    def regular(self) -> dict[str, ak.Array]:
        return {
            "x": ak.Array(np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])),
            "w": ak.Array([1.0, 2.0]),
            "met": ak.Array([5.0, 50.0]),
            "ow": ak.Array(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])),
        }

    def test_event_weight_broadcasts_along_events(self, regular: dict[str, ak.Array]) -> None:
        cols = prepare(regular, "x", weight="w")
        assert cols.weights is not None
        assert cols.weights.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        assert cols.per_object

    def test_object_selection_masks_objects(self, regular: dict[str, ak.Array]) -> None:
        cols = prepare(regular, "x", selection="x > 25")
        assert cols.values.tolist() == [30.0, 40.0, 50.0, 60.0]
        assert cols.n_selected_events == 2

    def test_event_selection_and_object_weight(self, regular: dict[str, ak.Array]) -> None:
        cols = prepare(regular, "x", selection="met > 10", weight="ow")
        assert cols.values.tolist() == [40.0, 50.0, 60.0]
        assert cols.weights is not None
        assert cols.weights.tolist() == [2.0, 2.0, 2.0]
        cols = prepare(regular, "x * 2", selection="x >= 50 and met > 10")
        assert cols.values.tolist() == [100.0, 120.0]

    def test_regular_reductions_are_per_event(self, regular: dict[str, ak.Array]) -> None:
        cols = prepare(regular, "max(x)", selection="count(x) == 3")
        assert cols.values.tolist() == [30.0, 60.0]
        assert not cols.per_object


class TestMissingCollections:
    def test_missing_list_drops_the_event(self) -> None:
        cols = prepare({"x": ak.Array([[1.0, 2.0], None, [3.0]])}, "x")
        assert cols.values.tolist() == [1.0, 2.0, 3.0]
        assert cols.n_missing == 1
        assert cols.n_selected_events == 2
        assert cols.n_events == 3

    def test_missing_event_weight_drops_its_objects(self) -> None:
        cols = prepare(
            {"x": ak.Array([[1.0, 2.0], [3.0]]), "w": ak.Array([None, 2.0])}, "x", weight="w"
        )
        assert cols.values.tolist() == [3.0]
        assert cols.weights is not None
        assert cols.weights.tolist() == [2.0]
        assert cols.n_missing == 2  # the two objects of the event without a weight

    def test_missing_object_weight_list(self) -> None:
        cols = prepare(
            {"x": ak.Array([[1.0, 2.0], [3.0]]), "w": ak.Array([[1.0, 1.0], None])},
            "x",
            weight="w",
        )
        assert cols.values.tolist() == [1.0, 2.0]
        assert cols.n_missing == 1

    def test_missing_inner_list_and_leaf(self) -> None:
        cols = prepare({"x": ak.Array([[[1.0], None], [[2.0, None]]])}, "x")
        assert cols.values.tolist() == [1.0, 2.0]
        assert cols.n_missing == 2

    def test_missing_list_in_an_object_selection_drops_the_event(self) -> None:
        # the selection's collection is missing where the variable's is (event 1), or only
        # the selection's (event 2); a missing value inside a list counts as False (event 3)
        arrays = {
            "x": ak.Array([[1.0, 2.0], None, [3.0], [4.0, 5.0], [6.0]]),
            "eta": ak.Array([[0.1, -0.2], None, None, [0.5, None], [0.7]]),
            "w": ak.Array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        cols = prepare(arrays, "x", selection="eta > 0", weight="w")
        assert cols.values.tolist() == [1.0, 4.0, 6.0]
        assert cols.weights is not None
        assert cols.weights.tolist() == [1.0, 4.0, 5.0]
        assert cols.n_missing == 2  # one per missing list; the None inside a list is False
        assert cols.n_selected_events == 3

    def test_missing_list_fails_a_cut_flow_step(self) -> None:
        from rootfig.selection import event_mask

        mask = event_mask("eta > 0", {"eta": ak.Array([[0.1], None, [-0.1, None]])})
        assert mask.tolist() == [True, False, False]

    def test_two_variables_stay_aligned(self) -> None:
        arrays = {
            "x": ak.Array([[1.0, 2.0], None, [3.0]]),
            "y": ak.Array([[10.0, 20.0], [5.0], None]),
        }
        cols = prepare(arrays, ["x", "y"])
        assert cols.arrays[0].tolist() == [1.0, 2.0]
        assert cols.arrays[1].tolist() == [10.0, 20.0]
        assert cols.n_missing == 2


class TestMissingVersusNonFinite:
    def test_missing_field_does_not_hide_a_nan(self) -> None:
        arrays = {"x": ak.Array([None, np.nan, 1.0]), "w": ak.Array([None, 1.0, 1.0])}
        with pytest.raises(SelectionError, match="non-finite"):
            prepare(arrays, "x", weight="w", nonfinite="error")
        with pytest.warns(RootfigWarning, match="1 non-finite"):
            cols = prepare(arrays, "x", weight="w")
        assert cols.values.tolist() == [1.0]
        assert (cols.n_missing, cols.n_nonfinite) == (1, 1)

    def test_missing_and_nan_in_one_entry_counts_once(self) -> None:
        arrays = {"x": ak.Array([None, 2.0]), "w": ak.Array([np.nan, 1.0])}
        cols = prepare(arrays, "x", weight="w", nonfinite="error")  # missing takes precedence
        assert (cols.n_missing, cols.n_nonfinite) == (1, 0)

    def test_2d_counts(self) -> None:
        arrays = {"x": ak.Array([1.0, np.nan, None, 4.0]), "y": ak.Array([1.0, 2.0, 3.0, np.inf])}
        with pytest.warns(RootfigWarning, match="2 non-finite"):
            cols = prepare(arrays, ["x", "y"])
        assert cols.arrays[0].tolist() == [1.0]
        assert (cols.n_missing, cols.n_nonfinite) == (1, 2)


class TestScaleAndConstants:
    def test_scale_must_be_finite(self) -> None:
        with pytest.raises(IncompatibleWeightError, match="finite"):
            prepare({"x": ak.Array([1.0])}, "x", scale=np.inf)

    def test_constant_expressions_use_n_events(self) -> None:
        cols = prepare({}, "1", n_events=3)
        assert cols.values.tolist() == [1.0, 1.0, 1.0]
        cols = prepare({}, "1", weight="2", selection="True", n_events=2)
        assert cols.weights is not None
        assert cols.weights.tolist() == [2.0, 2.0]

    def test_boolean_mask_rejects_numbers(self) -> None:
        from rootfig.selection import boolean_mask

        with pytest.raises(SelectionError, match="must be boolean"):
            boolean_mask("flag", {"flag": ak.Array([0, 1, 0])})
        assert boolean_mask("flag != 0", {"flag": ak.Array([0, 1, None])}).tolist() == [
            False,
            True,
            False,
        ]
        regular = boolean_mask("x > 1", {"x": ak.Array(np.arange(4.0).reshape(2, 2))})
        assert regular.tolist() == [[False, False], [True, True]]


class TestUnitWeights:
    """Weights of 0 and 1 fill counts, as without weights; judged before the selection."""

    ARRAYS: ClassVar[dict[str, ak.Array]] = {
        "x": ak.Array([1.0, 2.0, 3.0, 4.0]),
        "w": ak.Array([1.0, 0.0, 1.0, 1.0]),
        "half": ak.Array([0.5, 0.5, 0.5, 0.5]),
        "pt": ak.Array([[1.0, 2.0], [], None, [3.0]]),
        "ow": ak.Array([[1.0, 0.0], [], None, [1.0]]),
    }

    @pytest.mark.parametrize(
        ("weight", "options", "weighted"),
        [
            (None, {}, False),
            ("1", {}, False),
            ("w", {}, False),
            ("x > 2", {}, False),  # a boolean weight
            ("half", {}, True),
            ("w", {"scale": 2.0}, True),
            (None, {"scale": 2.0}, True),
            ("half", {"selection": "x > 10"}, True),  # nothing selected: still its weights
            ("w", {"selection": "x > 10"}, False),
        ],
    )
    def test_event_weights(self, weight: str | None, options: Any, weighted: bool) -> None:
        cols = prepare(self.ARRAYS, "x", weight=weight, **options)
        assert cols.weighted is weighted

    def test_object_weights_and_chunks(self) -> None:
        assert not prepare(self.ARRAYS, "pt", weight="ow").weighted
        assert prepare(self.ARRAYS, "pt", weight="ow * 3").weighted
        halves = [
            prepare({"x": ak.Array([1.0]), "w": ak.Array([w])}, "x", weight="w") for w in (1.0, 0.5)
        ]
        assert Columns.concatenate(halves).weighted
        assert not Columns.concatenate([halves[0], halves[0]]).weighted


class TestEventWeightsPolicy:
    def test_nonfinite_weights_are_reported_not_zeroed(self) -> None:
        from rootfig.selection import event_weights

        arrays = {"w": ak.Array([1.0, np.inf, None])}
        with pytest.warns(RootfigWarning, match="1 event"):
            weights = event_weights("w", arrays, 3)
        assert weights[0] == 1.0
        assert np.isnan(weights[1:]).all()
        with pytest.raises(SelectionError, match="non-finite weight"):
            event_weights("w", arrays, 3, nonfinite="error")
        assert event_weights("2", {}, 2).tolist() == [2.0, 2.0]


class TestPrepareChunks:
    """prepare_chunks() gives what prepare() gives for all events, a chunk at a time."""

    @pytest.fixture
    def events(self) -> dict[str, ak.Array]:
        return {
            "x": ak.Array([[1.0, np.nan], None, [3.0, 4.0, None], [], [5.0], [np.inf, 6.0]]),
            "pt": ak.Array([[1.0, np.nan], [2.0], [3.0, 4.0, None], [], [5.0], [np.inf, 6.0]]),
            "eta": ak.Array([[0.1, 0.2], [0.4], [-1.0, 1.0, 0.5], [], [2.0], [0.3, -0.3]]),
            "w": ak.Array([1.0, 2.0, None, 1.5, np.nan, 0.5]),
            "met": ak.Array([10.0, np.nan, 30.0, 40.0, 50.0, 60.0]),
        }

    @staticmethod
    def _chunks(events: dict[str, ak.Array], edges: list[int]) -> list[Any]:
        return [
            ({name: array[start:stop] for name, array in events.items()}, stop - start)
            for start, stop in pairwise(edges)
        ]

    @staticmethod
    def _assert_same(got: Columns, want: Columns) -> None:
        assert len(got.arrays) == len(want.arrays)
        for mine, theirs in zip(got.arrays, want.arrays, strict=True):
            np.testing.assert_array_equal(mine, theirs)
        if want.weights is None:
            assert got.weights is None
        else:
            assert got.weights is not None
            np.testing.assert_array_equal(got.weights, want.weights)
        assert (got.n_events, got.n_selected_events, got.n_missing, got.n_nonfinite) == (
            want.n_events,
            want.n_selected_events,
            want.n_missing,
            want.n_nonfinite,
        )
        assert got.per_object == want.per_object

    @pytest.mark.parametrize("threads", [1, 3])
    @pytest.mark.parametrize("edges", [[0, 6], [0, 1, 2, 3, 4, 5, 6], [0, 0, 2, 2, 5, 6]])
    @pytest.mark.parametrize(
        ("variables", "kwargs"),
        [
            (("pt",), {"selection": "eta > 0", "weight": "w"}),
            (("x",), {"selection": "count(x) > 1", "weight": "w", "scale": 2.0}),
            (("met",), {"weight": "w"}),
            (("pt", "eta"), {"selection": "met > 20"}),
        ],
    )
    def test_chunks_join_to_the_whole(
        self,
        events: dict[str, ak.Array],
        monkeypatch: pytest.MonkeyPatch,
        threads: int,
        edges: list[int],
        variables: tuple[str, ...],
        kwargs: dict[str, Any],
    ) -> None:
        import rootfig._threads as threads_module
        from rootfig.selection import Request, prepare_chunks

        monkeypatch.setattr(threads_module, "THREADS", threads)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RootfigWarning)
            want = prepare(events, list(variables), n_events=6, **kwargs)
            [got] = prepare_chunks(self._chunks(events, edges), [Request(variables, **kwargs)])
        self._assert_same(got, want)

    def test_nonfinite_values_are_reported_once_with_their_total(
        self, events: dict[str, ak.Array]
    ) -> None:
        from rootfig.selection import Request, prepare_chunks

        chunks = self._chunks(events, [0, 1, 2, 3, 4, 5, 6])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare_chunks(chunks, [Request(("x",), weight="w", context="S")])
        messages = [str(w.message) for w in caught if issubclass(w.category, RootfigWarning)]
        assert messages == ["S: dropped 3 non-finite (nan/inf) value(s) for 'x'"]
        with pytest.raises(SelectionError, match="dropped 3 non-finite"):
            prepare_chunks(chunks, [Request(("x",), weight="w", nonfinite="error")])

    def test_requests_share_the_chunks(self, events: dict[str, ak.Array]) -> None:
        from rootfig.selection import Request, prepare_chunks

        chunks = self._chunks(events, [0, 3, 6])
        shifted = {**events, "eta": events["eta"] * -1}
        requests = [Request(("pt",), selection="eta > 0"), Request(("pt",), selection="eta < 0")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RootfigWarning)
            nominal, flipped = prepare_chunks(chunks, requests)
            [substituted] = prepare_chunks(
                self._chunks({**events, "eta_down": shifted["eta"]}, [0, 3, 6]),
                [Request(("pt",), selection="eta > 0", substitutes={"eta": "eta_down"})],
            )
        self._assert_same(substituted, flipped)
        assert nominal.n_entries + flipped.n_entries == 6  # every finite pt but the one at 0

    def test_errors_carry_the_note(self, events: dict[str, ak.Array]) -> None:
        from rootfig.selection import Request, prepare_chunks

        request = Request(("met",), selection="eta > 0", note="while evaluating X")
        with pytest.raises(SelectionError, match="per-object") as info:
            prepare_chunks(self._chunks(events, [0, 3, 6]), [request])
        assert info.value.__notes__ == ["while evaluating X"]
        with pytest.raises(ValueError, match="no chunk"):
            prepare_chunks([], [request])

    def test_threads_see_the_callers_error_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rootfig._threads as threads_module
        from rootfig.selection import Request, prepare_chunks

        monkeypatch.setattr(threads_module, "THREADS", 3)
        events = {"x": ak.Array([[0.0, 1.0], [2.0], [0.0]] * 4)}
        chunks = self._chunks(events, list(range(13)))
        with np.errstate(divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # log(0) would warn in a worker
            warnings.simplefilter("ignore", RootfigWarning)  # the -inf are dropped, as usual
            [columns] = prepare_chunks(chunks, [Request(("log(x)",))])
        assert columns.n_nonfinite == 8

    @pytest.mark.parametrize("threads", [1, 3])
    def test_a_failing_chunk_releases_the_reader(
        self, events: dict[str, ak.Array], monkeypatch: pytest.MonkeyPatch, threads: int
    ) -> None:
        import rootfig._threads as threads_module
        from rootfig.selection import Request, prepare_chunks

        monkeypatch.setattr(threads_module, "THREADS", threads)
        closed: list[bool] = []

        def reader() -> Any:
            try:
                yield from self._chunks(events, [0, 2, 4, 6])
            finally:
                closed.append(True)

        with pytest.raises(SelectionError) as info:
            prepare_chunks(reader(), [Request(("met",), selection="eta > 0")])
        assert closed == [True]  # while the error, and the frame it holds, is still alive
        assert info.value is not None

    def test_threads_call_the_callers_error_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rootfig._threads as threads_module
        from rootfig.selection import Request, prepare_chunks

        monkeypatch.setattr(threads_module, "THREADS", 3)
        events = {"x": ak.Array([[0.0, 1.0], [2.0], [0.0]] * 4)}
        seen: list[str] = []

        def handler(kind: str, flag: int) -> None:
            seen.append(kind)

        with np.errstate(divide="call", call=handler), warnings.catch_warnings():
            warnings.simplefilter("ignore", RootfigWarning)
            prepare_chunks(self._chunks(events, list(range(13))), [Request(("log(x)",))])
        assert seen
        assert set(seen) == {"divide by zero"}
