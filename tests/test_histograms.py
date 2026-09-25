"""Tests for histogram filling, normalisation, comparisons, statistics and the pipeline."""

from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import hist
import numpy as np
import pytest

from rootfig.errors import (
    BinningError,
    ExpressionError,
    MissingBranchError,
    RootfigWarning,
    SelectionError,
    SourceError,
    SystematicError,
)
from rootfig.histograms import (
    COMPARISON_KINDS,
    Cutflow,
    CutflowStep,
    Efficiency,
    Histogram,
    Profile,
    Summary,
    as_weight_storage,
    build_histograms,
    build_histograms_2d,
    combined_selection,
    compare,
    compatible_binning,
    correlation_matrix,
    count_problem,
    describe_table,
    fill,
    is_unit_counts,
    load_columns,
    normalize,
    poisson_interval,
    sum_histograms,
    summarize,
    uncertainty,
)
from rootfig.histograms.binomial import clopper_pearson, normal_interval, wilson_interval
from rootfig.histograms.build import from_sample
from rootfig.histograms.groups import group_histogram, regroup_histograms
from rootfig.histograms.intervals import poisson_errors
from rootfig.histograms.normalize import normalization_label, normalize_hist
from rootfig.histograms.pipeline import combined_weight
from rootfig.model import Cut, Group, Sample, Systematic, Variable
from rootfig.plotting import fold_flow_bins, show_flow_bins
from rootfig.selection import Columns, prepare


def columns(values: list[float], weights: list[float] | None = None) -> Columns:
    return Columns(
        arrays=(np.asarray(values, dtype=float),),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        n_events=len(values),
        n_selected_events=len(values),
    )


class TestFill:
    def test_unweighted(self) -> None:
        h = fill([hist.axis.Regular(4, 0, 4)], columns([0.5, 0.5, 1.5, 9.0, -1.0]))
        assert h.values().tolist() == [2, 1, 0, 0]
        assert h.variances().tolist() == [2, 1, 0, 0]
        assert h.values(flow=True)[0] == 1  # underflow
        assert h.values(flow=True)[-1] == 1  # overflow

    def test_weighted(self) -> None:
        h = fill([hist.axis.Regular(2, 0, 2)], columns([0.5, 0.5, 1.5], [1.0, 2.0, 3.0]))
        assert h.values().tolist() == [3.0, 3.0]
        assert h.variances().tolist() == [5.0, 9.0]

    def test_empty(self) -> None:
        h = fill([hist.axis.Regular(2, 0, 2)], columns([]))
        assert h.values().tolist() == [0.0, 0.0]

    def test_axis_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="axes"):
            fill([hist.axis.Regular(2, 0, 2), hist.axis.Regular(2, 0, 2)], columns([1.0]))


class TestHistogram:
    @pytest.fixture
    def histogram(self) -> Histogram:
        cols = columns([0.5, 0.5, 1.5, 2.5, 9.0], [1.0, 1.0, 2.0, 1.0, 1.0])
        return Histogram(fill([hist.axis.Regular(3, 0, 3)], cols), label="h", stats=summarize(cols))

    def test_accessors(self, histogram: Histogram) -> None:
        assert histogram.ndim == 1
        assert histogram.edges.tolist() == [0, 1, 2, 3]
        assert histogram.centers.tolist() == [0.5, 1.5, 2.5]
        assert histogram.widths.tolist() == [1, 1, 1]
        assert histogram.values().tolist() == [2.0, 2.0, 1.0]
        assert _symmetric(histogram.errors()).tolist() == pytest.approx([np.sqrt(2), 2.0, 1.0])
        assert histogram.integral == 5.0
        assert histogram.overflow == 1.0
        assert histogram.underflow == 0.0
        assert histogram.entries == 5

    def test_scaled_and_with(self, histogram: Histogram) -> None:
        scaled = histogram.scaled(2.0)
        assert scaled.values().tolist() == [4.0, 4.0, 2.0]
        assert scaled.variances().tolist() == [8.0, 16.0, 4.0]
        assert histogram.replace(label="x").label == "x"

    def test_scaled_keeps_statistics_consistent(self, histogram: Histogram) -> None:
        assert histogram.stats is not None
        scaled = histogram.scaled(2.0)
        assert scaled.stats is not None
        assert scaled.stats.sum_weights == 12.0
        assert scaled.sum_weights == pytest.approx(scaled.stats.sum_weights)
        assert scaled.stats.effective_entries == pytest.approx(histogram.stats.effective_entries)
        assert scaled.stats.mean == histogram.stats.mean
        assert scaled.stats.std == histogram.stats.std
        assert scaled.stats.entries == scaled.entries == 5
        assert Histogram(histogram.hist, label="h").scaled(2.0).stats is None

    def test_entries_without_stats(self) -> None:
        h = Histogram(fill([hist.axis.Regular(3, 0, 3)], columns([0.5, 0.5])), label="h")
        assert h.entries is None  # a sum of weights is not an entry count
        assert h.sum_weights == 2.0
        weighted = Histogram(
            fill([hist.axis.Regular(3, 0, 3)], columns([0.5, 0.5, 1.5], [0.1, 0.2, 0.3])),
            label="w",
        )
        assert weighted.entries is None
        assert weighted.sum_weights == pytest.approx(0.6)

    def test_sum_weights_includes_flow_and_normalisation(self, histogram: Histogram) -> None:
        assert histogram.sum_weights == 6.0  # 5 visible + 1 overflow
        unity = normalize(histogram, True)
        assert unity.sum_weights == pytest.approx(1.2)
        assert unity.entries == 5  # the statistics are kept

    def test_plain_storage_is_converted_on_construction(self) -> None:
        plain = hist.Hist(hist.axis.Regular(2, 0, 4)).fill([1.0, 3.0, 3.0])
        h = Histogram(plain, label="h")
        assert h.hist.storage_type is hist.storage.Weight
        np.testing.assert_allclose(h.variances(), [1.0, 2.0])
        mean = hist.Hist(hist.axis.Regular(2, 0, 4), storage=hist.storage.Mean())
        mean.fill([1.0], sample=[2.0])
        with pytest.raises(TypeError, match="Mean storage"):
            Histogram(mean, label="m")


class TestNormalize:
    @pytest.fixture
    def h(self) -> hist.Hist:
        return fill(
            [hist.axis.Variable([0, 1, 3, 4])], columns([0.5, 0.5, 2.0, 3.5, 10.0], [1, 1, 2, 4, 1])
        )

    def test_none(self, h: hist.Hist) -> None:
        out = normalize_hist(h, False)
        assert out is not h
        assert out.values().tolist() == h.values().tolist()
        assert normalization_label(None) is None

    @pytest.mark.parametrize("spec", [True, "unity"])
    def test_unity(self, h: hist.Hist, spec: Any) -> None:
        out = normalize_hist(h, spec)
        assert out.values().sum() == pytest.approx(1.0)
        assert out.values().tolist() == pytest.approx([0.25, 0.25, 0.5])
        # variance scales with the square of the factor: factor = 1/8
        assert out.variances().tolist() == pytest.approx([2 / 64, 4 / 64, 16 / 64])
        # overflow bin scaled too
        assert out.values(flow=True)[-1] == pytest.approx(1 / 8)
        assert normalization_label(spec) == "Normalised to unity"

    def test_number(self, h: hist.Hist) -> None:
        out = normalize_hist(h, 100)
        assert out.values().sum() == pytest.approx(100.0)
        assert normalization_label(100) == "Normalised to 100"

    def test_density(self, h: hist.Hist) -> None:
        out = normalize_hist(h, "density")
        widths = np.array([1.0, 2.0, 1.0])
        assert (out.values() * widths).sum() == pytest.approx(1.0)
        assert out.values().tolist() == pytest.approx([2 / 8, 2 / 8 / 2, 4 / 8])
        assert normalization_label("density") == "Density"

    def test_width(self, h: hist.Hist) -> None:
        out = normalize_hist(h, "width")
        assert out.values().tolist() == pytest.approx([2.0, 1.0, 4.0])
        assert out.variances().tolist() == pytest.approx([2.0, 4.0 / 4, 16.0])
        assert normalization_label("width") == "Events / unit"

    def test_empty_warns(self) -> None:
        h = fill([hist.axis.Regular(2, 0, 2)], columns([]))
        with pytest.warns(RootfigWarning, match="no entries"):
            normalize_hist(h, True)
        with pytest.warns(RootfigWarning, match="no entries"):
            normalize_hist(h, "density")

    @pytest.mark.parametrize("bad", ["nope", -1, 0, np.nan])
    def test_invalid(self, h: hist.Hist, bad: Any) -> None:
        with pytest.raises(BinningError):
            normalize_hist(h, bad)

    def test_histogram_wrapper(self, h: hist.Hist) -> None:
        wrapped = Histogram(h, label="h")
        out = normalize(wrapped, True)
        assert out.normalization == "Normalised to unity"
        assert out.integral == pytest.approx(1.0)
        assert normalize(wrapped, None) is wrapped

    def test_negative_total_divides_by_signed_total(self) -> None:
        h = fill([hist.axis.Regular(2, 0, 2)], columns([0.5, 1.5], [1.0, -3.0]))
        with pytest.warns(RootfigWarning, match="negative total"):
            out = normalize_hist(h, True)
        np.testing.assert_allclose(out.values(), [-0.5, 1.5])
        assert out.values().sum() == pytest.approx(1.0)
        np.testing.assert_allclose(out.variances(), [0.25, 2.25])
        with pytest.warns(RootfigWarning, match="negative total"):
            assert normalize_hist(h, 100).values().sum() == pytest.approx(100.0)
        with pytest.warns(RootfigWarning, match="negative total"):
            assert normalize_hist(h, "density").values().sum() == pytest.approx(1.0)
        with pytest.warns(RootfigWarning, match="negative total"):
            assert normalize(Histogram(h, label="h"), True).normalization == "Normalised to unity"

    def test_cancelling_weights_skip_with_distinct_warning(self) -> None:
        h = fill([hist.axis.Regular(2, 0, 2)], columns([0.5, 1.5], [1.0, -1.0]))
        with pytest.warns(RootfigWarning, match="sum to zero"):
            out = normalize_hist(h, True)
        np.testing.assert_allclose(out.values(), [1.0, -1.0])
        with pytest.warns(RootfigWarning, match="sum to zero"):
            assert normalize(Histogram(h, label="h"), "density").normalization is None

    def test_skipped_normalisation_leaves_label_unset(self) -> None:
        empty = fill([hist.axis.Regular(2, 0, 2)], columns([]))
        wrapped = Histogram(empty, label="h")
        with pytest.warns(RootfigWarning, match="no entries"):
            out = normalize(wrapped, True)
        assert out.normalization is None
        assert out.values().tolist() == [0.0, 0.0]
        with pytest.warns(RootfigWarning, match="no entries"):
            assert normalize(wrapped, "density").normalization is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert normalize(wrapped, "width").normalization == "Events / unit"

    def test_2d_density(self) -> None:
        cols = Columns(
            arrays=(np.array([0.5, 1.5, 1.5]), np.array([0.5, 0.5, 2.5])),
            weights=None,
            n_events=3,
            n_selected_events=3,
        )
        h = fill([hist.axis.Regular(2, 0, 2), hist.axis.Variable([0, 1, 3])], cols)
        out = normalize_hist(h, "density")
        areas = np.outer([1.0, 1.0], [1.0, 2.0])
        assert (out.values() * areas).sum() == pytest.approx(1.0)


class TestNormalisedUncertainties:
    """Normalising scales the variances by the factor squared, the factor taken as a constant.

    The factor is the histogram's own total, which fluctuates with its bins; the
    multinomial variance of a shape, ``p (1 - p) / N``, is not what is drawn.
    """

    def _counts(self) -> Histogram:
        h = hist.Hist(hist.axis.Variable([0.0, 1.0, 3.0]), storage=hist.storage.Weight())
        h.fill(np.repeat([0.5, 2.0], [10, 30]))
        return Histogram(h, label="A")

    def test_the_factor_is_a_constant(self) -> None:
        unity = normalize(self._counts(), True)
        np.testing.assert_allclose(unity.values(), [0.25, 0.75])
        np.testing.assert_allclose(unity.variances(), [10 / 40**2, 30 / 40**2])
        multinomial = 0.25 * 0.75 / 40
        assert not np.allclose(unity.variances(), multinomial)
        np.testing.assert_allclose(uncertainty(unity).stat_up, np.sqrt([10, 30]) / 40)

    @pytest.mark.parametrize(
        ("spec", "factors"),
        [
            ("density", [1 / 40, 1 / 80]),  # per unit width, then to unit area
            ("width", [1.0, 0.5]),  # per unit width only
            (100, [2.5, 2.5]),  # to a total of 100
        ],
    )
    def test_every_mode_scales_the_variance_with_its_factor(
        self, spec: Any, factors: list[float]
    ) -> None:
        normalised = normalize(self._counts(), spec)
        np.testing.assert_allclose(normalised.values(), np.multiply([10, 30], factors))
        np.testing.assert_allclose(
            normalised.variances(), np.multiply([10, 30], np.square(factors))
        )

    def test_a_comparison_of_shapes_keeps_the_relative_errors(self) -> None:
        a = self._counts()
        b = Histogram(a.hist.copy(), label="B")
        b.hist.view().value = [20.0, 20.0]
        b.hist.view().variance = [20.0, 20.0]
        raw = compare(a, b)
        shapes = compare(normalize(a, True), normalize(b, True))
        np.testing.assert_allclose(shapes.values, raw.values)  # both totals are 40
        for got, want in zip(shapes.errors, raw.errors, strict=True):
            np.testing.assert_allclose(got, want)

    def test_a_normalisation_source_drops_out_of_a_shape(self) -> None:
        h = self._counts()
        varied = h.replace(variations={"lumi": (h.hist * 1.1, h.hist * 0.9)})
        np.testing.assert_allclose(uncertainty(normalize(varied, True)).syst_up, 0.0, atol=1e-15)
        per_width = uncertainty(normalize(varied, "width"))
        np.testing.assert_allclose(per_width.syst_up, 0.1 * np.array([10.0, 15.0]))


def _poisson(values: list[float]) -> hist.Hist:
    """Two bins holding ``values``, with Poisson variances."""
    h = hist.Hist(hist.axis.Regular(len(values), 0, len(values)), storage=hist.storage.Weight())
    h.view().value = values
    h.view().variance = values
    return h


def _symmetric(errors: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The one side of a ``(down, up)`` pair whose sides agree exactly."""
    down, up = errors
    np.testing.assert_array_equal(down, up)
    return down


def _normalisation(h: hist.Hist, label: str, size: float = 0.1) -> Histogram:
    """``h`` with a relative normalisation source ``"norm"`` of ``size``."""
    return Histogram(h, label=label, variations={"norm": (h * (1 + size), h * (1 - size))})


class TestCompare:
    """Per kind, with n = [4, 9] and d = [2, 3] and Poisson variances."""

    N = (4.0, 9.0)
    D = (2.0, 3.0)

    @pytest.mark.parametrize(
        ("kind", "values", "propagate", "numerator", "band"),
        [
            ("ratio", [2, 3], [np.sqrt(3), 2], [1, 1], [0.7071, 0.5774]),
            ("relative_difference", [1, 2], [np.sqrt(3), 2], [1, 1], [0.7071, 0.5774]),
            ("difference", [2, 6], [2.4495, 3.4641], [2, 3], [1.4142, 1.7321]),
            ("pull", [0.8165, 1.7321], [1, 1], None, None),
            # 2 sqrt(d^2 vn + n^2 vd) / (n + d)^2
            ("asymmetry", [1 / 3, 0.5], [2 * np.sqrt(48) / 36, 0.25], None, None),
            ("s/sqrt(b)", [2.8284, 5.1962], None, None, None),
            ("s/sqrt(s+b)", [1.6330, 2.5981], None, None, None),
        ],
    )
    def test_kinds(
        self,
        kind: Any,
        values: list[float],
        propagate: list[float] | None,
        numerator: list[float] | None,
        band: list[float] | None,
    ) -> None:
        result = compare(_poisson(list(self.N)), _poisson(list(self.D)), kind=kind)
        assert result.kind == kind
        np.testing.assert_allclose(result.values, values, atol=1e-4)
        if propagate is not None:
            np.testing.assert_allclose(_symmetric(result.errors), propagate, atol=1e-4)
        if band is None:
            assert result.band is None
            assert result.total_band() is None
        else:
            assert result.band is not None
            np.testing.assert_allclose(_symmetric(result.band), band, atol=1e-4)
        if numerator is not None:
            split = compare(
                _poisson(list(self.N)), _poisson(list(self.D)), kind=kind, uncertainty="numerator"
            )
            np.testing.assert_allclose(split.values, values, atol=1e-4)
            np.testing.assert_allclose(_symmetric(split.errors), numerator, atol=1e-4)
            assert split.band is not None
            np.testing.assert_allclose(_symmetric(split.band), band, atol=1e-4)
        else:
            with pytest.raises(ValueError, match="reference band"):
                compare(_poisson([1.0]), _poisson([1.0]), kind=kind, uncertainty="numerator")

    @pytest.mark.parametrize(
        ("kind", "syst", "values"),
        [
            ("ratio", [0, 0], [2, 3]),
            ("relative_difference", [0, 0], [1, 2]),
            ("difference", [0.2, 0.6], [2, 6]),
            ("asymmetry", [0, 0], [1 / 3, 0.5]),
        ],
    )
    def test_a_shared_normalisation_source(
        self, kind: Any, syst: list[float], values: list[float]
    ) -> None:
        num = _normalisation(_poisson(list(self.N)), "N")
        ref = _normalisation(_poisson(list(self.D)), "D")
        result = compare(num, ref, kind=kind)
        np.testing.assert_allclose(result.values, values)
        assert result.syst_errors is not None
        for side in result.syst_errors:  # cancels in a ratio, scales the difference
            np.testing.assert_allclose(side, syst, atol=1e-12)
        assert (result.label, result.reference) == ("N", "D")

    def test_a_pull_divides_by_the_systematics_it_faces(self) -> None:
        num = _normalisation(_poisson(list(self.N)), "N")
        ref = _normalisation(_poisson(list(self.D)), "D")
        result = compare(num, ref, kind="pull")
        np.testing.assert_allclose(result.values, [0.8138, 1.7066], atol=1e-4)
        assert result.syst_errors is None  # in the denominator, not beside it
        # a source that only raises n: n - d is uncertain by 3 upwards, not at all downwards
        up = Histogram(
            _poisson([2.0]), label="N", variations={"s": (_poisson([5.0]), _poisson([2.0]))}
        )
        below = compare(up, _poisson([4.0]), kind="pull")  # n < d: the upper side faces d
        np.testing.assert_allclose(below.values, [-2 / np.sqrt(2 + 4 + 9)])
        above = compare(up, _poisson([1.0]), kind="pull")  # n > d: the lower side faces d
        np.testing.assert_allclose(above.values, [1 / np.sqrt(2 + 1)])
        equal = compare(up, _poisson([2.0]), kind="pull")
        np.testing.assert_allclose(equal.values, [0.0])

    @pytest.mark.parametrize(
        "kind", ["ratio", "relative_difference", "s/sqrt(b)", "pull", "asymmetry"]
    )
    def test_an_empty_reference_bin_is_nan(self, kind: Any) -> None:
        result = compare(_poisson([0.0, 2.0]), _poisson([0.0, 1.0]), kind=kind)
        assert np.isnan(result.values[0])
        assert np.isfinite(result.values[1])

    @pytest.mark.parametrize("kind", COMPARISON_KINDS)
    def test_hist_and_histogram_inputs_agree(self, kind: Any) -> None:
        n, d = _poisson(list(self.N)), _poisson(list(self.D))
        plain = compare(n, d, kind=kind)
        for num, ref in ((Histogram(n, label="N"), d), (n, Histogram(d, label="D"))):
            wrapped = compare(num, ref, kind=kind)
            np.testing.assert_array_equal(wrapped.values, plain.values)
            np.testing.assert_array_equal(wrapped.errors, plain.errors)
        assert (plain.label, plain.reference) == ("", "")

    def test_the_reference_histogram_is_kept(self) -> None:
        n, d = _poisson(list(self.N)), _poisson(list(self.D))
        assert compare(n, d).reference_hist is d
        assert compare(Histogram(n, label="N"), Histogram(d, label="D")).reference_hist is d

    def test_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            compare(_poisson([1.0]), _poisson([1.0]), kind="significance")  # type: ignore[arg-type]

    def test_propagate(self) -> None:
        num = fill([hist.axis.Regular(3, 0, 3)], columns([0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2]))
        den = fill([hist.axis.Regular(3, 0, 3)], columns([0.5, 1.5, 2.5]))
        r = compare(num, den)
        assert r.values.tolist() == pytest.approx([2.0, 4.0, 0.0])
        errors = _symmetric(r.errors)
        # bin 0: n=2, vn=2, d=1, vd=1 -> sqrt(2/1 + 4*1/1) = sqrt(6)
        assert errors[0] == pytest.approx(np.sqrt(6))
        # bin 2: n=0, vn=0, d=1 -> 0
        assert errors[2] == pytest.approx(0.0)
        assert r.band is not None
        assert _symmetric(r.band).tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert r.centers.tolist() == [0.5, 1.5, 2.5]
        assert r.half_widths.tolist() == [0.5, 0.5, 0.5]

    def test_numerator_only(self) -> None:
        num = fill([hist.axis.Regular(2, 0, 2)], columns([0.5, 0.5, 0.5, 0.5]))
        den = fill([hist.axis.Regular(2, 0, 2)], columns([0.5, 0.5, 1.5], [2.0, 2.0, 1.0]))
        r = compare(num, den, uncertainty="numerator")
        assert r.values.tolist() == pytest.approx([1.0, 0.0])
        assert _symmetric(r.errors).tolist() == pytest.approx([2 / 4, 0.0])
        assert r.band is not None
        assert _symmetric(r.band).tolist() == pytest.approx([np.sqrt(8) / 4, 1.0])

    def test_zero_denominator_is_nan(self) -> None:
        num = fill([hist.axis.Regular(2, 0, 2)], columns([0.5]))
        den = fill([hist.axis.Regular(2, 0, 2)], columns([1.5]))
        r = compare(num, den)
        assert np.isnan(r.values[0])
        assert all(np.isnan(side[0]) for side in r.errors)
        assert r.band is not None
        assert all(np.isnan(side[0]) for side in r.band)
        assert r.values[1] == 0.0

    def test_incompatible(self) -> None:
        a = fill([hist.axis.Regular(2, 0, 2)], columns([0.5]))
        b = fill([hist.axis.Regular(3, 0, 2)], columns([0.5]))
        assert not compatible_binning(a, b)
        with pytest.raises(BinningError, match="identical bin edges"):
            compare(a, b)
        with pytest.raises(ValueError, match="uncertainty"):
            compare(a, a, uncertainty="bogus")  # type: ignore[arg-type]


class TestStats:
    def test_unweighted(self) -> None:
        s = summarize(columns([1.0, 2.0, 3.0, 4.0]))
        assert isinstance(s, Summary)
        assert s.entries == 4
        assert s.sum_weights == 4.0
        assert s.mean == 2.5
        assert s.std == pytest.approx(np.std([1, 2, 3, 4]))
        assert s.sem == pytest.approx(np.std([1, 2, 3, 4]) / 2)
        assert s.skewness == pytest.approx(0.0)
        assert s.rms == s.std
        assert s.effective_entries == pytest.approx(4.0)
        assert (s.minimum, s.maximum) == (1.0, 4.0)

    def test_weighted_equals_repeated(self) -> None:
        weighted = summarize(columns([1.0, 2.0, 5.0], [2.0, 1.0, 1.0]))
        repeated = summarize(columns([1.0, 1.0, 2.0, 5.0]))
        assert weighted.mean == pytest.approx(repeated.mean)
        assert weighted.std == pytest.approx(repeated.std)
        assert weighted.skewness == pytest.approx(repeated.skewness)
        assert weighted.skewness > 0

    def test_empty(self) -> None:
        s = summarize(columns([]))
        assert s.entries == 0
        assert np.isnan(s.mean)
        assert np.isnan(s.std)
        assert s.effective_entries == 0.0

    def test_format(self) -> None:
        s = summarize(columns([1.0, 2.0, 3.0]))
        text = s.format(precision=3)
        assert "N = 3" in text
        assert "$\\mu$ = 2" in text
        assert "$\\sigma$ = 0.816" in text
        assert "N =" not in s.format(include_entries=False)
        assert "e" in Summary(1, 1, 1.5e-9, 0, 0, 0, 0, 0).format()
        assert "nan" in summarize(columns([])).format()

    def test_table(self) -> None:
        table = describe_table(
            [("a", summarize(columns([1.0, 2.0]))), ("bb", summarize(columns([3.0])))]
        )
        lines = table.splitlines()
        assert lines[0].split()[:3] == ["entries", "mean", "std"]
        assert lines[1].startswith("a ")
        assert lines[2].startswith("bb")

    def test_correlation(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        cols = Columns(arrays=(x, 2 * x, -x + 1), weights=None, n_events=4, n_selected_events=4)
        m = correlation_matrix(cols)
        assert m.shape == (3, 3)
        np.testing.assert_allclose(m, [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])

    def test_correlation_weighted_and_constant(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        cols = Columns(
            arrays=(x, x**2),
            weights=np.array([1.0, 2.0, 1.0, 2.0]),
            n_events=4,
            n_selected_events=4,
        )
        m = correlation_matrix(cols)
        assert 0.9 < m[0, 1] < 1.0
        const = Columns(arrays=(x, np.ones(4)), weights=None, n_events=4, n_selected_events=4)
        m = correlation_matrix(const)
        assert np.isnan(m[0, 1])
        assert np.isnan(m[1, 1])

    def test_correlation_errors(self) -> None:
        x = np.array([1.0, 2.0])
        with pytest.raises(SelectionError, match="two variables"):
            correlation_matrix(Columns(arrays=(x,), weights=None, n_events=2, n_selected_events=2))
        with pytest.raises(SelectionError, match="two entries"):
            correlation_matrix(
                Columns(arrays=(x[:1], x[:1]), weights=None, n_events=1, n_selected_events=1)
            )
        with pytest.raises(SelectionError, match="negative"):
            correlation_matrix(
                Columns(
                    arrays=(x, x), weights=np.array([1.0, -1.0]), n_events=2, n_selected_events=2
                )
            )

    def test_correlation_degenerate_weights(self) -> None:
        x = np.array([1.0, 2.0, 3.0])

        def cols(weights: np.ndarray) -> Columns:
            return Columns(arrays=(x, 2 * x), weights=weights, n_events=3, n_selected_events=3)

        with pytest.raises(SelectionError, match="non-zero weight"):
            correlation_matrix(cols(np.zeros(3)))  # numpy would raise ZeroDivisionError
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # numpy would emit RuntimeWarnings
            with pytest.raises(SelectionError, match="non-zero weight"):
                correlation_matrix(cols(np.array([1.0, 0.0, 0.0])))
        m = correlation_matrix(cols(np.array([1.0, 1.0, 0.0])))
        assert m[0, 1] == pytest.approx(1.0)


class TestPipeline:
    def test_combined_selection_and_weight(self) -> None:
        sample = Sample({"x": [1]}, selection="x > 0", weight="w")
        assert combined_selection(sample, None) == Cut("x > 0")
        assert combined_selection(sample, "x < 5").expression == "(x > 0) & (x < 5)"  # type: ignore[union-attr]
        assert combined_selection(Sample({"x": [1]}), "x < 5") == Cut("x < 5")
        assert combined_selection(Sample({"x": [1]}), None) is None
        assert combined_weight(sample, None) == "w"
        assert combined_weight(sample, "v") == "(w) * (v)"
        assert combined_weight(sample, "  ") == "w"
        assert combined_weight(Sample({"x": [1]}), "v") == "v"

    def test_load_columns_reads_only_needed_branches(self, signal_file: Path) -> None:
        sample = Sample(signal_file, tree="events", selection="nMuon > 0", weight="weight")
        cols = load_columns(sample, ["Muon_pt"], selection="abs(Muon_eta) < 2.5", weight="2")
        assert cols.n_entries > 0
        assert cols.weights is not None
        assert cols.n_events == 2000
        # weights are event weights broadcast to muons, times the constant 2
        assert np.all(cols.weights > 0)

    def test_build_histograms_shared_binning(
        self, signal_file: Path, background_file: Path
    ) -> None:
        samples = [
            Sample(signal_file, tree="events"),
            Sample(background_file, tree="events", is_data=True),
        ]
        hists = build_histograms(samples, Variable("MET", bins=20), selection="nMuon >= 1")
        assert len(hists) == 2
        assert hists[0].edges.tolist() == hists[1].edges.tolist()
        assert hists[0].label == "signal"
        assert hists[1].is_data
        assert hists[0].stats is not None
        assert hists[0].stats.entries == hists[0].hist.sum(flow=True).value
        assert hists[0].axis.name == "MET"

    def test_build_histograms_contents_match_numpy(
        self, signal_arrays: dict[str, ak.Array]
    ) -> None:
        sample = Sample(signal_arrays, label="mem")
        [h] = build_histograms(
            [sample],
            Variable("Muon_pt", bins=(10, 0, 200)),
            selection="Muon_pt > 20",
            weight="weight",
        )
        pt = signal_arrays["Muon_pt"]
        w = ak.broadcast_arrays(signal_arrays["weight"], pt)[0]
        mask = pt > 20
        expected, _ = np.histogram(
            ak.flatten(pt[mask]), bins=10, range=(0, 200), weights=ak.flatten(w[mask])
        )
        assert h.values().tolist() == pytest.approx(expected.tolist())

    def test_build_histograms_2d(self, signal_arrays: dict[str, ak.Array]) -> None:
        sample = Sample(signal_arrays)
        [h] = build_histograms_2d(
            [sample], Variable("Muon_pt", bins=(5, 0, 100)), Variable("Muon_eta", bins=5)
        )
        assert h.ndim == 2
        assert h.hist.axes[0].name == "Muon_pt"
        assert h.hist.axes[1].name == "Muon_eta"
        assert h.hist.sum(flow=True).value == pytest.approx(ak.count(signal_arrays["Muon_pt"]))

    def test_build_histograms_2d_same_variable(self, signal_arrays: dict[str, ak.Array]) -> None:
        [h] = build_histograms_2d([Sample(signal_arrays)], "MET", "MET")
        assert h.hist.axes[1].name == "MET_y"

    def test_prepare_then_fill_roundtrip(self, signal_arrays: dict[str, ak.Array]) -> None:
        cols = prepare(signal_arrays, "nMuon")
        h = fill([hist.axis.Regular(10, -0.5, 9.5)], cols)
        assert h.sum(flow=True).value == 2000
        assert (
            h.values().tolist() == np.bincount(signal_arrays["nMuon"], minlength=10)[:10].tolist()
        )


def _hist(values: list[float], weights: list[float] | None = None) -> hist.Hist:
    h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
    h.fill(values, weight=weights)
    return h


def _contents(values: list[float], variances: list[float]) -> hist.Hist:
    """One bin per value, with the given variances."""
    h = hist.Hist(hist.axis.Regular(len(values), 0, len(values)), storage=hist.storage.Weight())
    h.view().value = values
    h.view().variance = variances
    return h


def _with_errors(histogram: Histogram, down: list[float], up: list[float]) -> Histogram:
    """``histogram`` reporting the given ``(down, up)`` statistical errors, as an asymmetric
    model of its contents would."""
    pair = (np.asarray(down, dtype=float), np.asarray(up, dtype=float))
    object.__setattr__(histogram, "errors", lambda *, flow=False: pair)
    return histogram


class TestAsymmetricErrors:
    """Statistical errors are ``(down, up)`` pairs from the histogram to the comparison."""

    N = [4.0, 9.0, 1.0, 30.0]
    VN = [2.0, 7.0, 0.5, 45.0]
    D = [2.0, 3.0, -2.0, 12.0]
    VD = [1.5, 4.0, 3.0, 20.0]

    def _sides(self) -> tuple[hist.Hist, hist.Hist]:
        return _contents(self.N, self.VN), _contents(self.D, self.VD)

    @staticmethod
    def _closed_form(kind: str, n: Any, d: Any, vn: Any, vd: Any) -> Any:
        """The symmetric error of ``kind`` from the textbook formula."""
        errors = {
            "ratio": np.sqrt(vn / d**2 + n**2 * vd / d**4),
            "relative_difference": np.sqrt(vn / d**2 + n**2 * vd / d**4),
            "difference": np.sqrt(vn + vd),
            "pull": np.ones(4),
            "asymmetry": 2 * np.sqrt(d**2 * vn + n**2 * vd) / (n + d) ** 2,
            "s/sqrt(b)": np.sqrt(vn / d + n**2 * vd / (4 * d**3)),
            "s/sqrt(s+b)": np.sqrt(
                ((n + 2 * d) / (2 * (n + d) ** 1.5)) ** 2 * vn
                + (n / (2 * (n + d) ** 1.5)) ** 2 * vd
            ),
        }
        return errors[kind]

    @pytest.mark.parametrize("kind", COMPARISON_KINDS)
    def test_symmetric_errors_match_the_closed_forms(self, kind: Any) -> None:
        n, d = np.array(self.N), np.array(self.D)
        vn, vd = np.array(self.VN), np.array(self.VD)
        with np.errstate(invalid="ignore"):  # a negative background has no significance
            expected = self._closed_form(kind, n, d, vn, vd)
            result = compare(*self._sides(), kind=kind)
        defined = np.isfinite(result.values)
        errors = _symmetric(result.errors)
        np.testing.assert_allclose(errors[defined], expected[defined], rtol=1e-13)
        assert np.isnan(errors[~defined]).all()
        if kind == "pull":
            np.testing.assert_allclose(result.values, (n - d) / np.sqrt(vn + vd), rtol=1e-13)
        if kind in ("ratio", "difference"):
            split = compare(*self._sides(), kind=kind, uncertainty="numerator")
            assert split.band is not None
            scale = np.abs(d) if kind == "ratio" else 1.0
            np.testing.assert_allclose(_symmetric(split.errors), np.sqrt(vn) / scale, rtol=1e-13)
            np.testing.assert_allclose(_symmetric(split.band), np.sqrt(vd) / scale, rtol=1e-13)

    def test_a_ratio_takes_the_side_that_moves_it(self) -> None:
        # n = 4 with errors (1, 3); d = 2 with errors (0.5, 1), and d = -2 in bin 1
        num = _with_errors(Histogram(_contents([4.0, 4.0], [4.0, 4.0]), "N"), [1, 1], [3, 3])
        ref = _with_errors(
            Histogram(_contents([2.0, -2.0], [1.0, 1.0]), "D"), [0.5, 0.5], [1.0, 1.0]
        )
        split = compare(num, ref, uncertainty="numerator")
        # n / d falls with n where d > 0 and rises with it where d < 0
        np.testing.assert_allclose(split.errors[0], [1 / 2, 3 / 2])
        np.testing.assert_allclose(split.errors[1], [3 / 2, 1 / 2])
        assert split.band is not None  # the reference's own relative interval, around 1
        np.testing.assert_allclose(split.band[0], [0.5 / 2, 1.0 / 2])
        np.testing.assert_allclose(split.band[1], [1.0 / 2, 0.5 / 2])
        both = compare(num, ref)
        # bin 0: lowering n / d = lowering n (1) or raising d (1): hypot(1/2, 4 * 1 / 4)
        np.testing.assert_allclose(both.errors[0][0], np.hypot(1 / 2, 4 * 1.0 / 4))
        np.testing.assert_allclose(both.errors[1][0], np.hypot(3 / 2, 4 * 0.5 / 4))

    def test_a_difference_and_a_pull_face_the_reference(self) -> None:
        num = _with_errors(Histogram(_contents([5.0, 1.0], [5.0, 1.0]), "N"), [2, 0.8], [3, 2.3])
        ref = _with_errors(Histogram(_contents([3.0, 3.0], [3.0, 3.0]), "D"), [1, 1], [1.5, 1.5])
        difference = compare(num, ref, kind="difference")
        np.testing.assert_allclose(difference.errors[0], np.hypot([2, 0.8], [1.5, 1.5]))
        np.testing.assert_allclose(difference.errors[1], np.hypot([3, 2.3], [1, 1]))
        split = compare(num, ref, kind="difference", uncertainty="numerator")
        np.testing.assert_allclose(split.errors[0], [2, 0.8])
        assert split.band is not None
        np.testing.assert_allclose(split.band[1], [1.5, 1.5])
        pull = compare(num, ref, kind="pull")
        # n > d in bin 0: the lower side of n - d faces zero; n < d in bin 1: the upper one
        np.testing.assert_allclose(
            pull.values, [2 / np.hypot(2, 1.5), -2 / np.hypot(2.3, 1)], rtol=1e-13
        )
        np.testing.assert_array_equal(_symmetric(pull.errors), [1.0, 1.0])

    def test_uncertainty_adds_each_statistical_side_to_its_systematic_side(self) -> None:
        nominal = _contents([10.0, 10.0], [10.0, 10.0])
        varied = Histogram(nominal, "A", variations={"s": (nominal * 1.3, nominal * 0.8)})
        u = uncertainty(_with_errors(varied, [2.0, 1.0], [4.0, 5.0]))
        np.testing.assert_allclose(u.stat_down, [2.0, 1.0])
        np.testing.assert_allclose(u.stat_up, [4.0, 5.0])
        np.testing.assert_allclose(u.total_down, np.hypot([2.0, 1.0], 2.0))
        np.testing.assert_allclose(u.total_up, np.hypot([4.0, 5.0], 3.0))


# Garwood bounds (count, lower, upper) from scipy.special.gammaincinv, at z = 1 and z = 2
GARWOOD = {
    1.0: [
        (0, 0.0, 1.8410216450092634),
        (1, 0.17275377902344996, 3.299526559115855),
        (2, 0.7081854398189713, 4.637859623455245),
        (3, 1.367295313890434, 5.918185832883396),
        (5, 2.8403088555932205, 8.382472652146888),
        (10, 6.891305560638357, 14.266949761009391),
        (20, 15.56555201778588, 25.54651922951146),
        (100, 90.0167451771134, 111.03336094114967),
        (1000, 968.3825014065658, 1032.633323474849),
        (1001, 969.3666913282727, 1033.6491256405543),
        (5000, 4929.291680397712, 5071.715393515044),
        (1000000, 999000.000166674, 1001001.0003333407),
    ],
    2.0: [
        (0, 0.0, 3.783184333682032),
        (1, 0.0230129093289635, 5.682707562895901),
        (10, 4.719233718620755, 18.577119961610656),
        (1000, 937.7596419630572, 1065.2718274168408),
        (1001, 938.7280245307875, 1066.303429205423),
    ],
}

# Computed with ROOT 6.40. TH1 without Sumw2, kPoisson: count -> (GetBinErrorLow, GetBinErrorUp)
ROOT_POISSON = {
    0: (0.0, 1.841021644577239),
    1: (0.8272462208950817, 2.2995265585528952),
    2: (1.291814559984522, 2.6378596227967464),
    5: (2.1596911439740243, 3.382472651278441),
    10: (3.1086944386636226, 4.266949759891316),
    100: (9.983254820245776, 11.033360938117326),
    1000: (31.61749858466783, 32.633323465689045),
    1001: (31.633308662952913, 32.649125631397055),
    10000: (99.99833255931117, 101.00333399898227),
    1000000: (999.9998331097886, 1001.0003342226846),
}
# TEfficiency::Wilson(total, passed, 0.6826894921370859, upper): (passed, total) -> (lower, upper)
ROOT_WILSON = {
    (0, 1): (0.0, 0.5),
    (1, 1): (0.5, 1.0),
    (0, 10): (0.0, 0.09090909090909091),
    (3, 10): (0.17882082075676461, 0.4575428156068717),
    (10, 10): (0.9090909090909092, 1.0),
    (37, 50): (0.6736930271670393, 0.7968952081270785),
    (999, 1000): (0.997385028286038, 0.999617968716959),
}
# TEfficiency::ClopperPearson(total, passed, 0.6826894921370859, upper)
ROOT_CLOPPER_PEARSON = {
    (0, 1): (0.0, 0.8413447460685429),
    (1, 1): (0.1586552539314571, 1.0),
    (0, 10): (0.0, 0.16814918613797644),
    (3, 10): (0.14167190110718023, 0.5082624819902524),
    (10, 10): (0.8318508138620236, 1.0),
    (37, 50): (0.663178241460404, 0.8058241366566161),
    (999, 1000): (0.9967042648821212, 0.9998272611420517),
    (5000, 10000): (0.4949502550122015, 0.5050497449877984),
    (99990, 100000): (0.9998573335462261, 0.9999310862177838),
    (123456, 1000000): (0.12312691723706515, 0.12378583699547208),
    (3, 10000000): (1.3672953621198285e-07, 5.918184985409525e-07),
    (9999999, 10000000): (0.9999996700473842, 0.999999982724622),
}
# TEfficiency(passed, total) with its default options, one bin filled with (weight, passes)
# entries: (efficiency, lower, upper). Weighted histograms get the normal approximation.
ROOT_TEFFICIENCY = {
    "unit": ([(1.0, True)] * 3 + [(1.0, False)], (0.75, 0.38159757449607973, 0.9577308936963108)),
    "uniform 2": (
        [(2.0, True)] * 3 + [(2.0, False)],
        (0.75, 0.5334936490539288, 0.9665063509460712),
    ),
    "toy": ([(10.0, True), (1.0, False)], (0.9090909090909091, 0.7922137551757983, 1.0)),
    "signed": (
        [(2.0, True), (-1.0, True), (1.0, False), (1.0, False)],
        (0.3333333333333333, 0.0, 0.8544906399802885),
    ),
    "all pass": ([(1.0, True), (2.0, True), (3.0, True)], (1.0, 1.0, 1.0)),
}


class TestRootReference:
    """rootfig against numbers computed with ROOT, where both implement the same interval."""

    def test_poisson_errors_are_th1_kpoisson(self) -> None:
        counts = np.array(list(ROOT_POISSON), dtype=float)
        down, up = (np.array(side) for side in zip(*ROOT_POISSON.values(), strict=True))
        ours = Histogram(_poisson(list(counts)), label="Data", is_data=True, poisson=True)
        # exact at every count; ROOT's coverage is the truncated 1 - 0.682689492
        for mine, root in zip(ours.errors(), (down, up), strict=True):
            np.testing.assert_array_less(np.abs(mine - root) / np.maximum(root, 1.0), 1e-9)

    def test_efficiency_intervals_are_tefficiency_wilson(self) -> None:
        passed, total = (np.array(side, dtype=float) for side in zip(*ROOT_WILSON, strict=True))
        lower, upper = (np.array(side) for side in zip(*ROOT_WILSON.values(), strict=True))
        mine = wilson_interval(passed, total, total)  # counts: the variance is the count
        np.testing.assert_allclose(mine[0], lower, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(mine[1], upper, rtol=1e-12)

    def test_clopper_pearson_is_tefficiency_clopper_pearson(self) -> None:
        passed, total = (np.array(s, dtype=float) for s in zip(*ROOT_CLOPPER_PEARSON, strict=True))
        lower, upper = (np.array(s) for s in zip(*ROOT_CLOPPER_PEARSON.values(), strict=True))
        mine = clopper_pearson(passed, total)
        # 3 of 1e7: ROOT misses the exact bounds by ~4e-9 (lgamma of 1e7), SciPy does not
        tolerance = np.where(total < 1e7, 1e-11, 1e-8)
        for got, want in zip(mine, (lower, upper), strict=True):
            np.testing.assert_array_less(np.abs(got - want), tolerance * np.maximum(want, 1e-300))
        # the exact bounds (mpmath, 40 digits): SciPy is within 2e-10 of them, ROOT 4e-9
        exact = [1.3672953571451469e-07, 5.918184969365128e-07]
        np.testing.assert_allclose(np.ravel(clopper_pearson([3.0], [1e7])), exact, rtol=1e-9)
        two = clopper_pearson([3.0], [10.0], z=2.0)
        np.testing.assert_allclose(np.ravel(two), [0.06440282972673787, 0.6581255125487373])
        # ROOT takes real counts too
        real = clopper_pearson([0.5, 2.5, 0.25], [3.7, 3.0, 0.5])
        np.testing.assert_allclose(
            real[0], [0.005049115171304538, 0.3843380720179961, 4.680016622133597e-4], rtol=1e-11
        )
        np.testing.assert_allclose(
            real[1], [0.5315598555042972, 0.9938725041908549, 0.9995319983377866], rtol=1e-11
        )

    @pytest.mark.parametrize(
        ("delta", "lower"),
        # ROOT 6.40, TH1D: TEfficiency treats weights of 1 + 1e-11 as weighted (normal
        # approximation) and 1 + 1e-13 as unweighted (Clopper-Pearson): a tolerance of 1e-12
        [(1e-11, 0.75 - 0.2165063509), (1e-13, 0.38159757449607973)],
    )
    def test_unweighted_is_decided_like_tefficiency_for_th1d(
        self, delta: float, lower: float
    ) -> None:
        from rootfig.histograms import efficiency

        weights = [1.0 + delta] * 4
        eff = efficiency(_hist([0.5] * 3, weights[:3]), _hist([0.5] * 4, weights))
        assert eff.lower[0] == pytest.approx(lower, abs=1e-9)

    @pytest.mark.parametrize("case", sorted(ROOT_TEFFICIENCY))
    def test_the_default_is_tefficiency(self, case: str) -> None:
        from rootfig.histograms import efficiency

        entries, (value, lower, upper) = ROOT_TEFFICIENCY[case]
        weights = np.array([w for w, _ in entries])
        passes = np.array([ok for _, ok in entries])
        total = _hist([0.5] * len(entries), list(weights))
        passed = _hist([0.5] * int(passes.sum()), list(weights[passes]))
        eff = efficiency(passed, total)
        assert (eff.values[0], eff.lower[0], eff.upper[0]) == pytest.approx(
            (value, lower, upper), rel=1e-12, abs=1e-15
        )

    def test_a_propagated_ratio_is_th1_divide(self) -> None:
        a = Histogram(_contents([3.0, 3.0, 0.5], [5.0, 3.0, 0.25]), label="A")
        b = Histogram(_contents([2.0, 3.0, 4.0], [2.0, 9.0, 6.0]), label="B")
        ratio = compare(a, b, uncertainty="propagate")
        np.testing.assert_allclose(ratio.values, [1.5, 1.0, 0.125])
        root = [1.541103500742244, 1.1547005383792515, 0.14657549249448218]
        np.testing.assert_allclose(_symmetric(ratio.errors), root, rtol=1e-12)


class TestPoissonIntervals:
    @pytest.mark.parametrize("z", sorted(GARWOOD))
    def test_garwood_bounds(self, z: float) -> None:
        counts, lower, upper = (
            np.array(column, dtype=float) for column in zip(*GARWOOD[z], strict=True)
        )
        low, high = poisson_interval(counts, z)
        # judged against the error bar, which is exact at every count
        tolerance = 1e-10
        down_bar = np.where(counts > 0, counts - lower, 1.0)
        np.testing.assert_array_less(np.abs(low - lower) / down_bar, tolerance)
        np.testing.assert_array_less(np.abs(high - upper) / (upper - counts), tolerance)
        assert low[0] == 0.0

    def test_many_counts_become_symmetric(self) -> None:
        counts = np.array([100.0, 1e4, 1e6])
        low, high = poisson_interval(counts)
        down, up = (counts - low) / np.sqrt(counts), (high - counts) / np.sqrt(counts)
        # both sides approach sqrt(n), and each other
        assert np.all(np.diff(np.abs(up - 1.0)) < 0)
        assert np.all(np.diff(np.abs(down - 1.0)) < 0)
        np.testing.assert_allclose([down[-1], up[-1]], 1.0, rtol=2e-3)
        only_large = poisson_interval([2e6, 3e6])
        np.testing.assert_allclose(only_large[1] - [2e6, 3e6], np.sqrt([2e6, 3e6]), rtol=1e-3)

    @pytest.mark.parametrize("counts", [[-1.0], [1.5], [2.0, np.nan], [np.inf]])
    def test_counts_must_be_whole_and_non_negative(self, counts: list[float]) -> None:
        with pytest.raises(ValueError, match="non-negative whole numbers"):
            poisson_interval(counts)

    @pytest.mark.parametrize("z", [0.0, -1.0, np.nan, np.inf])
    def test_z_must_be_positive_and_finite(self, z: float) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            poisson_interval([2.0], z)

    def test_round_off_is_a_whole_count(self) -> None:
        np.testing.assert_array_equal(poisson_interval([2.0 + 1e-12]), poisson_interval([2.0]))

    def test_equal_counts_agree(self) -> None:
        low, high = poisson_interval([3.0, 0.0, 3.0, 7.0, 0.0])
        assert (low[0], high[0]) == (low[2], high[2])
        assert high[1] == high[4]

    def test_scaled_counts(self) -> None:
        counts = np.array([0.0, 3.0, 0.0, 0.0, 5.0, 0.0])
        raw_down, raw_up = poisson_errors(counts, counts)
        low, high = poisson_interval(counts)
        np.testing.assert_allclose(raw_down, counts - low)
        np.testing.assert_allclose(raw_up, high - counts)
        # normalised to unity: one factor for every bin, empty ones included
        down, up = poisson_errors(counts / 8, counts / 64)
        np.testing.assert_allclose(down, raw_down / 8, rtol=1e-12)
        np.testing.assert_allclose(up, raw_up / 8, rtol=1e-12)
        # per-bin factors: an empty bin borrows the nearest filled bin's (ties: the lower one)
        factor = np.array([1.0, 2.0, 1.0, 1.0, 4.0, 1.0])
        _, borrowed = poisson_errors(counts / factor, counts / factor**2)
        np.testing.assert_allclose(borrowed, raw_up / [2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
        np.testing.assert_allclose(poisson_errors(np.zeros(2), np.zeros(2))[1], 1.8410216450, 1e-9)

    def test_unit_counts_and_count_problems(self) -> None:
        assert is_unit_counts([0.0, 3.0, 7.0], [0.0, 3.0, 7.0])
        assert not is_unit_counts([0.0, 1.5], [0.0, 1.5])  # not whole
        assert not is_unit_counts([2.0, 4.0], [4.0, 8.0])  # weight 2
        assert count_problem([0.0, 2.0, 4.0], [0.0, 4.0, 8.0]) is None  # counts scaled by 2
        assert count_problem([0.25, 0.75], [0.0625, 0.1875]) is None  # normalised counts
        assert "negative" in str(count_problem([-1.0, 2.0], [1.0, 2.0]))
        assert "cancel" in str(count_problem([0.0, 2.0], [2.0, 2.0]))
        assert "weighted" in str(count_problem([1.5, 2.0], [1.25, 2.0]))
        assert "without a variance" in str(count_problem([1.0], [0.0]))
        assert "non-finite" in str(count_problem([np.nan], [1.0]))


class TestPoissonHistograms:
    def _data(self, counts: list[float], **kwargs: Any) -> Histogram:
        return Histogram(_poisson(counts), label="Data", is_data=True, poisson=True, **kwargs)

    def test_errors_are_the_interval_of_the_counts(self) -> None:
        data = self._data([0.0, 1.0, 4.0])
        low, high = poisson_interval([0.0, 1.0, 4.0])
        down, up = data.errors()
        np.testing.assert_allclose(down, [0.0, 1.0, 4.0] - low)
        np.testing.assert_allclose(up, high - [0.0, 1.0, 4.0])
        assert up[0] == pytest.approx(1.8410216450)  # an empty bin still has an upper error
        u = uncertainty(data)
        np.testing.assert_array_equal(u.stat_up, up)
        np.testing.assert_array_equal(u.total_down, down)
        plain = Histogram(_poisson([0.0, 1.0, 4.0]), label="Data", is_data=True)
        np.testing.assert_array_equal(_symmetric(plain.errors()), [0.0, 1.0, 2.0])

    def test_the_flag_survives_display_transformations(self) -> None:
        data = self._data([2.0, 6.0])
        unity = normalize(data, True)
        assert unity.poisson
        np.testing.assert_allclose(unity.errors()[1], data.errors()[1] / 8, rtol=1e-12)
        density = normalize(self._data([2.0, 0.0, 6.0]), "density")
        np.testing.assert_allclose(density.errors()[1][1], 1.8410216450 / 8, rtol=1e-9)
        assert data.scaled(3.0).poisson

    def test_an_empty_bin_takes_its_own_width(self) -> None:
        h = hist.Hist(hist.axis.Variable([0.0, 1.0, 11.0]), storage=hist.storage.Weight())
        h.fill(np.full(10, 5.0))  # counts [0, 10] in bins 1 and 10 wide
        data = Histogram(h, label="Data", is_data=True, poisson=True)
        ten = poisson_interval([10.0])[1][0] - 10.0
        # per width, the empty bin's factor is 1 / 1, not the filled bin's 1 / 10
        np.testing.assert_allclose(normalize(data, "width").errors()[1], [1.84102164, ten / 10])
        density = normalize(data, "density")  # also divided by the total, 10
        np.testing.assert_allclose(density.errors()[1], [0.184102164, ten / 100])
        with pytest.raises(BinningError, match="count scale"):
            data.replace(hist=_poisson([1.0, 2.0, 3.0]), _unit=data._unit)

    def test_an_empty_histogram_keeps_its_scale(self) -> None:
        h = hist.Hist(hist.axis.Variable([0.0, 1.0, 11.0]), storage=hist.storage.Weight())
        empty = Histogram(h, label="Data", is_data=True, poisson=True)
        np.testing.assert_allclose(normalize(empty, "width").errors()[1], [1.84102164, 0.184102164])
        np.testing.assert_allclose(empty.scaled(3.0).errors()[1], [5.52306493] * 2)
        with pytest.warns(RootfigWarning, match="no entries"):  # nothing to normalise to
            unchanged = normalize(empty, "density")
        np.testing.assert_allclose(unchanged.errors()[1], [1.84102164] * 2)
        merged = empty.scaled(2.0).rebinned(2)
        np.testing.assert_allclose(merged.errors()[1], [3.68204329])
        assert empty.replace(poisson=False)._unit is None
        tripled = empty.scaled(3.0)
        np.testing.assert_allclose(tripled.replace(label="x").errors()[1], [5.52306493] * 2)
        with pytest.warns(RootfigWarning, match="no entries"):  # skipped: the scale stays
            np.testing.assert_allclose(normalize(tripled, "unity").errors()[1], [5.52306493] * 2)

    def test_sums_of_counts_with_one_scale_keep_the_interval(self) -> None:
        a, b = self._data([0.0, 1.0, 4.0]), self._data([0.0, 2.0, 0.0])
        total = sum_histograms([a, b])
        assert total.poisson
        low, high = poisson_interval([0.0, 3.0, 4.0])
        np.testing.assert_allclose(total.errors()[0], [0.0, 3.0, 4.0] - low)
        np.testing.assert_allclose(total.errors()[1], high - [0.0, 3.0, 4.0])
        empty_scaled = sum_histograms([a.scaled(2.0), b.scaled(2.0)])
        assert empty_scaled.errors()[1][0] == pytest.approx(2 * 1.8410216450)
        # a plain input, or counts scaled differently, are no longer counts of one scale
        plain = Histogram(_poisson([0.0, 2.0, 0.0]), label="MC")
        assert not sum_histograms([a, plain]).poisson
        assert not sum_histograms([a, b.scaled(2.0)]).poisson

    def test_zero_scaled_poisson_has_zero_errors(self) -> None:
        data = self._data([0.0, 1.0, 4.0])
        zero = data.scaled(0.0)
        np.testing.assert_array_equal(zero.values(), 0.0)
        np.testing.assert_array_equal(zero.errors()[0], 0.0)
        np.testing.assert_array_equal(zero.errors()[1], 0.0)
        np.testing.assert_array_equal(zero.errors(flow=True)[1], 0.0)
        np.testing.assert_array_equal(normalize(zero, "width").errors()[1], 0.0)
        total = sum_histograms([zero, self._data([0.0, 2.0, 0.0]).scaled(0.0)])
        assert total.poisson
        np.testing.assert_array_equal(total.errors()[1], 0.0)
        assert not sum_histograms([zero, data]).poisson  # different count scales

    def test_a_new_hist_brings_its_own_count_scale(self) -> None:
        # counts [0, 1, 2] scaled by 10, then replaced by counts scaled by 2 with the same
        # binning: the empty bin takes 2 x 1.84, not the old 10 x 1.84
        old = Histogram(_contents([0.0, 10.0, 20.0], [0.0, 100.0, 200.0]), "D", poisson=True)
        new = old.replace(hist=_contents([0.0, 2.0, 4.0], [0.0, 4.0, 8.0]))
        assert new.errors()[1][0] == pytest.approx(2 * 1.8410216450)
        assert old.errors()[1][0] == pytest.approx(10 * 1.8410216450)

    def test_shown_flow_bins_keep_the_width_they_were_divided_by(self) -> None:
        edges = [0.0, 10.0, 20.0, 100.0]
        h = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Weight())
        h.fill([150.0] * 3)  # three counts in the overflow, divided by the last width, 80
        mc = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Weight())
        mc.fill([-5.0, 5.0])  # an underflow, so that one is shown too (empty for data)
        data = normalize(Histogram(h, label="Data", is_data=True, poisson=True), "width")
        shown, _ = show_flow_bins([data, normalize(Histogram(mc, label="MC"), "width")])
        np.testing.assert_allclose(shown[0].values(), [0.0, 0.0, 0.0, 0.0, 3 / 80])
        three = poisson_interval([3.0])[1][0] - 3.0
        # the empty bins borrow the count scale of the shown overflow per unit of *its* size
        expected = np.array([1.84102164 / 10, 1.84102164 / 10, 1.84102164 / 10, 1.84102164 / 80])
        np.testing.assert_allclose(shown[0].errors()[1], [*expected, three / 80])

    def test_folded_flow_bins_take_the_interval_of_the_sum(self) -> None:
        h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        h.fill([-1.0, -1.0, 0.5, 1.5, 5.0])  # underflow 2, bins 1 and 1, overflow 1
        (folded,) = fold_flow_bins([Histogram(h, label="Data", is_data=True, poisson=True)])
        assert folded.poisson
        low, high = poisson_interval([3.0, 2.0])
        np.testing.assert_allclose(folded.errors()[1], high - [3.0, 2.0])
        np.testing.assert_allclose(folded.errors()[0], [3.0, 2.0] - low)

    @pytest.mark.parametrize(
        ("values", "variances", "problem"),
        [
            ([2.0, 3.0], [2.5, 3.5], "weighted"),
            ([-1.0, 3.0], [1.0, 3.0], "negative"),
            ([0.0, 3.0], [2.0, 3.0], "cancel"),
        ],
    )
    def test_contents_that_are_not_counts_are_refused(
        self, values: list[float], variances: list[float], problem: str
    ) -> None:
        with pytest.raises(ValueError, match=problem):
            Histogram(_contents(values, variances), label="Data", poisson=True)
        with pytest.raises(ValueError, match="negative"):
            self._data([1.0, 2.0]).scaled(-1.0)

    def test_a_ratio_to_data_counts_takes_its_interval(self) -> None:
        data = self._data([0.0, 1.0, 9.0])
        mc = Histogram(_poisson([2.0, 2.0, 8.0]), label="MC")
        low, high = poisson_interval([0.0, 1.0, 9.0])
        split = compare(data, mc, uncertainty="numerator")
        np.testing.assert_allclose(split.values, [0.0, 0.5, 9 / 8])
        np.testing.assert_allclose(split.errors[0], ([0.0, 1.0, 9.0] - low) / [2, 2, 8])
        np.testing.assert_allclose(split.errors[1], (high - [0.0, 1.0, 9.0]) / [2, 2, 8])
        assert split.errors[1][0] == pytest.approx(1.8410216450 / 2)  # not 0 +- 0
        # data as the reference: lowering mc / data raises data, so its upper error enters
        inverse = compare(mc, data, uncertainty="numerator")
        assert inverse.band is not None
        np.testing.assert_allclose(inverse.band[1][1:], (high[1:] - [1, 9]) / [1, 9])
        propagated = compare(mc, data)
        n, d = np.array([2.0, 8.0]), np.array([1.0, 9.0])
        d_up = (high - [0.0, 1.0, 9.0])[1:]
        np.testing.assert_allclose(
            propagated.errors[0][1:], np.hypot(np.sqrt(n) / d, n * d_up / d**2)
        )
        pull = compare(data, mc, kind="pull")
        # data below the prediction: its upper error faces the reference
        sigma = np.hypot((high - [0.0, 1.0, 9.0])[:2], np.sqrt(2.0))
        np.testing.assert_allclose(pull.values[:2], [-2.0, -1.0] / sigma)


class TestComparePoints:
    """Efficiencies and profiles: independent points, asymmetric intervals propagated."""

    EDGES = np.array([0.0, 1.0, 2.0])

    def _efficiencies(self) -> tuple[Efficiency, Efficiency]:
        a = Efficiency(
            values=np.array([0.5, 0.9]),
            lower=np.array([0.4, 0.8]),
            upper=np.array([0.55, 0.95]),
            edges=self.EDGES,
            label="A",
        )
        b = Efficiency(
            values=np.array([0.25, 0.9]),
            lower=np.array([0.2, 0.85]),
            upper=np.array([0.35, 0.92]),
            edges=self.EDGES,
            label="B",
        )
        return a, b

    def test_a_ratio_keeps_the_intervals_asymmetric(self) -> None:
        a, b = self._efficiencies()
        (a_down, a_up), (b_down, b_up) = a.errors, b.errors
        result = compare(a, b)
        np.testing.assert_allclose(result.values, [2.0, 1.0])
        down, up = result.errors
        # lowering a / b lowers a or raises b: a's lower error with b's upper one
        np.testing.assert_allclose(down, np.hypot(a_down / b.values, a.values * b_up / b.values**2))
        np.testing.assert_allclose(up, np.hypot(a_up / b.values, a.values * b_down / b.values**2))
        assert (result.label, result.reference, result.band) == ("A", "B", None)
        assert result.reference_hist is None
        assert result.reference_points is b
        np.testing.assert_array_equal(result.total_errors()[0], down)
        relative = compare(a, b, kind="relative_difference")
        np.testing.assert_allclose(relative.values, [1.0, 0.0])
        np.testing.assert_allclose(relative.errors, result.errors)

    def test_difference_asymmetry_and_pull(self) -> None:
        a, b = self._efficiencies()
        (a_down, a_up), (b_down, b_up) = a.errors, b.errors
        difference = compare(a, b, kind="difference")
        np.testing.assert_allclose(difference.values, [0.25, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            difference.errors, (np.hypot(a_down, b_up), np.hypot(a_up, b_down))
        )
        asymmetry = compare(a, b, kind="asymmetry")
        total = a.values + b.values
        np.testing.assert_allclose(asymmetry.values, (a.values - b.values) / total, atol=1e-12)
        pa, pb = 2 * b.values / total**2, 2 * a.values / total**2
        np.testing.assert_allclose(
            asymmetry.errors, (np.hypot(pa * a_down, pb * b_up), np.hypot(pa * a_up, pb * b_down))
        )
        pull = compare(a, b, kind="pull")
        # a > b in bin 0: the lower error of a - b faces zero; a == b in bin 1: the upper
        np.testing.assert_allclose(pull.values, [0.25 / np.hypot(0.1, 0.1), 0.0], atol=1e-12)
        np.testing.assert_array_equal(_symmetric(pull.errors), [1.0, 1.0])

    def test_profiles_propagate_symmetric_errors(self) -> None:
        def profile_(values: list[float], errors: list[float], label: str) -> Profile:
            return Profile(
                values=np.array(values),
                errors=np.array(errors),
                counts=np.ones(2),
                edges=self.EDGES,
                label=label,
            )

        a, b = profile_([2.0, 4.0], [0.2, 0.4], "A"), profile_([1.0, 2.0], [0.1, 0.1], "B")
        result = compare(a, b)
        expected = result.values * np.hypot(a.errors / a.values, b.errors / b.values)
        down, up = result.errors
        np.testing.assert_allclose(down, expected)
        np.testing.assert_allclose(up, expected)
        std = Profile(a.values, a.errors, a.counts, a.edges, statistic="std")
        with pytest.raises(ValueError, match="one statistic"):
            compare(std, b)

    def test_undefined_points_have_no_errors(self) -> None:
        a, b = self._efficiencies()
        empty = Efficiency(
            values=np.array([0.0, np.nan]),
            lower=np.array([0.0, np.nan]),
            upper=np.array([0.1, np.nan]),
            edges=self.EDGES,
        )
        result = compare(a, empty)
        assert np.isnan(result.values).all()  # an efficiency of zero, and an empty bin
        assert all(np.isnan(side).all() for side in result.errors)
        unknown = Efficiency(a.values, np.array([np.nan, 0.8]), a.upper, self.EDGES)
        down, up = compare(unknown, b).errors  # negative weights: no interval, no error bar
        assert np.isnan(down[0])
        assert np.isfinite(up[0])

    def test_refusals(self) -> None:
        a, b = self._efficiencies()
        profile_ = Profile(a.values, a.values, a.values, self.EDGES)
        with pytest.raises(TypeError, match="two efficiencies or two profiles"):
            compare(a, profile_)
        with pytest.raises(TypeError, match="two efficiencies or two profiles"):
            compare(_poisson([1.0, 2.0]), a)
        with pytest.raises(ValueError, match="counts signal and background"):
            compare(a, b, kind="s/sqrt(b)")
        with pytest.raises(ValueError, match="reference band"):
            compare(a, b, uncertainty="numerator")
        shifted = Efficiency(b.values, b.lower, b.upper, self.EDGES + 1)
        with pytest.raises(BinningError, match="identical bin edges"):
            compare(a, shifted)


class TestSignificance:
    def test_s_over_sqrt_b(self) -> None:
        signal = _hist([0.5, 0.5, 1.5])  # s = [2, 1, 0]
        background = _hist([0.5] * 4 + [1.5] * 1)  # b = [4, 1, 0]
        result = compare(signal, background, kind="s/sqrt(b)")
        np.testing.assert_allclose(result.values[:2], [2 / 2, 1 / 1])
        assert np.isnan(result.values[2])
        # var = vs/b + s^2 vb/(4 b^3): bin 0 -> 2/4 + 4*4/(4*64) = 0.5 + 0.0625
        assert _symmetric(result.errors)[0] == pytest.approx(np.sqrt(0.5625))
        assert result.band is None
        np.testing.assert_allclose(result.edges, [0, 1, 2, 3])

    def test_s_over_sqrt_s_plus_b(self) -> None:
        signal = _hist([0.5, 0.5])
        background = _hist([0.5, 0.5, 1.5])
        result = compare(signal, background, kind="s/sqrt(s+b)")
        assert result.values[0] == pytest.approx(2 / np.sqrt(4))
        assert result.values[1] == pytest.approx(0.0)
        assert np.isnan(result.values[2])
        assert np.isfinite(_symmetric(result.errors)[0])
        with pytest.raises(BinningError):
            compare(
                signal,
                hist.Hist(hist.axis.Regular(4, 0, 4), storage=hist.storage.Weight()),
                kind="s/sqrt(s+b)",
            )

    def test_statistical_only(self) -> None:
        signal = Histogram(_hist([0.5, 0.5]), label="S", variations={"lumi": (_hist([0.5]), None)})
        plain = compare(signal.hist, _hist([0.5, 1.5]), kind="s/sqrt(b)")
        result = compare(signal, _hist([0.5, 1.5]), kind="s/sqrt(b)")
        np.testing.assert_array_equal(result.values, plain.values)
        np.testing.assert_array_equal(result.errors, plain.errors)
        assert result.syst_errors is None


class TestEfficiency:
    def test_wilson_interval(self) -> None:
        from rootfig.histograms import efficiency

        total = _hist([0.5] * 10 + [1.5] * 4)
        passed = _hist([0.5] * 5 + [1.5] * 4)
        eff = efficiency(passed, total, label="tight", interval="wilson")
        assert eff.label == "tight"
        assert eff.values[0] == pytest.approx(0.5)
        assert eff.values[1] == pytest.approx(1.0)
        assert np.isnan(eff.values[2])
        # Wilson, z = 1, p = 0.5, n = 10: centre 0.5, half-width (1/1.1) sqrt(0.025 + 0.0025)
        half = np.sqrt(0.0275) / 1.1
        assert eff.lower[0] == pytest.approx(0.5 - half)
        assert eff.upper[0] == pytest.approx(0.5 + half)
        assert eff.upper[1] == 1.0  # clipped
        assert eff.lower[1] < 1.0
        low_err, up_err = eff.errors
        assert low_err[0] == pytest.approx(half)
        assert up_err[0] == pytest.approx(half)
        np.testing.assert_allclose(eff.centers, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(eff.half_widths, [0.5, 0.5, 0.5])

    def test_weights_use_effective_entries(self) -> None:
        from rootfig.histograms import efficiency

        unweighted = efficiency(_hist([0.5] * 5), _hist([0.5] * 10), interval="wilson")
        weighted = efficiency(
            _hist([0.5] * 5, [2.0] * 5), _hist([0.5] * 10, [2.0] * 10), interval="wilson"
        )
        assert weighted.values[0] == pytest.approx(unweighted.values[0])
        assert weighted.lower[0] == pytest.approx(unweighted.lower[0])  # same n_eff = 10
        # all of weights 1, 2 and 3 pass: n_eff = 36 / 14, and the interval keeps a width,
        # where ROOT's weighted normal approximation (the default) gives [1, 1]
        all_pass = _hist([0.5] * 3, [1.0, 2.0, 3.0])
        n_eff = 36 / 14
        wilson = efficiency(all_pass, all_pass, interval="wilson")
        assert wilson.lower[0] == pytest.approx(n_eff / (n_eff + 1))
        assert efficiency(all_pass, all_pass).lower[0] == 1.0
        with pytest.raises(BinningError):
            efficiency(
                _hist([0.5]), hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
            )


class TestEfficiencyIntervals:
    def test_the_default_follows_the_weights_and_explicit_choices_are_checked(self) -> None:
        from rootfig.histograms import efficiency
        from rootfig.histograms.binomial import is_unweighted, resolve_interval

        assert is_unweighted(3.0, 3.0)
        assert is_unweighted(0.0, 0.0)
        assert not is_unweighted(6.0, 12.0)
        assert resolve_interval("auto", True) == "clopper-pearson"
        assert resolve_interval("auto", False) == "normal"
        assert resolve_interval("wilson", False) == "wilson"
        with pytest.raises(ValueError, match="needs unweighted"):
            efficiency(
                _hist([0.5], [2.0]), _hist([0.5, 0.5], [2.0, 2.0]), interval="clopper-pearson"
            )
        with pytest.raises(ValueError, match="interval must be"):
            efficiency(_hist([0.5]), _hist([0.5]), interval="jeffreys")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="interval must be"):
            Cutflow("s", (), interval="exact")  # type: ignore[arg-type]
        weighted = CutflowStep(label="", expression="", events=2, yield_=3.0, error=np.sqrt(5.0))
        with pytest.raises(ValueError, match="needs unweighted"):
            Cutflow("w", (weighted,), interval="clopper-pearson")

    def test_undefined_inputs(self) -> None:
        lower, upper = clopper_pearson([0.0, 3.0, -1.0, 2.0], [0.0, 2.0, 2.0, -1.0])
        assert np.isnan(lower).all()
        assert np.isnan(upper).all()
        lower, upper = normal_interval([1.0, 3.0], [0.0, 2.0], [1.0, 3.0], [1.0, 2.0])
        assert np.isnan(lower).all()
        assert np.isnan(upper).all()
        empty = clopper_pearson(np.zeros(0), np.zeros(0))
        assert empty[0].size == empty[1].size == 0

    @pytest.mark.parametrize("scale", [2.5, -2.5, 0.0])
    def test_a_cutflow_leaves_the_sample_scale_out(self, scale: float) -> None:
        from rootfig.histograms import cutflow

        n = np.arange(20.0)
        cuts = ["n >= 5", "n >= 12"]
        for weight in (None, "1 + n % 3"):
            plain = cutflow(Sample({"n": n}, weight=weight), cuts)
            scaled = cutflow(Sample({"n": n}, weight=weight, scale=scale), cuts)
            assert scaled.interval == plain.interval
            np.testing.assert_allclose(scaled.yields, scale * plain.yields)
            np.testing.assert_allclose(scaled.efficiencies, plain.efficiencies)
            np.testing.assert_allclose(scaled.absolute_efficiencies, plain.absolute_efficiencies)
            for got, want in zip(scaled.efficiency_errors, plain.efficiency_errors, strict=True):
                np.testing.assert_allclose(got, want)
            assert not any(step.negative_weights for step in scaled.steps)

    def test_efficiency_bounds_needs_a_resolved_interval(self) -> None:
        from rootfig.histograms.binomial import efficiency_bounds

        with pytest.raises(ValueError, match="resolved interval"):
            efficiency_bounds("auto", [1.0], [2.0], [1.0], [2.0])

    def test_a_hand_made_cutflow_resolves_auto_from_its_yields(self) -> None:
        def step(events: int, yield_: float, error: float) -> CutflowStep:
            return CutflowStep(label="", expression="", events=events, yield_=yield_, error=error)

        counts = Cutflow("c", (step(10, 10.0, np.sqrt(10.0)), step(4, 4.0, 2.0)))
        lower, upper = clopper_pearson([4.0], [10.0])
        np.testing.assert_allclose(
            [e[1] for e in counts.efficiency_errors], [0.4 - lower[0], upper[0] - 0.4]
        )
        weighted = Cutflow("w", (step(10, 20.0, np.sqrt(40.0)), step(4, 8.0, 4.0)))
        lower, upper = normal_interval([8.0], [20.0], [16.0], [40.0])
        np.testing.assert_allclose(
            [e[1] for e in weighted.efficiency_errors], [0.4 - lower[0], upper[0] - 0.4]
        )


class TestProfile:
    def test_mean_and_std(self) -> None:
        from rootfig.histograms import profile

        x = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 5.0])  # last is outside
        y = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 99.0])
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mean = profile(x, y, edges)
        np.testing.assert_allclose(mean.values, [2.0, 5.0, 7.0, np.nan])
        np.testing.assert_allclose(mean.counts, [3, 2, 1, 0])
        assert mean.errors[0] == pytest.approx(np.sqrt(2 / 3) / np.sqrt(3))
        assert mean.errors[2] == 0.0
        std = profile(x, y, edges, statistic="std", label="res")
        assert std.label == "res"
        assert std.values[0] == pytest.approx(np.sqrt(2 / 3))
        assert std.errors[0] == pytest.approx(np.sqrt(2 / 3) / np.sqrt(6))
        np.testing.assert_allclose(std.centers, [0.5, 1.5, 2.5, 3.5])
        np.testing.assert_allclose(std.half_widths, 0.5)

    def test_weights_and_errors(self) -> None:
        from rootfig.histograms import profile

        x = np.array([0.5, 0.5])
        y = np.array([1.0, 3.0])
        weighted = profile(x, y, np.array([0.0, 1.0]), weights=np.array([3.0, 1.0]))
        assert weighted.values[0] == pytest.approx(1.5)
        with pytest.raises(BinningError, match="same length"):
            profile(x, y[:1], np.array([0.0, 1.0]))
        with pytest.raises(BinningError, match="statistic"):
            profile(x, y, np.array([0.0, 1.0]), statistic="median")  # type: ignore[arg-type]


class TestCutflow:
    @staticmethod
    def _sample(**kwargs: Any) -> Sample:
        arrays = {
            "n": np.array([0, 1, 2, 3, 4]),
            "w": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "Jet_pt": ak.Array([[], [10.0], [10.0, 40.0], [50.0], [5.0, 5.0, 60.0]]),
        }
        return Sample(arrays, label="toy", **kwargs)

    def test_steps_and_efficiencies(self) -> None:
        from rootfig.histograms import cutflow

        flow = cutflow(self._sample(), ["n >= 1", Cut("Jet_pt > 30", label="hard jet"), "n >= 4"])
        assert flow.labels == ["All", "n >= 1", "hard jet", "n >= 4"]
        assert flow.events.tolist() == [5, 4, 3, 1]  # per-object cut: any jet above 30
        assert flow.yields.tolist() == [5, 4, 3, 1]  # unit weights
        np.testing.assert_allclose(flow.efficiencies, [1, 0.8, 0.75, 1 / 3])
        np.testing.assert_allclose(flow.absolute_efficiencies, [1, 0.8, 0.6, 0.2])
        assert flow.steps[1].expression == "n >= 1"

    def test_weights_selection_and_scale(self) -> None:
        from rootfig.histograms import cutflow

        sample = self._sample(weight="w", selection=Cut("n >= 1", label="baseline"), scale=2.0)
        flow = cutflow(sample, ["n >= 3"], weight="2")
        assert flow.labels == ["baseline", "n >= 3"]
        assert flow.events.tolist() == [4, 2]
        # weights w * 2 (plot) * 2 (scale): (2+3+4+5)*4 = 56 and (4+5)*4 = 36
        np.testing.assert_allclose(flow.yields, [56, 36])
        assert flow.steps[1].error == pytest.approx(np.sqrt(16**2 + 20**2))

    def test_unlabelled_sample_selection_names_first_step(self) -> None:
        from rootfig.histograms import cutflow

        flow = cutflow(self._sample(selection="n >= 1"), ["n >= 3"])
        assert flow.labels == ["n >= 1", "n >= 3"]  # not "All": 4 of 5 events pass it
        assert flow.steps[0].expression == "n >= 1"
        assert flow.events.tolist() == [4, 2]

    def test_signed_yields_keep_efficiencies(self) -> None:
        from rootfig.histograms import cutflow

        # weights 1 - 3 = -2 for all events, then -3 after n >= 4: the ratios are defined
        arrays = {"n": np.array([0, 4]), "w": np.array([1.0, -3.0])}
        flow = cutflow(Sample(arrays, label="nlo", weight="w"), ["n >= 4", "n >= 9"])
        np.testing.assert_allclose(flow.yields, [-2.0, -3.0, 0.0])
        np.testing.assert_allclose(flow.efficiencies, [1.0, 1.5, 0.0])  # may leave [0, 1]
        np.testing.assert_allclose(flow.absolute_efficiencies, [1.0, 1.5, 0.0])
        cancelled = cutflow(
            Sample({"n": np.array([0, 4]), "w": np.array([1.0, -1.0])}, weight="w"), ["n >= 4"]
        )
        assert np.isnan(cancelled.efficiencies[1])  # zero yield: undefined
        assert np.isnan(cancelled.absolute_efficiencies).all()

    @staticmethod
    def _wilson(k: float, n: float, n_eff: float) -> tuple[float, float]:
        """The Wilson interval at one standard deviation, written out."""
        p = k / n
        centre = (p + 1 / (2 * n_eff)) / (1 + 1 / n_eff)
        half = np.sqrt(p * (1 - p) / n_eff + 1 / (4 * n_eff**2)) / (1 + 1 / n_eff)
        return p - (centre - half), (centre + half) - p

    def test_efficiency_errors_are_binomial(self) -> None:
        from rootfig.histograms import cutflow

        sample = Sample({"n": np.arange(100.0)}, label="counts")
        flow = cutflow(sample, ["n >= 40", "n >= 70"])  # 100 -> 60 -> 30
        assert flow.interval == "clopper-pearson"  # unweighted events, as TEfficiency
        down, up = flow.efficiency_errors
        assert (down[0], up[0]) == (0.0, 0.0)  # the first step is the reference itself

        def cp(k: float, n: float) -> list[float]:
            lower, upper = clopper_pearson([k], [n])
            return [k / n - lower[0], upper[0] - k / n]

        np.testing.assert_allclose([down[1], up[1]], cp(60, 100))
        np.testing.assert_allclose([down[2], up[2]], cp(30, 60))
        absolute = flow.absolute_efficiency_errors
        np.testing.assert_allclose([absolute[0][2], absolute[1][2]], cp(30, 100))
        assert not any(step.negative_weights for step in flow.steps)
        wilson = cutflow(sample, ["n >= 40", "n >= 70"], interval="wilson")
        down, up = wilson.efficiency_errors
        np.testing.assert_allclose([down[1], up[1]], self._wilson(60, 100, 100))
        # 60 of 100: 0.6 -0.0497 +0.0478, not the independent-yield sqrt(1/60 + 1/100)
        assert (down[1], up[1]) == pytest.approx((0.04975, 0.04777), abs=1e-5)

    def test_scale_keeps_events_unweighted_and_weights_do_not(self) -> None:
        from rootfig.histograms import cutflow

        cuts = ["n >= 40", "n >= 70"]
        plain = cutflow(Sample({"n": np.arange(100.0)}), cuts)
        scaled = cutflow(Sample({"n": np.arange(100.0)}, scale=2.5), cuts)
        assert scaled.interval == "clopper-pearson"  # a factor the efficiency cancels
        for got, want in zip(scaled.efficiency_errors, plain.efficiency_errors, strict=True):
            np.testing.assert_allclose(got, want)
        # an event weight makes the events weighted for ROOT, even one shared by all
        weighted = cutflow(Sample({"n": np.arange(100.0)}, weight="2.5"), cuts)
        assert weighted.interval == "normal"
        down, up = weighted.efficiency_errors
        lower, upper = normal_interval([60.0], [100.0], [60.0], [100.0])
        np.testing.assert_allclose([down[1], up[1]], [0.6 - lower[0], upper[0] - 0.6])
        with pytest.raises(ValueError, match="needs unweighted"):
            cutflow(Sample({"n": np.arange(100.0)}, weight="2.5"), cuts, interval="clopper-pearson")
        with pytest.raises(ValueError, match="interval must be"):
            cutflow(Sample({"n": np.arange(100.0)}), cuts, interval="bayes")  # type: ignore[arg-type]

    def test_weighted_events_use_the_effective_entries(self) -> None:
        from rootfig.histograms import cutflow

        weights = np.where(np.arange(100) % 2, 3.0, 1.0)  # 50 of weight 1, 50 of weight 3
        sample = Sample({"n": np.arange(100.0), "w": weights}, weight="w")
        flow = cutflow(sample, ["n >= 40"], interval="wilson")
        n, vn = 200.0, 50 * 1.0 + 50 * 9.0
        k, vk = float(weights[40:].sum()), float(np.sum(weights[40:] ** 2))
        down, up = flow.efficiency_errors
        np.testing.assert_allclose([down[1], up[1]], self._wilson(k, n, n**2 / vn))
        down, up = cutflow(sample, ["n >= 40"]).efficiency_errors  # ROOT: normal
        lower, upper = normal_interval([k], [n], [vk], [vn])
        np.testing.assert_allclose([down[1], up[1]], [k / n - lower[0], upper[0] - k / n])

    def test_an_empty_step_leaves_later_efficiencies_undefined(self) -> None:
        from rootfig.histograms import cutflow

        flow = cutflow(Sample({"n": np.arange(10.0)}), ["n > 100", "n > 200"])
        down, up = flow.efficiency_errors
        # 0 of 10: Clopper-Pearson's upper bound 1 - 0.1587^(1/10)
        assert (down[1], up[1]) == (0.0, pytest.approx(0.16814918613797644))
        assert np.isnan(flow.efficiencies[2])
        assert np.isnan(down[2])
        assert np.isnan(up[2])
        empty = cutflow(Sample({"n": np.arange(10.0)}, selection="n < 0"), ["n > 1"])
        assert np.isnan(empty.absolute_efficiency_errors[0]).all()

    def test_negative_weights_have_no_interval(self) -> None:
        from rootfig.histograms import cutflow

        # one negative weight among ten, cut away by the first cut
        weights = np.r_[-1.0, np.ones(9)]
        sample = Sample({"n": np.arange(10.0), "w": weights}, weight="w")
        flow = cutflow(sample, ["n >= 1", "n >= 5"], interval="wilson")
        assert [step.negative_weights for step in flow.steps] == [True, False, False]
        np.testing.assert_allclose(flow.efficiencies, [1.0, 9 / 8, 5 / 9])  # still reported
        down, up = flow.efficiency_errors
        assert np.isnan([down[1], up[1]]).all()  # measured against a signed step
        np.testing.assert_allclose([down[2], up[2]], self._wilson(5, 9, 9))
        absolute_down, _ = flow.absolute_efficiency_errors
        assert np.isnan(absolute_down[1:]).all()  # every step measured against the first
        assert absolute_down[0] == 0.0
        # the normal approximation (ROOT's for weighted events) holds for signed weights
        # too; only an efficiency outside [0, 1] has none
        down, up = cutflow(sample, ["n >= 1", "n >= 5"]).efficiency_errors
        assert np.isnan([down[1], up[1]]).all()  # 9 / 8
        lower, upper = normal_interval([5.0], [9.0], [5.0], [9.0])
        np.testing.assert_allclose([down[2], up[2]], [5 / 9 - lower[0], upper[0] - 5 / 9])

    def test_table(self) -> None:
        from rootfig.histograms import CutflowTable, cutflow

        table = CutflowTable(
            (cutflow(self._sample(), ["n >= 1"]), cutflow(self._sample(weight="w"), ["n >= 1"]))
        )
        text = str(table)
        assert "toy" in text
        assert "n >= 1" in text
        assert "80.0%" in text
        assert table.samples == ["toy", "toy"]
        assert table.labels == ["All", "n >= 1"]
        assert table.get("toy").events.tolist() == [5, 4]
        with pytest.raises(KeyError):
            table.get("other")
        assert str(CutflowTable(())) == ""
        assert cutflow(self._sample(), []).events.tolist() == [5]

    def test_errors(self) -> None:
        from rootfig.errors import IncompatibleWeightError
        from rootfig.histograms import cutflow

        with pytest.raises(IncompatibleWeightError, match="per-object"):
            cutflow(self._sample(), ["n >= 1"], weight="Jet_pt")
        with pytest.raises(SelectionError, match="depth"):
            cutflow(Sample({"a": ak.Array([[[1.0]], [[2.0]]])}), ["a > 1"])


class TestSignedWeights:
    def test_summary_with_negative_variance_is_nan_not_an_error(self) -> None:
        s = summarize(columns([0.0, 1.0], [1.0, -0.5]))
        assert s.mean == -1.0
        assert np.isnan(s.std)
        assert np.isnan(s.sem)
        assert np.isnan(s.skewness)
        assert "nan" in s.format()

    def test_histogram_fill_does_not_depend_on_statistics(self) -> None:
        sample = Sample({"x": np.array([0.0, 1.0]), "w": np.array([1.0, -0.5])})
        [h] = build_histograms([sample], Variable("x", bins=(2, 0, 2)), weight="w")
        np.testing.assert_allclose(h.values(), [1.0, -0.5])
        np.testing.assert_allclose(h.variances(), [1.0, 0.25])
        assert h.stats is not None
        assert np.isnan(h.stats.std)


class TestEfficiencyDomain:
    @staticmethod
    def _hists(passed: int, total: int, *, weights: list[float] | None = None) -> tuple[Any, Any]:
        axis = hist.axis.Regular(1, 0, 1)
        pass_h = hist.Hist(axis, storage=hist.storage.Weight())
        total_h = hist.Hist(axis, storage=hist.storage.Weight())
        total_h.fill(np.full(total, 0.5), weight=weights)
        pass_h.fill(np.full(passed, 0.5), weight=None if weights is None else weights[:passed])
        return pass_h, total_h

    @pytest.mark.parametrize("n", [1, 3, 6, 50])
    def test_all_pass_and_all_fail_have_nonnegative_errors(self, n: int) -> None:
        from rootfig.histograms import efficiency

        full = efficiency(*self._hists(n, n))
        assert full.values[0] == 1.0
        assert full.upper[0] == 1.0
        assert 0.0 < full.lower[0] < 1.0
        none = efficiency(*self._hists(0, n))
        assert none.values[0] == 0.0
        assert none.lower[0] == 0.0
        assert 0.0 < none.upper[0] < 1.0
        for eff in (full, none):
            low, high = eff.errors
            assert (low >= 0).all()
            assert (high >= 0).all()

    def test_weighted_endpoints(self) -> None:
        from rootfig.histograms import efficiency

        eff = efficiency(*self._hists(4, 4, weights=[0.5, 1.5, 2.0, 0.25]))
        assert eff.values[0] == 1.0
        assert eff.upper[0] == 1.0
        assert all((e >= 0).all() for e in eff.errors)

    def test_outside_unit_interval_is_undefined_with_warning(self) -> None:
        from rootfig.histograms import efficiency

        axis = hist.axis.Regular(1, 0, 1)
        pass_h = hist.Hist(axis, storage=hist.storage.Weight()).fill([0.5, 0.5], weight=[1.0, 1.0])
        total_h = hist.Hist(axis, storage=hist.storage.Weight()).fill(
            [0.5, 0.5, 0.5], weight=[1.0, 1.0, -0.5]
        )
        with pytest.warns(RootfigWarning, match="outside"):
            eff = efficiency(pass_h, total_h, label="nlo")
        assert eff.values[0] == pytest.approx(2 / 1.5)
        assert np.isnan(eff.lower[0])
        assert np.isnan(eff.upper[0])

    @staticmethod
    def _weighted(passed: list[float], total: list[float]) -> tuple[Any, Any]:
        axis = hist.axis.Regular(1, 0, 1)
        pass_h = hist.Hist(axis, storage=hist.storage.Weight()).fill(
            np.full(len(passed), 0.5), weight=passed
        )
        total_h = hist.Hist(axis, storage=hist.storage.Weight()).fill(
            np.full(len(total), 0.5), weight=total
        )
        return pass_h, total_h

    def test_negative_total_weight_keeps_value_without_interval(self) -> None:
        from rootfig.histograms import efficiency

        # total = 1 - 2 = -1: the ratio is defined, an interval is not
        with pytest.warns(RootfigWarning, match="negative weights"):
            eff = efficiency(*self._weighted([-0.5], [1.0, -2.0]), interval="wilson")
        assert eff.values[0] == pytest.approx(0.5)
        assert np.isnan(eff.lower[0])
        assert np.isnan(eff.upper[0])
        with pytest.warns(RootfigWarning, match="negative weights"):
            eff = efficiency(*self._weighted([-2.0], [1.0, -2.0]), interval="wilson")
        assert eff.values[0] == pytest.approx(2.0)
        assert np.isnan(eff.lower[0])
        with pytest.warns(RootfigWarning, match="negative total"):  # the normal approximation
            eff = efficiency(*self._weighted([-0.5], [1.0, -2.0]))
        assert np.isnan([eff.lower[0], eff.upper[0]]).all()

    def test_cancelling_total_weight_is_empty(self) -> None:
        from rootfig.histograms import efficiency

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eff = efficiency(*self._weighted([1.0], [1.0, -1.0]))
        assert np.isnan(eff.values[0])
        assert np.isnan(eff.lower[0])
        assert np.isnan(eff.upper[0])

    def test_negative_weights_inside_the_unit_interval_have_no_interval(self) -> None:
        from rootfig.histograms import efficiency

        # p = 1 / 1.5 lies in [0, 1], but the failing entries {1, -0.5} sum to 0.5 with
        # squares summing to 1.25 > 0.25, which non-negative weights never give
        with pytest.warns(RootfigWarning, match="negative weights"):
            eff = efficiency(*self._weighted([1.0], [1.0, 1.0, -0.5]), interval="wilson")
        assert eff.values[0] == pytest.approx(2 / 3)
        assert np.isnan(eff.lower[0])
        assert np.isnan(eff.upper[0])

    def test_the_normal_approximation_holds_for_signed_weights(self) -> None:
        from rootfig.histograms import efficiency

        # ROOT 6.40, TEfficiency and TGraphAsymmErrors::Divide alike: {2, -1} pass, {1, 1}
        # fail gives 1/3 in [0, 0.854491]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eff = efficiency(*self._weighted([2.0, -1.0], [2.0, -1.0, 1.0, 1.0]))
        assert eff.values[0] == pytest.approx(1 / 3)
        assert (eff.lower[0], eff.upper[0]) == pytest.approx((0.0, 0.854490637), abs=1e-8)

    def test_negative_weights_the_sums_hide_need_the_flag(self) -> None:
        from rootfig.histograms import efficiency

        # passing {1, 1, -0.1} and failing {1}: every sum looks like non-negative weights
        hists = self._weighted([1.0, 1.0, -0.1], [1.0, 1.0, -0.1, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            unknown = efficiency(*hists, interval="wilson")
            normal = efficiency(*hists, negative_weights=[True])  # needs no flag
        assert np.isfinite(unknown.lower[0])  # nothing in the sums says so
        assert np.isfinite(normal.lower[0])
        with pytest.warns(RootfigWarning, match="negative weights"):
            flagged = efficiency(*hists, negative_weights=[True], interval="wilson")
        assert flagged.values[0] == pytest.approx(1.9 / 2.9)
        assert np.isnan(flagged.lower[0])
        assert np.isnan(flagged.upper[0])

    def test_positive_weights_keep_their_interval(self) -> None:
        from rootfig.histograms import efficiency

        # positive weights of different sizes, all passing or all failing included
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eff = efficiency(*self._weighted([0.5, 3.0], [0.5, 3.0, 1.0, 0.2]), interval="wilson")
            ends = [
                efficiency(*self._weighted(w, [0.5, 3.0]), interval="wilson")
                for w in ([], [0.5, 3.0])
            ]
        assert np.isfinite(eff.lower[0])
        assert np.isfinite(eff.upper[0])
        n, vn = 4.7, 0.25 + 9.0 + 1.0 + 0.04
        n_eff = n**2 / vn
        p = 3.5 / n
        centre = (p + 1 / (2 * n_eff)) / (1 + 1 / n_eff)
        half = np.sqrt(p * (1 - p) / n_eff + 1 / (4 * n_eff**2)) / (1 + 1 / n_eff)
        assert (eff.lower[0], eff.upper[0]) == pytest.approx((centre - half, centre + half))
        assert [e.values[0] for e in ends] == [0.0, 1.0]
        assert all(np.isfinite(e.lower[0]) and np.isfinite(e.upper[0]) for e in ends)

    @pytest.mark.parametrize("z", [0.0, -1.0, np.inf, np.nan])
    def test_z_is_validated(self, z: float) -> None:
        from rootfig.histograms import efficiency

        with pytest.raises(BinningError, match="z must be"):
            efficiency(*self._hists(1, 2), z=z)


class TestProfileNumerics:
    def test_variance_is_translation_invariant(self) -> None:
        from rootfig.histograms import profile

        x = np.array([0.5, 0.5])
        edges = np.array([0.0, 1.0])
        near = profile(x, np.array([0.0, 1.0]), edges, statistic="std")
        far = profile(x, np.array([1e9, 1e9 + 1.0]), edges, statistic="std")
        np.testing.assert_allclose(near.values, [0.5])
        np.testing.assert_allclose(far.values, [0.5])
        np.testing.assert_allclose(far.errors, near.errors)
        weighted = profile(x, np.array([1e9, 1e9 + 1.0]), edges, weights=np.array([3.0, 1.0]))
        np.testing.assert_allclose(weighted.values, [1e9 + 0.25])

    def test_negative_weighted_variance_is_nan(self) -> None:
        from rootfig.histograms import profile

        result = profile(
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            weights=np.array([1.0, -0.5]),
        )
        np.testing.assert_allclose(result.values, [-1.0])
        assert np.isnan(result.errors[0])
        as_std = profile(
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            weights=np.array([1.0, -0.5]),
            statistic="std",
        )
        assert np.isnan(as_std.values[0])

    def test_negative_total_weight_keeps_mean_without_error(self) -> None:
        from rootfig.histograms import profile

        x, y, edges = np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        result = profile(x, y, edges, weights=np.array([0.5, -1.0]))
        np.testing.assert_allclose(result.values, [2.0])  # (0*0.5 + 1*-1) / -0.5
        assert np.isnan(result.errors[0])
        np.testing.assert_allclose(result.counts, [-0.5])
        as_std = profile(x, y, edges, weights=np.array([0.5, -1.0]), statistic="std")
        assert np.isnan(as_std.values[0])

    def test_cancelling_weights_are_empty(self) -> None:
        from rootfig.histograms import profile

        x, y, edges = np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        result = profile(x, y, edges, weights=np.array([1.0, -1.0]))
        assert np.isnan(result.values[0])
        assert np.isnan(result.errors[0])
        assert result.counts[0] == 0.0


class TestFlowNormalisation:
    @staticmethod
    def _flow_hist() -> Any:
        h = hist.Hist(hist.axis.Variable([0.0, 2.0, 6.0]), storage=hist.storage.Weight())
        h.fill([-1.0, 1.0, 3.0, 7.0], weight=[8.0, 10.0, 12.0, 16.0])
        return h

    def test_width_divides_flow_bins_by_neighbouring_width(self) -> None:
        result = normalize_hist(self._flow_hist(), "width")
        np.testing.assert_allclose(result.values(flow=True), [4.0, 5.0, 3.0, 4.0])
        np.testing.assert_allclose(result.variances(flow=True), [16.0, 25.0, 9.0, 16.0])
        density = normalize_hist(self._flow_hist(), "density")
        np.testing.assert_allclose(density.values(flow=True), np.array([4.0, 5.0, 3.0, 4.0]) / 22)
        np.testing.assert_allclose((density.values() * np.array([2.0, 4.0])).sum(), 1.0)

    def test_2d_flow_cells(self) -> None:
        h = hist.Hist(
            hist.axis.Variable([0.0, 2.0, 6.0]),
            hist.axis.Variable([0.0, 1.0, 3.0]),
            storage=hist.storage.Weight(),
        )
        h.fill([-1.0, 1.0, 7.0], [-1.0, 0.5, 5.0])
        values = normalize_hist(h, "width").values(flow=True)
        assert values[0, 0] == pytest.approx(1 / (2 * 1))  # both underflows
        assert values[1, 1] == pytest.approx(1 / (2 * 1))
        assert values[-1, -1] == pytest.approx(1 / (4 * 2))  # both overflows

    def test_axes_without_flow_bins(self) -> None:
        axis = hist.axis.Regular(2, 0, 4, underflow=False, overflow=False)
        h = hist.Hist(axis, storage=hist.storage.Weight()).fill([1.0, 3.0])
        np.testing.assert_allclose(normalize_hist(h, "width").values(flow=True), [0.5, 0.5])


class TestWeightStorage:
    def test_plain_storages_are_converted(self) -> None:
        from rootfig.histograms import as_weight_storage

        double = hist.Hist(hist.axis.Regular(2, 0, 4)).fill([1.0, 3.0, 3.0])
        converted = as_weight_storage(double)
        assert converted.storage_type is hist.storage.Weight
        np.testing.assert_allclose(converted.values(), [1.0, 2.0])
        np.testing.assert_allclose(converted.variances(), [1.0, 2.0])
        np.testing.assert_allclose(double.values(), [1.0, 2.0])  # original untouched
        integer = hist.Hist(hist.axis.Regular(2, 0, 4), storage=hist.storage.Int64()).fill([1.0])
        np.testing.assert_allclose(as_weight_storage(integer).variances(), [1.0, 0.0])
        weighted = hist.Hist(hist.axis.Regular(2, 0, 4), storage=hist.storage.Weight()).fill([1.0])
        assert as_weight_storage(weighted) is weighted
        np.testing.assert_allclose(normalize_hist(double, "width").values(), [0.5, 1.0])

    def test_weighted_plain_storage_needs_explicit_poisson_assumption(self) -> None:
        from rootfig.histograms import as_weight_storage

        # a count storage forgets the sum of squared weights: hist reports no variances
        double = hist.Hist(hist.axis.Regular(2, 0, 4)).fill([1.0, 3.0], weight=[2.0, -3.0])
        assert double.variances() is None
        with pytest.raises(ValueError, match="assume_poisson=True"):
            as_weight_storage(double)
        with pytest.raises(ValueError, match="no variances"):
            Histogram(double, label="h")
        with pytest.warns(RootfigWarning, match="Poisson guess"):
            converted = as_weight_storage(double, assume_poisson=True)
        np.testing.assert_allclose(converted.values(), [2.0, -3.0])
        np.testing.assert_allclose(converted.variances(), [2.0, 3.0])  # never negative
        rescaled = hist.Hist(hist.axis.Regular(2, 0, 4)).fill([1.0]) * 2
        with pytest.raises(ValueError, match="rescaled"):
            as_weight_storage(rescaled)
        with pytest.warns(RootfigWarning, match="Poisson guess"):
            converted = as_weight_storage(rescaled, assume_poisson=True)
        np.testing.assert_allclose(converted.variances(), [2.0, 0.0])

    def test_unweighted_plain_storage_is_silent(self) -> None:
        from rootfig.histograms import as_weight_storage

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            double = as_weight_storage(hist.Hist(hist.axis.Regular(2, 0, 4)).fill([1.0, 3.0, 3.0]))
            integer = as_weight_storage(
                hist.Hist(hist.axis.Regular(2, 0, 4), storage=hist.storage.Int64()).fill([1.0])
            )
        np.testing.assert_allclose(double.variances(), [1.0, 2.0])
        np.testing.assert_allclose(integer.variances(), [1.0, 0.0])

    def test_unsupported_storage(self) -> None:
        from rootfig.histograms import as_weight_storage

        mean = hist.Hist(hist.axis.Regular(2, 0, 4), storage=hist.storage.Mean())
        mean.fill([1.0], sample=[2.0])
        with pytest.raises(TypeError, match="Mean storage"):
            as_weight_storage(mean)


class TestNegativeVariances:
    def test_count_storage_with_negative_contents(self) -> None:
        h = hist.Hist(hist.axis.Regular(2, 0, 2))
        h[...] = np.array([2.0, -1.0])
        assert h.variances() is not None  # boost reports the counts, one of them negative
        with pytest.raises(ValueError, match=r"negative bin contents.*assume_poisson=True"):
            as_weight_storage(h)
        with pytest.warns(RootfigWarning, match="Poisson guess"):
            converted = as_weight_storage(h, assume_poisson=True)
        np.testing.assert_allclose(converted.variances(), [2.0, 1.0])

    def test_weight_storage_with_negative_variances(self) -> None:
        h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        h.view().variance = np.array([1.0, -1.0])
        with pytest.raises(ValueError, match="negative variances"):
            as_weight_storage(h)
        with pytest.warns(RootfigWarning):
            converted = as_weight_storage(h, assume_poisson=True)
        np.testing.assert_allclose(converted.variances(), np.abs(h.values()))


class TestCategoryCompatibility:
    @staticmethod
    def _cutflow(categories: list[str], values: list[float]) -> Histogram:
        h = hist.Hist(hist.axis.StrCategory(categories, name="cut"), storage=hist.storage.Weight())
        for category, value in zip(categories, values, strict=True):
            h.fill([category], weight=value)
        return Histogram(h, label=" ".join(categories))

    def test_same_index_edges_different_categories(self) -> None:
        from rootfig.histograms import compatible_binning, sum_histograms

        a = self._cutflow(["all", "preselection", "final"], [10.0, 5.0, 2.0])
        b = self._cutflow(["all", "preselection", "control"], [10.0, 4.0, 1.0])
        same = self._cutflow(["all", "preselection", "final"], [8.0, 4.0, 1.0])
        assert compatible_binning(a.hist, same.hist)
        assert not compatible_binning(a.hist, b.hist)
        numeric = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
        assert not compatible_binning(a.hist, numeric)  # index edges agree, kinds do not
        np.testing.assert_allclose(sum_histograms([a, same]).values(), [18.0, 9.0, 3.0])
        with pytest.raises(BinningError, match="different bin edges"):
            sum_histograms([a, b])
        with pytest.raises(BinningError):
            compare(a.hist, b.hist)
        np.testing.assert_allclose(compare(a.hist, same.hist).values, [1.25, 1.25, 2.0])


class TestRebinnedTo:
    def test_counts_per_axis(self) -> None:
        h = hist.Hist(
            hist.axis.Regular(12, 0, 12, name="x"),
            hist.axis.Regular(4, 0, 4, name="y"),
            storage=hist.storage.Weight(),
        )
        h.fill([0.5] * 3, [0.5] * 3)
        merged = Histogram(h, label="h").rebinned_to([3, None])
        assert [a.size for a in merged.hist.axes] == [3, 4]
        assert Histogram(h, label="h").rebinned_to(2).hist.axes[1].size == 2
        assert Histogram(h, label="h").rebinned_to([None, None]) is not None
        with pytest.raises(BinningError, match=r"has 12 bins.*\[12, 6, 4, 3, 2, 1\].*not 5"):
            Histogram(h, label="h").rebinned_to([5, None])
        with pytest.raises(BinningError, match="bin edges must be one-dimensional"):
            Histogram(h, label="h").rebinned_to([2.5, None])  # type: ignore[list-item]
        with pytest.raises(BinningError, match="got 1 bin specifications for a 2D"):
            Histogram(h, label="h").rebinned_to([3])

    def test_category_axis_keeps_its_count_only(self) -> None:
        h = hist.Hist(
            hist.axis.StrCategory(["a", "b", "c"], name="cut"), storage=hist.storage.Weight()
        )
        histogram = Histogram(h, label="h")
        assert histogram.rebinned_to(3) is histogram  # the same count is a no-op
        with pytest.raises(BinningError, match="categories"):
            histogram.rebinned_to(1)
        with pytest.raises(BinningError, match="categories of axis 'cut' into bin edges"):
            histogram.rebinned_to([0, 3])

    def test_edges_merge_aligned_bins(self) -> None:
        h = hist.Hist(
            hist.axis.Regular(6, 0, 6, name="x", label="X"), storage=hist.storage.Weight()
        )
        h.fill([0.5, 1.5, 2.5, 5.5], weight=[1.0, 2.0, 3.0, 4.0])
        h.fill([-1.0, 7.0])  # flow bins
        histogram = Histogram(h, label="h", variations={"s": (h * 2.0, None)})
        assert histogram.rebinned_to([0, 1, 2, 3, 4, 5, 6]) is histogram  # its own edges
        uniform = histogram.rebinned_to([0, 2, 4, 6])
        assert isinstance(uniform.axis, hist.axis.Regular)  # a plain rebin keeps the axis type
        np.testing.assert_allclose(uniform.values(), [3.0, 3.0, 4.0])
        merged = histogram.rebinned_to([0, 1, 3, 6])
        assert isinstance(merged.axis, hist.axis.Variable)
        assert (merged.axis.name, merged.axis.label) == ("x", "X")
        np.testing.assert_allclose(merged.edges, [0, 1, 3, 6])
        np.testing.assert_allclose(merged.values(), [1.0, 5.0, 4.0])
        np.testing.assert_allclose(merged.variances(), [1.0, 13.0, 16.0])
        assert (merged.underflow, merged.overflow) == (1.0, 1.0)
        np.testing.assert_allclose(merged.variations["s"][0].values(), [2.0, 10.0, 8.0])
        np.testing.assert_allclose(merged.variations["s"][1].values(), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            histogram.rebinned_to(np.array([0.0, 3.0, 6.0])).values(), [6.0, 4.0]
        )
        cropped = histogram.rebinned_to([1, 3, 6])
        np.testing.assert_allclose(cropped.values(flow=True), [2, 5, 4, 1])
        np.testing.assert_allclose(cropped.variances(flow=True), [2, 13, 16, 1])
        np.testing.assert_allclose(cropped.variations["s"][0].values(), [10, 8])
        cropped = histogram.rebinned_to(None, range=(1, 4))
        assert isinstance(cropped.axis, hist.axis.Regular)
        np.testing.assert_allclose(cropped.edges, [1, 2, 3, 4])
        np.testing.assert_allclose(cropped.values(flow=True), [2, 2, 3, 0, 5])
        merged_crop = histogram.rebinned_to((2, 2, 6))
        assert isinstance(merged_crop.axis, hist.axis.Regular)
        np.testing.assert_allclose(merged_crop.values(flow=True), [4, 3, 4, 1])
        normalised = normalize(histogram.replace(variations={}), "unity")
        assert normalised.rebinned_to(histogram.edges) is normalised
        with pytest.raises(BinningError, match=r"axis 'X'.*crop and rebin before normalising"):
            normalised.rebinned_to(None, range=(1, 4))
        for bad, message in (
            ([0, 2.5, 6], "no bin edge at 2.5"),
            ([0, 3, 7], "no bin edge at 7"),
            ([0, 6, 3], "strictly increasing"),
            ([0, 2.9999999, 3.0000001, 6], "resolve to the same edge 3"),
            ([3, None], "got 2 bin specifications for a 1D"),
        ):
            with pytest.raises(BinningError, match=message):
                histogram.rebinned_to(bad)
        # a uniform merge of a transformed axis keeps the transform
        log = hist.Hist(
            hist.axis.Regular(4, 1, 1e4, name="l", transform=hist.axis.transform.log),
            storage=hist.storage.Weight(),
        )
        kept = Histogram(log, label="l").rebinned_to([1, 100, 1e4]).axis
        assert isinstance(kept, hist.axis.Regular)
        np.testing.assert_allclose(kept.edges, [1, 100, 1e4])
        cropped_log = Histogram(log, label="l").rebinned_to(None, range=(10, 1000)).axis
        assert isinstance(cropped_log, hist.axis.Regular)
        np.testing.assert_allclose(cropped_log.edges, [10, 100, 1000])  # linear would put 505

    @pytest.mark.parametrize(
        ("bins", "window"),
        [
            ((2, 2, 6), None),
            (2, (2, 6)),
            ([2, 4, 6], None),
            (hist.axis.Regular(2, 2, 6), None),
            (hist.axis.Variable([2, 4, 6]), None),
        ],
    )
    def test_crop_specifications_and_statistics(self, bins: Any, window: Any) -> None:
        [original] = build_histograms(
            [Sample({"x": [0.5, 2.5, 5.5, 7.0]})], Variable("x", bins=(6, 0, 6))
        )
        cropped = original.rebinned_to(bins, range=window)
        np.testing.assert_allclose(cropped.values(flow=True), [1, 1, 1, 1])
        assert cropped.stats is original.stats
        assert cropped.entries == 4
        np.testing.assert_allclose(original.values(flow=True), [0, 1, 0, 1, 0, 0, 1, 1])

    def test_invalid_specs(self) -> None:
        h = Histogram(
            hist.Hist(hist.axis.Regular(6, 0, 6, name="x"), storage=hist.storage.Weight()),
            label="h",
        )
        with pytest.raises(BinningError, match="got a bool"):
            h.rebinned_to(True)
        with pytest.raises(BinningError, match="got 2 ranges for a 1D"):
            h.rebinned_to(None, range=[None, None])
        with pytest.raises(BinningError, match=r"range must be \(low, high\).*got \[1, 3\]"):
            h.rebinned_to(None, range=[1, 3])

    def test_one_range_for_every_axis(self) -> None:
        axes = [hist.axis.Regular(4, 0, 4, name=name) for name in "xy"]
        h = Histogram(hist.Hist(*axes, storage=hist.storage.Weight()), label="h")
        for axis in h.rebinned_to(None, range=(1, 3)).hist.axes:
            np.testing.assert_allclose(axis.edges, [1, 2, 3])
        assert h.rebinned_to(None, range="robust") is h
        one = Histogram(hist.Hist(axes[0], storage=hist.storage.Weight()), label="one")
        np.testing.assert_allclose(one.rebinned_to([None], range=((1, 3),)).edges, [1, 2, 3])

    def test_edges_per_axis_and_flowless_axes(self) -> None:
        h = hist.Hist(
            hist.axis.Regular(4, 0, 4, name="x"),
            hist.axis.Regular(6, 0, 6, name="y"),
            storage=hist.storage.Weight(),
        )
        h.fill([0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 4.5, 5.5])
        h.fill([-1.0], [7.0])
        merged = Histogram(h, label="h").rebinned_to([[0, 1, 4], 3])
        assert [(type(a).__name__, a.size) for a in merged.hist.axes] == [
            ("Variable", 2),
            ("Regular", 3),
        ]
        np.testing.assert_allclose(merged.values(), [[1, 0, 0], [1, 0, 2]])
        assert merged.hist.values(flow=True).sum() == h.values(flow=True).sum()
        with pytest.raises(BinningError, match="got 3 bin specifications for a 2D"):
            Histogram(h, label="h").rebinned_to([0, 1, 4])
        no_flow = hist.Hist(
            hist.axis.Regular(4, 0, 4, name="x", flow=False), storage=hist.storage.Weight()
        ).fill([0.5, 3.5])
        merged = Histogram(no_flow, label="n").rebinned_to([0, 1, 4])
        np.testing.assert_allclose(merged.values(), [1.0, 1.0])
        assert not merged.axis.traits.underflow
        assert not merged.axis.traits.overflow
        for window, side in (((1, 4), "underflow"), ((0, 3), "overflow")):
            with pytest.raises(BinningError, match=f"axis 'x'.*{side} bin is missing"):
                Histogram(no_flow, label="n").rebinned_to(None, range=window)
        cropped = Histogram(h, label="h").rebinned_to(None, range=[(1, 3), None])
        np.testing.assert_allclose(cropped.hist.axes[0].edges, [1, 2, 3])
        np.testing.assert_allclose(cropped.values(flow=True).sum(), h.values(flow=True).sum())
        np.testing.assert_allclose(cropped.variances(flow=True).sum(), h.variances(flow=True).sum())


class TestAxisRenaming:
    def test_renames_a_copy_only(self) -> None:
        from rootfig.histograms.stored import _rename_axis

        for axis in (
            hist.axis.Regular(4, 0, 4, name="xaxis", label="X"),
            hist.axis.StrCategory(["a", "b"], name="xaxis", label="Cut"),
        ):
            h = hist.Hist(axis, storage=hist.storage.Weight())
            copy_ = h.copy()
            _rename_axis(copy_.axes[0], "mz")
            assert copy_.axes[0].name == "mz"
            assert copy_.axes[0].label == h.axes[0].label
            assert h.axes[0].name == "xaxis"
            assert type(copy_.axes[0]) is type(axis)


class TestBinwiseAddition:
    def test_add_hists_consumes_an_iterator(self) -> None:
        from rootfig._storage import add_hists, add_into

        def make(label: str, value: float) -> Any:
            h = hist.Hist(hist.axis.Regular(2, 0, 2, label=label), storage=hist.storage.Weight())
            h.fill([0.5], weight=value)
            return h

        parts = [make("a", 1.0), make("b", 2.0), make("c", 3.0)]
        total = add_hists(h for h in parts)  # a generator: nothing is held back
        np.testing.assert_allclose(total.values(), [6.0, 0.0])
        np.testing.assert_allclose(total.variances(), [14.0, 0.0])
        assert total.axes[0].label == "a"  # the first histogram's axes
        np.testing.assert_allclose(parts[0].values(), [1.0, 0.0])  # inputs untouched
        add_into(total, parts[0])
        np.testing.assert_allclose(total.values(), [7.0, 0.0])


class TestBinningTolerance:
    @staticmethod
    def _hist(edges: list[float], value: float) -> Any:
        return hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Weight()).fill([value])

    def test_whole_bin_shift_at_large_coordinates_is_rejected(self) -> None:
        a = self._hist([1e6, 1e6 + 1, 1e6 + 2], 1e6 + 0.5)
        b = self._hist([1e6 + 1, 1e6 + 2, 1e6 + 3], 1e6 + 1.5)
        assert not compatible_binning(a, b)
        with pytest.raises(BinningError):
            compare(a, b)

    def test_tiny_coordinates_and_round_off(self) -> None:
        a = self._hist([1e-9, 2e-9, 3e-9], 1.5e-9)
        b = self._hist([2e-9, 3e-9, 4e-9], 2.5e-9)
        assert not compatible_binning(a, b)
        regular = hist.Hist(hist.axis.Regular(7, 0, 1), storage=hist.storage.Weight())
        variable = hist.Hist(
            hist.axis.Variable(np.linspace(0, 1, 8) * (1 + 1e-13)), storage=hist.storage.Weight()
        )
        assert compatible_binning(regular, variable)


class TestCutflowPolicy:
    def test_nonfinite_weights_follow_the_policy(self) -> None:
        from rootfig.histograms import cutflow

        sample = Sample({"x": np.arange(3.0), "w": np.array([1.0, np.inf, 1.0])})
        with pytest.raises(SelectionError, match="non-finite weight"):
            cutflow(sample, ["x > 0"], weight="w", nonfinite="error")
        with pytest.warns(RootfigWarning, match="1 event"):
            flow = cutflow(sample, ["x > 0"], weight="w")
        np.testing.assert_allclose(flow.yields, [2.0, 1.0])
        assert flow.events.tolist() == [2, 1]

    def test_constant_cuts_and_weights_know_the_event_count(self) -> None:
        from rootfig.histograms import cutflow

        sample = Sample({"x": np.arange(3.0)})
        np.testing.assert_allclose(cutflow(sample, [], weight="2").yields, [6.0])
        np.testing.assert_allclose(cutflow(sample, ["True"], weight="2").yields, [6.0, 6.0])
        assert cutflow(sample, ["x > 0"], weight="2").events.tolist() == [3, 2]


class TestConstantExpressions:
    def test_constant_variable_has_one_entry_per_event(self) -> None:
        sample = Sample({"x": np.arange(3.0)})
        [h] = build_histograms([sample], Variable("1", bins=(1, 0, 2)))
        assert h.values().tolist() == [3.0]
        cols = load_columns(sample, ["1"], weight="2", selection="True")
        assert cols.n_events == 3
        assert cols.sum_weights == 6.0

    def test_custom_source_without_num_entries(self) -> None:
        from rootfig.histograms.pipeline import source_length

        class Custom:
            def branches(self) -> list[str]:
                return ["x"]

            def arrays(self, branches: Any) -> dict[str, ak.Array]:
                return {b: ak.Array([1.0, 2.0]) for b in branches}

            def describe(self) -> str:
                return "custom"

        assert source_length(Custom()) == 2
        [h] = build_histograms([Sample(Custom())], Variable("1", bins=(1, 0, 2)))
        assert h.values().tolist() == [2.0]


def filled(values: list[float], *, edges: tuple[int, float, float] = (4, 0.0, 4.0)) -> Any:
    h = hist.Hist(hist.axis.Regular(*edges), storage=hist.storage.Weight())
    h.fill(values)
    return h


def contents(values: list[float]) -> Any:
    h = hist.Hist(
        hist.axis.Regular(len(values), 0.0, float(len(values))), storage=hist.storage.Weight()
    )
    view = h.view()
    view.value = values
    view.variance = values
    return h


class TestVariations:
    def test_variations_are_read_only_and_updates_are_validated(self) -> None:
        nominal = contents([10.0, 20.0])
        up = contents([12.0, 18.0])
        given = {"shape": (up, None)}
        histogram = Histogram(nominal, "MC", variations=given)
        given.clear()
        assert list(histogram.variations) == ["shape"]
        assert histogram.variations["shape"][0] is up
        np.testing.assert_allclose(histogram.variations["shape"][1].values(), [8, 22])
        with pytest.raises(TypeError):
            histogram.variations["new"] = (up, up)  # type: ignore[index]
        with pytest.raises(TypeError):
            del histogram.variations["shape"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            Histogram(nominal, "MC").variations["new"] = (up, up)  # type: ignore[index]
        changed = histogram.replace(variations={"norm": (nominal * 1.1, None)})
        assert list(changed.variations) == ["norm"]
        assert list(histogram.variations) == ["shape"]
        with pytest.raises(TypeError):
            changed.variations["new"] = (up, up)  # type: ignore[index]
        with pytest.raises(SystematicError, match="binning"):
            histogram.replace(variations={"bad": (contents([1.0]), None)})

    @pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
    def test_variations_support_copy_and_pickle(self, operation: str) -> None:
        original = Histogram(
            contents([10.0, 20.0]), "MC", variations={"shape": (contents([12.0, 18.0]), None)}
        )
        restored = (
            pickle.loads(pickle.dumps(original))
            if operation == "pickle"
            else getattr(copy, operation)(original)
        )
        assert restored.label == "MC"
        np.testing.assert_allclose(restored.values(), original.values())
        np.testing.assert_allclose(restored.variations["shape"][1].values(), [8, 22])
        with pytest.raises(TypeError):
            restored.variations["new"] = (original.hist, original.hist)

    def test_constructor_preserves_positional_metadata_and_replace(self) -> None:
        nominal = contents([1.0])
        sample = Sample({"x": [0.5]})
        stats = summarize(prepare({"x": np.array([0.5])}, ["x"]))
        histogram = Histogram(
            nominal, "MC", sample, stats, False, "red", "step", "unity", {"s": (nominal, None)}
        )
        changed = histogram.replace(label="renamed")
        assert changed.label == "renamed"
        assert changed.sample is sample
        assert changed.stats is stats
        assert not changed.is_data
        assert changed.color == "red"
        assert changed.histtype == "step"
        assert changed.normalization == "unity"
        assert list(changed.variations) == ["s"]

    def test_dataclass_subclass_still_validates_variations(self) -> None:
        @dataclass(frozen=True)
        class TaggedHistogram(Histogram):
            tag: str = "tagged"

        nominal = contents([1.0])
        histogram = TaggedHistogram(nominal, "MC", variations={"s": (nominal, None)})
        assert histogram.tag == "tagged"
        np.testing.assert_allclose(histogram.variations["s"][1].values(), [1])
        with pytest.raises(TypeError):
            histogram.variations["new"] = (nominal, nominal)  # type: ignore[index]
        with pytest.raises(SystematicError, match="binning"):
            TaggedHistogram(nominal, "MC", variations={"s": (contents([1.0, 2.0]), None)})

    def test_dataclass_asdict_can_copy_immutable_mappings(self) -> None:
        sample = Sample({"x": [1.0]}, systematics={"s": {"x": "up"}})
        nominal = contents([1.0])
        original = Histogram(nominal, "MC", sample=sample, variations={"s": (nominal, None)})
        result = asdict(original)
        assert result["sample"]["systematics"]["s"].up == {"x": "up"}
        np.testing.assert_allclose(result["variations"]["s"][1].values(), [1])

    def test_data_histograms_cannot_carry_variations(self) -> None:
        nominal = contents([1.0])
        with pytest.raises(SystematicError, match="observed data"):
            Histogram(nominal, "Data", is_data=True, variations={"s": (nominal, None)})
        simulated = Histogram(nominal, "MC", variations={"s": (nominal, None)})
        with pytest.raises(SystematicError, match="observed data"):
            simulated.replace(is_data=True)
        assert simulated.replace(is_data=True, variations={}).is_data

    def test_down_is_mirrored_and_binning_checked(self) -> None:
        nominal = contents([10.0, 20.0])
        h = Histogram(nominal, label="A", variations={"s": (contents([12.0, 18.0]), None)})
        up, down = h.variations["s"]
        np.testing.assert_allclose(down.values(), [8.0, 22.0])
        np.testing.assert_allclose(down.variances(), [12.0, 18.0])
        with pytest.raises(SystematicError, match="binning"):
            Histogram(nominal, label="A", variations={"s": (contents([1.0, 2.0, 3.0]), None)})
        with pytest.raises(SystematicError, match="pair"):
            Histogram(nominal, label="A", variations={"s": nominal})  # type: ignore[dict-item]

    def test_uncertainty_combines_per_side(self) -> None:
        h = Histogram(
            contents([100.0, 100.0]),
            label="A",
            variations={
                "a": (contents([103.0, 104.0]), contents([96.0, 102.0])),  # bin 2: same sign
                "b": (contents([104.0, 97.0]), contents([97.0, 103.0])),
            },
        )
        u = uncertainty(h)
        np.testing.assert_allclose(u.syst_up, [5.0, 5.0])
        np.testing.assert_allclose(u.syst_down, [5.0, 3.0])
        np.testing.assert_allclose(u.stat_down, [10.0, 10.0])
        np.testing.assert_allclose(u.stat_up, [10.0, 10.0])
        np.testing.assert_allclose(u.total_up, np.hypot(10.0, [5.0, 5.0]))
        np.testing.assert_allclose(u.components["a"][0], [3.0, 4.0])
        assert u.has_systematics
        assert not uncertainty(Histogram(contents([1.0]), label="B")).has_systematics

    def test_uncertainty_needs_1d(self) -> None:
        h2 = hist.Hist(hist.axis.Regular(2, 0, 1), hist.axis.Regular(2, 0, 1))
        with pytest.raises(BinningError, match="one-dimensional"):
            uncertainty(Histogram(h2, label="2D"))

    def test_sum_correlates_by_name(self) -> None:
        a = Histogram(contents([10.0]), label="A", variations={"s": (contents([11.0]), None)})
        b = Histogram(
            contents([20.0]),
            label="B",
            variations={"s": (contents([22.0]), None), "t": (contents([25.0]), None)},
        )
        total = sum_histograms([a, b])
        assert total.label == "Total"
        np.testing.assert_allclose(total.values(), [30.0])
        np.testing.assert_allclose(total.variations["s"][0].values(), [33.0])  # linear
        np.testing.assert_allclose(total.variations["t"][0].values(), [35.0])  # a nominal
        np.testing.assert_allclose(uncertainty(total).syst_up, [np.hypot(3.0, 5.0)])
        with pytest.raises(BinningError, match="different bin edges"):
            sum_histograms([a, Histogram(contents([1.0, 2.0]), label="C")])
        with pytest.raises(BinningError, match="no histograms"):
            sum_histograms([])

    def test_normalize_and_scale_apply_to_variations(self) -> None:
        h = Histogram(
            contents([10.0, 30.0]),
            label="A",
            variations={
                "norm": (contents([11.0, 33.0]), None),
                "shape": (contents([20.0, 20.0]), None),
            },
        )
        normalised = normalize(h, True)
        np.testing.assert_allclose(normalised.variations["norm"][0].values(), [0.25, 0.75])
        np.testing.assert_allclose(normalised.variations["shape"][0].values(), [0.5, 0.5])
        scaled = h.scaled(2.0)
        np.testing.assert_allclose(scaled.variations["norm"][1].values(), [18.0, 54.0])

    def test_ratio_systematics(self) -> None:
        num = Histogram(contents([10.0]), label="N", variations={"s": (contents([12.0]), None)})
        den = Histogram(
            contents([20.0]), label="D", variations={"s": (contents([24.0]), contents([19.0]))}
        )
        split = compare(num, den, uncertainty="numerator")
        assert split.syst_band is not None
        assert split.syst_errors is not None
        np.testing.assert_allclose(split.syst_band, [[1.0 / 20.0], [4.0 / 20.0]])
        np.testing.assert_allclose(split.syst_errors, [[2.0 / 20.0], [2.0 / 20.0]])
        both = compare(num, den)
        assert both.syst_errors is not None
        # one source "s" varies both sides together: up 12 / 24, down (mirrored) 8 / 19
        np.testing.assert_allclose(both.syst_errors[0], [0.5 - 8 / 19])
        np.testing.assert_allclose(both.syst_errors[1], [0.0])
        down, up = both.total_errors()
        np.testing.assert_allclose(up, np.hypot(both.errors[1], both.syst_errors[1]))
        band_down, _ = split.total_band()
        assert split.band is not None
        np.testing.assert_allclose(band_down, np.hypot(split.band[0], 1.0 / 20.0))

        plain = compare(num.hist, den.hist)
        assert plain.syst_errors is None
        assert plain.syst_band is None
        assert plain.band is not None
        np.testing.assert_allclose(plain.total_errors()[0], plain.errors[0])
        np.testing.assert_allclose(plain.total_band()[1], plain.band[1])
        no_num = compare(Histogram(contents([10.0]), label="N"), den, uncertainty="numerator")
        assert no_num.syst_errors is None
        assert no_num.syst_band is not None


class TestSystematicsPipeline:
    @pytest.fixture
    def arrays(self) -> dict[str, Any]:
        rng = np.random.default_rng(7)
        x = rng.uniform(0.0, 10.0, 500)
        return {"x": x, "x_up": x + 0.5, "w": np.ones(500), "w_up": np.full(500, 1.2), "cut": x}

    def test_each_kind(self, arrays: dict[str, Any]) -> None:
        variant = {**arrays, "x": arrays["x"] + 1.0}
        sample = Sample(
            arrays,
            label="MC",
            weight="w",
            selection="x > 2",
            systematics={
                "weight": "w_up",
                "norm": 0.1,
                "jes": {"x": "x_up"},
                "alt": Systematic.samples(variant),
            },
        )
        (h,) = build_histograms([sample], Variable("x", bins=(10, 0, 10)))
        reference = {
            "weight": Sample(arrays, weight="w_up", selection="x > 2"),
            "jes": Sample({**arrays, "x": arrays["x_up"]}, weight="w", selection="x > 2"),
            "alt": Sample(variant, weight="w", selection="x > 2"),
        }
        for name, other in reference.items():
            (expected,) = build_histograms([other], Variable("x", bins=(10, 0, 10)))
            np.testing.assert_allclose(
                h.variations[name][0].values(), expected.values(), err_msg=name
            )
        np.testing.assert_allclose(h.variations["norm"][1].values(), 0.9 * h.values())
        np.testing.assert_allclose(
            h.variations["weight"][1].values(), 2 * h.values() - h.variations["weight"][0].values()
        )

    def test_replace_edge_cases(self) -> None:
        a = np.array([1.5, 2.5, 3.5])
        b = np.array([0.5, 0.5, 2.5])
        columns = {"a": a, "b": b, "Jet.pt": a, "Jet.pt_up": b, "c-d": b}
        systematics = {
            "swap": {"a": "b", "b": "a"},
            "dotted": {"Jet.pt": "Jet.pt_up"},
            "unused": {"nothing": "missing_branch"},
        }
        bins = Variable("a - b + Jet.pt", bins=(8, -4, 4))
        (h,) = build_histograms([Sample(columns, systematics=systematics)], bins)
        (swapped,) = build_histograms(
            [Sample(columns)], Variable("b - a + Jet.pt", bins=(8, -4, 4))
        )
        np.testing.assert_allclose(h.variations["swap"][0].values(), swapped.values())
        (dotted,) = build_histograms([Sample(columns)], Variable("a - b + `c-d`", bins=(8, -4, 4)))
        np.testing.assert_allclose(h.variations["dotted"][0].values(), dotted.values())
        np.testing.assert_allclose(h.variations["unused"][0].values(), h.values())

    def test_replacing_branch_must_exist(self, arrays: dict[str, Any]) -> None:
        sample = Sample(arrays, systematics={"s": {"x": "x_typo"}})
        with pytest.raises(MissingBranchError, match="x_typo"):
            build_histograms([sample], "x")

    def test_plot_level_merge_and_data(self, arrays: dict[str, Any]) -> None:
        mc = Sample(arrays, label="MC", systematics={"lumi": 0.5})
        data = Sample(arrays, label="Data", is_data=True)
        hists = build_histograms([mc, data], "x", systematics={"lumi": 0.1, "xsec": 0.2})
        assert set(hists[0].variations) == {"lumi", "xsec"}
        np.testing.assert_allclose(hists[0].variations["lumi"][0].values(), 1.5 * hists[0].values())
        assert hists[1].variations == {}
        with pytest.raises(SystematicError, match="observed data"):
            build_histograms([data.replace(systematics={"s": 0.1})], "x")

    def test_binning_from_nominal_and_single_read(self, arrays: dict[str, Any]) -> None:
        sample = Sample(
            {**arrays, "x_far": arrays["x"] + 100.0},
            label="MC",
            weight="w",
            systematics={"w": ("w_up", "w"), "far": {"x": "x_far"}},
        )
        calls: list[list[str]] = []
        original = sample.source.arrays

        def spy(names: Any) -> Any:
            calls.append(list(names))
            return original(names)

        object.__setattr__(sample.source, "arrays", spy)
        (h,) = build_histograms([sample], Variable("x", bins=10, range="auto"))
        assert len(calls) == 1
        assert set(calls[0]) >= {"x", "x_far", "w", "w_up"}
        assert h.edges[-1] < 11.0
        assert h.variations["far"][0].values().sum() == 0.0  # all in the overflow

    def test_nonfinite_warnings_identify_each_variation(self, arrays: dict[str, Any]) -> None:
        values = arrays["x"].copy()
        values[0] = np.nan
        shifted = values + 1.0
        shifted[1] = np.inf
        sample = Sample(
            {"x": values, "x_up": shifted, "w": arrays["w"], "w_up": arrays["w_up"]},
            label="MC",
            systematics={"w": "w_up", "jes": {"x": "x_up"}},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_histograms([sample], Variable("x", bins=(10, 0, 11)))
        messages = [str(w.message) for w in caught if issubclass(w.category, RootfigWarning)]
        assert len(messages) == 3
        assert "w up" in messages[1]
        assert "jes up" in messages[2]

    def test_invalid_varied_data(self, arrays: dict[str, Any]) -> None:
        sample = Sample(arrays, label="MC", systematics={"alt": Systematic.samples(42)})
        with pytest.raises(SystematicError, match=r"MC \[alt up\]"):
            build_histograms([sample], "x")


class TestNuisanceIdentity:
    """A source's name is its nuisance: one name is one fully correlated source, two names
    are independent sources, in a sum as in a comparison."""

    @staticmethod
    def _pair(first: str, second: str) -> tuple[Histogram, Histogram]:
        a = Histogram(
            contents([100.0]), label="A", variations={first: (contents([110.0]), contents([90.0]))}
        )
        b = Histogram(
            contents([200.0]),
            label="B",
            variations={second: (contents([220.0]), contents([180.0]))},
        )
        return a, b

    def test_one_name_is_one_nuisance(self) -> None:
        a, b = self._pair("scale", "scale")
        total = uncertainty(sum_histograms([a, b]))
        assert list(total.components) == ["scale"]
        np.testing.assert_allclose(total.syst_up, [30.0])  # 10 + 20, linearly
        ratio = compare(a, b)
        assert ratio.syst_errors is not None
        np.testing.assert_allclose(ratio.syst_errors, [[0.0], [0.0]], atol=1e-12)  # cancels

    def test_two_names_are_independent(self) -> None:
        a, b = self._pair("scale_a", "scale_b")
        total = uncertainty(sum_histograms([a, b]))
        assert list(total.components) == ["scale_a", "scale_b"]
        np.testing.assert_allclose(total.syst_up, [np.hypot(10.0, 20.0)])
        ratio = compare(a, b)
        assert ratio.syst_errors is not None
        # each source varies its own side against the other's nominal contents
        np.testing.assert_allclose(ratio.syst_errors[1], [np.hypot(0.05, 100 / 180 - 0.5)])
        np.testing.assert_allclose(ratio.syst_errors[0], [np.hypot(0.05, 0.5 - 100 / 220)])

    def test_a_plot_level_source_is_shared_by_every_sample(self) -> None:
        x = Variable("x", bins=(1, 0.0, 1.0))
        a = Sample({"x": np.full(100, 0.5)}, label="A")
        b = Sample({"x": np.full(300, 0.5)}, label="B")
        shared = sum_histograms(build_histograms([a, b], x, systematics={"xsec": 0.1}))
        np.testing.assert_allclose(uncertainty(shared).syst_up, [40.0])  # 10 % of 400
        # independent cross sections per process: one name per sample
        own = sum_histograms(
            build_histograms(
                [a.replace(systematics={"xsec_a": 0.1}), b.replace(systematics={"xsec_b": 0.1})],
                x,
            )
        )
        np.testing.assert_allclose(uncertainty(own).syst_up, [np.hypot(10.0, 30.0)])


class TestSystematicsRegressions:
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_variation_rejects_shifted_edges_at_large_coordinates(self, ndim: int) -> None:
        axes = [hist.axis.Regular(2, 1e9, 1e9 + 2, name=f"x{i}") for i in range(ndim)]
        nominal = hist.Hist(*axes, storage=hist.storage.Weight())
        shifted = hist.Hist(
            hist.axis.Regular(2, 1e9 + 1, 1e9 + 3), *axes[1:], storage=hist.storage.Weight()
        )
        with pytest.raises(SystematicError, match="binning"):
            Histogram(nominal, label="MC", variations={"s": (shifted, None)})

    @pytest.mark.parametrize("name", ["", " ", 1])
    def test_variation_names_are_validated(self, name: Any) -> None:
        nominal = contents([1.0])
        with pytest.raises(SystematicError, match="non-empty strings"):
            Histogram(nominal, label="MC", variations={name: (nominal, None)})

    def test_sum_allows_axis_metadata_and_type_differences(self) -> None:
        a = Histogram(contents([10.0, 20.0]), label="A")
        other = hist.Hist(
            hist.axis.Variable([0, 1, 2], name="other", label="Other axis"),
            storage=hist.storage.Weight(),
        )
        other.view().value = [3.0, 4.0]
        other.view().variance = [5.0, 6.0]
        other.view(flow=True).value[0] = 2.0
        b = Histogram(other, label="B", variations={"s": (other * 2, None)})
        total = sum_histograms([a, b])
        np.testing.assert_allclose(total.values(), [13, 24])
        np.testing.assert_allclose(total.variances(), [15, 26])
        np.testing.assert_allclose(total.variations["s"][0].values(), [16, 28])
        assert total.underflow == 2
        assert total.variations["s"][0].values(flow=True)[0] == 4
        np.testing.assert_allclose(a.values(), [10, 20])
        assert total.axis.name == a.axis.name

    def test_sum_rejects_flow_mismatch_and_single_2d(self) -> None:
        a = Histogram(contents([1.0]), label="A")
        b = Histogram(hist.Hist(hist.axis.Regular(1, 0, 1, underflow=False)), label="B")
        with pytest.raises(BinningError, match="flow-bin traits"):
            sum_histograms([a, b])
        h2 = Histogram(
            hist.Hist(hist.axis.Regular(1, 0, 1), hist.axis.Regular(1, 0, 1)), label="2D"
        )
        with pytest.raises(BinningError, match="one-dimensional"):
            sum_histograms([h2])

    @pytest.mark.parametrize("spec", [True, "density", 100.0])
    def test_normalization_rejects_zero_total_variations(self, spec: Any) -> None:
        h = Histogram(
            contents([10.0, 20.0]), label="MC", variations={"zero": (contents([0.0, 0.0]), None)}
        )
        with (
            pytest.warns(RootfigWarning, match="normalisation skipped"),
            pytest.raises(SystematicError, match="'zero' up cannot be normalised"),
        ):
            normalize(h, spec)

    def test_failed_nominal_normalization_keeps_variations_raw(self) -> None:
        h = Histogram(
            contents([0.0, 0.0]), label="MC", variations={"s": (contents([2.0, 4.0]), None)}
        )
        with pytest.warns(RootfigWarning, match="normalisation skipped"):
            result = normalize(h, True)
        assert result.normalization is None
        np.testing.assert_allclose(result.variations["s"][0].values(), [2, 4])
        np.testing.assert_allclose(result.variations["s"][1].values(), [-2, -4])

    @pytest.mark.parametrize("n", [-10.0, 10.0])
    @pytest.mark.parametrize("d", [-20.0, 20.0])
    def test_signed_asymmetric_ratio(self, n: float, d: float) -> None:
        def make(value: float) -> Any:
            h = contents([abs(value)])
            h.view().value = [value]
            return h

        num = Histogram(make(n), label="N", variations={"n": (make(n + 3), make(n - 1))})
        den = Histogram(make(d), label="D", variations={"d": (make(d + 4), make(d - 2))})
        low_n, high_n = (1, 3) if d > 0 else (3, 1)
        split = compare(num, den, uncertainty="numerator")
        np.testing.assert_allclose(split.syst_errors, [[low_n / abs(d)], [high_n / abs(d)]])
        band_low, band_high = (2, 4) if d > 0 else (4, 2)
        np.testing.assert_allclose(split.syst_band, [[band_low / abs(d)], [band_high / abs(d)]])
        both = compare(num, den)
        shifts = [(3 / d, -1 / d), (n / (d + 4) - n / d, n / (d - 2) - n / d)]
        low = np.sqrt(sum(min(up, down, 0.0) ** 2 for up, down in shifts))
        high = np.sqrt(sum(max(up, down, 0.0) ** 2 for up, down in shifts))
        np.testing.assert_allclose(both.syst_errors, [[low], [high]])

    def test_ratio_cancels_shared_sources(self) -> None:
        a = Histogram(contents([100.0]), label="A", variations={"lumi": (contents([110.0]), None)})
        b = Histogram(
            contents([200.0]),
            label="B",
            variations={"lumi": (contents([220.0]), None), "xsec": (contents([240.0]), None)},
        )
        for mode in ("propagate", "numerator"):
            result = compare(a, b, uncertainty=mode)  # type: ignore[arg-type]
            assert result.syst_errors is not None
            if mode == "propagate":  # lumi cancels, only b's xsec is left: 100/240 and 100/160
                np.testing.assert_allclose(
                    result.syst_errors, [[0.5 - 100 / 240], [100 / 160 - 0.5]]
                )
            else:  # a's own lumi against b's nominal; b's sources are the band
                np.testing.assert_allclose(result.syst_errors, [[0.05], [0.05]])
        assert result.syst_band is not None
        np.testing.assert_allclose(result.syst_band, [[np.hypot(0.1, 0.2)], [np.hypot(0.1, 0.2)]])

    def test_ratio_warns_when_a_variation_empties_the_denominator(self) -> None:
        num = Histogram(contents([10.0, 10.0]), label="N")
        den = Histogram(
            contents([20.0, 20.0]), label="D", variations={"shape": (contents([20.0, 0.0]), None)}
        )
        with pytest.warns(RootfigWarning, match="'shape' up empties the reference in 1 bin"):
            result = compare(num, den)
        assert result.syst_errors is not None
        assert np.isfinite(result.syst_errors[0][0])
        assert np.isnan(result.syst_errors[0][1])

    def test_sum_keeps_only_a_shared_normalization(self) -> None:
        raw = Histogram(contents([1.0]), label="raw")
        unity = Histogram(contents([1.0]), label="unity", normalization="Normalised to unity")
        assert sum_histograms([unity, unity]).normalization == "Normalised to unity"
        assert sum_histograms([unity, raw]).normalization is None
        assert sum_histograms([raw, unity]).normalization is None

    def test_different_nonfinite_entries_with_equal_counts_are_reported(self) -> None:
        sample = Sample({"x": [np.nan, 1.0], "up": [1.0, np.nan]}, systematics={"s": {"x": "up"}})
        with pytest.warns(RootfigWarning) as caught:
            build_histograms([sample], Variable("x", bins=(2, 0, 2)))
        assert len(caught) == 2
        assert "[s up]" in str(caught[1].message)

    def test_variation_respects_warning_error_filter(self) -> None:
        sample = Sample({"x": [1.0], "up": [np.nan]}, systematics={"s": {"x": "up"}})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RootfigWarning)
            with pytest.raises(RootfigWarning, match=r"\[s up\]"):
                build_histograms([sample], Variable("x", bins=(2, 0, 2)))

    @pytest.mark.parametrize("as_record", [False, True])
    def test_varied_arrays_inherit_entry_range(self, as_record: bool) -> None:
        varied: Any = {"x": [0.5, 1.5, 2.5, 3.5]}
        if as_record:
            varied = ak.Array(varied)
        sample = Sample(
            {"x": [0.2, 1.2, 2.2, 3.2]},
            entry_start=1,
            entry_stop=3,
            systematics={"s": Systematic.samples(varied)},
        )
        (h,) = build_histograms([sample], Variable("x", bins=(4, 0, 4)))
        np.testing.assert_allclose(h.values(), [0, 1, 1, 0])
        np.testing.assert_allclose(h.variations["s"][0].values(), [0, 1, 1, 0])

    def test_replacement_cannot_use_an_expression_constant_as_a_branch(self) -> None:
        sample = Sample({"x": [1.0]}, systematics={"s": {"x": "pi"}})
        with pytest.raises(MissingBranchError, match="pi"):
            build_histograms([sample], Variable("x", bins=(2, 0, 2)))
        # A name resolving to a mathematical constant is not a branch to replace.
        sample = sample.replace(systematics={"s": {"pi": "missing"}})
        (h,) = build_histograms([sample], Variable("pi", bins=(4, 0, 4)))
        np.testing.assert_allclose(h.variations["s"][0].values(), h.values())

    def test_jagged_variations_keep_plot_weight_and_luminosity(self) -> None:
        arrays = {
            "x": ak.Array([[0.5, 1.5], [2.5]]),
            "shifted": ak.Array([[1.5, 2.5], [3.5]]),
            "w": [2.0, 3.0],
            "up": [4.0, 5.0],
            "global_w": [2.0, 3.0],
        }
        sample = Sample(
            arrays,
            weight="w",
            scale=2,
            xsec=0.001,
            ngen=2,
            selection="x > 1",
            systematics={"weight": "up", "shape": {"x": "shifted"}},
        )
        (h,) = build_histograms([sample], Variable("x", bins=(4, 0, 4)), weight="global_w", lumi=2)
        np.testing.assert_allclose(h.values(), [0, 8, 18, 0])
        np.testing.assert_allclose(h.variations["weight"][0].values(), [0, 16, 30, 0])
        np.testing.assert_allclose(h.variations["weight"][0].variances(), [0, 256, 900, 0])
        np.testing.assert_allclose(h.variations["shape"][0].values(), [0, 8, 8, 18])

    def test_alternate_sample_uses_its_own_configuration(self) -> None:
        alternative = Sample(
            {"x": [0.5, 1.5], "alt_weight": [3.0, 4.0]},
            weight="alt_weight",
            scale=2,
            selection="x > 1",
        )
        nominal = Sample(
            {"x": [0.5, 1.5], "w": [1.0, 1.0]},
            weight="w",
            scale=10,
            selection="x < 1",
            systematics={"s": Systematic.samples(alternative)},
        )
        (h,) = build_histograms([nominal], Variable("x", bins=(2, 0, 2)), weight="2")
        np.testing.assert_allclose(h.values(), [20, 0])
        np.testing.assert_allclose(h.variations["s"][0].values(), [0, 16])


class TestRebinned:
    def test_merges_bins_and_variations(self) -> None:
        h = hist.Hist(hist.axis.Regular(6, 0, 6, name="x"), storage=hist.storage.Weight())
        h.fill([0.5, 1.5, 2.5, 5.5], weight=[1.0, 2.0, 3.0, 4.0])
        h.fill([-1.0, 7.0])  # flow bins
        histogram = Histogram(h, label="h", variations={"s": (h * 2.0, None)})
        merged = histogram.rebinned(2)
        np.testing.assert_allclose(merged.values(), [3.0, 3.0, 4.0])
        np.testing.assert_allclose(merged.variances(), [5.0, 9.0, 16.0])
        assert merged.underflow == 1.0
        assert merged.overflow == 1.0
        up, down = merged.variations["s"]
        np.testing.assert_allclose(up.values(), [6.0, 6.0, 8.0])
        np.testing.assert_allclose(down.values(), [0.0, 0.0, 0.0])
        assert merged.label == "h"

    def test_two_dimensional_factors(self) -> None:
        h = hist.Hist(
            hist.axis.Regular(4, 0, 4), hist.axis.Regular(6, 0, 6), storage=hist.storage.Weight()
        )
        merged = Histogram(h, label="h").rebinned((2, 3))
        assert [a.size for a in merged.hist.axes] == [2, 2]
        with pytest.raises(BinningError, match="got 1 rebin factors for a 2D"):
            Histogram(h, label="h").rebinned([2])

    def test_indivisible(self) -> None:
        h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"), storage=hist.storage.Weight())
        with pytest.raises(BinningError, match=r"grouped by \[1, 2, 5, 10\]"):
            Histogram(h, label="h").rebinned(3)

    def test_factors_must_be_positive_integers(self) -> None:
        h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"), storage=hist.storage.Weight())
        histogram = Histogram(h, label="h")
        for bad in (2.9, [2.9], True, 0, -2):
            with pytest.raises(BinningError, match="positive integers"):
                histogram.rebinned(bad)  # type: ignore[arg-type]
        assert histogram.rebinned(np.int64(5)).axis.size == 2

    def test_normalised_histograms_are_refused(self) -> None:
        h = hist.Hist(hist.axis.Regular(4, 0, 4, name="x"), storage=hist.storage.Weight())
        h.fill([0.5, 1.5, 2.5, 3.5])
        density = normalize(Histogram(h, label="h"), "density")
        with pytest.raises(BinningError, match=r"normalised .*rebin before normalising"):
            density.rebinned(2)
        # asking for the bins it already has is not a merge, so the Variable a histogram was
        # filled with still describes it after normalising; any real merge is refused
        assert density.rebinned_to(density.axis.size) is density
        assert density.rebinned_to(density.edges) is density
        assert density.rebinned_to(None) is density
        for merge in (2, [0, 2, 4], [0, 1, 4]):
            with pytest.raises(BinningError, match="rebin before normalising"):
                density.rebinned_to(merge)
        # rebinning first, then normalising, keeps the area at one
        merged = normalize(Histogram(h, label="h").rebinned(2), "density")
        assert (merged.values() * merged.widths).sum() == pytest.approx(1.0)

    def test_category_axes_are_refused(self) -> None:
        h = hist.Hist(
            hist.axis.StrCategory(["a", "b"], name="cut", label="Cut"),
            storage=hist.storage.Weight(),
        )
        with pytest.raises(BinningError, match="categories of axis 'Cut'"):
            Histogram(h, label="h").rebinned(2)
        assert Histogram(h, label="h").rebinned(1).values().tolist() == [0.0, 0.0]  # a no-op


class TestStoredHistograms:
    @staticmethod
    def _zh(stored_dir: Path, **kwargs: Any) -> Sample:
        return Sample(stored_dir / "ZH_sel0_histo.root", label="ZH", **kwargs)

    def test_stored_mode_rule(self, stored_dir: Path) -> None:
        from rootfig.histograms import stored_mode

        zh = self._zh(stored_dir)
        assert stored_mode([zh], [Variable("mz")])
        assert not stored_mode([zh], [Variable("mz * 2")])  # an expression fills
        assert not stored_mode([zh], [Variable("nope")])
        # explicit intent wins: a tree name or an entry range means a branch
        assert not stored_mode(
            [Sample(stored_dir / "ZH_sel0_histo.root", tree="events")], [Variable("mz")]
        )
        assert not stored_mode(
            [Sample(stored_dir / "ZH_sel0_histo.root", entry_stop=5)], [Variable("mz")]
        )
        # a branch of the same name wins over the histogram
        assert not stored_mode([Sample(stored_dir / "branch_and_histogram.root")], [Variable("mz")])
        # a tree without that branch does not
        assert stored_mode([Sample(stored_dir / "tree_without_branch.root")], [Variable("mz")])
        # histograms inside directories are not matched by a bare name
        assert not stored_mode([Sample(stored_dir / "in_directory.root")], [Variable("mz")])
        with pytest.raises(SourceError, match="several trees"):
            stored_mode([Sample(stored_dir / "two_trees.root")], [Variable("mz")])
        with pytest.raises(
            SourceError,
            match=r"stored in the files of \['ZH'\], but not for every sample \('b': tree "
            r"'events' has a branch of that name, which wins\)",
        ):
            stored_mode(
                [zh, Sample(stored_dir / "branch_and_histogram.root", label="b")], [Variable("mz")]
            )
        # equally labelled samples are still judged one by one, in either order
        twin = Sample(stored_dir / "branch_and_histogram.root", label="ZH")
        for order in ([zh, twin], [twin, zh]):
            with pytest.raises(
                SourceError, match=r"\['ZH'\], but not for every sample \('ZH': tree"
            ):
                stored_mode(order, [Variable("mz")])
        # the diagnostic says why each other sample is not read that way
        for other, reason in (
            (Sample({"mz": [1.0, 2.0]}, label="m"), "'m': in-memory arrays .* hold no stored"),
            (
                Sample(stored_dir / "ZH_sel0_histo.root", tree="events", label="t"),
                "'t': tree='events' addresses a branch",
            ),
            (
                Sample(stored_dir / "ZH_sel0_histo.root", entry_stop=5, label="e"),
                "'e': an entry range addresses a tree",
            ),
            (
                Sample(stored_dir / "in_directory.root", label="d"),
                r"'d': '.*in_directory\.root' holds no histogram of that name",
            ),
        ):
            with pytest.raises(SourceError, match=reason):
                stored_mode([zh, other], [Variable("mz")])
        assert not stored_mode([Sample({"mz": [1.0, 2.0]})], [Variable("mz")])
        # a histogram inside a directory is named by its path in backticks
        assert stored_mode([Sample(stored_dir / "in_directory.root")], [Variable("`sub/mz`")])
        (nested,) = build_histograms([Sample(stored_dir / "in_directory.root")], "`sub/mz`")
        assert nested.sum_weights == 500

    def test_read_scaled_and_labelled(self, stored_dir: Path) -> None:
        zh = self._zh(stored_dir, scale=2.0, color="C3")
        vv = Sample(
            [stored_dir / "WW_sel0_histo.root", stored_dir / "ZZ_sel0_histo.root"], label="VV"
        )
        h_zh, h_vv = build_histograms([zh, vv], "mz")
        assert h_zh.label == "ZH"
        assert h_zh.color == "C3"
        assert h_zh.sample is zh
        assert h_zh.stats is None
        assert h_zh.axis.name == "mz"
        assert h_zh.axis.label == "m_{Z} [GeV]"
        assert h_zh.sum_weights == pytest.approx(2.0 * 0.5 * 4000)
        assert h_vv.sum_weights == pytest.approx(0.5 * 3000)
        # variances scale with the square of the factor
        assert h_zh.variances(flow=True).sum() == pytest.approx(4.0 * 0.25 * 4000)
        # the stored title is kept unless the variable has a label; a unit is appended once
        (h,) = build_histograms([zh], Variable("mz", label="$m_Z$", unit="GeV"))
        assert h.axis.label == "$m_Z$ [GeV]"
        (h,) = build_histograms([zh], Variable("mz", unit="GeV"))
        assert h.axis.label == "m_{Z} [GeV]"
        (h,) = build_histograms([zh], Variable("mz", unit="MeV"))
        assert h.axis.label == "m_{Z} [GeV] [MeV]"
        # ROOT's placeholder title gives way to the variable
        (h,) = build_histograms([Sample(stored_dir / "other_binning.root")], "mz")
        assert h.axis.label == "mz"

    def test_luminosity_and_plain_storage(self, stored_dir: Path) -> None:
        zh = self._zh(stored_dir, xsec=2.0, ngen="eventsProcessed")  # 2 pb, 4000 events
        (h,) = build_histograms(
            [zh], "mz", lumi=1.0
        )  # 1 fb^-1 -> 2000 pb^-1... in pb: 2 * 1000 / 4000
        assert h.sum_weights == pytest.approx(0.5 * 4000 * 2.0 * 1000.0 / 4000)
        (plain,) = build_histograms([self._zh(stored_dir)], "mz_raw")
        assert plain.hist.storage_type is hist.storage.Weight
        np.testing.assert_allclose(plain.variances(flow=True), plain.values(flow=True))

    def test_rebin_by_bins(self, stored_dir: Path) -> None:
        zh = self._zh(stored_dir)
        (h,) = build_histograms([zh], Variable("mz", bins=25))
        assert h.axis.size == 25
        assert h.sum_weights == pytest.approx(0.5 * 4000)
        (h,) = build_histograms([zh], Variable("mz"))  # no preference keeps the stored binning
        assert h.axis.size == 100
        with pytest.raises(BinningError, match=r"has 100 bins.*not 30"):
            build_histograms([zh], Variable("mz", bins=30))
        # edges that coincide with the stored ones merge the bins between them
        (h,) = build_histograms([zh], Variable("mz", bins=(50, 0, 250)))
        assert h.axis.size == 50
        (h,) = build_histograms([zh], Variable("mz", bins=10, range=(0, 250)))
        assert h.axis.size == 10
        (h,) = build_histograms([zh], Variable("mz", bins=[0, 100, 150, 250]))
        assert isinstance(h.axis, hist.axis.Variable)
        np.testing.assert_allclose(h.edges, [0, 100, 150, 250])
        assert h.sum_weights == pytest.approx(0.5 * 4000)
        (h,) = build_histograms([zh], Variable("mz", bins=(10, 0, 100)))
        assert h.axis.size == 10
        assert h.edges[[0, -1]].tolist() == [0, 100]
        assert h.sum_weights == pytest.approx(0.5 * 4000)
        with pytest.raises(BinningError, match="no bin edge at 101"):
            build_histograms([zh], Variable("mz", bins=[0, 101, 250]))
        (h,) = build_histograms([zh], Variable("mz", range=(0, 100)))
        assert h.axis.size == 40
        assert h.sum_weights == pytest.approx(0.5 * 4000)

    def test_event_options_are_refused(self, stored_dir: Path) -> None:
        zh = self._zh(stored_dir)
        with pytest.raises(SelectionError, match="selection cannot be applied"):
            build_histograms([zh], "mz", selection="mz > 1")
        with pytest.raises(SelectionError, match="weight cannot be applied"):
            build_histograms([zh], "mz", weight="w")
        with pytest.raises(SelectionError, match="nonfinite= applies only"):
            build_histograms([zh], "mz", nonfinite="error")
        with pytest.raises(SelectionError, match="sample's selection"):
            build_histograms([self._zh(stored_dir, selection="x > 1")], "mz")
        with pytest.raises(SelectionError, match="sample's weight"):
            build_histograms([self._zh(stored_dir, weight="w")], "mz")
        with pytest.raises(SourceError, match="2D histogram; draw it with plot2d"):
            build_histograms([zh], "mz_recoil_2D")
        with pytest.raises(SourceError, match="1D histogram; draw it with plot"):
            build_histograms_2d([zh], "mz")

    def test_systematics(self, stored_dir: Path) -> None:
        alternative = stored_dir / "WW_sel0_histo.root"
        zh = self._zh(
            stored_dir,
            systematics={"norm": 0.1, "model": Systematic.samples(alternative)},
        )
        (h,) = build_histograms([zh], "mz", systematics={"lumi": (1.02, 0.97)})
        assert set(h.variations) == {"norm", "model", "lumi"}
        up, down = h.variations["norm"]
        assert up.values().sum() == pytest.approx(1.1 * h.integral)
        assert down.values().sum() == pytest.approx(0.9 * h.integral)
        assert h.variations["lumi"][1].values().sum() == pytest.approx(0.97 * h.integral)
        assert h.variations["model"][0].values(flow=True).sum() == pytest.approx(0.5 * 2000)
        with pytest.raises(SystematicError, match="varies the weight"):
            build_histograms([self._zh(stored_dir, systematics={"w": "w_up"})], "mz")
        with pytest.raises(SystematicError, match="varies the branches"):
            build_histograms([self._zh(stored_dir, systematics={"r": {"mz": ("a", "b")}})], "mz")
        with pytest.raises(SystematicError, match="other ROOT files"):
            build_histograms(
                [self._zh(stored_dir, systematics={"s": Systematic.samples({"mz": [1.0]})})], "mz"
            )
        # rebinning applies to the variations too
        (h,) = build_histograms([zh], Variable("mz", bins=50))
        assert h.variations["model"][0].axes[0].size == 50

    def test_variation_samples_are_checked_like_the_nominal(self, stored_dir: Path) -> None:
        ww = stored_dir / "WW_sel0_histo.root"
        for bad, message in (
            (Sample(ww, selection="x > 100"), "selection cannot be applied"),
            (Sample(ww, weight="100"), "weight cannot be applied"),
            (Sample(ww, entry_stop=0), "entry range"),
            (Sample(ww, tree="nope"), "tree='nope' addresses a branch"),
            (f"{ww}:events", "tree='events' addresses a branch"),
            ({"mz": [1.0]}, "other ROOT files"),
        ):
            zh = self._zh(stored_dir, systematics={"model": Systematic.samples(bad)})
            with pytest.raises(SystematicError, match=message) as info:
                build_histograms([zh], "mz")
            assert "[model up]" in str(info.value) or any(
                "[model up]" in note for note in getattr(info.value, "__notes__", [])
            )
        # a file lacking the histogram names itself
        zh = self._zh(
            stored_dir, systematics={"model": Systematic.samples(stored_dir / "in_directory.root")}
        )
        with pytest.raises(SourceError, match=r"in_directory\.root.*histograms present"):
            build_histograms([zh], "mz")

    def test_category_axis_is_kept(self, stored_dir: Path) -> None:
        (h,) = build_histograms([self._zh(stored_dir, scale=2.0)], "cutflow")
        assert type(h.axis).__name__ == "StrCategory"
        assert list(h.axis) == ["all", "sel0", "sel1"]
        assert h.axis.name == "cutflow"
        assert h.axis.label == "Selection"
        np.testing.assert_allclose(h.values(), [4000.0, 2000.0, 1000.0])
        with pytest.raises(BinningError, match="categories of axis"):
            build_histograms([self._zh(stored_dir)], Variable("cutflow", bins=1))

    def test_read_stored_needs_a_bare_name(self, stored_dir: Path) -> None:
        from rootfig.histograms import read_stored

        with pytest.raises(SourceError, match="bare name of a stored histogram"):
            read_stored([self._zh(stored_dir)], [Variable("mz * 2")])

    def test_two_dimensional(self, stored_dir: Path) -> None:
        zh = self._zh(stored_dir)
        (h,) = build_histograms_2d([zh], "mz_recoil_2D")
        assert h.ndim == 2
        assert [a.name for a in h.hist.axes] == ["mz_recoil_2D", "mz_recoil_2D_y"]
        assert [a.label for a in h.hist.axes] == ["m_{Z} [GeV]", "recoil [GeV]"]
        (h,) = build_histograms_2d(
            [zh], Variable("mz_recoil_2D", bins=5), Variable("mz_recoil_2D", bins=4)
        )
        assert [a.size for a in h.hist.axes] == [5, 4]
        # distinct variable names name the axes as a tree fill would, without a suffix
        (h,) = build_histograms_2d(
            [zh], Variable("mz_recoil_2D", name="mass"), Variable("mz_recoil_2D", name="recoil")
        )
        assert [a.name for a in h.hist.axes] == ["mass", "recoil"]
        # 2D plots draw no variations, so systematics are ignored as for trees: a weight
        # systematic that a 1D read would refuse is not even looked at, and the sample stays
        varied = self._zh(stored_dir, systematics={"w": ("w_up", "w_down"), "lumi": 0.02})
        (h,) = build_histograms_2d([varied], "mz_recoil_2D")
        assert h.variations == {}
        assert h.sample is varied
        with pytest.raises(SourceError, match="needs two variables"):
            build_histograms_2d([zh], "mz * 2")

    def test_unsupported_objects_are_named(self, stored_dir: Path) -> None:
        from rootfig.histograms import stored_mode

        # an object of that name exists but is no TH1/TH2: say so, instead of failing later
        # on the missing tree or branch
        alone = Sample(stored_dir / "unsupported.root")
        with pytest.raises(
            SourceError, match=r"'prof' .* is a TProfile, which rootfig cannot plot"
        ):
            build_histograms([alone], "prof")
        with pytest.raises(SourceError, match=r"is a TH3D, .*read as TH1 and TH2 only"):
            build_histograms_2d([alone], "h3")
        beside_tree = Sample(stored_dir / "unsupported_with_tree.root")
        with pytest.raises(SourceError, match=r"'h3' .* is a TH3D"):
            build_histograms([beside_tree], "h3")
        # a branch of the same name wins over any stored object
        assert not stored_mode([beside_tree], [Variable("prof")])
        (h,) = build_histograms([beside_tree], Variable("prof", bins=(3, -0.5, 2.5)))
        np.testing.assert_allclose(h.values(), [1.0, 1.0, 1.0])
        # with several trees the tree lookup reports the ambiguity, as for any branch
        assert not stored_mode([Sample(stored_dir / "two_trees.root")], [Variable("prof")])
        with pytest.raises(SourceError, match="several trees"):
            build_histograms([Sample(stored_dir / "two_trees.root")], "prof")
        # a name no object carries is left to the tree lookup as well
        with pytest.raises(SourceError, match="no TTree or RNTuple"):
            build_histograms([alone], "missing")

    def test_two_dimensional_placeholder_titles(self, stored_dir: Path) -> None:
        # the object name labels the x axis, not the y axis it merely names: that one has
        # no label of its own, which hist presents as the axis name
        untitled = Sample(stored_dir / "untitled_2D.root")
        for name in ("mz_recoil_2D", "hist_2D"):  # ROOT's empty titles, uproot's "Axis 0"
            (h,) = build_histograms_2d([untitled], name)
            assert [a.label for a in h.hist.axes] == [name, f"{name}_y"]
            assert h.hist.axes[1]._raw_metadata["label"] == ""
        (h,) = build_histograms_2d([untitled], Variable("mz_recoil_2D", label="Mass", unit="GeV"))
        assert [a.label for a in h.hist.axes] == ["Mass [GeV]", "mz_recoil_2D_y"]
        (h,) = build_histograms_2d(
            [untitled],
            Variable("mz_recoil_2D", label="Mass"),
            Variable("mz_recoil_2D", label="Recoil", unit="GeV"),
        )
        assert [a.label for a in h.hist.axes] == ["Mass", "Recoil [GeV]"]


class TestStoredNames:
    """``stored_names``: the stored half of ``rf.ALL``, one rule with ``stored_mode``."""

    def test_agrees_with_stored_mode_for_every_th1(self, stored_dir: Path) -> None:
        from rootfig.expressions import quote_name
        from rootfig.histograms import stored_mode, stored_names

        for path in sorted(stored_dir.glob("*.root")):
            if path.name == "two_trees.root":
                continue
            sample = Sample(path)
            by_rule = stored_names(sample)
            assert by_rule == sorted(by_rule)
            for name in sample.source.histograms(ndim=1):
                expression = quote_name(name)
                assert expression is not None
                assert stored_mode([sample], [Variable(expression)]) == (name in by_rule), name

    def test_cases(self, stored_dir: Path) -> None:
        from rootfig.histograms import stored_names

        assert stored_names(Sample(stored_dir / "ZH_sel0_histo.root")) == [
            "cutflow",
            "eventsProcessed",
            "mz",
            "mz_raw",
        ]
        assert stored_names(Sample(stored_dir / "in_directory.root")) == ["sub/mz"]
        assert stored_names(Sample(stored_dir / "tree_without_branch.root")) == ["mz"]
        assert stored_names(Sample(stored_dir / "branch_and_histogram.root")) == []  # branch wins
        assert stored_names(Sample(stored_dir / "unsupported.root")) == []
        # explicit intent addresses a tree; in-memory data holds no stored histograms
        assert stored_names(Sample(stored_dir / "ZH_sel0_histo.root", tree="events")) == []
        assert stored_names(Sample(stored_dir / "ZH_sel0_histo.root", entry_stop=3)) == []
        assert stored_names(Sample({"mz": [1.0]})) == []
        with pytest.raises(SourceError, match=r"holds 1 stored histograms and several trees"):
            stored_names(Sample(stored_dir / "two_trees.root"))


class TestGroupedHistograms:
    @staticmethod
    def _sample(values: list[float], label: str, **kwargs: Any) -> Sample:
        return Sample({"x": values, "w": [2.0] * len(values)}, label=label, **kwargs)

    def test_group_is_the_sum_of_its_samples(self) -> None:
        a = self._sample([0.2, 0.4], "A", color="C1")
        b = self._sample([0.6, 0.8], "B")
        group = Group([a, b], label="AB", color="C0", histtype="fill")
        (h,) = build_histograms([group], Variable("x", bins=(4, 0, 1)))
        ha, hb = build_histograms([a, b], Variable("x", bins=(4, 0, 1)))
        np.testing.assert_allclose(h.values(), ha.values() + hb.values())
        np.testing.assert_allclose(h.variances(), ha.variances() + hb.variances())
        assert (h.label, h.color, h.histtype) == ("AB", "C0", "fill")
        assert (h.sample, h.stats, h.entries, h.per_object) == (None, None, None, False)
        assert not h.is_data

    def test_top_level_order_is_kept(self) -> None:
        a, b, s = self._sample([0.1], "A"), self._sample([0.2], "B"), self._sample([0.9], "S")
        group = Group([a, b], label="AB")
        var = Variable("x", bins=(2, 0, 1))
        assert [h.label for h in build_histograms([group, s], var)] == ["AB", "S"]
        assert [h.label for h in build_histograms([s, group], var)] == ["S", "AB"]
        (h,) = build_histograms([Group([group, s], label="All")], var)
        np.testing.assert_allclose(h.values(), [2.0, 1.0])

    def test_binning_is_shared_by_every_leaf(self) -> None:
        low = self._sample([0.0, 1.0, 2.0], "low")
        high = self._sample([100.0, 101.0, 102.0], "high")
        alone = self._sample([50.0], "alone")
        grouped, single = build_histograms(
            [Group([low, high], label="G"), alone], Variable("x", bins=10, range="auto")
        )
        assert (grouped.edges[0], grouped.edges[-1]) == pytest.approx((0.0, 102.0), abs=0.2)
        np.testing.assert_allclose(single.edges, grouped.edges)
        assert grouped.integral == 6.0  # nothing in the flow bins

    def test_each_sample_keeps_its_scaling_selection_and_weight(self) -> None:
        a = self._sample([0.3, 0.5, 0.5, 0.5], "A", xsec="2 pb", ngen=1000, selection="x > 0.4")
        b = self._sample([0.5, 0.5], "B", xsec="8 pb", ngen=100, weight="w", scale=3.0)
        (h,) = build_histograms(
            [Group([a, b], label="AB")], Variable("x", bins=(1, 0, 1)), lumi=1.0
        )
        # per entry: xsec [pb] * 1e3 * lumi [fb^-1] / ngen, times weight and scale
        assert h.integral == pytest.approx(3 * 2.0 + 2 * 2.0 * 3.0 * 80.0)

    def test_systematics_combine_as_a_sum(self) -> None:
        a = self._sample([0.5], "A", systematics={"lumi": 0.10, "gen": 0.50})
        b = self._sample([0.5, 0.5], "B", systematics={"lumi": 0.10})
        (h,) = build_histograms(
            [Group([a, b], label="AB")],
            Variable("x", bins=(1, 0, 1)),
            systematics={"theory": (1.2, 0.9)},
        )
        assert set(h.variations) == {"lumi", "gen", "theory"}
        np.testing.assert_allclose(h.variations["lumi"][0].values(), [3.3])  # linear: 1.1 + 2.2
        np.testing.assert_allclose(h.variations["lumi"][1].values(), [2.7])
        np.testing.assert_allclose(h.variations["gen"][0].values(), [1.5 + 2.0])  # B: its nominal
        np.testing.assert_allclose(h.variations["gen"][1].values(), [0.5 + 2.0])
        np.testing.assert_allclose(h.variations["theory"][0].values(), [3.6])
        np.testing.assert_allclose(uncertainty(h).syst_up, [np.sqrt(0.3**2 + 0.5**2 + 0.6**2)])
        normalised = normalize(h, True)  # the sum, then each variation by its own total
        assert normalised.integral == pytest.approx(1.0)
        np.testing.assert_allclose(normalised.variations["gen"][0].values(), [1.0])

    def test_data_group(self) -> None:
        d1 = self._sample([0.5], "D1", is_data=True)
        d2 = self._sample([0.5], "D2", is_data=True)
        (h,) = build_histograms(
            [Group([d1, d2], label="Data")],
            Variable("x", bins=(1, 0, 1)),
            systematics={"lumi": 0.1},
        )
        assert h.is_data
        assert not h.variations
        np.testing.assert_allclose(h.values(), [2.0])

    def test_stored_histograms_are_summed(self, stored_dir: Path) -> None:
        vv = Group(
            [
                Sample(stored_dir / "WW_sel0_histo.root", label="WW"),
                Sample(stored_dir / "ZZ_sel0_histo.root", label="ZZ"),
            ],
            label="VV",
            color="C0",
        )
        zh = Sample(stored_dir / "ZH_sel0_histo.root", label="ZH")
        h_vv, h_zh = build_histograms([vv, zh], Variable("mz", bins=50))
        assert (h_vv.label, h_vv.color, h_vv.axis.size) == ("VV", "C0", 50)
        assert h_vv.sum_weights == pytest.approx(0.5 * 3000)
        assert h_zh.sum_weights == pytest.approx(0.5 * 4000)
        other = Group(
            [Sample(stored_dir / "ZH_sel0_histo.root"), Sample(stored_dir / "other_binning.root")],
            label="G",
        )
        with pytest.raises(BinningError, match="different bin edges") as info:
            build_histograms([other], "mz")
        assert "group 'G'" in "".join(info.value.__notes__)

    def test_group_histogram_and_regroup(self) -> None:
        a = Histogram(contents([1.0, 2.0]), label="A", color="C1", per_object=True)
        b = Histogram(contents([3.0, 4.0]), label="B")
        group = Group(
            [Sample({"x": [1.0]}, label="A"), Sample({"x": [1.0]}, label="B")], label="AB"
        )
        sample = Sample({"x": [1.0]}, label="S")
        h = group_histogram(group, [a, b])
        np.testing.assert_allclose(h.values(), [4.0, 6.0])
        assert (h.label, h.color, h.histtype, h.sample) == ("AB", None, None, None)
        assert h.per_object  # any component
        regrouped = regroup_histograms([group, sample], [a, b, a])
        assert regrouped[1] is a
        np.testing.assert_allclose(regrouped[0].values(), [4.0, 6.0])
        with pytest.raises(ValueError, match="got 2 histograms for 3 samples"):
            regroup_histograms([group, sample], [a, b])

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_group_histogram_requires_one_histogram_per_leaf(self, count: int) -> None:
        sample = Sample({"x": [0.5]}, label="A")
        inner = Group([sample, sample.replace(label="B")], label="AB")
        group = Group([inner], label="outer")
        h = Histogram(contents([1.0, 2.0]), label="A")
        with pytest.raises(ValueError, match=f"got {count} histograms for 2 samples"):
            group_histogram(group, [h] * count)
        np.testing.assert_allclose(group_histogram(group, [h, h]).values(), [2.0, 4.0])

    def test_only_top_level_samples_are_summarised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rootfig.histograms import pipeline

        summarised: list[str] = []

        def counting(columns: Columns) -> Summary:
            summarised.append(str(columns.n_entries))
            return summarize(columns)

        monkeypatch.setattr(pipeline, "summarize", counting)
        jagged = Sample({"pt": ak.Array([[1.0, 2.0], [3.0], []])}, label="A")
        flat = Sample({"pt": [1.0, 2.0]}, label="F")
        var = Variable("pt", bins=(4, 0, 4))
        grouped, single = build_histograms(
            [Group([jagged, jagged.replace(label="B")], label="AB"), flat], var
        )
        assert summarised == ["2"]  # the flat sample only
        assert grouped.per_object
        assert not single.per_object
        assert grouped.stats is None
        assert single.entries == 2
        # the same sample inside a group and on its own: summarised once, for the latter
        summarised.clear()
        grouped, alone = build_histograms([Group([jagged, flat], label="G"), jagged], var)
        assert summarised == ["3"]
        assert grouped.stats is None
        assert alone.entries == 3

    def test_per_object_follows_statistics_and_sums(self) -> None:
        stats = summarize(prepare({"pt": ak.Array([[1.0, 2.0], [3.0]])}, ["pt"]))
        assert stats.per_object
        h = Histogram(contents([1.0, 2.0]), "A", stats=stats)
        assert h.per_object
        assert not Histogram(contents([1.0, 2.0]), "B", stats=stats, per_object=False).per_object
        assert not Histogram(contents([1.0, 2.0]), "C").per_object
        assert from_sample(Sample({"pt": [1.0]}), contents([1.0, 2.0]), stats=stats).per_object
        assert sum_histograms([h, Histogram(contents([1.0, 2.0]), "D")]).per_object
        assert not sum_histograms([Histogram(contents([1.0, 2.0]), "D")]).per_object


class TestChunkedFilling:
    """Histograms prepared a chunk of events at a time, in threads, equal those of one read."""

    @staticmethod
    def _chunked(monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        for name, value in (
            ("rootfig.io.sources:CHUNK_BYTES", 2_000),
            ("rootfig.histograms.pipeline:CHUNK_BYTES", 2_000),
            ("rootfig._threads:THREADS", 4),
        ):
            module, attr = name.split(":")
            monkeypatch.setattr(importlib.import_module(module), attr, value)

    @pytest.mark.parametrize("source", ["tree", "rntuple", "two files", "arrays"])
    def test_same_histograms_and_variations(
        self,
        data_dir: Path,
        signal_columns: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        source: str,
    ) -> None:
        data: Any = {
            "tree": str(data_dir / "signal.root"),
            "rntuple": str(data_dir / "signal_rntuple.root"),
            "two files": [str(data_dir / "bkg_part1.root"), str(data_dir / "bkg_part2.root")],
            "arrays": signal_columns,
        }[source]
        sample = Sample(
            data,
            label="S",
            weight="weight",
            systematics={
                "w": ("weight * 1.1", "weight * 0.8"),
                "shift": {"Muon_pt": ("Muon_eta", "Muon_phi"), "MET": ("sentinel", "with_nan")},
                "norm": 0.05,
            },
        )
        cases = [("Muon_pt", "Muon_eta > 0"), ("MET", "nMuon > 0"), ("with_nan", None)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RootfigWarning)
            whole = [build_histograms([sample], v, selection=s)[0] for v, s in cases]
            self._chunked(monkeypatch)
            parts = [build_histograms([sample], v, selection=s)[0] for v, s in cases]
        for got, want in zip(parts, whole, strict=True):
            np.testing.assert_array_equal(got.values(flow=True), want.values(flow=True))
            np.testing.assert_array_equal(got.variances(flow=True), want.variances(flow=True))
            assert got.stats == want.stats
            assert sorted(got.variations) == sorted(want.variations)
            for name, pair in got.variations.items():
                for mine, theirs in zip(pair, want.variations[name], strict=True):
                    np.testing.assert_array_equal(mine.values(flow=True), theirs.values(flow=True))
                    np.testing.assert_array_equal(
                        mine.variances(flow=True), theirs.variances(flow=True)
                    )

    def test_constant_variables_are_chunked_too(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rootfig.selection.chunks import Request

        sample = Sample(str(signal_file))
        variable = Variable("1", bins=(3, 0, 3))  # reads no branch, broadcast to every event
        [whole] = build_histograms([sample], variable)
        self._chunked(monkeypatch)
        sizes: list[int] = []
        original = Request.prepare

        def counting(request: Request, arrays: Any, n_events: int) -> Columns:
            sizes.append(n_events)
            return original(request, arrays, n_events)

        monkeypatch.setattr(Request, "prepare", counting)
        [parts] = build_histograms([sample], variable)
        assert len(sizes) > 1
        assert sum(sizes) == 2000
        np.testing.assert_array_equal(parts.values(flow=True), whole.values(flow=True))
        assert parts.stats is not None
        assert whole.stats is not None
        np.testing.assert_equal(asdict(parts.stats), asdict(whole.stats))  # nan skewness alike

    def test_slices_hold_at_most_the_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rootfig.histograms import pipeline

        monkeypatch.setattr(pipeline, "CHUNK_BYTES", 1_000)
        for n_events in (1, 125, 126, 199, 250, 1_000):  # 8 B to 8 kB of float64
            values = np.arange(float(n_events))
            slices = list(pipeline._slices({"x": values}, n_events))
            assert all(arrays["x"].nbytes <= 1_000 for arrays, _ in slices)
            assert [n for _, n in slices] == [len(arrays["x"]) for arrays, _ in slices]
            np.testing.assert_array_equal(np.concatenate([a["x"] for a, _ in slices]), values)

    def test_reading_and_preparing_share_one_pool(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        import rootfig._threads as threads_module
        from rootfig.selection.chunks import Request

        self._chunked(monkeypatch)
        pools: list[ThreadPoolExecutor] = []
        prepared: list[int] = []

        class Counted(ThreadPoolExecutor):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                pools.append(self)

        original = Request.prepare

        def counting(request: Request, arrays: Any, n_events: int) -> Columns:
            prepared.append(n_events)
            return original(request, arrays, n_events)

        monkeypatch.setattr(threads_module, "ThreadPoolExecutor", Counted)
        monkeypatch.setattr(Request, "prepare", counting)
        sample = Sample(str(signal_file), weight="weight")
        build_histograms([sample], Variable("Muon_pt", bins=(10, 0, 100)), selection="nMuon > 0")
        assert len(prepared) > 1  # read and prepared a chunk at a time
        assert len(pools) == 1

    @pytest.mark.parametrize(
        ("dtypes", "expression"),
        [
            (("int32", "int64"), "x * x"),
            (("float32", "float64"), "x * 1.1"),
            (("int64", "int64"), "x * x"),
        ],
    )
    @pytest.mark.parametrize("threads", [1, 4])  # one: the first file is prepared before
    def test_files_of_different_types_are_prepared_as_one_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        dtypes: tuple[str, str],
        expression: str,
        threads: int,
    ) -> None:
        import importlib

        import uproot

        from rootfig.histograms import load_columns
        from rootfig.io import FileSource, ReadCache

        parts = []
        for i, dtype in enumerate(dtypes):
            values = (np.arange(1, 1001) * 100_003 + i).astype(dtype)  # squares overflow int32
            with uproot.recreate(tmp_path / f"part{i}.root") as file:
                file.mktree("events", {"x": values.dtype})
                file["events"].extend({"x": values})
            parts.append(values)
        sample = Sample([str(tmp_path / f"part{i}.root") for i in range(2)])
        joined = np.concatenate(parts)  # one type for both, as reading them at once gives
        want = joined * joined if expression == "x * x" else joined * 1.1
        cached = load_columns(sample, [expression], cache=ReadCache())
        self._chunked(monkeypatch)
        monkeypatch.setattr(importlib.import_module("rootfig._threads"), "THREADS", threads)
        whole_reads: list[list[str]] = []
        original = FileSource.arrays

        def counting(source: FileSource, branches: Any) -> Any:
            whole_reads.append(list(branches))
            return original(source, branches)

        monkeypatch.setattr(FileSource, "arrays", counting)
        chunked = load_columns(sample, [expression])
        np.testing.assert_array_equal(cached.arrays[0], want)
        np.testing.assert_array_equal(chunked.arrays[0], want)
        # read whole only when the files differ, after the chunks showed it
        assert whole_reads == ([["x"]] if dtypes[0] != dtypes[1] else [])

    @pytest.mark.parametrize("threads", [1, 4])
    @pytest.mark.parametrize("strict", ["errstate", "warnings", "call", "log"])
    def test_a_floating_point_error_in_a_narrower_file_reads_the_sample_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threads: int, strict: str
    ) -> None:
        import contextlib
        import importlib

        import uproot

        from rootfig.histograms import load_columns
        from rootfig.io import ReadCache

        def write(name: str, dtype: str, n_events: int = 1_000) -> str:
            with uproot.recreate(tmp_path / name) as file:
                file.mktree("events", {"x": dtype})
                if n_events:
                    file["events"].extend({"x": np.full(n_events, 1e20, dtype=dtype)})
            return str(tmp_path / name)

        # x * x overflows float32, not float64; an empty tree has types all the same
        mixed = Sample([write("a.root", "float32"), write("b.root", "float64")])
        narrow = Sample([write("c.root", "float32"), write("d.root", "float32", n_events=0)])

        class Refusing:
            """A NumPy error handler, or log, that raises an error of its own."""

            def __call__(self, kind: str, flag: int) -> None:
                raise RuntimeError(kind)

            def write(self, message: str) -> None:
                raise RuntimeError(message)

        def raising() -> contextlib.ExitStack:
            stack = contextlib.ExitStack()
            if strict == "warnings":
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("error", RuntimeWarning)
            else:
                mode = {"errstate": "raise"}.get(strict, strict)
                stack.enter_context(np.errstate(over=mode, call=Refusing()))
            return stack

        with raising():
            cached = load_columns(mixed, ["x * x"], cache=ReadCache())
        self._chunked(monkeypatch)
        monkeypatch.setattr(importlib.import_module("rootfig._threads"), "THREADS", threads)
        with raising():
            chunked = load_columns(mixed, ["x * x"])
            with pytest.raises(ExpressionError, match="overflow"):  # as a whole read of it
                load_columns(narrow, ["x * x"])
        np.testing.assert_array_equal(chunked.arrays[0], cached.arrays[0])

    @pytest.mark.parametrize("wrapped", [False, True])
    def test_running_out_of_memory_is_not_retried_as_a_whole_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrapped: bool
    ) -> None:
        import importlib

        import uproot

        from rootfig.histograms import load_columns, pipeline
        from rootfig.selection.chunks import Request

        paths = []
        for name, dtype in (("a.root", "float32"), ("b.root", "float64")):
            with uproot.recreate(tmp_path / name) as file:
                file.mktree("events", {"x": dtype})
                file["events"].extend({"x": np.ones(1_000, dtype=dtype)})
            paths.append(str(tmp_path / name))

        def out_of_memory(request: Request, arrays: Any, n_events: int) -> Columns:
            if not wrapped:
                raise MemoryError
            try:
                raise MemoryError
            except MemoryError as exc:
                raise ExpressionError("failed to evaluate expression 'x'") from exc

        def whole_read(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a whole read after running out of memory")

        self._chunked(monkeypatch)
        # one thread: the first file's first chunk fails before the second file is read
        monkeypatch.setattr(importlib.import_module("rootfig._threads"), "THREADS", 1)
        monkeypatch.setattr(Request, "prepare", out_of_memory)
        monkeypatch.setattr(pipeline, "read_arrays", whole_read)
        with pytest.raises(ExpressionError if wrapped else MemoryError):
            load_columns(Sample(paths), ["x"])


class TestPrefetch:
    """prefetch() warms a ReadCache with what build_histograms reads; results do not change."""

    @staticmethod
    def _reads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        from rootfig.io import FileSource

        calls: list[list[str]] = []
        original = FileSource.arrays

        def counting(self: FileSource, branches: Any) -> Any:
            calls.append(list(branches))
            return original(self, branches)

        monkeypatch.setattr(FileSource, "arrays", counting)
        return calls

    @staticmethod
    def _assert_same(cached: list[Any], plain: list[Any]) -> None:
        for got, want in zip(cached, plain, strict=True):
            np.testing.assert_array_equal(got.values(flow=True), want.values(flow=True))
            np.testing.assert_array_equal(got.variances(flow=True), want.variances(flow=True))
            assert got.stats == want.stats
            assert got.per_object == want.per_object
            assert sorted(got.variations) == sorted(want.variations)
            for name, (up, down) in got.variations.items():
                want_up, want_down = want.variations[name]
                np.testing.assert_array_equal(up.values(flow=True), want_up.values(flow=True))
                np.testing.assert_array_equal(down.values(flow=True), want_down.values(flow=True))

    def test_lower_level_callers_reuse_the_cached_source(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uproot

        import rootfig as rf
        from rootfig.histograms import load_columns, read_arrays
        from rootfig.io import ReadCache

        opens: list[str] = []
        original = uproot.open

        def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
            opens.append(Path(path).name)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(uproot, "open", counting)
        reads = self._reads(monkeypatch)
        cache = ReadCache()
        # samples built apart from the same files: equal sources, each blank to begin with
        first = load_columns(rf.Sample(signal_file, tree="events"), ["MET"], cache=cache)
        again = load_columns(rf.Sample(signal_file, tree="events"), ["MET"], cache=cache)
        arrays, _ = read_arrays(
            rf.Sample(signal_file, tree="events"), [rf.Variable("MET").parsed()], cache=cache
        )
        assert opens == ["signal.root"]  # the branch list was learnt once, not three times
        assert reads == [["MET"]]
        np.testing.assert_array_equal(first.values, again.values)
        np.testing.assert_array_equal(np.asarray(arrays["MET"]), first.values)

    def test_union_read_once_per_source(
        self, signal_file: Path, background_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rootfig as rf
        from rootfig.histograms import build_histograms, prefetch
        from rootfig.io import ReadCache

        samples = [
            rf.Sample(signal_file, tree="events", weight="weight", label="S"),
            rf.Sample(background_file, tree="events", label="B"),
        ]
        variables = [rf.Variable("MET"), rf.Variable("Muon_pt"), rf.Variable("nMuon * 2")]
        selections = [None, "nMuon > 0"]
        plain = {
            (variable.expression, selection): build_histograms(
                samples, variable, selection=selection
            )
            for variable in variables
            for selection in selections
        }
        reads = self._reads(monkeypatch)
        cache = ReadCache()
        prefetch(cache, samples, variables, selections=selections)
        assert [set(call) for call in reads] == [
            {"MET", "Muon_pt", "nMuon", "weight"},
            {"MET", "Muon_pt", "nMuon"},
        ]
        for (expression, selection), want in plain.items():
            cached = build_histograms(samples, expression, selection=selection, cache=cache)
            assert len(reads) == 2  # every histogram of the batch is filled from the two reads
            self._assert_same(cached, want)

    def test_systematics_join_the_plan(
        self, signal_file: Path, background_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rootfig as rf
        from rootfig.histograms import build_histograms, prefetch
        from rootfig.io import ReadCache

        sample = rf.Sample(
            signal_file,
            tree="events",
            weight="weight",
            label="S",
            systematics={
                "w": ("weight * 1.1", "weight * 0.9"),
                "scale": {"Muon_pt": ("Muon_eta", "Muon_phi")},
                "norm": 0.05,
                "alt": rf.Systematic.samples(background_file),
            },
        )
        plain = build_histograms([sample], "Muon_pt")
        reads = self._reads(monkeypatch)
        cache = ReadCache()
        prefetch(cache, [sample], [rf.Variable("Muon_pt"), rf.Variable("MET")])
        # one read per file: the sample's own, with the weight variations' and the
        # replacement branches, and the file the "alt" variation fills from
        assert len(reads) == 2
        assert set(reads[0]) == {"Muon_pt", "weight", "MET", "Muon_eta", "Muon_phi"}
        assert set(reads[1]) == {"Muon_pt", "weight", "MET"}  # no systematics of its own
        cached = build_histograms([sample], "Muon_pt", cache=cache)
        build_histograms([sample], "MET", cache=cache)
        assert len(reads) == 2  # both variables, nominal and variations, from those two reads
        assert sorted(cached[0].variations) == ["alt", "norm", "scale", "w"]
        self._assert_same(cached, plain)

    def test_unusable_variations_are_left_to_build_histograms(
        self, signal_file: Path, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rootfig as rf
        from rootfig.histograms import build_histograms, prefetch
        from rootfig.io import ReadCache

        # a variation whose data is not a dataset is not planned; the task raises for it
        broken = rf.Systematic.samples(12345)
        sample = rf.Sample(signal_file, tree="events", label="S", systematics={"alt": broken})
        reads = self._reads(monkeypatch)
        cache = ReadCache()
        prefetch(cache, [sample], [rf.Variable("MET")])
        assert reads == [["MET"]]  # the nominal file only
        with pytest.raises(rf.SystematicError, match="cannot use 12345 as varied data"):
            build_histograms([sample], "MET", cache=cache)
        stored = rf.Sample(
            stored_dir / "WW_sel0_histo.root",
            label="WW",
            systematics={"alt": rf.Systematic.samples({"mz": [1.0]}), "norm": 0.05},
        )
        prefetch(cache, [stored], [rf.Variable("mz")])  # the variation is not files: skipped
        with pytest.raises(rf.SystematicError, match="takes its variations from other ROOT"):
            build_histograms([stored], "mz", cache=cache)
        # a sample that cannot provide the stored histogram at all stops there
        weighted = rf.Sample(stored_dir / "WW_sel0_histo.root", label="W", weight="w")
        prefetch(cache, [weighted], [rf.Variable("mz")])
        with pytest.raises(rf.SelectionError, match="weight cannot be applied"):
            build_histograms([weighted], "mz", cache=cache)

    def test_stored_variation_files_join_the_plan(
        self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uproot

        import rootfig as rf
        from rootfig.histograms import build_histograms, prefetch
        from rootfig.io import ReadCache

        opens: list[str] = []
        original = uproot.open

        def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
            opens.append(Path(path).name)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(uproot, "open", counting)
        sample = rf.Sample(
            stored_dir / "WW_sel0_histo.root",
            label="WW",
            systematics={"alt": rf.Systematic.samples(stored_dir / "ZZ_sel0_histo.root")},
        )
        variables = [rf.Variable("mz"), rf.Variable("mz_raw")]
        plain = build_histograms([sample], "mz")
        del opens[:]
        cache = ReadCache()
        prefetch(cache, [sample], variables)
        # each file: opens to learn its objects, then one that reads both histograms
        assert opens.count("ZZ_sel0_histo.root") <= 2
        del opens[:]
        cached = build_histograms([sample], "mz", cache=cache)
        build_histograms([sample], "mz_raw", cache=cache)
        assert opens == []  # both names of both files came from the batch read
        assert sorted(cached[0].variations) == ["alt"]
        self._assert_same(cached, plain)

    def test_skips_what_build_histograms_refuses(
        self, signal_file: Path, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rootfig as rf
        from rootfig.histograms import build_histograms, prefetch
        from rootfig.io import ReadCache

        samples = [rf.Sample(signal_file, tree="events", label="S")]
        reads = self._reads(monkeypatch)
        cache = ReadCache()
        prefetch(
            cache,
            samples,
            [rf.Variable("MET"), rf.Variable("nosuch")],
            selections=[None, "bogus > 1"],
        )
        assert reads == [["MET"]]
        with pytest.raises(rf.MissingBranchError, match="'nosuch'"):
            build_histograms(samples, "nosuch", cache=cache)
        # a stored name that not every sample provides is refused by stored_mode, so not planned
        mixed = [rf.Sample(stored_dir / "WW_sel0_histo.root"), samples[0]]
        prefetch(cache, mixed, [rf.Variable("mz")])
        with pytest.raises(rf.SourceError, match="every sample of one plot"):
            build_histograms(mixed, "mz", cache=cache)
        # a sample whose own selection forbids reading a stored histogram is skipped, not read
        selected = [rf.Sample(stored_dir / "WW_sel0_histo.root", selection="mz > 1")]
        prefetch(cache, selected, [rf.Variable("mz")])
        with pytest.raises(rf.SelectionError, match="sample's selection cannot be applied"):
            build_histograms(selected, "mz", cache=cache)
        prefetch(cache, samples, [rf.Variable("MET")], systematics={"bad": object()})  # type: ignore[dict-item]
        assert len(reads) == 1

    def test_stored_names_read_in_one_pass(
        self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uproot

        import rootfig as rf
        from rootfig.histograms import build_histograms, prefetch
        from rootfig.io import ReadCache

        opens: list[str] = []
        original = uproot.open

        def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
            opens.append(Path(path).name)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(uproot, "open", counting)
        samples = [
            rf.Sample(stored_dir / "WW_sel0_histo.root", label="WW"),
            rf.Sample(stored_dir / "ZZ_sel0_histo.root", label="ZZ"),
        ]
        cache = ReadCache()
        prefetch(cache, samples, [rf.Variable("mz"), rf.Variable("mz_raw")])
        # each file: one open to list its objects, one to read both histograms
        assert sorted(opens) == [
            "WW_sel0_histo.root",
            "WW_sel0_histo.root",
            "ZZ_sel0_histo.root",
            "ZZ_sel0_histo.root",
        ]
        cached = build_histograms(samples, "mz_raw", cache=cache)
        assert len(opens) == 4
        self._assert_same(cached, build_histograms(samples, "mz_raw"))

    def test_read_failure_is_left_to_build_histograms(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rootfig as rf
        from rootfig.histograms import build_histograms, prefetch
        from rootfig.io import FileSource, ReadCache

        original = FileSource.arrays
        failed: list[list[str]] = []

        def flaky(self: FileSource, branches: Any) -> Any:
            if not failed:
                failed.append(list(branches))
                msg = "transient"
                raise OSError(msg)
            return original(self, branches)

        monkeypatch.setattr(FileSource, "arrays", flaky)
        samples = [rf.Sample(signal_file, tree="events")]
        cache = ReadCache()
        prefetch(cache, samples, [rf.Variable("MET")])  # the failure is left to the task
        assert failed == [["MET"]]
        self._assert_same(
            build_histograms(samples, "MET", cache=cache), build_histograms(samples, "MET")
        )


def test_tree_and_ready_made_variable_agree() -> None:
    import rootfig as rf

    x = np.random.default_rng(73).uniform(-10, 220, 30000)
    variable = rf.Variable("x", bins=(20, 120, 140))
    [tree] = rf.histograms({"x": x}, variable)
    fine = hist.Hist(hist.axis.Regular(2000, 0, 200, name="x"), storage=hist.storage.Weight())
    fine.fill(x)
    plot = rf.plot(fine, variable)
    [ready] = plot.histograms
    np.testing.assert_allclose(ready.values(flow=True), tree.values(flow=True))
    np.testing.assert_allclose(ready.variances(flow=True), tree.variances(flow=True))
    plot.close()
