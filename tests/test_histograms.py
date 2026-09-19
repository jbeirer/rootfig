"""Tests for histogram filling, normalisation, ratios, statistics and the pipeline."""

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
    MissingBranchError,
    RootfigWarning,
    SelectionError,
    SourceError,
    SystematicError,
)
from rootfig.histograms import (
    Histogram,
    Summary,
    as_weight_storage,
    build_histograms,
    build_histograms_2d,
    combined_selection,
    combined_weight,
    compatible_binning,
    correlation_matrix,
    describe_table,
    fill,
    from_sample,
    group_histogram,
    load_columns,
    normalization_label,
    normalize,
    normalize_hist,
    ratio,
    regroup_histograms,
    sum_histograms,
    summarize,
    uncertainty,
)
from rootfig.model import Cut, Group, Sample, Systematic, Variable
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
        assert histogram.errors().tolist() == pytest.approx([np.sqrt(2), 2.0, 1.0])
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


class TestRatio:
    def test_propagate(self) -> None:
        num = fill([hist.axis.Regular(3, 0, 3)], columns([0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2]))
        den = fill([hist.axis.Regular(3, 0, 3)], columns([0.5, 1.5, 2.5]))
        r = ratio(num, den)
        assert r.values.tolist() == pytest.approx([2.0, 4.0, 0.0])
        # bin 0: n=2, vn=2, d=1, vd=1 -> sqrt(2/1 + 4*1/1) = sqrt(6)
        assert r.errors[0] == pytest.approx(np.sqrt(6))
        # bin 2: n=0, vn=0, d=1 -> 0
        assert r.errors[2] == pytest.approx(0.0)
        assert r.band.tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert r.centers.tolist() == [0.5, 1.5, 2.5]
        assert r.half_widths.tolist() == [0.5, 0.5, 0.5]

    def test_numerator_only(self) -> None:
        num = fill([hist.axis.Regular(2, 0, 2)], columns([0.5, 0.5, 0.5, 0.5]))
        den = fill([hist.axis.Regular(2, 0, 2)], columns([0.5, 0.5, 1.5], [2.0, 2.0, 1.0]))
        r = ratio(num, den, uncertainty="numerator")
        assert r.values.tolist() == pytest.approx([1.0, 0.0])
        assert r.errors.tolist() == pytest.approx([2 / 4, 0.0])
        assert r.band.tolist() == pytest.approx([np.sqrt(8) / 4, 1.0])

    def test_zero_denominator_is_nan(self) -> None:
        num = fill([hist.axis.Regular(2, 0, 2)], columns([0.5]))
        den = fill([hist.axis.Regular(2, 0, 2)], columns([1.5]))
        r = ratio(num, den)
        assert np.isnan(r.values[0])
        assert np.isnan(r.errors[0])
        assert np.isnan(r.band[0])
        assert r.values[1] == 0.0

    def test_incompatible(self) -> None:
        a = fill([hist.axis.Regular(2, 0, 2)], columns([0.5]))
        b = fill([hist.axis.Regular(3, 0, 2)], columns([0.5]))
        assert not compatible_binning(a, b)
        with pytest.raises(BinningError, match="identical bin edges"):
            ratio(a, b)
        with pytest.raises(BinningError, match="uncertainty"):
            ratio(a, a, uncertainty="bogus")  # type: ignore[arg-type]


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


class TestSignificance:
    def test_s_over_sqrt_b(self) -> None:
        from rootfig.histograms import significance

        signal = _hist([0.5, 0.5, 1.5])  # s = [2, 1, 0]
        background = _hist([0.5] * 4 + [1.5] * 1)  # b = [4, 1, 0]
        result = significance(signal, background)
        np.testing.assert_allclose(result.values[:2], [2 / 2, 1 / 1])
        assert np.isnan(result.values[2])
        # var = vs/b + s^2 vb/(4 b^3): bin 0 -> 2/4 + 4*4/(4*64) = 0.5 + 0.0625
        assert result.errors[0] == pytest.approx(np.sqrt(0.5625))
        assert np.isnan(result.band).all()
        np.testing.assert_allclose(result.edges, [0, 1, 2, 3])

    def test_s_over_sqrt_s_plus_b(self) -> None:
        from rootfig.histograms import significance

        signal = _hist([0.5, 0.5])
        background = _hist([0.5, 0.5, 1.5])
        result = significance(signal, background, kind="s/sqrt(s+b)")
        assert result.values[0] == pytest.approx(2 / np.sqrt(4))
        assert result.values[1] == pytest.approx(0.0)
        assert np.isnan(result.values[2])
        assert np.isfinite(result.errors[0])
        with pytest.raises(BinningError):
            significance(
                signal, hist.Hist(hist.axis.Regular(4, 0, 4), storage=hist.storage.Weight())
            )


class TestEfficiency:
    def test_wilson_interval(self) -> None:
        from rootfig.histograms import efficiency

        total = _hist([0.5] * 10 + [1.5] * 4)
        passed = _hist([0.5] * 5 + [1.5] * 4)
        eff = efficiency(passed, total, label="tight")
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

        unweighted = efficiency(_hist([0.5] * 5), _hist([0.5] * 10))
        weighted = efficiency(_hist([0.5] * 5, [2.0] * 5), _hist([0.5] * 10, [2.0] * 10))
        assert weighted.values[0] == pytest.approx(unweighted.values[0])
        assert weighted.lower[0] == pytest.approx(unweighted.lower[0])  # same n_eff = 10
        with pytest.raises(BinningError):
            efficiency(
                _hist([0.5]), hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
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

        # total = 1 - 2 = -1: the ratio is defined, a binomial interval is not
        with pytest.warns(RootfigWarning, match="negative total weight"):
            eff = efficiency(*self._weighted([-0.5], [1.0, -2.0]))
        assert eff.values[0] == pytest.approx(0.5)
        assert np.isnan(eff.lower[0])
        assert np.isnan(eff.upper[0])
        with pytest.warns(RootfigWarning, match="negative total weight"):
            eff = efficiency(*self._weighted([-2.0], [1.0, -2.0]))
        assert eff.values[0] == pytest.approx(2.0)
        assert np.isnan(eff.lower[0])

    def test_cancelling_total_weight_is_empty(self) -> None:
        from rootfig.histograms import efficiency

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eff = efficiency(*self._weighted([1.0], [1.0, -1.0]))
        assert np.isnan(eff.values[0])
        assert np.isnan(eff.lower[0])
        assert np.isnan(eff.upper[0])

    def test_positive_total_with_negative_weights_keeps_interval(self) -> None:
        from rootfig.histograms import efficiency

        # total = 1.5 with sum w^2 = 2.25, so n_eff = 1; p = 2/3 with a Wilson band
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eff = efficiency(*self._weighted([1.0], [1.0, 1.0, -0.5]))
        assert eff.values[0] == pytest.approx(2 / 3)
        assert eff.lower[0] == pytest.approx(0.2397411978652, rel=1e-6)
        assert eff.upper[0] == pytest.approx(0.9269254688015, rel=1e-6)

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
        from rootfig.histograms import compatible_binning, ratio, sum_histograms

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
            ratio(a.hist, b.hist)
        np.testing.assert_allclose(ratio(a.hist, same.hist).values, [1.25, 1.25, 2.0])


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
        with pytest.raises(BinningError, match=r"not 2\.5"):
            Histogram(h, label="h").rebinned_to([2.5, None])  # type: ignore[list-item]
        with pytest.raises(BinningError, match="got 1 bin counts for a 2D"):
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
        for bad, message in (
            ([0, 2.5, 6], "no bin edge at 2.5"),
            ([1, 3, 6], "range of a histogram that already exists is fixed"),
            ([0, 3, 7], "range of a histogram that already exists is fixed"),
            ([0, 6, 3], "strictly increasing"),
            ([0, 2.9999999, 3.0000001, 6], "resolve to the same edge 3"),
            ([3, None], "got 2 bin counts for a 1D"),
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
        with pytest.raises(BinningError, match="got 3 bin counts for a 2D"):
            Histogram(h, label="h").rebinned_to([0, 1, 4])
        no_flow = hist.Hist(
            hist.axis.Regular(4, 0, 4, name="x", flow=False), storage=hist.storage.Weight()
        ).fill([0.5, 3.5])
        merged = Histogram(no_flow, label="n").rebinned_to([0, 1, 4])
        np.testing.assert_allclose(merged.values(), [1.0, 1.0])
        assert not merged.axis.traits.underflow
        assert not merged.axis.traits.overflow


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
            ratio(a, b)

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
        from rootfig.histograms import source_length

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
        np.testing.assert_allclose(u.stat, [10.0, 10.0])
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
        split = ratio(num, den, uncertainty="numerator")
        assert split.syst_band is not None
        assert split.syst_errors is not None
        np.testing.assert_allclose(split.syst_band, [[1.0 / 20.0], [4.0 / 20.0]])
        np.testing.assert_allclose(split.syst_errors, [[2.0 / 20.0], [2.0 / 20.0]])
        both = ratio(num, den)
        assert both.syst_errors is not None
        # one source "s" varies both sides together: up 12 / 24, down (mirrored) 8 / 19
        np.testing.assert_allclose(both.syst_errors[0], [0.5 - 8 / 19])
        np.testing.assert_allclose(both.syst_errors[1], [0.0])
        down, up = both.total_errors()
        np.testing.assert_allclose(up, np.hypot(both.errors, both.syst_errors[1]))
        band_down, _ = split.total_band()
        np.testing.assert_allclose(band_down, np.hypot(split.band, 1.0 / 20.0))

        plain = ratio(num.hist, den.hist)
        assert plain.syst_errors is None
        assert plain.syst_band is None
        np.testing.assert_allclose(plain.total_errors()[0], plain.errors)
        np.testing.assert_allclose(plain.total_band()[1], plain.band)
        no_num = ratio(Histogram(contents([10.0]), label="N"), den, uncertainty="numerator")
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
        split = ratio(num, den, uncertainty="numerator")
        np.testing.assert_allclose(split.syst_errors, [[low_n / abs(d)], [high_n / abs(d)]])
        band_low, band_high = (2, 4) if d > 0 else (4, 2)
        np.testing.assert_allclose(split.syst_band, [[band_low / abs(d)], [band_high / abs(d)]])
        both = ratio(num, den)
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
            result = ratio(a, b, uncertainty=mode)  # type: ignore[arg-type]
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
        with pytest.warns(RootfigWarning, match="'shape' up empties the denominator in 1 bin"):
            result = ratio(num, den)
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
        # the stored range is fixed and only its own edges can be kept
        with pytest.raises(BinningError, match="range of a histogram that already exists is fixed"):
            build_histograms([zh], Variable("mz", bins=(10, 0, 100)))
        with pytest.raises(BinningError, match="no bin edge at 101"):
            build_histograms([zh], Variable("mz", bins=[0, 101, 250]))
        with pytest.raises(BinningError, match="axis range is fixed"):
            build_histograms([zh], Variable("mz", range=(0, 100)))

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
