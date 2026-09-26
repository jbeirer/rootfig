"""The statistical error model of a Histogram: Poisson intervals, given errors, counts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hist
import numpy as np
import pytest

from helpers import filled, hist_of, symmetric, with_errors
from rootfig.errors import (
    BinningError,
    RootfigWarning,
)
from rootfig.histograms import (
    COMPARISON_KINDS,
    Efficiency,
    Histogram,
    compare,
    normalize,
    poisson_interval,
    read_stored,
    sum_histograms,
    uncertainty,
)
from rootfig.histograms.normalize import normalize_hist
from rootfig.histograms.provenance import Provenance
from rootfig.model import Sample, Variable
from rootfig.plotting import fold_flow_bins, show_flow_bins


class TestAsymmetricErrors:
    """Statistical errors are ``(down, up)`` pairs from the histogram to the comparison."""

    N = [4.0, 9.0, 1.0, 30.0]
    VN = [2.0, 7.0, 0.5, 45.0]
    D = [2.0, 3.0, -2.0, 12.0]
    VD = [1.5, 4.0, 3.0, 20.0]

    def _sides(self) -> tuple[hist.Hist, hist.Hist]:
        return hist_of(self.N, self.VN), hist_of(self.D, self.VD)

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
        errors = symmetric(result.errors)
        np.testing.assert_allclose(errors[defined], expected[defined], rtol=1e-13)
        assert np.isnan(errors[~defined]).all()
        if kind == "pull":
            np.testing.assert_allclose(result.values, (n - d) / np.sqrt(vn + vd), rtol=1e-13)
        if kind in ("ratio", "difference"):
            split = compare(*self._sides(), kind=kind, uncertainty="numerator")
            assert split.band is not None
            scale = np.abs(d) if kind == "ratio" else 1.0
            np.testing.assert_allclose(symmetric(split.errors), np.sqrt(vn) / scale, rtol=1e-13)
            np.testing.assert_allclose(symmetric(split.band), np.sqrt(vd) / scale, rtol=1e-13)

    def test_a_ratio_takes_the_side_that_moves_it(self) -> None:
        # n = 4 with errors (1, 3); d = 2 with errors (0.5, 1), and d = -2 in bin 1
        num = with_errors(Histogram(hist_of([4.0, 4.0], [4.0, 4.0]), "N"), [1, 1], [3, 3])
        ref = with_errors(Histogram(hist_of([2.0, -2.0], [1.0, 1.0]), "D"), [0.5, 0.5], [1.0, 1.0])
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
        num = with_errors(Histogram(hist_of([5.0, 1.0], [5.0, 1.0]), "N"), [2, 0.8], [3, 2.3])
        ref = with_errors(Histogram(hist_of([3.0, 3.0], [3.0, 3.0]), "D"), [1, 1], [1.5, 1.5])
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
        np.testing.assert_array_equal(symmetric(pull.errors), [1.0, 1.0])

    def test_uncertainty_adds_each_statistical_side_to_its_systematic_side(self) -> None:
        nominal = hist_of([10.0, 10.0], [10.0, 10.0])
        varied = Histogram(nominal, "A", variations={"s": (nominal * 1.3, nominal * 0.8)})
        u = uncertainty(with_errors(varied, [2.0, 1.0], [4.0, 5.0]))
        np.testing.assert_allclose(u.stat_down, [2.0, 1.0])
        np.testing.assert_allclose(u.stat_up, [4.0, 5.0])
        np.testing.assert_allclose(u.total_down, np.hypot([2.0, 1.0], 2.0))
        np.testing.assert_allclose(u.total_up, np.hypot([4.0, 5.0], 3.0))


class TestPoissonHistograms:
    def test_what_was_filled_with_weights_is_never_counts(self) -> None:
        empty = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        weighted = Provenance(weighted=True)
        assert Histogram(empty, label="Data", poisson=True).poisson  # looks like counts
        with pytest.raises(ValueError, match="filled with weights or scaled"):
            Histogram(empty, label="Data", poisson=True, _provenance=weighted)
        counts = Histogram(empty, label="Data")
        assert not counts.scaled(1.0)._provenance.weighted
        scaled = counts.scaled(2.0)
        # scaled by rootfig: known counts of factor 2, whose empty bins get 0 +2 x 1.84
        assert scaled.replace(poisson=True).errors()[1] == pytest.approx([2 * 1.8410216450] * 2)
        with pytest.raises(ValueError, match="filled with weights or scaled"):
            Histogram(scaled.hist, label="Data", poisson=True, _provenance=weighted)
        assert scaled.rebinned(2)._provenance.weighted  # rebinning keeps what filled it
        assert sum_histograms([counts, scaled])._provenance.weighted
        assert normalize(counts, "width")._provenance.weighted
        assert not scaled.replace(hist=empty.copy())._provenance.weighted  # judged afresh

    def test_only_a_linear_map_keeps_the_interval(self) -> None:
        data = self._data([1.0, 4.0])
        assert data.map_hists(lambda h: h * 2.0, linear=True).poisson

        def square(h: hist.Hist) -> hist.Hist:
            new = h.copy()
            view: Any = new.view(flow=True)
            view.value = view.value**2
            return new

        squared = data.map_hists(square)  # no count relation left: sqrt(variances)
        assert not squared.poisson
        np.testing.assert_allclose(squared.errors()[1], np.sqrt(squared.variances()))

    def _data(self, counts: list[float], **kwargs: Any) -> Histogram:
        return Histogram(hist_of(counts), label="Data", is_data=True, poisson=True, **kwargs)

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
        plain = Histogram(hist_of([0.0, 1.0, 4.0]), label="Data", is_data=True)
        np.testing.assert_array_equal(symmetric(plain.errors()), [0.0, 1.0, 2.0])

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
            data.replace(hist=hist_of([1.0, 2.0, 3.0]), _provenance=data._provenance)

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
        # the error model changes, what is known about the counts stays
        plain = empty.scaled(2.0).replace(poisson=False)
        np.testing.assert_array_equal(plain.errors()[1], 0.0)
        np.testing.assert_allclose(plain.replace(poisson=True).errors()[1], [3.68204329] * 2)
        tripled = empty.scaled(3.0)
        np.testing.assert_allclose(tripled.replace(label="x").errors()[1], [5.52306493] * 2)
        with pytest.warns(RootfigWarning, match="no entries"):  # skipped: the scale stays
            np.testing.assert_allclose(normalize(tripled, "unity").errors()[1], [5.52306493] * 2)

    def test_sums_of_counts_with_one_scale_keep_the_interval(self) -> None:
        a, b = self._data([0.0, 1.0, 4.0]), self._data([0.0, 2.0, 0.0])
        total = sum_histograms([a, b])
        assert total.poisson
        assert total.is_data  # two data periods add up to data
        assert not sum_histograms([a, Histogram(hist_of([0.0, 2.0, 0.0]), label="MC")]).is_data
        low, high = poisson_interval([0.0, 3.0, 4.0])
        np.testing.assert_allclose(total.errors()[0], [0.0, 3.0, 4.0] - low)
        np.testing.assert_allclose(total.errors()[1], high - [0.0, 3.0, 4.0])
        empty_scaled = sum_histograms([a.scaled(2.0), b.scaled(2.0)])
        assert empty_scaled.errors()[1][0] == pytest.approx(2 * 1.8410216450)
        # a plain input, or counts scaled differently, are no longer counts of one scale
        plain = Histogram(hist_of([0.0, 2.0, 0.0]), label="MC")
        assert not sum_histograms([a, plain]).poisson
        assert not sum_histograms([a, b.scaled(2.0)]).poisson

    def test_sums_of_counts_rebinned_differently_are_counts(self) -> None:
        # four bins merged in pairs record (2, 2) per bin, two bins filled directly (1, 1):
        # both hold counts of factor 1, which add up to counts
        h = hist.Hist(hist.axis.Regular(4, 0, 4), storage=hist.storage.Weight())
        h.fill([0.5, 1.5, 1.5, 3.5])
        merged = Histogram(h, label="A", is_data=True, poisson=True).rebinned(2)
        direct = Histogram(
            hist.Hist(hist.axis.Regular(2, 0, 4), storage=hist.storage.Weight()).fill([1.0, 3.0]),
            label="B",
            is_data=True,
            poisson=True,
        )
        assert not np.allclose(merged._provenance.unit.values(), direct._provenance.unit.values())  # type: ignore[union-attr]
        total = sum_histograms([merged, direct])
        assert total.poisson
        low, high = poisson_interval([4.0, 2.0])
        np.testing.assert_allclose(total.errors()[0], [4.0, 2.0] - low)
        np.testing.assert_allclose(total.errors()[1], high - [4.0, 2.0])

    def test_sums_compare_the_factor_of_a_count(self) -> None:
        # merged pairs of factors (1, 0.5) and (0.75, 0.75) record one count per bin as
        # 1.5 alike, but 1.25 and 1.125 as its square: factors of 5/6 and 3/4 per count
        h = hist.Hist(hist.axis.Variable([0.0, 1.0, 3.0, 4.0, 6.0]), storage=hist.storage.Weight())
        h.fill([0.5, 2.0, 2.0, 5.0])
        data = Histogram(h, label="Data", is_data=True, poisson=True)
        wide = data.map_hists(lambda h: normalize_hist(h, "width")[:: hist.rebin(2)], linear=True)
        flat = data.scaled(0.75).rebinned(2)
        np.testing.assert_allclose(wide._provenance.unit.values(), flat._provenance.unit.values())  # type: ignore[union-attr]
        assert not sum_histograms([wide, flat]).poisson
        assert sum_histograms([flat, flat]).poisson

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

    def test_counts_scaled_by_a_negative_factor_have_no_interval(self) -> None:
        # the empty one keeps its contents at zero: only the record of counts knows the sign
        for data in (self._data([0.0, 0.0, 0.0]), self._data([0.0, 1.0, 4.0])):
            with pytest.raises(ValueError, match="scaled by a negative factor"):
                data.scaled(-1.0)
            flipped = data.replace(poisson=False).scaled(-1.0)  # sqrt(sum w^2) stays possible
            np.testing.assert_array_equal(flipped.errors()[1], np.sqrt(data.values()))
            with pytest.raises(ValueError, match="scaled by a negative factor"):
                flipped.replace(poisson=True)

    def test_a_new_hist_brings_its_own_count_scale(self) -> None:
        # counts [0, 1, 2] scaled by 10, then replaced by the counts with the same binning:
        # the empty bin takes 1.84, not the old 10 x 1.84
        old = Histogram(hist_of([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]), "D", poisson=True).scaled(10)
        new = old.replace(hist=hist_of([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
        assert new.errors()[1][0] == pytest.approx(1.8410216450)
        assert old.errors()[1][0] == pytest.approx(10 * 1.8410216450)
        with pytest.raises(ValueError, match="not known to hold counts"):
            old.replace(hist=old.hist)  # its own contents again, but no longer known as counts

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
            ([0.0, 3.0], [2.0, 3.0], "weighted"),  # weights that cancel
            ([6.0], [18.0], "weighted"),  # weights [1, 1, 4]: 2 effective entries, not counts
            ([0.0, 2.0, 4.0], [0.0, 4.0, 8.0], "scaled"),  # build counts and scale them
            ([0.5, 3.0], [0.5, 3.0], "whole"),
        ],
    )
    def test_contents_that_are_not_counts_are_refused(
        self, values: list[float], variances: list[float], problem: str
    ) -> None:
        with pytest.raises(ValueError, match=problem):
            Histogram(hist_of(values, variances), label="Data", poisson=True)
        with pytest.raises(ValueError, match="negative"):
            self._data([1.0, 2.0]).scaled(-1.0)

    def test_a_ratio_to_data_counts_takes_its_interval(self) -> None:
        data = self._data([0.0, 1.0, 9.0])
        mc = Histogram(hist_of([2.0, 2.0, 8.0]), label="MC")
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


ERROR_OPTIONS = Path(__file__).parent / "data" / "error_options.root"
"""Counts 0, 1, 4 saved by ROOT 6.40 with kNormal, kPoisson and kPoisson2 (see CONTRIBUTING.md)."""


class TestPoissonLevels:
    def test_kpoisson2_on_a_histogram(self) -> None:
        data = Histogram(hist_of([0.0, 1.0, 4.0]), label="Data", is_data=True, poisson=0.95)
        down, up = data.errors()
        # ROOT 6.40, TH1::kPoisson2
        np.testing.assert_allclose(down, [0.0, 0.9746821920157102, 2.910134626373674], atol=1e-12)
        np.testing.assert_allclose(up, [3.688879454113936, 4.571643390938899, 6.241588675403699])
        np.testing.assert_allclose(data.scaled(2.0).errors()[1], 2 * up)
        with pytest.raises(ValueError, match="confidence level"):
            Histogram(hist_of([1.0]), label="Data", poisson=1.5)
        one_sigma = Histogram(hist_of([0.0, 1.0, 4.0]), label="Data", poisson=True)
        assert not sum_histograms([data, one_sigma]).poisson  # different levels
        assert sum_histograms([data, data]).poisson == 0.95

    def test_counts_behind_the_contents(self) -> None:
        counts, factor = Histogram(hist_of([0.0, 3.0]), label="A").counts()
        np.testing.assert_array_equal(counts, [0.0, 3.0])
        np.testing.assert_array_equal(factor, [1.0, 1.0])
        scaled = Histogram(hist_of([0.0, 3.0]), label="A", poisson=True).scaled(2.5)
        counts, factor = scaled.counts()
        np.testing.assert_array_equal(counts, [0.0, 3.0])
        np.testing.assert_allclose(factor, [2.5, 2.5])
        with pytest.raises(ValueError, match="holds no known counts"):
            Histogram(hist_of([2.0], [4.0]), label="W").counts()
        with pytest.raises(ValueError, match="negative factor"):
            Histogram(hist_of([0.0, 3.0]), label="A").scaled(-1.0).counts()
        wrong = Provenance(errors=(hist_of([1.0]), hist_of([1.0])))
        with pytest.raises(BinningError, match="statistical errors do not have its binning"):
            Histogram(hist_of([0.0, 3.0]), label="A", _provenance=wrong)


class TestStatErrors:
    """Statistical errors given with the histogram: stat_errors=(down, up)."""

    def _given(self, **kwargs: Any) -> Histogram:
        return Histogram(
            hist_of([4.0, 9.0, 1.0, 1.0], [4.0, 9.0, 1.0, 1.0]),
            label="Fit",
            stat_errors=([1.0, 2.0, 0.5, 0.5], [3.0, 4.0, 1.0, 1.0]),
            **kwargs,
        )

    def test_they_are_the_errors_and_follow_the_contents(self) -> None:
        given = self._given()
        np.testing.assert_array_equal(given.errors()[0], [1.0, 2.0, 0.5, 0.5])
        np.testing.assert_array_equal(given.errors()[1], [3.0, 4.0, 1.0, 1.0])
        np.testing.assert_array_equal(given.errors(flow=True)[1], [0.0, 3.0, 4.0, 1.0, 1.0, 0.0])
        # a negative factor turns the interval over: 4 -1 +3 becomes -8 -6 +2
        flipped = given.scaled(-2.0).errors()
        np.testing.assert_allclose(flipped[0], [6.0, 8.0, 2.0, 2.0])
        np.testing.assert_allclose(flipped[1], [2.0, 4.0, 1.0, 1.0])
        np.testing.assert_array_equal(given.scaled(-1.0).scaled(-1.0).errors(), given.errors())
        negative = Histogram(
            hist_of([-4.0, -9.0, -1.0, -1.0], [4.0, 9.0, 1.0, 1.0]),
            label="N",
            stat_errors=([1.0, 2.0, 0.5, 0.5], [3.0, 4.0, 1.0, 1.0]),
        )
        for spec in (True, "density"):  # the negative total, -15, flips the sign too
            with pytest.warns(RootfigWarning, match="negative total"):
                unity = normalize(negative, spec)
            np.testing.assert_allclose(unity.errors()[0], np.array([3.0, 4.0, 1.0, 1.0]) / 15)
            np.testing.assert_allclose(unity.errors()[1], np.array([1.0, 2.0, 0.5, 0.5]) / 15)
        merged = given.rebinned(2)  # in quadrature, side by side
        np.testing.assert_allclose(merged.errors()[0], [np.hypot(1, 2), np.hypot(0.5, 0.5)])
        unity = normalize(given, True)  # the total, 15, taken as a constant
        np.testing.assert_allclose(unity.errors()[1], np.array([3.0, 4.0, 1.0, 1.0]) / 15)
        assert given.map_hists(lambda h: h.copy())._provenance.errors is None  # not declared linear
        assert given.replace(hist=given.hist.copy())._provenance.errors is None  # new contents
        assert given.replace(label="B")._provenance.errors is not None

    def test_they_add_in_quadrature_in_a_sum_and_enter_comparisons(self) -> None:
        given, plain = self._given(), Histogram(hist_of([1.0] * 4, [4.0] * 4), label="P")
        total = sum_histograms([given, plain])
        np.testing.assert_allclose(total.errors()[1], np.hypot([3.0, 4.0, 1.0, 1.0], 2.0))
        ratio = compare(given, Histogram(hist_of([2.0] * 4, [0.0] * 4), label="R"))
        np.testing.assert_allclose(ratio.errors[1], np.array([3.0, 4.0, 1.0, 1.0]) / 2.0)

    @pytest.mark.parametrize(
        ("errors", "match"),
        [
            (([1.0], [1.0]), "one error per bin"),
            (([1.0, -1.0, 0.0, 0.0], [1.0] * 4), "non-negative"),
            (([1.0] * 4, [np.nan] * 4), "non-negative"),
            ((1.0,), "pair of arrays"),
        ],
    )
    def test_bad_errors_are_refused(self, errors: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Histogram(hist_of([4.0] * 4, [4.0] * 4), label="Fit", stat_errors=errors)

    def test_one_model_at_a_time(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            self._given(poisson=True)
        counts = Histogram(hist_of([1.0, 2.0]), label="C", stat_errors=([0.5, 0.5], [1.0, 1.0]))
        assert (
            counts.replace(poisson=True)._provenance.errors is None
        )  # the Poisson interval replaces them


class TestPoissonRatio:
    # ROOT 6.40, TGraphAsymmErrors::Divide(a, b, "pois"): (ratio, down, up); b = 0 is skipped
    A, B = [3.0, 0.0, 7.0, 5.0], [5.0, 4.0, 2.0, 0.0]
    ROOT = [
        (0.6, 0.38208537342310667, 0.9637507888049045),
        (0.0, 0.0, 0.5844786269875022),
        (3.5, 2.2459790068163716, 7.983773592298714),
    ]

    def test_the_ratio_of_two_poisson_means_is_divide_pois(self) -> None:
        a, b = Histogram(hist_of(self.A), label="A"), Histogram(hist_of(self.B), label="B")
        ratio = compare(a, b, uncertainty="poisson-ratio")
        np.testing.assert_allclose(ratio.values[:3], [r for r, _, _ in self.ROOT])
        np.testing.assert_allclose(ratio.errors[0][:3], [d for _, d, _ in self.ROOT], atol=1e-12)
        np.testing.assert_allclose(ratio.errors[1][:3], [u for _, _, u in self.ROOT], rtol=1e-10)
        assert np.isnan([ratio.values[3], ratio.errors[0][3], ratio.errors[1][3]]).all()
        relative = compare(a, b, kind="relative_difference", uncertainty="poisson-ratio")
        np.testing.assert_allclose(relative.errors[1], ratio.errors[1])
        # counts scaled by known factors: the interval of the counts, scaled like the ratio
        scaled = compare(a.replace(poisson=True).scaled(2.0), b, uncertainty="poisson-ratio")
        np.testing.assert_allclose(scaled.errors[1][:3], 2 * ratio.errors[1][:3])

    def test_the_level_belongs_to_the_ratio(self) -> None:
        # drawing a side with a 95 % Poisson interval does not change the ratio's interval
        a, b = Histogram(hist_of(self.A), label="A"), Histogram(hist_of(self.B), label="B")
        plain = compare(a, b, uncertainty="poisson-ratio")
        for sides in ((a.replace(poisson=0.95), b), (a, b.replace(poisson=0.95))):
            wide = compare(*sides, uncertainty="poisson-ratio")
            np.testing.assert_array_equal(wide.errors, plain.errors)

    def test_it_needs_counts_and_a_ratio(self) -> None:
        a = Histogram(hist_of(self.A), label="A")
        weighted = Histogram(hist_of([1.0] * 4, [2.0] * 4), label="W")
        with pytest.raises(ValueError, match="holds no known counts"):
            compare(a, weighted, uncertainty="poisson-ratio")
        with pytest.raises(ValueError, match="interval of a ratio"):
            compare(a, a, kind="difference", uncertainty="poisson-ratio")
        with pytest.raises(ValueError, match="needs counts"):
            compare(efficiency_of_counts(), efficiency_of_counts(), uncertainty="poisson-ratio")


def efficiency_of_counts() -> Efficiency:
    from rootfig.histograms import efficiency

    return efficiency(filled([0.5]), filled([0.5, 0.5]))


class TestSavedErrorOptions:
    """TH1s saved by ROOT with kNormal, kPoisson, kPoisson2 and weighted kPoisson."""

    # ROOT 6.40, TH1::GetBinErrorLow/Up of the saved histograms, bins 1-3
    ROOT = {
        "normal": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        "poisson": [
            (0.0, 1.841021644577239),
            (0.8272462208950817, 2.2995265585528952),
            (1.9143391858579828, 3.1627531714544537),
        ],
        "poisson2": [
            (0.0, 3.688879454113936),
            (0.9746821920157102, 4.571643390938899),
            (2.910134626373674, 6.241588675403699),
        ],
        "weighted": [(0.0, 0.0), (2.0, 2.0), (1.0, 1.0)],  # ROOT's fallback for weights
    }

    @pytest.mark.parametrize("name", sorted(ROOT))
    def test_stored_histograms_bring_their_errors(self, name: str) -> None:
        (histogram,) = read_stored([Sample(ERROR_OPTIONS)], [Variable(name)])
        down, up = histogram.errors()
        # kPoisson's coverage is ROOT's truncated 1 - 0.682689492: 3e-10 apart
        np.testing.assert_allclose(down, [d for d, _ in self.ROOT[name]], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(up, [u for _, u in self.ROOT[name]], rtol=1e-9)
        assert histogram.poisson == {"poisson": True, "poisson2": 0.95}.get(name, False)

    def test_a_sample_scale_makes_it_weighted(self) -> None:
        # as TH1::Scale in ROOT, and as a sample's scale when filling from a tree
        (scaled,) = read_stored([Sample(ERROR_OPTIONS, scale=2.0)], [Variable("poisson")])
        assert not scaled.poisson
        for side in scaled.errors():
            np.testing.assert_allclose(side, [0.0, 2.0, 4.0])  # ROOT 6.40: 2 sqrt(N)
        with pytest.raises(ValueError, match="weighted or scaled"):
            scaled.counts()
