"""goodness_of_fit(): ROOT's chi-square and Kolmogorov tests, and the absolute chi-square."""

from __future__ import annotations

import math

import hist
import numpy as np
import pytest

from helpers import hist_of
from rootfig.errors import BinningError
from rootfig.histograms import (
    GoodnessOfFit,
    Histogram,
    compare,
    goodness_of_fit,
    sum_histograms,
)

# The histograms of the ROOT references below (recipe in CONTRIBUTING.md): counts, and
# weighted contents with their variances (sums of squared weights).
COUNTS_A = [12, 25, 40, 31, 0, 9, 3]
COUNTS_B = [20, 41, 58, 50, 0, 11, 0]
DATA = [3, 8, 15, 22, 18, 9, 4, 1, 0, 0, 0]
MC = [2.5, 9.1, 13.8, 24.0, 16.2, 10.5, 3.3, 0.0, 1.2, 0.0, 4.0]
MC_VARIANCES = [0.9, 2.1, 3.0, 4.4, 3.5, 2.6, 1.4, 0.0, 1.44, 0.0, 8.0]
W1 = [5.2, 10.4, 0.0, 7.7, 3.1, 0.5]
W1_VARIANCES = [2.0, 4.5, 0.0, 3.1, 1.2, 0.3]
W2 = [6.0, 9.0, 0.0, 8.5, 2.0, 1.0]
W2_VARIANCES = [1.5, 3.0, 0.0, 4.0, 1.1, 0.8]
SHIFTED_A = [30, 25, 18, 10, 5, 2]
SHIFTED_B = [10, 18, 25, 30, 20, 8]

# ROOT 6.40, TH1::Chi2TestX: (chi2, ndf, p-value, igood)
ROOT_UU = (5.043437970025272, 5, 0.410602010948533, 2)  # also "UU NORM", B scaled by 0.37
ROOT_UW = (3.3883470095678825, 9, 0.9468920490332712, 3)
# the same with MC scaled: ROOT gives an empty MC bin the variance sum(w^2) / sum(w), which
# scales like the contents, not like a variance
ROOT_UW_SCALED = {
    0.01: (1.995577482198153, 9, 0.9915373494886918, 3),
    250.0: (30.879139388656277, 9, 0.00031049966735732803, 3),
}
ROOT_WW = (1.2799837054803667, 4, 0.8647623248568614, 3)
ROOT_UU_SHIFTED = (33.045753195055525, 5, 3.685518791226889e-06, 0)
# data [0, 4, 6, 2] against MC [1, 4, 5, 2] with variances [1, 3, 6, 0]: in the first bin the
# expected value vanishes exactly, the last one has no MC uncertainty
ROOT_UW_EXACT = (0.057011889414134896, 3, 0.9964408107348687, 3)
# ROOT 6.40, TH1::KolmogorovTest: (p-value, distance from option "M"). TMath::KolmogorovProb
# truncates sqrt(2 pi) in its series for small distances, so p-values agree to 1e-9 there.
ROOT_KS_UU = (0.9999087813182189, 0.038888888888889084)
ROOT_KS_UW = (0.9747436326967583, 0.061465721040189214)
ROOT_KS_FUNCTION = (0.9230514896558095, 0.061465721040189214)  # MC without errors
ROOT_KS_SHIFTED = (5.515559354493481e-06, 0.35885885885885893)


def counts(values: list[int], label: str = "") -> Histogram:
    return Histogram(hist_of(values), label=label)


def weighted(values: list[float], variances: list[float], label: str = "") -> Histogram:
    return Histogram(hist_of(values, variances), label=label)


def assert_chi2(result: GoodnessOfFit, reference: tuple[float, int, float, int]) -> None:
    chi2, ndf, p_value, _ = reference
    assert result.statistic == pytest.approx(chi2, rel=1e-12)
    assert result.ndf == ndf
    assert result.p_value == pytest.approx(p_value, rel=1e-12)
    assert result.chi2_ndf == pytest.approx(chi2 / ndf, rel=1e-12)


class TestRootChi2Test:
    def test_counts_with_counts(self) -> None:
        result = goodness_of_fit(counts(COUNTS_A, "A"), counts(COUNTS_B, "B"))
        assert (result.test, result.method) == ("chi2", "UU")
        assert_chi2(result, ROOT_UU)
        # the bin empty in both does not enter, and lowers the degrees of freedom
        np.testing.assert_array_equal(result.bins, [True, True, True, True, False, True, True])
        assert result.notes == ("'B' has a bin with less than 1 event",)  # igood 2
        assert (result.label, result.reference, result.systematics) == ("A", "B", ())
        assert_chi2(goodness_of_fit(counts(SHIFTED_A), counts(SHIFTED_B)), ROOT_UU_SHIFTED)

    def test_scaled_counts_are_counts(self) -> None:
        # ROOT's "UU NORM" recovers the counts from the errors; rootfig knows them
        scaled = counts(COUNTS_B, "B").scaled(0.37)
        result = goodness_of_fit(counts(COUNTS_A, "A"), scaled)
        assert result.method == "UU"
        assert_chi2(result, ROOT_UU)

    def test_counts_with_a_weighted_histogram(self) -> None:
        # an empty MC bin takes the variance of one entry of the average weight, and the
        # empty data bins whose expectation would vanish get ROOT's added event
        data, mc = counts(DATA, "Data"), weighted(MC, MC_VARIANCES, "MC")
        result = goodness_of_fit(data, mc)
        assert result.method == "UW"
        assert_chi2(result, ROOT_UW)
        assert result.notes == (  # igood 3
            "'Data' has a bin with less than 1 event",
            "'MC' has a bin with fewer than 10 effective entries",
        )
        # ROOT needs the counts first; rootfig puts them there
        swapped = goodness_of_fit(mc, data)
        assert swapped.method == "UW"
        assert_chi2(swapped, ROOT_UW)
        assert (swapped.label, swapped.reference) == ("MC", "Data")

    def test_exact_cases_of_counts_with_a_weighted_histogram(self) -> None:
        mc = weighted([1.0, 4.0, 5.0, 2.0], [1.0, 3.0, 6.0, 0.0])
        assert_chi2(goodness_of_fit(counts([0, 4, 6, 2]), mc), ROOT_UW_EXACT)

    @pytest.mark.parametrize("factor", sorted(ROOT_UW_SCALED))
    def test_an_empty_weighted_bin_depends_on_the_normalisation(self, factor: float) -> None:
        mc = weighted(MC, MC_VARIANCES).scaled(factor)
        assert_chi2(goodness_of_fit(counts(DATA), mc), ROOT_UW_SCALED[factor])

    def test_two_weighted_histograms(self) -> None:
        result = goodness_of_fit(weighted(W1, W1_VARIANCES, "W1"), weighted(W2, W2_VARIANCES))
        assert result.method == "WW"
        assert_chi2(result, ROOT_WW)
        assert result.notes == (
            "'W1' has a bin with fewer than 10 effective entries",
            "a histogram has a bin with fewer than 10 effective entries",
        )

    def test_weighted_counts_are_never_taken_for_counts(self) -> None:
        # whole contents equal to their variances, but filled with weights: weighted
        h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
        h.fill([0.5, 1.5, 1.5, 2.5], weight=[2.0, 1.0, 1.0, 3.0])
        mc = Histogram(h, label="MC").scaled(1.0)
        assert goodness_of_fit(counts([3, 5, 4]), mc).method == "UW"
        lumi = counts([30, 50, 40]).scaled(0.1)  # scaled counts stay counts
        assert goodness_of_fit(counts([3, 5, 4]), lumi).method == "UU"

    @pytest.mark.parametrize("factor", [0.01, 1.0, 250.0])
    @pytest.mark.parametrize("test", ["chi2", "ks"])
    def test_only_the_shapes_matter(self, test: str, factor: float) -> None:
        data, mc = counts(DATA[:7]), weighted(MC[:7], MC_VARIANCES[:7])  # no empty MC bin
        plain = goodness_of_fit(data, mc, test=test)  # type: ignore[arg-type]
        for a, b in ((data.scaled(factor), mc), (data, mc.scaled(factor))):
            result = goodness_of_fit(a, b, test=test)  # type: ignore[arg-type]
            assert result.statistic == pytest.approx(plain.statistic, rel=1e-12)
            assert result.p_value == pytest.approx(plain.p_value, rel=1e-12)

    def test_without_degrees_of_freedom_there_is_no_p_value(self) -> None:
        result = goodness_of_fit(counts([0, 4, 0]), counts([0, 7, 0]))
        assert (result.statistic, result.ndf) == (0.0, 0)
        assert math.isnan(result.p_value)
        assert result.chi2_ndf is not None
        assert math.isnan(result.chi2_ndf)


class TestRootKolmogorovTest:
    @pytest.mark.parametrize(
        ("a", "b", "reference", "tolerance"),
        [
            (counts(COUNTS_A), counts(COUNTS_B), ROOT_KS_UU, 1e-9),
            (counts(DATA), weighted(MC, MC_VARIANCES), ROOT_KS_UW, 1e-9),
            (counts(SHIFTED_A), counts(SHIFTED_B), ROOT_KS_SHIFTED, 1e-12),
        ],
    )
    def test_root_references(
        self, a: Histogram, b: Histogram, reference: tuple[float, float], tolerance: float
    ) -> None:
        result = goodness_of_fit(a, b, test="ks")
        assert (result.test, result.method, result.ndf, result.chi2_ndf) == ("ks", "ks", None, None)
        assert result.p_value == pytest.approx(reference[0], rel=tolerance)
        assert result.statistic == pytest.approx(reference[1], rel=1e-12)
        assert result.bins.all()
        assert result.notes == ()

    def test_a_histogram_without_errors_is_a_function(self) -> None:
        function = weighted(MC, [0.0] * len(MC), "Model")
        result = goodness_of_fit(counts(DATA), function, test="ks")
        assert result.p_value == pytest.approx(ROOT_KS_FUNCTION[0], rel=1e-9)
        assert result.notes == ("'Model' has no uncertainties and is compared as a function",)
        flipped = goodness_of_fit(function, counts(DATA), test="ks")
        assert flipped.p_value == pytest.approx(result.p_value, rel=1e-12)
        with pytest.raises(ValueError, match="both have no uncertainties"):
            goodness_of_fit(function, function, test="ks")


class TestAbsoluteChi2:
    def test_without_systematics_it_is_the_sum_of_squared_pulls(self) -> None:
        # Poisson data enters with the error facing the prediction, as in the pull panel
        data = Histogram(hist_of(DATA[:7]), label="Data", is_data=True, poisson=True)
        mc = weighted(MC[:7], MC_VARIANCES[:7], "MC")
        result = goodness_of_fit(data, mc, test="chi2-absolute")
        pulls = compare(data, mc, kind="pull").values
        assert (result.test, result.method) == ("chi2-absolute", "absolute")
        assert result.statistic == pytest.approx(np.sum(pulls**2), rel=1e-12)
        assert result.ndf == 7  # the normalisation is tested too
        assert result.p_value == pytest.approx(_chi2_probability(result.statistic, 7), rel=1e-12)
        assert result.notes == (
            "'Data' has a bin with fewer than 10 effective entries",
            "'MC' has a bin with fewer than 10 effective entries",
        )

    def test_bins_empty_on_both_sides_do_not_enter(self) -> None:
        result = goodness_of_fit(counts([4, 0, 6]), counts([5, 0, 2]), test="chi2-absolute")
        np.testing.assert_array_equal(result.bins, [True, False, True])
        assert result.ndf == 2
        assert result.statistic == pytest.approx(1 / 9 + 16 / 8)

    def test_a_correlated_source_is_a_rank_one_covariance(self) -> None:
        data, mc = counts([110, 95, 130], "Data"), weighted([100.0, 90.0, 110.0], [4.0, 4, 9])
        varied = mc.replace(
            variations={"lumi": (hist_of([110.0, 99, 121], [4, 4, 9]), hist_of([90.0, 81, 99]))}
        )
        result = goodness_of_fit(data, varied, test="chi2-absolute")
        # Sherman-Morrison: r D^-1 r - (r D^-1 s)^2 / (1 + s D^-1 s), s the shift of r
        r = np.array([10.0, 5.0, 20.0])
        variances = np.array([110 + 4, 95 + 4, 130 + 9.0])
        s = -0.1 * np.array([100.0, 90.0, 110.0])
        expected = np.sum(r**2 / variances) - np.sum(r * s / variances) ** 2 / (
            1 + np.sum(s**2 / variances)
        )
        assert result.statistic == pytest.approx(expected, rel=1e-12)
        assert result.systematics == ("lumi",)
        assert result.statistic < goodness_of_fit(data, mc, test="chi2-absolute").statistic

    def test_a_source_shared_by_both_sides_cancels(self) -> None:
        a = weighted([12.0, 20.0], [12.0, 20.0], "A")
        b = weighted([10.0, 25.0], [10.0, 25.0], "B")
        plain = goodness_of_fit(a, b, test="chi2-absolute")

        def shifted(h: Histogram, by: float) -> Histogram:
            values = h.values() + by
            return h.replace(variations={"jes": (hist_of(values), hist_of(values - 2 * by))})

        both = goodness_of_fit(shifted(a, 3.0), shifted(b, 3.0), test="chi2-absolute")
        assert both.statistic == pytest.approx(plain.statistic, rel=1e-12)
        one = goodness_of_fit(shifted(a, 3.0), b, test="chi2-absolute")
        assert one.statistic < plain.statistic

    def test_a_singular_covariance_names_the_bins(self) -> None:
        exact = weighted([5.0, 6.0, 7.0], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match=r"singular: bins \[0\] have no statistical"):
            goodness_of_fit(counts([0, 6, 9]), exact, test="chi2-absolute")
        # one correlated source cannot make up for two bins without statistical uncertainty
        varied = exact.replace(
            variations={"lumi": (hist_of([5.5, 6.6, 7.7]), hist_of([4.5, 5.4, 6.3]))}
        )
        with pytest.raises(ValueError, match=r"singular: bins \[0, 1\] have no statistical"):
            goodness_of_fit(counts([0, 0, 9]), varied, test="chi2-absolute")


class TestStatErrors:
    @staticmethod
    def given(values: list[float], variances: list[float], sumw2: float = 0.0) -> Histogram:
        sigma = np.sqrt(variances)
        return Histogram(
            hist_of(values, [sumw2] * len(values)), label="Fit", stat_errors=(sigma, sigma)
        )

    @pytest.mark.parametrize("sumw2", [0.0, 9.0])  # no sums of squared weights, or others
    def test_symmetric_errors_are_the_variances(self, sumw2: float) -> None:
        fit = self.given(W1, W1_VARIANCES, sumw2)
        result = goodness_of_fit(fit, weighted(W2, W2_VARIANCES))
        assert result.method == "WW"
        assert_chi2(result, ROOT_WW)
        mc = self.given(MC, MC_VARIANCES, sumw2)
        assert_chi2(goodness_of_fit(counts(DATA), mc), ROOT_UW)
        ks = goodness_of_fit(counts(DATA), mc, test="ks")
        assert ks.p_value == pytest.approx(ROOT_KS_UW[0], rel=1e-9)

    def test_counts_with_errors_of_their_own_are_weighted(self) -> None:
        sigma = np.sqrt([3.0, 5.0, 4.0])
        given = Histogram(hist_of([3, 5, 4]), label="Data", stat_errors=(sigma, sigma))
        assert goodness_of_fit(given, counts([4, 4, 5])).method == "UW"

    def test_a_stack_total_carries_them(self) -> None:
        half = [0.5 * v for v in W1]
        parts = [self.given(half, [0.5 * v for v in W1_VARIANCES]) for _ in range(2)]
        total = sum_histograms(parts)
        result = goodness_of_fit(total, weighted(W2, W2_VARIANCES))
        assert_chi2(result, ROOT_WW)

    @pytest.mark.parametrize("test", ["chi2", "ks"])
    def test_asymmetric_errors_have_no_variance(self, test: str) -> None:
        fit = Histogram(hist_of(W1), label="Fit", stat_errors=(np.full(6, 1.0), np.full(6, 2.0)))
        with pytest.raises(ValueError, match="'Fit' has asymmetric statistical errors"):
            goodness_of_fit(fit, weighted(W2, W2_VARIANCES), test=test)  # type: ignore[arg-type]
        # the absolute chi-square takes each side's error towards the other
        assert goodness_of_fit(fit, weighted(W2, W2_VARIANCES), test="chi2-absolute").ndf == 5

    def test_the_absolute_chi2_counts_the_entries_of_the_errors_it_takes(self) -> None:
        sigma = np.full(3, 3.0)  # 25 / 9 effective entries in the first bin, none in sumw2
        fit = Histogram(
            hist_of([5.0, 60.0, 70.0], [0.0] * 3), label="Fit", stat_errors=(sigma, sigma)
        )
        result = goodness_of_fit(fit, counts([100, 100, 100], "Data"), test="chi2-absolute")
        assert result.notes == ("'Fit' has a bin with fewer than 10 effective entries",)


class TestRequests:
    def test_unknown_test(self) -> None:
        with pytest.raises(ValueError, match="test must be one of"):
            goodness_of_fit(counts([1, 2]), counts([2, 1]), test="anderson")  # type: ignore[arg-type]

    def test_the_binning_must_agree(self) -> None:
        with pytest.raises(BinningError, match="identical bin edges"):
            goodness_of_fit(counts([1, 2]), counts([2, 1, 3]))

    def test_categories_have_no_order_to_accumulate(self) -> None:
        def categories(names: list[str], values: list[float]) -> Histogram:
            h = hist.Hist(hist.axis.StrCategory(names), storage=hist.storage.Weight())
            h.fill(names, weight=values)
            return Histogram(h, label="")

        a = categories(["e", "mu", "tau"], [4.0, 9.0, 2.0])
        b = categories(["e", "mu", "tau"], [5.0, 7.0, 3.0])
        assert goodness_of_fit(a, b).method == "WW"
        with pytest.raises(ValueError, match="categories have no order"):
            goodness_of_fit(a, b, test="ks")

    @pytest.mark.parametrize("test", ["chi2", "ks"])
    def test_an_empty_histogram_has_no_shape(self, test: str) -> None:
        with pytest.raises(ValueError, match=r"the second histogram .* is empty"):
            goodness_of_fit(counts([1, 2]), counts([0, 0]), test=test)  # type: ignore[arg-type]

    def test_bins_without_any_uncertainty_cannot_be_judged(self) -> None:
        a = weighted([1.0, 2.0, 3.0], [0.0, 0.5, 0.5], "A")
        b = weighted([2.0, 2.0, 2.0], [0.0, 0.5, 0.5], "B")
        with pytest.raises(ValueError, match=r"both have no uncertainty in bins \[0\]"):
            goodness_of_fit(a, b)
        exact = weighted([0.0, 4.0], [0.0, 0.0], "Model")
        with pytest.raises(ValueError, match="no uncertainties, and no content in bin 0"):
            goodness_of_fit(counts([3, 4]), exact)
        # ROOT's adjustment of the counts would never end
        negative = weighted([-1.0, 4.0], [0.0, 1.5], "Model")
        with pytest.raises(ValueError, match="negative content without uncertainty in bin 0"):
            goodness_of_fit(counts([3, 4]), negative)
        with pytest.raises(ValueError, match="negative total"):
            goodness_of_fit(counts([3, 4]), weighted([-6.0, 4.0], [2.0, 1.5]))

    def test_a_plain_hist_is_judged_by_its_contents(self) -> None:
        plain = goodness_of_fit(hist_of(COUNTS_A), hist_of(COUNTS_B))
        assert plain.method == "UU"
        assert (plain.label, plain.reference) == ("", "")
        assert_chi2(plain, ROOT_UU)


def _chi2_probability(chi2: float, ndf: int) -> float:
    """The upper tail of a chi-square of odd ``ndf``, in closed form."""
    x = chi2 / 2
    series = sum(x ** (k + 0.5) / math.gamma(k + 1.5) for k in range((ndf - 1) // 2))
    return math.erfc(math.sqrt(x)) + math.exp(-x) * series
