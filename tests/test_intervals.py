"""Confidence intervals: Poisson counts, binomial and Bayesian efficiencies, ROOT references."""

from __future__ import annotations

import math
import subprocess
import sys
from typing import Any

import hist
import numpy as np
import pytest

from rootfig.errors import (
    RootfigWarning,
)
from rootfig.histograms import (
    Cutflow,
    CutflowStep,
    Histogram,
    compare,
    poisson_interval,
    shape_covariance,
)
from rootfig.histograms.bayesian import Bayesian, bayesian_interval
from rootfig.histograms.binomial import (
    agresti_coull,
    clopper_pearson,
    normal_interval,
    wilson_interval,
)
from rootfig.histograms.intervals import ONE_SIGMA, count_problem, count_scale, poisson_errors
from rootfig.model import Sample
from test_histograms import _contents, _hist, _poisson, _symmetric

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
    # TGraphAsymmErrors::Divide gives the same with "", "cp", "wilson" and "n": its
    # frequentist options fall back to the normal approximation for weighted histograms
    "divide": (
        [(2.0, True), (2.0, True), (2.0, True), (1.0, False), (3.0, True)],
        (0.9, 0.7990049506163972, 1.0),
    ),
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
        two = clopper_pearson([3.0], [10.0], cl=math.erf(2.0 / math.sqrt(2.0)))
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
        low, high = poisson_interval(counts, math.erf(z / math.sqrt(2.0)))  # z standard deviations
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

    @pytest.mark.parametrize("cl", [0.0, 1.0, -0.5, 1.5, np.nan, np.inf, True])
    def test_cl_must_be_a_confidence_level(self, cl: float) -> None:
        with pytest.raises(ValueError, match="confidence level between 0 and 1"):
            poisson_interval([2.0], cl)

    def test_kpoisson2_is_a_95_percent_level(self) -> None:
        # ROOT 6.40, TH1::kPoisson2 for counts 0, 1 and 4: (down, up)
        root = [(0.0, 3.688879454113936), (0.9746821920157102, 4.571643390938899)]
        root.append((2.910134626373674, 6.241588675403699))
        low, high = poisson_interval([0.0, 1.0, 4.0], 0.95)
        np.testing.assert_allclose([0.0, 1.0, 4.0] - low, [d for d, _ in root], atol=1e-12)
        np.testing.assert_allclose(high - [0.0, 1.0, 4.0], [u for _, u in root], rtol=1e-12)

    def test_round_off_is_a_whole_count(self) -> None:
        np.testing.assert_array_equal(
            poisson_interval([2.0 + np.spacing(2.0)]), poisson_interval([2.0])
        )

    @pytest.mark.parametrize("count", [2.0 + 1e-12, 5e8 + 0.4, 5e8 + 0.25])
    def test_fractions_are_no_counts_at_any_magnitude(self, count: float) -> None:
        # a relative tolerance would let a fraction pass once counts are large enough
        with pytest.raises(ValueError, match="whole numbers"):
            poisson_interval([count])
        assert "not whole" in str(count_problem([count], [count]))
        assert count_problem([5e8], [5e8]) is None

    def test_equal_counts_agree(self) -> None:
        low, high = poisson_interval([3.0, 0.0, 3.0, 7.0, 0.0])
        assert (low[0], high[0]) == (low[2], high[2])
        assert high[1] == high[4]

    def test_scaled_counts(self) -> None:
        counts = np.array([0.0, 3.0, 0.0, 0.0, 5.0, 0.0])
        raw_down, raw_up = poisson_errors(counts, np.ones_like(counts))
        low, high = poisson_interval(counts)
        np.testing.assert_allclose(raw_down, counts - low)
        np.testing.assert_allclose(raw_up, high - counts)
        # a factor per bin, empty bins included: the interval of the count times it
        factor = np.array([0.125, 2.0, 1.0, 0.5, 4.0, 1.0])
        down, up = poisson_errors(counts * factor, factor)
        np.testing.assert_allclose(down, raw_down * factor, rtol=1e-12)
        np.testing.assert_allclose(up, raw_up * factor, rtol=1e-12)
        np.testing.assert_array_equal(poisson_errors(np.zeros(2), np.zeros(2)), 0.0)

    def test_count_scale_comes_from_the_record_of_one_count(self) -> None:
        # a cell of m merged counts of factor c holds (m c, m c**2); a cell the record
        # lacks borrows the nearest factor (ties: the lower cell's)
        ones, squares = [0.0, 2.0, 0.0, 0.0, 1.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.25, 0.0]
        np.testing.assert_allclose(count_scale(ones, squares), [2, 2, 2, 0.25, 0.25, 0.25])
        np.testing.assert_array_equal(count_scale([0.0, 0.0], [0.0, 0.0]), 0.0)  # scaled by 0

    def test_only_unit_weight_counts_are_counts(self) -> None:
        assert count_problem([0.0, 3.0, 7.0], [0.0, 3.0, 7.0]) is None
        assert "whole" in str(count_problem([0.0, 1.5], [0.0, 1.5]))
        # counts scaled by 2 sum like weights of 2, and weights [1, 1, 4] like two entries of
        # weight 3 (6 and 18): whole effective counts do not make the contents counts
        assert "scaled" in str(count_problem([2.0, 4.0], [4.0, 8.0]))
        assert "weighted" in str(count_problem([6.0], [18.0]))
        assert "weighted" in str(count_problem([0.0, 2.0], [2.0, 2.0]))  # weights that cancel
        assert "negative" in str(count_problem([-1.0, 2.0], [1.0, 2.0]))
        assert "non-finite" in str(count_problem([np.nan], [1.0]))
        # equal relative to both, as TMath::AreEqualRel: a variance of 1e-13 is no empty count
        assert "weighted" in str(count_problem([0.0], [1e-13]))
        assert count_problem([1e6], [1e6 * (1 + 1e-13)]) is None


class TestEfficiencyIntervals:
    def test_the_default_follows_the_weights_and_explicit_choices_are_checked(self) -> None:
        from rootfig.histograms import efficiency
        from rootfig.histograms.binomial import is_unweighted, resolve_interval

        assert is_unweighted(3.0, 3.0)
        assert is_unweighted(0.0, 0.0)
        assert not is_unweighted(6.0, 12.0)
        assert resolve_interval("auto", True) == "clopper-pearson"
        assert resolve_interval("auto", False) == "normal"
        assert resolve_interval("wilson", True) == "wilson"
        assert resolve_interval("wilson-effective", False) == "wilson-effective"
        # ROOT falls back to the normal approximation for methods of counts with weights
        # (with a warning); an explicit choice here never silently changes
        for method in ("clopper-pearson", "wilson"):
            with pytest.raises(ValueError, match=f"'{method}' needs unweighted"):
                efficiency(_hist([0.5], [2.0]), _hist([0.5, 0.5], [2.0, 2.0]), interval=method)
        with pytest.raises(ValueError, match="interval must be"):
            efficiency(_hist([0.5]), _hist([0.5]), interval="bayes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="interval must be"):
            Cutflow("s", (), interval="exact")  # type: ignore[arg-type]
        weighted = CutflowStep(label="", expression="", events=2, yield_=3.0, error=np.sqrt(5.0))
        for method in ("clopper-pearson", "wilson"):
            with pytest.raises(ValueError, match="needs unweighted"):
                Cutflow("w", (weighted,), interval=method)  # type: ignore[arg-type]

    def test_undefined_inputs(self) -> None:
        lower, upper = clopper_pearson([0.0, 3.0, -1.0, 2.0], [0.0, 2.0, 2.0, -1.0])
        assert np.isnan(lower).all()
        assert np.isnan(upper).all()
        lower, upper = normal_interval([1.0, 3.0], [0.0, 2.0], [1.0, 3.0], [1.0, 2.0])
        assert np.isnan(lower).all()
        assert np.isnan(upper).all()
        empty = clopper_pearson(np.zeros(0), np.zeros(0))
        assert empty[0].size == empty[1].size == 0

    def test_the_normal_approximation_refuses_variances_no_subset_has(self) -> None:
        # ROOT 6.40, TEfficiency: 8 (variance 100) of 10 (variance 10) gives a variance of
        # -0.536 and nan bounds; 2 (variance 4) of 10 (variance 3) is no subset either, but its
        # variance is positive and ROOT's interval is 0.2 +- 0.158745
        lower, upper = normal_interval([8.0, 2.0], [10.0, 10.0], [100.0, 4.0], [10.0, 3.0])
        assert np.isnan([lower[0], upper[0]]).all()  # not 0.8 +- 0
        np.testing.assert_allclose([0.2 - lower[1], upper[1] - 0.2], 0.15874507866384727)
        # every entry passes, the passed variance above the total's by round-off: the
        # variance is -1e-16, no width rather than nan
        vk, vn = 0.1 + 0.2 + 0.3, 0.3 + 0.2 + 0.1
        assert vk > vn
        lower, upper = normal_interval([6.0], [6.0], [vk], [vn])
        assert (lower[0], upper[0]) == (1.0, 1.0)
        from rootfig.histograms import efficiency

        passed = _hist([0.5] * 2, [5.0, 5.0])  # 10 of variance 50, from a total of variance 20
        total = _hist([0.5] * 5, [2.0] * 5)
        with pytest.warns(RootfigWarning, match="passed variance too large"):
            eff = efficiency(passed, total)
        assert eff.values[0] == 1.0
        assert np.isnan([eff.lower[0], eff.upper[0]]).all()

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

    def test_efficiency_interval_needs_a_resolved_method(self) -> None:
        from rootfig.histograms.binomial import efficiency_interval

        for unresolved in ("auto", "jeffreys"):
            with pytest.raises(ValueError, match="resolved method"):
                efficiency_interval(unresolved, [1.0], [2.0], [1.0], [2.0])  # type: ignore[arg-type]

    def test_a_negative_weight_on_either_side_has_no_binomial_interval(self) -> None:
        # a hand-made cut flow may flag the numerator alone; rf.cutflow never does
        def step(negative: bool, events: int) -> CutflowStep:
            error = float(np.sqrt(events))
            return CutflowStep("", "", events, float(events), error, negative_weights=negative)

        flow = Cutflow("c", (step(False, 10), step(True, 4), step(False, 2)))
        down, up = flow.efficiency_errors
        assert np.isnan([down[1], up[1]]).all()  # the numerator holds a negative weight
        assert np.isnan([down[2], up[2]]).all()  # measured against it
        absolute_down, _ = flow.absolute_efficiency_errors
        assert np.isfinite(absolute_down[2])  # 2 of the clean first step

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


# ROOT 6.40, TEfficiency::AgrestiCoull at one standard deviation of (passed, total):
# (lower, upper)
INTERVAL_CASES = [(0, 10), (3, 10), (10, 10), (1, 1), (0, 1), (7, 50), (2.5, 3.0)]


ROOT_AGRESTI_COULL = (
    [0.0, 0.17774673166969893, 0.8917409745720222, 0.44381378215210276, 0.0,
     0.09746586744522794, 0.5334936490538904],
    [0.10825902542797795, 0.45861690469393745, 1.0, 1.0, 0.5561862178478972,
     0.1966517796135956, 0.9665063509461096],
)  # fmt: skip


# ROOT 6.40, TEfficiency::BetaMean/BetaMode/BetaCentralInterval/BetaShortestInterval of the
# posterior of (alpha, beta) after k of n: (k, n, mean, mode, central, shortest)
ROOT_BAYESIAN = {
    (0.5, 0.5): [
        (0, 10, 0.045454545454545456, 0.0, (0.0019521139378112355, 0.0923340292577423),
         (0.0, 0.047591440043902505)),
        (3, 10, 0.3181818181818182, 0.2777777777777778, (0.17993225994165202, 0.45775062705439984),
         (0.1552304088634837, 0.42818453457827166)),
        (10, 10, 0.9545454545454546, 1.0, (0.9076659707422576, 0.9980478860621887),
         (0.9524085599560971, 1.0)),
        (0, 0, 0.5, 0.5, (0.060832954022950504, 0.9391670459770495),
         (0.060832954022950504, 0.9391670459770495)),
    ],
    (1.0, 1.0): [
        (0, 10, 0.08333333333333333, 0.0, (0.015582210290977427, 0.154109706155839),
         (0.0, 0.09909207989752448)),
        (3, 10, 0.3333333333333333, 0.3, (0.19887440293470748, 0.4687996314527011),
         (0.1781714807019396, 0.4446654484640067)),
        (10, 10, 0.9166666666666666, 1.0, (0.845890293844161, 0.9844177897090226),
         (0.9009079201024756, 1.0)),
    ],
    (2.0, 3.0): [
        (0, 10, 0.13333333333333333, 0.07692307692307693,
         (0.051148247014909366, 0.21710908928540418), (0.02360311532112108, 0.17290139632393697)),
        (3, 10, 0.3333333333333333, 0.3076923076923077, (0.21271220323344883, 0.4545914642896819),
         (0.19663122992027987, 0.43620806507697685)),
        (0, 0, 0.4, 0.3333333333333333, (0.18530110612745485, 0.61840242550396),
         (0.14385588952247427, 0.5698333054032234)),
    ],
}  # fmt: skip


class TestMoreEfficiencyIntervals:
    """ROOT's other TEfficiency methods: Agresti-Coull and Bayesian."""

    def test_agresti_coull_is_tefficiency(self) -> None:
        passed, total = (np.array(c, dtype=float) for c in zip(*INTERVAL_CASES, strict=True))
        lower, upper = agresti_coull(passed, total)
        np.testing.assert_allclose(lower, ROOT_AGRESTI_COULL[0], atol=1e-12)
        np.testing.assert_allclose(upper, ROOT_AGRESTI_COULL[1], atol=1e-12)
        lower, upper = agresti_coull([1.0, -1.0, 3.0], [0.0, 2.0, 2.0])  # no valid counts
        assert np.isnan(lower).all()
        assert np.isnan(upper).all()

    @pytest.mark.parametrize("prior", sorted(ROOT_BAYESIAN))
    def test_bayesian_intervals_are_tefficiency(self, prior: tuple[float, float]) -> None:
        rows = ROOT_BAYESIAN[prior]
        passed = [row[0] for row in rows]
        total = [row[1] for row in rows]
        mean, lower, upper = bayesian_interval(Bayesian(*prior), passed, total)
        np.testing.assert_allclose(mean, [row[2] for row in rows], rtol=1e-12)
        np.testing.assert_allclose(lower, [row[4][0] for row in rows], rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(upper, [row[4][1] for row in rows], rtol=1e-10)
        # the mode with its shortest interval, as TEfficiency::SetPosteriorMode
        mode, low, high = bayesian_interval(Bayesian(*prior, mode=True), passed, total)
        np.testing.assert_allclose(mode, [row[3] for row in rows], rtol=1e-12)
        # ROOT minimises the length to 1e-10; the shortest bounds here are exact
        np.testing.assert_allclose(low, [row[5][0] for row in rows], atol=2e-9)
        np.testing.assert_allclose(high, [row[5][1] for row in rows], atol=2e-9)

    @pytest.mark.parametrize(
        ("prior", "expected"),
        [
            (Bayesian(0.5, 0.5), (0.8278688524590164, 0.14888207817862908, 0.13897275550806953)),
            (Bayesian(0.5, 0.5, mode=True), (1.0, 0.21232310008693722, 0.0)),
            (Bayesian(1, 1), (0.7777777777777778, 0.1569838557407377, 0.150963364005811)),
            (
                Bayesian(1, 1, mode=True),
                (0.9000000000000001, 0.18635582386397764, 0.08412177278672028),
            ),
            (Bayesian(2, 3), (0.6380952380952382, 0.15502397938104, 0.15369230026958902)),
        ],
    )
    def test_weighted_bayesian_is_tefficiency(
        self, prior: Bayesian, expected: tuple[float, float, float]
    ) -> None:
        from rootfig.histograms import efficiency

        # ROOT 6.40, TEfficiency with weights 2, 2, 2 passing, 1 failing, 3 passing: the sums
        # scaled to the total's effective entries, sum w / sum w^2 = 10 / 22
        weights, passes = [2.0, 2.0, 2.0, 1.0, 3.0], [True, True, True, False, True]
        total = _hist([0.5] * 5, weights)
        passed = _hist([0.5] * 4, [w for w, ok in zip(weights, passes, strict=True) if ok])
        eff = efficiency(passed, total, interval=prior)
        down, up = eff.errors
        np.testing.assert_allclose([eff.values[0], down[0], up[0]], expected, atol=2e-9)

    def test_named_priors_and_their_checks(self) -> None:
        from rootfig.histograms import efficiency
        from rootfig.histograms.binomial import resolve_interval

        assert resolve_interval("jeffreys", False) == Bayesian(0.5, 0.5)
        assert resolve_interval("uniform", True) == Bayesian(1.0, 1.0)
        custom = Bayesian(2.0, 3.0, shortest=True)
        assert resolve_interval(custom, False) is custom
        passed, total = _hist([0.5] * 3), _hist([0.5] * 10)
        jeffreys = efficiency(passed, total, interval="jeffreys")
        assert jeffreys.values[0] == pytest.approx(0.3181818181818182)  # the posterior mean
        with pytest.raises(ValueError, match="'agresti-coull' needs unweighted"):
            efficiency(_hist([0.5], [2.0]), _hist([0.5, 0.5], [2.0, 2.0]), interval="agresti-coull")
        for removed in ("feldman-cousins", "mid-p"):  # ROOT's, left out of rootfig
            with pytest.raises(ValueError, match="interval must be"):
                efficiency(_hist([0.5]), _hist([0.5, 0.5]), interval=removed)  # type: ignore[arg-type]
        for bad in ({"alpha": 0.0}, {"beta": -1.0}, {"alpha": np.inf}, {"alpha": True}):
            with pytest.raises(ValueError, match="positive finite"):
                Bayesian(**bad)

    def test_the_mode_takes_the_shortest_interval_unless_told(self) -> None:
        from rootfig.histograms import efficiency

        assert Bayesian().shortest is False
        assert Bayesian(mode=True) == Bayesian(mode=True, shortest=True)
        assert Bayesian(mode=True, shortest=False).shortest is False
        # ROOT 6.40, TEfficiency (kBUniform) of 0 of 10 after SetPosteriorMode(): 0 +0.0991
        eff = efficiency(_hist([]), _hist([0.5] * 10), interval=Bayesian(1, 1, mode=True))
        down, up = eff.errors
        assert (eff.values[0], down[0]) == (0.0, 0.0)
        assert up[0] == pytest.approx(0.0990920798975024, abs=2e-9)

    def test_a_mode_outside_its_central_interval_has_one_sided_errors(self) -> None:
        from rootfig.histograms import efficiency

        # no passing entry: the posterior's mode is 0, below its central interval
        central = Bayesian(1, 1, mode=True, shortest=False)
        eff = efficiency(_hist([]), _hist([0.5] * 10), interval=central)
        down, up = eff.errors
        assert eff.values[0] == 0.0
        assert down[0] == 0.0  # not negative, where ROOT reports -0.0156
        assert up[0] == pytest.approx(0.154109706155839)

    def test_empty_bins_are_shown_as_root_does_with_e0(self) -> None:
        from rootfig.histograms import efficiency

        # ROOT 6.40, TGraphAsymmErrors::Divide of 2 of 3 and an empty bin: (value, down, up)
        passed, total = _hist([0.5, 0.5]), _hist([0.5, 0.5, 0.5])
        hidden = efficiency(passed, total)
        assert np.isnan([hidden.values[1], hidden.lower[1], hidden.upper[1]]).all()
        shown = efficiency(passed, total, show_empty=True)  # "e0"
        assert (shown.values[1], shown.lower[1], shown.upper[1]) == (0.0, 0.0, 1.0)
        normal = efficiency(passed, total, interval="normal", show_empty=True)  # "e0 n"
        assert (normal.values[1], normal.lower[1], normal.upper[1]) == (0.0, 0.0, 1.0)
        for prior, bins in (
            ("uniform", [(0.6, 0.21840242550392025, 0.21469889387251606), (0.5, 0.3413447460685)]),
            (
                "jeffreys",
                [(0.625, 0.2406619279819589, 0.23326735972005685), (0.5, 0.4391670459770172)],
            ),
        ):
            eff = efficiency(passed, total, interval=prior, show_empty=True)  # "b(a,b) e0"
            down, up = eff.errors
            np.testing.assert_allclose([eff.values[0], down[0], up[0]], bins[0], rtol=1e-10)
            np.testing.assert_allclose([eff.values[1], down[1], up[1]], [*bins[1], bins[1][1]])

    def test_weights_that_cancel_are_no_empty_bin(self) -> None:
        from rootfig.histograms import efficiency

        # bin 0: weights +1 and -1, a total of 0 with a variance of 2: entries whose weights
        # cancel, so no efficiency (ROOT 6.40 has none either: TEfficiency 0 with nan errors,
        # Divide "e0" 0 +- 0); bin 2: no entries at all, the only empty bin
        total = _hist([0.5, 0.5, 1.5, 1.5], [1.0, -1.0, 1.0, 1.0])
        passed = _hist([1.5], [1.0])
        for interval in ("auto", "uniform", "wilson-effective"):
            eff = efficiency(passed, total, interval=interval, show_empty=True)  # type: ignore[arg-type]
            assert np.isnan([eff.values[0], eff.lower[0], eff.upper[0]]).all()
        normal = efficiency(passed, total, show_empty=True)  # weighted: see below
        assert (normal.values[2], normal.lower[2], normal.upper[2]) == (0.0, 0.0, 0.0)

    def test_empty_bins_of_weighted_histograms_are_shown_as_root_does(self) -> None:
        from rootfig.histograms import efficiency

        # ROOT 6.40, TGraphAsymmErrors::Divide(..., "e0") of weights 2, 2 passing and 1 failing,
        # beside empty bins: the normal approximation ("", "n", "cp" and "w" alike for weighted
        # histograms) draws an empty bin at 0 +- 0, a Bayesian interval draws no point there
        total, passed = _hist([1.5] * 3, [2.0, 2.0, 1.0]), _hist([1.5] * 2, [2.0, 2.0])
        normal = efficiency(passed, total, show_empty=True)
        assert (normal.values[0], normal.lower[0], normal.upper[0]) == (0.0, 0.0, 0.0)
        np.testing.assert_allclose(
            [normal.values[1], *(side[1] for side in normal.errors)], [0.8, 0.196, 0.196], atol=5e-5
        )
        bayes = efficiency(passed, total, interval="uniform", show_empty=True)
        assert np.isnan([bayes.values[0], bayes.lower[0], bayes.upper[0]]).all()
        np.testing.assert_allclose(
            [bayes.values[1], *(side[1] for side in bayes.errors)],
            [0.6744, 0.2121, 0.2049],
            atol=5e-5,
        )
        # the Wilson interval of effective entries is rootfig's own: nothing is known, [0, 1]
        effective = efficiency(passed, total, interval="wilson-effective", show_empty=True)
        assert (effective.values[0], effective.lower[0], effective.upper[0]) == (0.0, 0.0, 1.0)

    def test_a_confidence_level(self) -> None:
        from rootfig.histograms import efficiency

        eff = efficiency(_hist([0.5] * 3), _hist([0.5] * 10), cl=0.95)
        np.testing.assert_allclose(
            [eff.lower[0], eff.upper[0]], np.ravel(clopper_pearson([3.0], [10.0], 0.95))
        )


# every k of n for a few totals, where the named intervals must hold their invariants
_K, _N = (
    np.array(side, dtype=float)
    for side in zip(*((k, n) for n in (1, 2, 7, 30, 250) for k in range(n + 1)), strict=True)
)


_BINOMIAL = {
    "clopper-pearson": clopper_pearson,
    "wilson": lambda k, n, cl: wilson_interval(k, n, n, cl),
    "agresti-coull": agresti_coull,
    "normal": lambda k, n, cl: normal_interval(k, n, k, n, cl),
    "jeffreys": lambda k, n, cl: bayesian_interval(Bayesian(0.5, 0.5), k, n, cl=cl)[1:],
    "uniform": lambda k, n, cl: bayesian_interval(Bayesian(1.0, 1.0), k, n, cl=cl)[1:],
}


class TestIntervalInvariants:
    """What every interval must satisfy, on grids of counts and confidence levels."""

    @pytest.mark.parametrize("cl", [ONE_SIGMA, 0.9, 0.95, 0.99])
    def test_poisson(self, cl: float) -> None:
        n = np.array([0.0, 1.0, 2.0, 5.0, 17.0, 100.0, 1e4, 1e7])
        low, high = poisson_interval(n, cl)
        assert np.all((low >= 0) & (low <= n) & (n <= high))
        assert low[0] == 0.0
        assert high[0] > 0.0
        wider = poisson_interval(n, (1.0 + cl) / 2)
        assert np.all((wider[0] <= low) & (wider[1] >= high))

    @pytest.mark.parametrize("cl", [ONE_SIGMA, 0.95])
    @pytest.mark.parametrize("method", sorted(_BINOMIAL))
    def test_binomial(self, method: str, cl: float) -> None:
        interval = _BINOMIAL[method]
        lower, upper = interval(_K, _N, cl)
        assert np.all((lower >= 0.0) & (lower <= upper) & (upper <= 1.0))
        if method not in ("jeffreys", "uniform"):  # a posterior's central interval need not
            assert np.all((lower <= _K / _N) & (upper >= _K / _N))
        # failures are passes of the complement, for these symmetric methods and priors
        mirrored = interval(_N - _K, _N, cl)
        np.testing.assert_allclose(lower, 1.0 - mirrored[1], atol=1e-12)
        np.testing.assert_allclose(upper, 1.0 - mirrored[0], atol=1e-12)
        wider = interval(_K, _N, (1.0 + cl) / 2)
        assert np.all((wider[0] <= lower + 1e-15) & (wider[1] >= upper - 1e-15))

    @pytest.mark.parametrize("spec", [True, 3.0, "density"])
    def test_a_normalised_shape_keeps_its_total(self, spec: Any) -> None:
        rng = np.random.default_rng(3)
        h = hist.Hist(
            hist.axis.Variable([0.0, 1.0, 1.5, 4.0, 4.2, 6.0]), storage=hist.storage.Weight()
        )
        h.fill(rng.uniform(0, 6, 400), weight=rng.uniform(0.2, 2.0, 400))
        covariance = shape_covariance(Histogram(h, label="W"), spec)
        # the normalised bins (times their widths for a density) sum to a constant
        sizes = np.diff(h.axes[0].edges) if spec == "density" else np.ones(5)
        np.testing.assert_allclose(covariance @ sizes, 0.0, atol=1e-15)
        np.testing.assert_allclose(covariance, covariance.T)
        assert np.all(np.linalg.eigvalsh(covariance) > -1e-15)


class TestUpstreamIntervals:
    """The kernels agree with ``hist.intervals``, which rootfig does not import at run time.

    ``hist.intervals`` imports ``scipy.stats`` (0.4 s or more); rootfig computes the same
    quantiles from ``scipy.special`` and keeps its own validation of counts.
    """

    @pytest.mark.parametrize("cl", [ONE_SIGMA, 0.95])
    def test_poisson(self, cl: float) -> None:
        from hist.intervals import poisson_interval as upstream

        counts = np.r_[np.arange(200.0), 1e4, 1e7]
        np.testing.assert_allclose(
            poisson_interval(counts, cl), upstream(counts, coverage=cl), rtol=1e-12
        )

    @pytest.mark.parametrize("cl", [ONE_SIGMA, 0.95])
    def test_clopper_pearson(self, cl: float) -> None:
        from hist.intervals import clopper_pearson_interval as upstream

        k, n = np.r_[_K, 3.0, 1e7 - 3.0], np.r_[_N, 1e7, 1e7]
        np.testing.assert_allclose(
            clopper_pearson(k, n, cl), upstream(k, n, coverage=cl), rtol=1e-12
        )

    def test_poisson_ratio(self) -> None:
        from hist.intervals import ratio_uncertainty as upstream

        n = np.array([0.0, 1.0, 3.0, 7.0, 40.0, 0.0])
        d = np.array([1.0, 1.0, 5.0, 2.0, 35.0, 0.0])
        a = Histogram(_contents(n.tolist(), n.tolist()), label="A")
        b = Histogram(_contents(d.tolist(), d.tolist()), label="B")
        errors = compare(a, b, uncertainty="poisson-ratio").errors
        expected = upstream(n[:-1], d[:-1], uncertainty_type="poisson-ratio")
        np.testing.assert_allclose(np.asarray(errors)[:, :-1], expected, rtol=1e-12)
        assert np.isnan(np.asarray(errors)[:, -1]).all()  # upstream's inf and nan: no ratio

    def test_rootfig_does_not_import_scipy_stats(self) -> None:
        code = """
import sys
from rootfig.histograms import poisson_interval
from rootfig.histograms.bayesian import Bayesian, bayesian_interval
from rootfig.histograms.binomial import clopper_pearson
poisson_interval([0.0, 3.0], 0.95)
clopper_pearson([1.0], [4.0])
bayesian_interval(Bayesian(2.0, 2.0, mode=True), [3.0], [5.0])
assert "scipy.special" in sys.modules and "scipy.stats" not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)
