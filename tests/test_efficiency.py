"""Efficiencies, profiles and cut flows built from histograms and columns."""

from __future__ import annotations

import warnings
from typing import Any

import awkward as ak
import hist
import numpy as np
import pytest

from rootfig.errors import (
    BinningError,
    RootfigWarning,
    SelectionError,
)
from rootfig.histograms.bayesian import Bayesian, bayesian_interval
from rootfig.histograms.binomial import (
    agresti_coull,
    clopper_pearson,
    normal_interval,
    wilson_interval,
)
from rootfig.model import Cut, Sample
from test_histograms import _contents, _hist


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

    def test_effective_entries_need_a_variance(self) -> None:
        from rootfig.histograms import efficiency

        # a positive total without a variance comes from no weights: no n_eff to invent
        lower, upper = wilson_interval([5.0, 5.0], [10.0, 10.0], [0.0, 10.0])
        assert np.isnan([lower[0], upper[0]]).all()
        np.testing.assert_allclose(
            [lower[1], upper[1]], np.ravel(wilson_interval([5.0], [10.0], [10.0]))
        )
        with pytest.warns(RootfigWarning, match="without a variance"):
            eff = efficiency(
                _contents([5.0], [0.0]), _contents([10.0], [0.0]), interval="wilson-effective"
            )
        assert eff.values[0] == 0.5
        assert np.isnan([eff.lower[0], eff.upper[0]]).all()

    def test_weights_use_effective_entries(self) -> None:
        from rootfig.histograms import efficiency

        unweighted = efficiency(_hist([0.5] * 5), _hist([0.5] * 10), interval="wilson")
        weighted = efficiency(
            _hist([0.5] * 5, [2.0] * 5),
            _hist([0.5] * 10, [2.0] * 10),
            interval="wilson-effective",
        )
        assert weighted.values[0] == pytest.approx(unweighted.values[0])
        assert weighted.lower[0] == pytest.approx(unweighted.lower[0])  # same n_eff = 10
        # all of weights 1, 2 and 3 pass: n_eff = 36 / 14, and the interval keeps a width,
        # where ROOT's weighted normal approximation (the default) gives [1, 1]
        all_pass = _hist([0.5] * 3, [1.0, 2.0, 3.0])
        n_eff = 36 / 14
        wilson = efficiency(all_pass, all_pass, interval="wilson-effective")
        assert wilson.lower[0] == pytest.approx(n_eff / (n_eff + 1))
        counts = efficiency(_hist([0.5] * 5), _hist([0.5] * 10), interval="wilson-effective")
        np.testing.assert_array_equal(counts.lower, unweighted.lower)  # the same for counts
        assert efficiency(all_pass, all_pass).lower[0] == 1.0
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
        assert flow.interval == "auto"  # Clopper-Pearson: unweighted events, as TEfficiency
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

    def test_weights_of_zero_and_one_are_unweighted_as_for_tefficiency(self) -> None:
        from rootfig.histograms import cutflow

        # ROOT 6.40: TH1D totals filled with weights 1, 0, 1, 0, ... for n = 0..19 and the
        # passing n >= 8; TEfficiency counts them as unweighted and takes Clopper-Pearson of
        # the contents, 6 of 10 (not of the 12 events of which 6 have weight 0)
        sample = Sample({"n": np.arange(20.0), "w": np.tile([1.0, 0.0], 10)}, weight="w")
        flow = cutflow(sample, ["n >= 8"])
        assert [step.events for step in flow.steps] == [20, 12]
        down, up = flow.efficiency_errors
        root = (0.39540314576424296, 0.7800092538943881)
        np.testing.assert_allclose([0.6 - down[1], 0.6 + up[1]], root, rtol=1e-11)

    def test_auto_is_decided_per_pair_of_steps_as_for_tefficiency(self) -> None:
        from rootfig.histograms import cutflow

        # ROOT 6.40, TEfficiency of TH1D steps filled with weights 2, 2, 1, 1, 1, 1: the first
        # cut removes both weights of 2, so 2/1 is unweighted (Clopper-Pearson) while 1/0 and
        # 2/0 are weighted (normal): (value, lower, upper)
        root = {
            (1, 0): (0.5, 0.2834936490539288, 0.7165063509460712),
            (2, 1): (0.5, 0.18530110612748396, 0.814698893872516),
            (2, 0): (0.25, 0.08464054305849245, 0.41535945694150755),
        }
        weights = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        sample = Sample({"n": np.arange(6.0), "w": weights}, weight="w")
        flow = cutflow(sample, ["n >= 2", "n >= 4"])
        assert flow.interval == "auto"
        down, up = flow.efficiency_errors
        absolute_down, absolute_up = flow.absolute_efficiency_errors
        errors = {
            (1, 0): (down[1], up[1]),
            (2, 1): (down[2], up[2]),
            (2, 0): (absolute_down[2], absolute_up[2]),
        }
        for pair, (d, u) in errors.items():
            value, lower, upper = root[pair]
            np.testing.assert_allclose([value - d, value + u], [lower, upper], rtol=1e-11)
        # an explicit method of counts still needs every step unweighted
        with pytest.raises(ValueError, match="needs unweighted"):
            cutflow(sample, ["n >= 2", "n >= 4"], interval="clopper-pearson")

    def test_scale_keeps_events_unweighted_and_weights_do_not(self) -> None:
        from rootfig.histograms import cutflow

        cuts = ["n >= 40", "n >= 70"]
        plain = cutflow(Sample({"n": np.arange(100.0)}), cuts)
        scaled = cutflow(Sample({"n": np.arange(100.0)}, scale=2.5), cuts)
        # a factor the efficiency cancels: still Clopper-Pearson
        for got, want in zip(scaled.efficiency_errors, plain.efficiency_errors, strict=True):
            np.testing.assert_allclose(got, want)
        lower, upper = clopper_pearson([60.0], [100.0])
        down, up = scaled.efficiency_errors
        np.testing.assert_allclose([down[1], up[1]], [0.6 - lower[0], upper[0] - 0.6])
        # an event weight makes the events weighted for ROOT, even one shared by all
        weighted = cutflow(Sample({"n": np.arange(100.0)}, weight="2.5"), cuts)
        down, up = weighted.efficiency_errors
        lower, upper = normal_interval([60.0], [100.0], [60.0], [100.0])
        np.testing.assert_allclose([down[1], up[1]], [0.6 - lower[0], upper[0] - 0.6])
        for method in ("clopper-pearson", "wilson"):
            with pytest.raises(ValueError, match="needs unweighted"):
                cutflow(Sample({"n": np.arange(100.0)}, weight="2.5"), cuts, interval=method)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="interval must be"):
            cutflow(Sample({"n": np.arange(100.0)}), cuts, interval="bayes")  # type: ignore[arg-type]

    def test_weighted_events_use_the_effective_entries(self) -> None:
        from rootfig.histograms import cutflow

        weights = np.where(np.arange(100) % 2, 3.0, 1.0)  # 50 of weight 1, 50 of weight 3
        sample = Sample({"n": np.arange(100.0), "w": weights}, weight="w")
        flow = cutflow(sample, ["n >= 40"], interval="wilson-effective")
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
        flow = cutflow(sample, ["n >= 1", "n >= 5"], interval="wilson-effective")
        assert [step.negative_weights for step in flow.steps] == [True, False, False]
        np.testing.assert_allclose(flow.efficiencies, [1.0, 9 / 8, 5 / 9])  # still reported
        down, up = flow.efficiency_errors
        assert np.isnan([down[1], up[1]]).all()  # measured against a signed step
        np.testing.assert_allclose([down[2], up[2]], self._wilson(5, 9, 9))
        absolute_down, _ = flow.absolute_efficiency_errors
        assert np.isnan(absolute_down[1:]).all()  # every step measured against the first
        assert absolute_down[0] == 0.0
        # the default: the normal approximation (ROOT's for weighted events) holds for signed
        # weights too, so only 9 / 8, outside [0, 1], has none; step 2 against step 1, both
        # of weight-1 events only, is Clopper-Pearson, as TEfficiency of those two histograms
        default = cutflow(sample, ["n >= 1", "n >= 5"])
        down, up = default.efficiency_errors
        assert np.isnan([down[1], up[1]]).all()  # 9 / 8
        lower, upper = clopper_pearson([5.0], [9.0])
        np.testing.assert_allclose([down[2], up[2]], [5 / 9 - lower[0], upper[0] - 5 / 9])
        absolute_down, absolute_up = default.absolute_efficiency_errors
        lower, upper = normal_interval([5.0], [8.0], [5.0], [10.0])  # against the signed step
        np.testing.assert_allclose(
            [absolute_down[2], absolute_up[2]], [5 / 8 - lower[0], upper[0] - 5 / 8]
        )

    def test_a_negative_weight_that_survives_the_cuts(self) -> None:
        from rootfig.histograms import cutflow

        # the numerators hold the negative weight, and so do the steps they are measured
        # against, whose flag alone decides: every step keeps a subset of the one before
        weights = np.r_[np.ones(9), -1.0]
        sample = Sample({"n": np.arange(10.0), "w": weights}, weight="w")
        flow = cutflow(sample, ["n >= 1", "n >= 5"], interval="wilson-effective")
        assert [step.negative_weights for step in flow.steps] == [True, True, True]
        np.testing.assert_allclose(flow.efficiencies, [1.0, 7 / 8, 3 / 7])
        for down, up in (flow.efficiency_errors, flow.absolute_efficiency_errors):
            assert np.isnan(down[1:]).all()
            assert np.isnan(up[1:]).all()
        default = cutflow(sample, ["n >= 1", "n >= 5"])  # normal: holds for signed weights
        assert np.isfinite(default.efficiency_errors[0][1:]).all()

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
            eff = efficiency(*self._weighted([-0.5], [1.0, -2.0]), interval="wilson-effective")
        assert eff.values[0] == pytest.approx(0.5)
        assert np.isnan(eff.lower[0])
        assert np.isnan(eff.upper[0])
        with pytest.warns(RootfigWarning, match="negative weights"):
            eff = efficiency(*self._weighted([-2.0], [1.0, -2.0]), interval="wilson-effective")
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
            eff = efficiency(*self._weighted([1.0], [1.0, 1.0, -0.5]), interval="wilson-effective")
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
            unknown = efficiency(*hists, interval="wilson-effective")
            normal = efficiency(*hists, negative_weights=[True])  # needs no flag
        assert np.isfinite(unknown.lower[0])  # nothing in the sums says so
        assert np.isfinite(normal.lower[0])
        with pytest.warns(RootfigWarning, match="negative weights"):
            flagged = efficiency(*hists, negative_weights=[True], interval="wilson-effective")
        assert flagged.values[0] == pytest.approx(1.9 / 2.9)
        assert np.isnan(flagged.lower[0])
        assert np.isnan(flagged.upper[0])

    def test_positive_weights_keep_their_interval(self) -> None:
        from rootfig.histograms import efficiency

        # positive weights of different sizes, all passing or all failing included
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hists = self._weighted([0.5, 3.0], [0.5, 3.0, 1.0, 0.2])
            eff = efficiency(*hists, interval="wilson-effective")
            ends = [
                efficiency(*self._weighted(w, [0.5, 3.0]), interval="wilson-effective")
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

    @pytest.mark.parametrize("cl", [0.0, 1.0, -1.0, np.inf, np.nan])
    def test_cl_is_validated(self, cl: float) -> None:
        from rootfig.histograms import efficiency

        with pytest.raises(ValueError, match="confidence level"):
            efficiency(*self._hists(1, 2), cl=cl)
        # the interval functions check it themselves
        for interval in (
            lambda: clopper_pearson([1.0], [2.0], cl),
            lambda: normal_interval([1.0], [2.0], [1.0], [2.0], cl),
            lambda: wilson_interval([1.0], [2.0], [2.0], cl),
            lambda: agresti_coull([1.0], [2.0], cl),
            lambda: bayesian_interval(Bayesian(), [1.0], [2.0], cl=cl),
        ):
            with pytest.raises(ValueError, match="confidence level"):
                interval()


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


class TestCutflowIntervals:
    def test_every_frequentist_method_and_a_level(self) -> None:
        from rootfig.histograms import cutflow

        sample = Sample({"n": np.arange(10.0)})
        for method, function in (
            ("agresti-coull", agresti_coull),
            ("clopper-pearson", clopper_pearson),
        ):
            flow = cutflow(sample, ["n >= 7"], interval=method, cl=0.95)  # type: ignore[arg-type]
            assert flow.cl == 0.95
            lower, upper = function([3.0], [10.0], 0.95)
            down, up = flow.efficiency_errors
            np.testing.assert_allclose([0.3 - down[1], 0.3 + up[1]], [lower[0], upper[0]])

    def test_bayesian_intervals_are_refused(self) -> None:
        from rootfig.histograms import cutflow

        for prior in ("jeffreys", Bayesian(2, 3)):
            with pytest.raises(ValueError, match=r"use rf\.efficiency for Bayesian"):
                cutflow(Sample({"n": np.arange(10.0)}), ["n >= 7"], interval=prior)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="confidence level"):
            cutflow(Sample({"n": np.arange(10.0)}), ["n >= 7"], cl=2.0)
