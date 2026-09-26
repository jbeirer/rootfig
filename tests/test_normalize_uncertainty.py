"""normalize(..., uncertainty=): what normalising does to the statistical uncertainty."""

from __future__ import annotations

import warnings
from typing import Any

import hist
import numpy as np
import pytest

from helpers import hist_of
from rootfig.histograms import (
    Histogram,
    compare,
    normalize,
    shape_covariance,
    uncertainty,
)
from rootfig.histograms.binomial import (
    clopper_pearson,
)


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


class TestShapeUncertainty:
    """normalize(..., uncertainty="shape"): the own total fluctuates with the bins."""

    def test_first_order_propagation(self) -> None:
        h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
        h.fill([0.5, 1.5, 1.5, 2.5, 5.0], weight=[1.0, 2.0, 0.5, 3.0, 1.5])
        histogram = Histogram(h, label="W")
        x, v = histogram.values(), histogram.variances()
        total = x.sum()
        jacobian = np.eye(3) / total - np.outer(x, np.ones(3)) / total**2
        expected = jacobian @ np.diag(v) @ jacobian.T
        np.testing.assert_allclose(shape_covariance(histogram), expected)
        np.testing.assert_allclose(shape_covariance(histogram).sum(axis=1), 0.0, atol=1e-15)
        shape = normalize(histogram, True, uncertainty="shape")
        np.testing.assert_allclose(shape.variances(), np.diag(expected))
        # the overflow (1.5 of weight 1.5) is divided by the total without entering it
        overflow = shape.variances(flow=True)[-1]
        assert overflow == pytest.approx(1.5**2 / total**2 + 1.5**2 * v.sum() / total**4)
        density = normalize(histogram, "density", uncertainty="shape")
        np.testing.assert_allclose(density.variances(), np.diag(expected))  # widths of 1
        np.testing.assert_allclose(shape_covariance(histogram, 4.0), 16 * expected)

    def test_counts_are_binomial_fractions(self) -> None:
        counts = Histogram(hist_of([1.0, 3.0, 6.0]), label="Data", is_data=True)
        plain = normalize(counts, True, uncertainty="shape")
        fraction = np.array([0.1, 0.3, 0.6])
        np.testing.assert_allclose(plain.variances(), fraction * (1 - fraction) / 10)
        exact = normalize(counts.replace(poisson=True), True, uncertainty="shape")
        assert not exact.poisson
        # the hist carries the shape's first-order variances, the errors the exact interval
        np.testing.assert_allclose(exact.variances(), fraction * (1 - fraction) / 10)
        lower, upper = clopper_pearson([1.0, 3.0, 6.0], 10.0)
        np.testing.assert_allclose(exact.errors()[0], fraction - lower)
        np.testing.assert_allclose(exact.errors()[1], upper - fraction)
        edges = hist.Hist(hist.axis.Variable([0, 1, 3, 4]), storage=hist.storage.Weight())
        edges.fill(np.repeat([0.5, 2.0, 3.5], [1, 3, 6]))
        density = normalize(Histogram(edges, "D", poisson=True), "density", uncertainty="shape")
        np.testing.assert_allclose(density.errors()[1], (upper - fraction) / [1, 2, 1])

    def test_it_needs_a_normalisation_to_the_own_total(self) -> None:
        counts = Histogram(hist_of([1.0, 3.0]), label="C")
        for spec in ("width", None, False):
            with pytest.raises(ValueError, match="own total"):
                normalize(counts, spec, uncertainty="shape")
        with pytest.raises(ValueError, match="'scale' or 'shape'"):
            normalize(counts, True, uncertainty="exact")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="own total"):
            shape_covariance(counts, "width")


class TestShapeEdges:
    def test_given_errors_and_empty_counts(self) -> None:
        # the total enters every bin with the opposite sign, so a bin's lower error takes the
        # others' upper ones: no side-by-side propagation, refused like shape_covariance
        given = Histogram(
            hist_of([1.0, 3.0], [1.0, 3.0]), label="G", stat_errors=([0.5, 1.0], [1.0, 2.0])
        )
        with pytest.raises(ValueError, match=r"stat_errors.*normalize_uncertainty='scale'"):
            normalize(given, True, uncertainty="shape")
        np.testing.assert_allclose(normalize(given, True).errors()[1], [0.25, 0.5])  # scale
        # counts only in the overflow: no visible total, the first-order spread instead
        h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        h.fill([0.5, 5.0, 5.0])
        counts = Histogram(h, label="C", poisson=True)
        visible = normalize(counts, True, uncertainty="shape")
        assert not visible.poisson
        assert np.all(np.isfinite(visible.errors()[1]))

    def test_a_covariance_needs_a_shape(self) -> None:
        # no visible total to divide by: a clear error, not nan with divide-by-zero warnings
        for values, variances in (
            ([0.0, 0.0], [0.0, 0.0]),  # empty
            ([2.0, -2.0], [4.0, 4.0]),  # signed weights that cancel
            ([np.inf, 1.0], [1.0, 1.0]),  # not finite
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with pytest.raises(ValueError, match="no shape to normalise to"):
                    shape_covariance(Histogram(hist_of(values, variances), label="H"))
                with pytest.raises(ValueError, match="no shape to normalise to"):
                    shape_covariance(hist_of(values, variances), "density")
        given = Histogram(
            hist_of([1.0, 3.0], [1.0, 3.0]), label="G", stat_errors=([0.5, 1.0], [1.0, 2.0])
        )
        with pytest.raises(ValueError, match="stat_errors"):
            shape_covariance(given)
