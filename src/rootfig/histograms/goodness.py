"""Goodness of fit of two histograms: ROOT's chi-square and Kolmogorov tests, and more.

ROOT's tests compare shapes; the absolute chi-square tests the normalisation
too and takes systematic uncertainties as a covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from rootfig._storage import is_category
from rootfig._typing import FloatArray, Hist
from rootfig.errors import BinningError
from rootfig.histograms.build import Histogram, compatible_binning
from rootfig.histograms.comparison import _propagated, _Side, _source_shifts

__all__ = ["GOODNESS_OF_FIT_TESTS", "GoodnessOfFit", "GoodnessOfFitTest", "goodness_of_fit"]

GoodnessOfFitTest: TypeAlias = Literal["chi2", "chi2-absolute", "ks"]
"""The tests :func:`goodness_of_fit` runs."""

GOODNESS_OF_FIT_TESTS: tuple[GoodnessOfFitTest, ...] = ("chi2", "chi2-absolute", "ks")
"""Every :data:`GoodnessOfFitTest`, for checking strings at runtime."""


@dataclass(frozen=True)
class GoodnessOfFit:
    """How well two histograms agree: the result of :func:`goodness_of_fit`.

    Attributes
    ----------
    test
        The :data:`GoodnessOfFitTest` run.
    method
        ``"UU"``, ``"UW"`` or ``"WW"`` for ``"chi2"`` (ROOT's names for
        comparing unweighted counts and weighted histograms), ``"absolute"``
        for ``"chi2-absolute"`` and ``"ks"`` for ``"ks"``.
    statistic
        The chi-square, or for ``"ks"`` the largest distance between the two
        cumulative shapes.
    ndf
        The degrees of freedom; ``None`` for ``"ks"``.
    p_value
        The probability of a statistic at least as large for compatible
        histograms; ``nan`` without degrees of freedom.
    label, reference
        The labels of the two histograms, in the order given.
    bins
        Which bins entered the test (bins empty on both sides do not).
    notes
        Conditions that make the p-value unreliable, as ROOT reports them:
        bins with less than one event or fewer than ten effective entries.
    systematics
        The systematic sources in the covariance of ``"chi2-absolute"``.
    """

    test: GoodnessOfFitTest
    method: str
    statistic: float
    ndf: int | None
    p_value: float
    label: str
    reference: str
    bins: np.ndarray
    notes: tuple[str, ...] = ()
    systematics: tuple[str, ...] = ()

    @property
    def chi2_ndf(self) -> float | None:
        """The chi-square per degree of freedom; ``None`` for ``"ks"``, ``nan`` without any."""
        if self.ndf is None:
            return None
        return self.statistic / self.ndf if self.ndf > 0 else math.nan


def goodness_of_fit(
    a: Hist | Histogram, b: Hist | Histogram, *, test: GoodnessOfFitTest = "chi2"
) -> GoodnessOfFit:
    """Test whether two one-dimensional histograms agree, from their own uncertainties.

    * ``"chi2"`` is ROOT's ``TH1::Chi2Test``, a test of homogeneity: whether
      both are drawn from one shape, whatever their normalisations (``ndf`` is
      the number of bins less one). Histograms known to hold counts
      (:meth:`Histogram.counts() <rootfig.histograms.Histogram.counts>`, scaled
      counts included, as ROOT's ``"UU NORM"``) enter as Poisson counts, every
      other one as weighted, so a weighted histogram is never taken for
      counts: counts with counts is ROOT's ``"UU"``, counts with a weighted
      histogram ``"UW"`` (in either order) and two weighted histograms
      ``"WW"``. Statistical only. As in ROOT, an empty bin of the weighted
      histogram of ``"UW"`` takes the variance ``sum(w^2) / sum(w)``, which
      does not scale like a variance, so there the result depends on that
      histogram's normalisation. A histogram with ``stat_errors`` is weighted
      and enters with their squares, whatever counts it holds.
    * ``"chi2-absolute"`` also tests the normalisation: ``r C^-1 r`` with
      ``r = a - b`` and ``C`` the statistical variances of ``r``, each side's
      error taken towards the other histogram as in a pull, plus one matrix
      per systematic source, fully correlated across bins (a source shared by
      both histograms moves them together, as in :func:`compare`); ``ndf`` is
      the number of bins. A Gaussian approximation, poor for bins with few
      entries.
    * ``"ks"`` is ROOT's ``TH1::KolmogorovTest``: the largest distance between
      the cumulative shapes, its probability from the effective entries of
      both. A histogram without uncertainties is compared as a function. For
      binned data the p-value is biased high, the less so the finer the
      binning relative to the features compared (ROOT's NOTE 3). Categories
      have no order to accumulate along, so it refuses a category axis.

    ``"chi2"`` and ``"ks"`` need one variance per bin: the sum of squared
    weights, or the square of symmetric ``stat_errors``; asymmetric
    ``stat_errors`` are refused, as no variance describes them.

    Bins empty on both histograms do not enter the chi-squares; flow bins do
    not enter at all. For ``"chi2-absolute"`` a bin is empty only without an
    uncertainty too: one whose weights cancel to zero still enters, with its
    sum of squared weights.

    Raises
    ------
    BinningError
        If the histograms do not share one one-dimensional binning.
    ValueError
        For an unknown ``test``; for ``"chi2"`` and ``"ks"``, for a histogram
        whose contents sum to zero or with asymmetric ``stat_errors``, or a
        bin they cannot test (no uncertainty on either side); for ``"ks"``, for
        a category axis; for ``"chi2-absolute"``, for a singular covariance
        (bins without statistical uncertainty that the systematic sources do
        not make up for).
    """
    if test not in GOODNESS_OF_FIT_TESTS:
        msg = f"test must be one of {GOODNESS_OF_FIT_TESTS}, got {test!r}"
        raise ValueError(msg)
    first, second = (h if isinstance(h, Histogram) else Histogram(h, label="") for h in (a, b))
    if not compatible_binning(first.hist, second.hist):
        msg = "goodness_of_fit requires two one-dimensional histograms with identical bin edges"
        raise BinningError(msg)
    if test == "ks" and is_category(first.axis):
        msg = "the Kolmogorov test needs ordered bins, and categories have no order"
        raise ValueError(msg)
    if test == "chi2-absolute":
        return _absolute(first, second)
    for position, side in (("first", first), ("second", second)):
        if side.values().sum() == 0:
            msg = f"the {position} histogram {side.label!r} is empty: there is no shape to test"
            raise ValueError(msg)
    if test == "ks":
        return _kolmogorov(first, second)
    return _chi2(first, second)


def _chi2(first: Histogram, second: Histogram) -> GoodnessOfFit:
    """``TH1::Chi2Test``, its kind of comparison chosen from which sides hold known counts."""
    counts = [_counts(first), _counts(second)]
    swapped = counts[0] is None and counts[1] is not None  # UW needs the counts first
    one, two = (second, first) if swapped else (first, second)
    n1, n2 = counts[::-1] if swapped else counts
    if n1 is not None and n2 is not None:
        method, (chi2, used, notes) = "UU", _uu(n1, n2, one, two)
    elif n1 is not None:
        method, (chi2, used, notes) = "UW", _uw(n1, one, two)
    else:
        method, (chi2, used, notes) = "WW", _ww(one, two)
    ndf = int(used.sum()) - 1
    return GoodnessOfFit(
        test="chi2",
        method=method,
        statistic=chi2,
        ndf=ndf,
        p_value=_probability(chi2, ndf),
        label=first.label,
        reference=second.label,
        bins=used,
        notes=notes,
    )


def _counts(histogram: Histogram) -> FloatArray | None:
    """Return the counts behind ``histogram``, or ``None`` if it is not known to hold any.

    A histogram with ``stat_errors`` holds no counts here: its errors are its own.
    """
    if histogram._provenance.errors is not None:
        return None
    try:
        return histogram.counts()[0]
    except ValueError:
        return None


def _variances(histogram: Histogram) -> FloatArray:
    """Return each bin's statistical variance: its sum of squared weights or ``stat_errors``².

    Raises
    ------
    ValueError
        For asymmetric ``stat_errors``, which no variance describes.
    """
    if histogram._provenance.errors is None:
        return histogram.variances()
    down, up = histogram.errors()
    if not np.array_equal(down, up):
        msg = (
            f"histogram {histogram.label!r} has asymmetric statistical errors of its own "
            "(stat_errors), which no variance describes"
        )
        raise ValueError(msg)
    return up * up


_Result: TypeAlias = tuple[float, np.ndarray, tuple[str, ...]]
"""A chi-square, the bins it took and ROOT's warnings."""

_FEW_EVENTS = "less than 1 event"
_FEW_ENTRIES = "fewer than 10 effective entries"


def _uu(n1: FloatArray, n2: FloatArray, one: Histogram, two: Histogram) -> _Result:
    """ROOT's ``"UU"``: two sets of Poisson counts."""
    sum1, sum2 = float(n1.sum()), float(n2.sum())
    used = (n1 != 0) | (n2 != 0)
    a, b = n1[used], n2[used]
    chi2 = float(np.sum((sum2 * a - sum1 * b) ** 2 / (a + b)) / (sum1 * sum2))
    notes = _note(one, _FEW_EVENTS, np.any(a < 1)) + _note(two, _FEW_EVENTS, np.any(b < 1))
    return chi2, used, notes


def _uw(n1: FloatArray, one: Histogram, two: Histogram) -> _Result:
    """ROOT's ``"UW"``: Poisson counts and a weighted histogram (Gagunashvili).

    Bin by bin, as ``TH1::Chi2TestX``: an empty weighted bin takes the
    variance ``sum(w^2) / sum(w)``, and where the counts would leave the
    expected value undefined, ROOT adds an event to the bin and to the total,
    which the later bins keep.
    """
    w2, s2 = two.values(), _variances(two)
    sum1, sum2, sumw2 = float(n1.sum()), float(w2.sum()), float(s2.sum())
    if sum2 < 0:
        msg = f"histogram {two.label!r} has a negative total, which has no shape to test"
        raise ValueError(msg)
    used = (n1 * n1 != 0) | (w2 * w2 != 0)
    chi2 = 0.0
    few_events = few_entries = False
    for index in np.flatnonzero(used):
        cnt1, cnt2, e2sq = float(n1[index]), float(w2[index]), float(s2[index])
        if cnt2 * cnt2 == 0 and e2sq == 0:
            if sumw2 <= 0:
                msg = (
                    f"histogram {two.label!r} has no uncertainties, and no content in bin "
                    f"{index}, where {one.label!r} has events"
                )
                raise ValueError(msg)
            e2sq = sumw2 / sum2
        if e2sq == 0 and cnt2 < 0:  # ROOT's adjustment would never end
            msg = (
                f"histogram {two.label!r} has a negative content without uncertainty in bin {index}"
            )
            raise ValueError(msg)
        few_events |= cnt1 < 1
        few_entries |= e2sq > 0 and cnt2 * cnt2 / e2sq < 10
        term, sum1 = _uw_bin(cnt1, cnt2, e2sq, sum1, sum2)
        chi2 += term
    notes = _note(one, _FEW_EVENTS, few_events) + _note(two, _FEW_ENTRIES, few_entries)
    return chi2, used, notes


def _uw_bin(cnt1: float, cnt2: float, e2sq: float, sum1: float, sum2: float) -> tuple[float, float]:
    """One bin's term of ROOT's ``"UW"`` chi-square, and the total of counts after it."""

    def terms(cnt1: float, sum1: float) -> tuple[float, float]:
        var1 = sum2 * cnt2 - sum1 * e2sq
        return var1, var1 * var1 + 4.0 * sum2 * sum2 * cnt1 * e2sq

    def adjusted(cnt1: float, sum1: float) -> tuple[float, float, float, float]:
        var1, var2 = terms(cnt1, sum1)
        while var1 * var1 + cnt1 == 0 or var1 + var2 == 0:
            sum1, cnt1 = sum1 + 1, cnt1 + 1
            var1, var2 = terms(cnt1, sum1)
        return cnt1, sum1, var1, math.sqrt(var2)

    cnt1, sum1, var1, var2 = adjusted(cnt1, sum1)
    while var1 + var2 == 0:
        cnt1, sum1, var1, var2 = adjusted(cnt1 + 1, sum1 + 1)
    probability = (var1 + var2) / (2.0 * sum2 * sum2)
    nexp1, nexp2 = probability * sum1, probability * sum2
    term = (cnt1 - nexp1) ** 2 / nexp1
    if e2sq > 0:
        term += (cnt2 - nexp2) ** 2 / e2sq
    return term, sum1


def _ww(one: Histogram, two: Histogram) -> _Result:
    """ROOT's ``"WW"``: two weighted histograms."""
    w1, w2, s1, s2 = one.values(), two.values(), _variances(one), _variances(two)
    sum1, sum2 = float(w1.sum()), float(w2.sum())
    used = (w1 * w1 != 0) | (w2 * w2 != 0)
    exact = used & (s1 == 0) & (s2 == 0)
    if exact.any():
        msg = (
            f"histograms {one.label!r} and {two.label!r} both have no uncertainty in bins "
            f"{np.flatnonzero(exact).tolist()}, whose difference cannot be judged"
        )
        raise ValueError(msg)
    sigma = sum1 * sum1 * s2[used] + sum2 * sum2 * s1[used]
    chi2 = float(np.sum((sum2 * w1[used] - sum1 * w2[used]) ** 2 / sigma))
    notes = _note(one, _FEW_ENTRIES, _few(w1, s1, used)) + _note(
        two, _FEW_ENTRIES, _few(w2, s2, used)
    )
    return chi2, used, notes


def _kolmogorov(first: Histogram, second: Histogram) -> GoodnessOfFit:
    """``TH1::KolmogorovTest`` with its default options (shape only, flow bins excluded)."""
    w1, w2 = first.values(), second.values()
    sum1, sum2 = float(w1.sum()), float(w2.sum())
    v1, v2 = float(_variances(first).sum()), float(_variances(second).sum())
    if v1 <= 0 and v2 <= 0:
        msg = (
            f"histograms {first.label!r} and {second.label!r} both have no uncertainties, "
            "so there are no entries to test"
        )
        raise ValueError(msg)
    # as ROOT: the running sums of contents times the inverse totals
    distance = float(np.max(np.abs(np.cumsum(w1 * (1 / sum1)) - np.cumsum(w2 * (1 / sum2)))))
    if v1 <= 0:  # a function: only the other side fluctuates
        z = distance * math.sqrt(sum2 * sum2 / v2)
    elif v2 <= 0:
        z = distance * math.sqrt(sum1 * sum1 / v1)
    else:
        e1, e2 = sum1 * sum1 / v1, sum2 * sum2 / v2
        z = distance * math.sqrt(e1 * e2 / (e1 + e2))
    from scipy.special import kolmogorov  # noqa: PLC0415 - 0.3 s to import, only when used

    notes = tuple(
        f"{_name(side)} has no uncertainties and is compared as a function"
        for side, variance in ((first, v1), (second, v2))
        if variance <= 0
    )
    return GoodnessOfFit(
        test="ks",
        method="ks",
        statistic=distance,
        ndf=None,
        p_value=float(kolmogorov(z)),
        label=first.label,
        reference=second.label,
        bins=np.ones(len(w1), dtype=bool),
        notes=notes,
    )


def _absolute(first: Histogram, second: Histogram) -> GoodnessOfFit:
    """``r C^-1 r`` of the difference ``r``, statistical and systematic covariance ``C``."""
    num, ref = _Side.of(first), _Side.of(second)
    difference = num.values - ref.values
    used = _occupied(first) | _occupied(second)
    stat = np.where(difference > 0, *_propagated(1.0, -1.0, num.errors, ref.errors))
    covariance = np.diag(stat**2)
    shifts = _source_shifts(np.subtract, difference, num, ref)
    for up, down in shifts.values():
        delta = 0.5 * (up - down)  # signed: a shift moving bins oppositely anticorrelates them
        covariance += np.outer(delta, delta)
    covariance = covariance[np.ix_(used, used)]
    residual = difference[used]
    # singular up to round-off (numpy's matrix_rank tolerance), on every platform alike
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = eigenvalues.max(initial=0.0) * len(eigenvalues) * np.finfo(float).eps
    if np.any(eigenvalues <= tolerance):
        # the systematic outer products only add, so some bin's own variance is that small
        bins = np.flatnonzero(used)[stat[used] ** 2 <= tolerance].tolist()
        msg = (
            f"the covariance of the difference of {first.label!r} and {second.label!r} is "
            f"singular: bins {bins} have no statistical uncertainty, and the systematic "
            "sources do not make up for it; give the histograms statistical uncertainties "
            "there (e.g. data_errors='auto' for empty data bins)"
        )
        raise ValueError(msg)
    chi2 = float(np.sum((eigenvectors.T @ residual) ** 2 / eigenvalues))
    ndf = int(used.sum())
    # the effective entries of the errors that entered: each side's towards the other
    towards = difference > 0
    facing = (
        np.where(towards, num.errors[0], num.errors[1]),
        np.where(towards, ref.errors[1], ref.errors[0]),
    )
    notes = _note(first, _FEW_ENTRIES, _few(num.values, _entries(first, facing[0]), used)) + _note(
        second, _FEW_ENTRIES, _few(ref.values, _entries(second, facing[1]), used)
    )
    return GoodnessOfFit(
        test="chi2-absolute",
        method="absolute",
        statistic=chi2,
        ndf=ndf,
        p_value=_probability(chi2, ndf),
        label=first.label,
        reference=second.label,
        bins=used,
        notes=notes,
        systematics=tuple(shifts),
    )


def _occupied(histogram: Histogram) -> np.ndarray:
    """Return which bins are not empty: with contents, or an uncertainty of their own.

    Weights that cancel leave a bin with nothing but its sum of squared weights;
    the Poisson interval of no counts does not make a bin occupied.
    """
    if histogram._provenance.errors is not None:
        down, up = histogram.errors()
        uncertain = (down != 0) | (up != 0)
    else:
        uncertain = histogram.variances() != 0
    return np.asarray((histogram.values() != 0) | uncertain)


def _entries(histogram: Histogram, error: FloatArray) -> FloatArray:
    """Return the variances behind the effective entries: ``stat_errors`` as they entered."""
    return error * error if histogram._provenance.errors is not None else histogram.variances()


def _probability(chi2: float, ndf: int) -> float:
    """``TMath::Prob``: the chi-square probability, ``nan`` without degrees of freedom.

    ROOT returns 0 without degrees of freedom, which would read as a failed test.
    """
    if ndf <= 0:
        return math.nan
    from scipy.special import chdtrc  # noqa: PLC0415 - 0.3 s to import, only when used

    return float(chdtrc(ndf, chi2))


def _few(values: FloatArray, variances: FloatArray, used: np.ndarray) -> bool:
    """Whether a used bin with an uncertainty has fewer than ten effective entries."""
    with np.errstate(divide="ignore", invalid="ignore"):
        effective = values * values / variances
    return bool(np.any(used & (variances > 0) & (effective < 10)))


def _note(histogram: Histogram, condition: str, applies: bool | np.bool_) -> tuple[str, ...]:
    """ROOT's warning that ``histogram`` has a bin with ``condition``, if it ``applies``."""
    return (f"{_name(histogram)} has a bin with {condition}",) if applies else ()


def _name(histogram: Histogram) -> str:
    """Return the label of ``histogram`` quoted, or ``"a histogram"`` without one."""
    return repr(histogram.label) if histogram.label else "a histogram"
