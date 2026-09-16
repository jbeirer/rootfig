"""``Weight``-storage conversion, binning comparison and bin-wise addition of ``hist`` objects.

Shared by :mod:`rootfig.io` (histograms read from files) and
:mod:`rootfig.histograms`, so both apply one policy for uncertainties and for
what counts as the same binning.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import hist
import numpy as np

from rootfig._typing import Hist
from rootfig.errors import RootfigWarning

__all__ = ["add_hists", "add_into", "as_weight_storage", "is_category", "same_axis", "same_binning"]

_COUNT_STORAGES = (
    hist.storage.Double,
    hist.storage.Int64,
    hist.storage.AtomicInt64,
    hist.storage.Unlimited,
)


def as_weight_storage(histogram: Hist, *, assume_poisson: bool = False) -> Hist:
    """Return ``histogram`` with ``Weight`` storage (a copy if it had another storage).

    Plain count storages (``Double``, ``Int64``, ...) carry no sum of squared
    weights; their variances are taken as ``hist`` reports them, i.e. the
    counts (Poisson) for unweighted fills. Two cases leave no usable variances:
    after a weighted fill or arithmetic on such a storage ``hist`` reports none
    at all, and a count storage with negative contents reports negative ones
    (the counts). Either is an error unless ``assume_poisson=True``, which uses
    the absolute bin contents as variances (the Poisson guess; a
    :class:`~rootfig.errors.RootfigWarning` says so). A ``Weight`` storage is
    returned as it is unless it reports negative variances, which get the same
    treatment.

    Raises
    ------
    TypeError
        If the storage is not a count or ``Weight`` storage (``Mean``, ...).
    ValueError
        If the histogram has no usable variances and ``assume_poisson`` is False.
    """
    if histogram.storage_type is hist.storage.Weight:
        weight_variances = np.asarray(histogram.variances(flow=True), dtype=float)
        if not np.any(weight_variances < 0):
            return histogram
        what = "histogram with Weight storage reports negative variances"
        reported: Any = None
    else:
        if histogram.ndim and histogram.storage_type not in _COUNT_STORAGES:
            msg = (
                f"histograms with {histogram.storage_type.__name__} storage are not supported; "
                "use Weight (or a plain count) storage"
            )
            raise TypeError(msg)
        reported = histogram.variances(flow=True)
        if reported is None:
            what = (
                f"histogram with {histogram.storage_type.__name__} storage was filled with "
                "weights or rescaled, so hist reports no variances (the sum of squared weights "
                "is lost)"
            )
        elif np.any(np.asarray(reported) < 0):
            what = (
                f"histogram with {histogram.storage_type.__name__} storage has negative bin "
                "contents, so the counts it reports as variances are negative (the sum of "
                "squared weights is unknown)"
            )
            reported = None
    values = np.asarray(histogram.values(flow=True), dtype=float)
    if reported is None:
        if not assume_poisson:
            msg = (
                f"{what}. Fill it with hist.storage.Weight() to keep the uncertainties, or pass "
                "assume_poisson=True to use the absolute bin contents as variances"
            )
            raise ValueError(msg)
        warnings.warn(
            f"{what}; using the absolute bin contents as variances (Poisson guess)",
            RootfigWarning,
            stacklevel=3,
        )
        variances = np.abs(values)  # never a negative variance for signed contents
    else:
        variances = np.asarray(reported, dtype=float)
    result = hist.Hist(*histogram.axes, storage=hist.storage.Weight())
    view: Any = result.view(flow=True)
    view.value = values
    view.variance = variances
    return result


def same_binning(a: Hist, b: Hist, *, flow: bool = True) -> bool:
    """Return True if two histograms have the same axes up to their names and labels.

    Axis by axis as :func:`same_axis`; ``flow`` asks for the same flow bins too.
    """
    if a.ndim != b.ndim:
        return False
    return all(
        same_axis(axis_a, axis_b, flow=flow) for axis_a, axis_b in zip(a.axes, b.axes, strict=True)
    )


def same_axis(axis_a: Any, axis_b: Any, *, flow: bool = True) -> bool:
    """Return True if two axes bin the same way, whatever their names and labels.

    Two category axes agree when they list the same categories in the same
    order; two numeric axes (``Regular``, ``Variable`` and ``Integer``
    interchangeably) when their edges agree to a millionth of the smallest bin
    width, so bins shifted by a whole width at large coordinates are rejected.
    A category and a numeric axis never agree. With ``flow`` the under- and
    overflow bins must be present or absent alike.
    """
    if flow and (axis_a.traits.underflow, axis_a.traits.overflow) != (
        axis_b.traits.underflow,
        axis_b.traits.overflow,
    ):
        return False
    if is_category(axis_a) or is_category(axis_b):
        both = is_category(axis_a) and is_category(axis_b)
        return both and list(axis_a) == list(axis_b)
    edges_a, edges_b = np.asarray(axis_a.edges, dtype=float), np.asarray(axis_b.edges, dtype=float)
    if edges_a.shape != edges_b.shape:
        return False
    tolerance = 1e-6 * float(min(np.diff(edges_a).min(), np.diff(edges_b).min()))
    return bool(np.allclose(edges_a, edges_b, rtol=0.0, atol=tolerance))


def is_category(axis: Any) -> bool:
    """Return True for a category axis (``StrCategory``/``IntCategory``: labelled bins)."""
    return isinstance(axis, hist.axis.StrCategory | hist.axis.IntCategory)


def add_into(total: Hist, other: Hist) -> None:
    """Add ``other`` into ``total`` in place, bin by bin, flow bins included.

    Both have ``Weight`` storage and the same binning (:func:`same_binning`);
    ``total`` keeps its axes (names and labels), which is what ``hist``'s own
    addition refuses when the metadata differ.
    """
    view: Any = total.view(flow=True)
    view.value += other.values(flow=True)
    view.variance += other.variances(flow=True)


def add_hists(hists: Iterable[Hist]) -> Hist:
    """Return the bin-wise sum of ``Weight``-storage histograms with the same binning.

    The first histogram is copied and the others added into the copy one at a
    time (see :func:`add_into`), so an iterator of histograms is summed without
    holding them all.
    """
    iterator = iter(hists)
    total = next(iterator).copy()
    for other in iterator:
        add_into(total, other)
    return total
