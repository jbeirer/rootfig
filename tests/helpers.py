"""Histogram constructors shared by the test modules of the histograms layer."""

from __future__ import annotations

from collections.abc import Sequence

import hist
import numpy as np

from rootfig.histograms import Histogram


def hist_of(values: Sequence[float], variances: Sequence[float] | None = None) -> hist.Hist:
    """One bin per value over ``[0, len(values))``, with ``variances`` (counts: the values)."""
    h = hist.Hist(hist.axis.Regular(len(values), 0, len(values)), storage=hist.storage.Weight())
    h.view().value = values
    h.view().variance = values if variances is None else variances
    return h


def filled(values: Sequence[float], weights: Sequence[float] | None = None) -> hist.Hist:
    """Three bins over ``[0, 3)`` filled with ``values``, weighted by ``weights`` if given."""
    h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight())
    h.fill(values, weight=weights)
    return h


def symmetric(errors: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The one side of a ``(down, up)`` pair whose sides agree exactly."""
    down, up = errors
    np.testing.assert_array_equal(down, up)
    return down


def with_errors(histogram: Histogram, down: Sequence[float], up: Sequence[float]) -> Histogram:
    """``histogram`` reporting the given ``(down, up)`` statistical errors, as an asymmetric
    model of its contents would."""
    pair = (np.asarray(down, dtype=float), np.asarray(up, dtype=float))
    object.__setattr__(histogram, "errors", lambda *, flow=False: pair)
    return histogram
