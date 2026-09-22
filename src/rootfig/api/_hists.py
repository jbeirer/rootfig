"""Histogram objects given directly to the api functions instead of data to fill from."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import hist

from rootfig._typing import Hist
from rootfig.histograms import Histogram, as_weight_storage
from rootfig.model.binning import Bins, RangeSpec

_UNIT_SUFFIX = re.compile(r"\[([^\[\]]+)\]\s*$")


def histogram_objects(data: Any) -> list[Histogram | Hist] | None:
    """Return ``data`` as a list of histogram objects, or ``None`` if it is something to fill from.

    A single :class:`~rootfig.histograms.Histogram` or ``hist.Hist``, or a list
    or tuple of them, counts as histogram objects; a list of paths, samples or
    arrays does not.
    """
    if isinstance(data, Histogram | hist.Hist):
        return [data]
    if isinstance(data, list | tuple) and all(
        isinstance(item, Histogram | hist.Hist) for item in data
    ):
        return list(data)  # an empty list counts: there is nothing to fill from either
    return None


def wrap_histograms(
    items: Sequence[Histogram | Hist],
    labels: str | Sequence[str] | None = None,
    *,
    assume_poisson: bool = False,
    is_data: bool = False,
) -> list[Histogram]:
    """Turn histogram objects into :class:`~rootfig.histograms.Histogram` objects.

    Plain ``hist.Hist`` objects are labelled from ``labels`` (or after their
    first axis, or numbered) and converted to ``Weight`` storage;
    ``assume_poisson`` accepts a count storage that lost its variances.
    ``labels`` also relabels ``Histogram`` objects; ``is_data`` marks every
    result as observed data.
    """
    label_list = None if labels is None else ([labels] if isinstance(labels, str) else list(labels))
    if label_list is not None and len(label_list) != len(items):
        msg = f"got {len(label_list)} labels for {len(items)} histograms"
        raise ValueError(msg)
    wrapped: list[Histogram] = []
    for index, item in enumerate(items):
        if isinstance(item, Histogram):
            histogram_ = item if label_list is None else item.replace(label=label_list[index])
        else:
            if label_list is not None:
                label = label_list[index]
            else:
                axis_name = item.axes[0].name if item.ndim == 1 else ""
                label = axis_name or f"hist {index + 1}"
            converted = as_weight_storage(item, assume_poisson=assume_poisson)
            histogram_ = Histogram(converted, label=str(label))
        if is_data and not histogram_.is_data:
            histogram_ = histogram_.replace(is_data=True)
        wrapped.append(histogram_)
    return wrapped


def reject_fill_options(what: str, **options: Any) -> None:
    """Raise if any of ``options`` (name -> value) was given for already filled histograms.

    An option counts as given when it is not ``None``; ``nonfinite`` when it is
    not ``"drop"``.
    """
    given = sorted(
        name
        for name, value in options.items()
        if (value != "drop" if name == "nonfinite" else value is not None)
    )
    if given:
        msg = (
            f"{', '.join(given)} appl{'ies' if len(given) == 1 else 'y'} when filling from "
            f"event data; {what} are already filled"
        )
        raise ValueError(msg)


def require_dimension(histograms: Sequence[Histogram], ndim: int, function: str) -> None:
    """Raise if a histogram object has another dimensionality than ``function`` draws.

    Checked before anything is done to the histograms, so the message names the
    function to use instead of failing on an axis that does not exist.
    """
    drawn_by = {1: "plot", 2: "plot2d"}
    for histogram_ in histograms:
        if histogram_.ndim != ndim:
            msg = (
                f"{function}() draws {'one' if ndim == 1 else 'two'}-dimensional histograms, got "
                f"{histogram_.ndim}D ({histogram_.label!r})"
            )
            if histogram_.ndim in drawn_by:
                msg += f"; use {drawn_by[histogram_.ndim]}()"
            raise ValueError(msg)


def rebin_ready_made(
    histograms: Sequence[Histogram],
    bins: Sequence[Bins | None],
    ranges: Sequence[RangeSpec] | None = None,
) -> list[Histogram]:
    """Crop and merge ready-made histograms with one bins/range specification per axis.

    Requested edges must coincide with existing edges; content outside the
    requested range moves into the flow bins. A range without bins keeps the
    bins between its ends. See :meth:`~rootfig.histograms.Histogram.rebinned_to`.
    """
    return [histogram_.rebinned_to(bins, range=ranges) for histogram_ in histograms]


def unit_of(axis_label: str | None) -> str | None:
    """Return the unit a label ends with (``"GeV"`` for ``"m [GeV]"``), or ``None``."""
    if not axis_label:
        return None
    match = _UNIT_SUFFIX.search(axis_label)
    return match.group(1) if match else None
