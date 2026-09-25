"""Helpers shared by the api modules."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from rootfig.errors import RootfigWarning, SourceError
from rootfig.histograms import (
    DataErrors,
    Histogram,
    NormalizeSpec,
    count_problem,
)
from rootfig.histograms import normalize as normalize_histogram
from rootfig.model import (
    Group,
    PlotItem,
    Sample,
    StyleLike,
    as_plot_items,
    as_style,
    map_samples,
)


def single_sample(
    data: Any,
    *,
    function: str,
    tree: str | None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> Sample:
    """Return the one sample ``data`` describes for ``function``, which takes no more."""
    items = as_plot_items(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
    if len(items) != 1:
        msg = f"{function} takes a single sample, got {len(items)}"
        raise SourceError(msg)
    (item,) = items
    if isinstance(item, Group):
        msg = (
            f"{function} takes one sample and {item.label!r} is a group of {len(item.samples)}; "
            f"pass one of group.samples, or call {function} once per component"
        )
        raise TypeError(msg)
    return item


def as_observed(item: PlotItem) -> PlotItem:
    """Mark a sample, or every sample of a group, as observed data."""
    return map_samples(item, lambda s: s if s.is_data else s.replace(is_data=True, systematics={}))


def style_for(
    style: StyleLike, text: str | Sequence[str] | None, lumi: float | str | None = None
) -> Any:
    """Resolve ``style`` and add free text lines and the luminosity used for scaling."""
    resolved = as_style(style)
    if lumi is not None and resolved.lumi is None:
        resolved = resolved.replace(lumi=lumi)
    if text is None:
        return resolved
    existing = list(resolved.text_lines)
    extra = [text] if isinstance(text, str) else list(text)
    return resolved.replace(text=[*existing, *extra])


def normalize_for_plot(histogram_: Histogram, spec: NormalizeSpec) -> Histogram:
    if histogram_.normalization is not None:
        warnings.warn(
            f"histogram {histogram_.label!r} is already normalised ({histogram_.normalization}); "
            "normalising again",
            RootfigWarning,
            stacklevel=3,
        )
    return normalize_histogram(histogram_, spec)


def with_data_errors(histograms: Sequence[Histogram], mode: DataErrors | None) -> list[Histogram]:
    """Give the observed histograms the error model ``mode`` asks for (``plot(data_errors=)``).

    Decided on the histograms as filled or read, before normalisation or flow
    bins change their contents, so the model does not depend on how they are
    drawn. ``None`` keeps every histogram's own model, ``"poisson"`` gives the
    Poisson interval to every observed histogram and ``"auto"`` to those holding
    unit-weight counts; a histogram that already has
    :attr:`~rootfig.histograms.Histogram.poisson` keeps it. Non-data histograms
    are returned as they are.

    Raises
    ------
    ValueError
        For an unknown ``mode``, or ``"poisson"`` for a histogram that does not
        hold unit-weight counts.
    """
    if mode not in (None, "poisson", "auto"):
        msg = f"data_errors must be None, 'poisson' or 'auto', got {mode!r}"
        raise ValueError(msg)
    result = []
    for histogram_ in histograms:
        if mode is None or not histogram_.is_data or histogram_.poisson:
            result.append(histogram_)
            continue
        problem = count_problem(histogram_.values(flow=True), histogram_.variances(flow=True))
        if problem is None:
            result.append(histogram_.replace(poisson=True))
            continue
        if mode == "poisson":
            msg = (
                f"data_errors='poisson' draws the Poisson interval of counts, but "
                f"{histogram_.label!r} {problem}; leave data_errors unset for sqrt(sum of "
                "squared weights), or use 'auto' to keep that where data is not counts"
            )
            raise ValueError(msg)
        result.append(histogram_)
    return result
