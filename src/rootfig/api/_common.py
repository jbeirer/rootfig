"""Helpers shared by the api modules."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from rootfig.errors import RootfigWarning, SourceError
from rootfig.histograms import Histogram, NormalizeSpec
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
