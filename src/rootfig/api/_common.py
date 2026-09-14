"""Helpers shared by the api modules."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from rootfig.errors import RootfigWarning, SourceError
from rootfig.histograms import (
    Histogram,
    NormalizeSpec,
)
from rootfig.histograms import normalize as normalize_histogram
from rootfig.model import (
    Sample,
    StyleLike,
    as_samples,
    as_style,
)


def single_sample(
    data: Any,
    *,
    tree: str | None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> Sample:
    samples = as_samples(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
    if len(samples) != 1:
        msg = f"expected a single sample, got {len(samples)}"
        raise SourceError(msg)
    return samples[0]


def style_for(
    style: StyleLike, text: str | Sequence[str] | None, lumi: float | str | None = None
) -> Any:
    """Resolve ``style`` and add free text lines and the luminosity used for scaling."""
    resolved = as_style(style)
    if lumi is not None and resolved.lumi is None:
        resolved = resolved.with_(lumi=lumi)
    if text is None:
        return resolved
    existing = list(resolved.text_lines)
    extra = [text] if isinstance(text, str) else list(text)
    return resolved.with_(text=[*existing, *extra])


def normalize_for_plot(histogram_: Histogram, spec: NormalizeSpec) -> Histogram:
    if histogram_.normalization is not None:
        warnings.warn(
            f"histogram {histogram_.label!r} is already normalised ({histogram_.normalization}); "
            "normalising again",
            RootfigWarning,
            stacklevel=3,
        )
    return normalize_histogram(histogram_, spec)
