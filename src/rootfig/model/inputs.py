"""Normalising the ``data`` argument of the api functions into samples and groups."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, TypeAlias

import awkward as ak

from rootfig.errors import SourceError
from rootfig.model.groups import Group
from rootfig.model.samples import Sample

__all__ = ["PlotItem", "as_plot_items", "as_samples", "leaf_samples", "map_samples"]

PlotItem: TypeAlias = Sample | Group
"""A sample or a group of samples: what one histogram of a plot is made from."""


def as_plot_items(
    data: Any,
    *,
    tree: str | None = None,
    labels: str | Sequence[str] | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> list[PlotItem]:
    """Normalise the ``data`` argument of :func:`rootfig.plot` into samples and groups.

    * A :class:`Sample` or :class:`Group` gives one item.
    * A single path/glob/mapping/array gives one sample.
    * A list gives one item per element: samples and groups as they are, every
      raw file specification as a sample of its own (``["sig.root", "bkg.root"]``
      is two samples; wrap several files in a ``Sample`` or use a glob for one).
    * A mapping from label to file specification, arrays, ``Sample`` or ``Group``
      gives one labelled item per entry (a ``Sample`` or ``Group`` only takes the
      label). A mapping whose values are columns (arrays or lists of numbers) is
      one in-memory sample instead.

    A group is never inferred: only ``Group(...)`` makes one.

    Raises
    ------
    SourceError
        If the list is empty, ``labels`` does not match the number of items, or
        a mapping holding a group has keys that are not strings or values that
        are not datasets.
    """
    items: list[PlotItem]
    if isinstance(data, Sample | Group):
        items = [data]
    elif isinstance(data, Mapping) and _looks_like_label_map(data):
        items = [
            _labelled(value, key, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
            for key, value in data.items()
        ]
    elif isinstance(data, Mapping) and any(isinstance(v, Group) for v in data.values()):
        msg = "a mapping holding groups must map labels to samples, groups or files"
        raise SourceError(msg)
    elif isinstance(data, list | tuple):
        if not data:
            msg = "no samples given"
            raise SourceError(msg)
        items = [
            item
            if isinstance(item, Sample | Group)
            else Sample(item, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
            for item in data
        ]
    else:
        items = [Sample(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)]
    if labels is None:
        return items
    label_list = [labels] if isinstance(labels, str) else list(labels)
    if len(label_list) != len(items):
        msg = f"got {len(label_list)} labels for {len(items)} samples"
        raise SourceError(msg)
    return [item.replace(label=lab) for item, lab in zip(items, label_list, strict=True)]


def as_samples(
    data: Any,
    *,
    tree: str | None = None,
    labels: str | Sequence[str] | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> list[Sample]:
    """Normalise ``data`` into samples only; :func:`as_plot_items` lists the accepted forms.

    Raises
    ------
    TypeError
        If ``data`` holds a :class:`Group`: only ``plot()``, ``histograms()``
        and ``histogram()`` draw several samples as one histogram.
    SourceError
        As :func:`as_plot_items`.
    """
    items = as_plot_items(
        data, tree=tree, labels=labels, entry_start=entry_start, entry_stop=entry_stop
    )
    groups = [item.label for item in items if isinstance(item, Group)]
    if groups:
        msg = (
            f"groups are not accepted here ({groups}); only plot(), histograms() and "
            "histogram() draw several samples as one histogram. Pass group.samples to use "
            "the components as separate samples"
        )
        raise TypeError(msg)
    return [item for item in items if isinstance(item, Sample)]


def leaf_samples(items: Iterable[PlotItem]) -> list[Sample]:
    """Return the samples of ``items`` with every group expanded, in order."""
    return [
        leaf for item in items for leaf in (item.samples if isinstance(item, Group) else (item,))
    ]


def map_samples(item: PlotItem, transform: Callable[[Sample], Sample]) -> PlotItem:
    """Apply ``transform`` to every sample of ``item``; a group is rebuilt and re-validated."""
    if isinstance(item, Sample):
        return transform(item)
    return item.replace(components=[map_samples(c, transform) for c in item.components])


def _looks_like_label_map(data: Mapping[Any, Any]) -> bool:
    """Distinguish ``{"Signal": "sig.root"}`` from a mapping of column arrays.

    Values that describe a dataset (paths, lists of paths, ``Sample`` or ``Group``
    objects, mappings of arrays, Awkward record arrays) make a label map;
    anything else (NumPy arrays, lists of numbers, flat Awkward arrays) is a column.
    """
    return bool(data) and all(
        isinstance(key, str) and (isinstance(value, Group) or _is_dataset_spec(value))
        for key, value in data.items()
    )


def _is_dataset_spec(value: Any) -> bool:
    if isinstance(value, str | PathLike | Sample | Mapping):
        return True
    if isinstance(value, ak.Array):
        return bool(value.fields)
    if isinstance(value, list | tuple):
        return bool(value) and all(isinstance(item, str | PathLike) for item in value)
    return False


def _labelled(
    value: Any, label: str, *, tree: str | None, entry_start: int | None, entry_stop: int | None
) -> PlotItem:
    if isinstance(value, Sample | Group):
        return value.replace(label=label)
    return Sample(value, tree=tree, label=label, entry_start=entry_start, entry_stop=entry_stop)
