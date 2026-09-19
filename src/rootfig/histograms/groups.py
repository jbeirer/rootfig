"""Histograms of sample groups: the components filled apart, then summed."""

from __future__ import annotations

from collections.abc import Sequence

from rootfig.errors import annotate
from rootfig.histograms.build import Histogram
from rootfig.histograms.systematics import sum_histograms
from rootfig.model.groups import Group
from rootfig.model.inputs import PlotItem, leaf_samples

__all__ = ["group_histogram", "regroup_histograms"]


def group_histogram(group: Group, components: Sequence[Histogram]) -> Histogram:
    """Sum the histograms of a group's samples into one with the group's label and drawing hints.

    Same-named systematic variations add linearly (:func:`sum_histograms`). The
    result has no ``sample`` and no unbinned ``stats``: it was not filled from one.
    """
    with annotate(f"while summing the components of group {group.label!r}"):
        total = sum_histograms(components, label=group.label)
    return total.replace(
        is_data=group.is_data,
        color=group.color,
        histtype=group.histtype,
        per_object=any(component.per_object for component in components),
    )


def regroup_histograms(
    items: Sequence[PlotItem], histograms: Sequence[Histogram]
) -> list[Histogram]:
    """Return one histogram per item from the ``histograms`` of the items' leaf samples, in order.

    A sample's histogram is returned as it is; the histograms of a group's
    samples are summed with :func:`group_histogram`.
    """
    expected = len(leaf_samples(items))
    if len(histograms) != expected:
        msg = f"got {len(histograms)} histograms for {expected} samples"
        raise ValueError(msg)
    result: list[Histogram] = []
    position = 0
    for item in items:
        count = len(item.samples) if isinstance(item, Group) else 1
        part = histograms[position : position + count]
        result.append(group_histogram(item, part) if isinstance(item, Group) else part[0])
        position += count
    return result
