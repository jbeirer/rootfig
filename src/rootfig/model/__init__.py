"""Declarative descriptions of what to plot: samples, variables, cuts, binning, style."""

from rootfig.model.binning import (
    DEFAULT_BINS,
    DEFAULT_RANGE,
    Axis,
    Bins,
    MergeTarget,
    RangeSpec,
    auto_range,
    log_bins,
    merge_target,
    resolve_axis,
)
from rootfig.model.cuts import Cut, CutLike, as_cut
from rootfig.model.groups import Group
from rootfig.model.inputs import PlotItem, as_plot_items, as_samples, leaf_samples, map_samples
from rootfig.model.samples import Sample
from rootfig.model.style import Style, StyleLike, as_style
from rootfig.model.systematics import (
    Systematic,
    SystematicKind,
    SystematicLike,
    as_systematics,
)
from rootfig.model.variables import Variable, as_variable

__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_RANGE",
    "Axis",
    "Bins",
    "Cut",
    "CutLike",
    "Group",
    "MergeTarget",
    "PlotItem",
    "RangeSpec",
    "Sample",
    "Style",
    "StyleLike",
    "Systematic",
    "SystematicKind",
    "SystematicLike",
    "Variable",
    "as_cut",
    "as_plot_items",
    "as_samples",
    "as_style",
    "as_systematics",
    "as_variable",
    "auto_range",
    "leaf_samples",
    "log_bins",
    "map_samples",
    "merge_target",
    "resolve_axis",
]
