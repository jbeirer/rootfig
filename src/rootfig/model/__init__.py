"""Declarative descriptions of what to plot: samples, variables, cuts, binning, style."""

from rootfig.model.binning import Axis, Bins, RangeSpec, auto_range, log_bins, resolve_axis
from rootfig.model.cuts import Cut, CutLike, as_cut
from rootfig.model.samples import Sample, as_samples
from rootfig.model.style import Style, StyleLike, as_style
from rootfig.model.variables import Variable, as_variable

__all__ = [
    "Axis",
    "Bins",
    "Cut",
    "CutLike",
    "RangeSpec",
    "Sample",
    "Style",
    "StyleLike",
    "Variable",
    "as_cut",
    "as_samples",
    "as_style",
    "as_variable",
    "auto_range",
    "log_bins",
    "resolve_axis",
]
