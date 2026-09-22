"""Histogram construction, normalisation, comparisons and statistics."""

from rootfig.histograms.build import (
    Histogram,
    as_weight_storage,
    compatible_binning,
    fill,
    from_sample,
)
from rootfig.histograms.comparison import (
    COMPARISON_KINDS,
    Comparison,
    ComparisonKind,
    UncertaintyMode,
    compare,
)
from rootfig.histograms.cutflow import Cutflow, CutflowStep, CutflowTable, cutflow
from rootfig.histograms.efficiency import Efficiency, Profile, ProfileStatistic, efficiency, profile
from rootfig.histograms.groups import group_histogram, regroup_histograms
from rootfig.histograms.normalize import (
    NormalizeSpec,
    normalization_label,
    normalize,
    normalize_hist,
)
from rootfig.histograms.pipeline import (
    branch_names,
    build_histograms,
    build_histograms_2d,
    combined_selection,
    combined_weight,
    load_columns,
    load_columns_each,
    read_arrays,
    source_length,
)
from rootfig.histograms.prefetch import ReadPlan, prefetch
from rootfig.histograms.stats import Summary, correlation_matrix, describe_table, summarize
from rootfig.histograms.stored import describe_axes, read_stored, stored_mode, stored_names
from rootfig.histograms.systematics import Uncertainty, sum_histograms, uncertainty

__all__ = [
    "COMPARISON_KINDS",
    "Comparison",
    "ComparisonKind",
    "Cutflow",
    "CutflowStep",
    "CutflowTable",
    "Efficiency",
    "Histogram",
    "NormalizeSpec",
    "Profile",
    "ProfileStatistic",
    "ReadPlan",
    "Summary",
    "Uncertainty",
    "UncertaintyMode",
    "as_weight_storage",
    "branch_names",
    "build_histograms",
    "build_histograms_2d",
    "combined_selection",
    "combined_weight",
    "compare",
    "compatible_binning",
    "correlation_matrix",
    "cutflow",
    "describe_axes",
    "describe_table",
    "efficiency",
    "fill",
    "from_sample",
    "group_histogram",
    "load_columns",
    "load_columns_each",
    "normalization_label",
    "normalize",
    "normalize_hist",
    "prefetch",
    "profile",
    "read_arrays",
    "read_stored",
    "regroup_histograms",
    "source_length",
    "stored_mode",
    "stored_names",
    "sum_histograms",
    "summarize",
    "uncertainty",
]
