"""Histogram construction, normalisation, comparisons and statistics."""

from rootfig.histograms.binomial import EfficiencyInterval
from rootfig.histograms.build import (
    Histogram,
    as_weight_storage,
    compatible_binning,
    fill,
    from_sample,
    negative_bins,
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
from rootfig.histograms.intervals import (
    DataErrors,
    count_problem,
    poisson_interval,
)
from rootfig.histograms.normalize import (
    NormalizeSpec,
    normalize,
)
from rootfig.histograms.pipeline import (
    build_histograms,
    build_histograms_2d,
    combined_selection,
    load_columns,
    load_columns_each,
    read_arrays,
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
    "DataErrors",
    "Efficiency",
    "EfficiencyInterval",
    "Histogram",
    "NormalizeSpec",
    "Profile",
    "ProfileStatistic",
    "ReadPlan",
    "Summary",
    "Uncertainty",
    "UncertaintyMode",
    "as_weight_storage",
    "build_histograms",
    "build_histograms_2d",
    "combined_selection",
    "compare",
    "compatible_binning",
    "correlation_matrix",
    "count_problem",
    "cutflow",
    "describe_axes",
    "describe_table",
    "efficiency",
    "fill",
    "from_sample",
    "load_columns",
    "load_columns_each",
    "negative_bins",
    "normalize",
    "poisson_interval",
    "prefetch",
    "profile",
    "read_arrays",
    "read_stored",
    "stored_mode",
    "stored_names",
    "sum_histograms",
    "summarize",
    "uncertainty",
]
