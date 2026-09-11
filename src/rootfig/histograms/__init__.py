"""Histogram construction, normalisation, ratios and statistics."""

from rootfig.histograms.build import Histogram, as_weight_storage, fill
from rootfig.histograms.cutflow import Cutflow, CutflowStep, CutflowTable, cutflow
from rootfig.histograms.efficiency import Efficiency, Profile, ProfileStatistic, efficiency, profile
from rootfig.histograms.normalize import (
    NormalizeSpec,
    normalization_label,
    normalize,
    normalize_hist,
)
from rootfig.histograms.pipeline import (
    build_histograms,
    build_histograms_2d,
    combined_selection,
    combined_weight,
    load_columns,
    read_arrays,
    source_length,
)
from rootfig.histograms.ratio import (
    SIGNIFICANCE_KINDS,
    Ratio,
    RatioUncertainty,
    SignificanceKind,
    compatible_binning,
    ratio,
    significance,
)
from rootfig.histograms.stats import Summary, correlation_matrix, describe_table, summarize

__all__ = [
    "SIGNIFICANCE_KINDS",
    "Cutflow",
    "CutflowStep",
    "CutflowTable",
    "Efficiency",
    "Histogram",
    "NormalizeSpec",
    "Profile",
    "ProfileStatistic",
    "Ratio",
    "RatioUncertainty",
    "SignificanceKind",
    "Summary",
    "as_weight_storage",
    "build_histograms",
    "build_histograms_2d",
    "combined_selection",
    "combined_weight",
    "compatible_binning",
    "correlation_matrix",
    "cutflow",
    "describe_table",
    "efficiency",
    "fill",
    "load_columns",
    "normalization_label",
    "normalize",
    "normalize_hist",
    "profile",
    "ratio",
    "read_arrays",
    "significance",
    "source_length",
    "summarize",
]
