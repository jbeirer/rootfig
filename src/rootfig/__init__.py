"""rootfig: publication-quality figures straight from ROOT trees and histograms, without ROOT.

The quick path::

    import rootfig as rf

    rf.plot("events.root", "Muon_pt", tree="events", selection="Muon_pt > 20", bins=50)

The composable path uses :class:`Sample`, :class:`Variable`, :class:`Cut` and
:class:`Style` objects with the same :func:`plot` function; :class:`PlotBook`
runs it over many variables, selections and variants, the variables listed or
discovered from the inputs (:data:`ALL`). Lower layers are
exposed as sub-packages: :mod:`rootfig.io`, :mod:`rootfig.expressions`,
:mod:`rootfig.selection`, :mod:`rootfig.histograms`, :mod:`rootfig.plotting`.
"""

from rootfig.api import (
    ALL,
    PlotBook,
    PlotTask,
    SummaryTable,
    correlation,
    cutflow,
    discover_variables,
    efficiency,
    histogram,
    histograms,
    load,
    plot,
    plot2d,
    profile,
    summarize,
)
from rootfig.errors import (
    BinningError,
    ExpressionError,
    IncompatibleWeightError,
    LuminosityError,
    MissingBranchError,
    RootfigError,
    RootfigWarning,
    SelectionError,
    SourceError,
    SystematicError,
)
from rootfig.expressions import evaluate
from rootfig.histograms import (
    ONE_SIGMA,
    Bayesian,
    Comparison,
    Cutflow,
    CutflowTable,
    Efficiency,
    Histogram,
    Profile,
    Summary,
    Uncertainty,
    compare,
)
from rootfig.model import Cut, Group, Sample, Style, Systematic, Variable, log_bins
from rootfig.plotting import Plot, dark_theme, use_style

__version__ = "0.9.0"

__all__ = [
    "ALL",
    "ONE_SIGMA",
    "Bayesian",
    "BinningError",
    "Comparison",
    "Cut",
    "Cutflow",
    "CutflowTable",
    "Efficiency",
    "ExpressionError",
    "Group",
    "Histogram",
    "IncompatibleWeightError",
    "LuminosityError",
    "MissingBranchError",
    "Plot",
    "PlotBook",
    "PlotTask",
    "Profile",
    "RootfigError",
    "RootfigWarning",
    "Sample",
    "SelectionError",
    "SourceError",
    "Style",
    "Summary",
    "SummaryTable",
    "Systematic",
    "SystematicError",
    "Uncertainty",
    "Variable",
    "__version__",
    "compare",
    "correlation",
    "cutflow",
    "dark_theme",
    "discover_variables",
    "efficiency",
    "evaluate",
    "histogram",
    "histograms",
    "load",
    "log_bins",
    "plot",
    "plot2d",
    "profile",
    "summarize",
    "use_style",
]
