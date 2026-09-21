"""rootfig: publication-quality histograms straight from ROOT trees, without ROOT.

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
    Cutflow,
    CutflowTable,
    Efficiency,
    Histogram,
    Profile,
    Ratio,
    Summary,
    Uncertainty,
    ratio,
    significance,
)
from rootfig.model import Cut, Group, Sample, Style, Systematic, Variable, log_bins
from rootfig.plotting import Plot, dark_theme, use_style

__version__ = "0.6.0"

__all__ = [
    "ALL",
    "BinningError",
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
    "Ratio",
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
    "ratio",
    "significance",
    "summarize",
    "use_style",
]
