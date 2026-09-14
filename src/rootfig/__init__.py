"""rootfig: publication-quality histograms straight from ROOT trees, without ROOT.

The quick path::

    import rootfig as rf

    rf.plot("events.root", "Muon_pt", tree="events", selection="Muon_pt > 20", bins=50)

The composable path uses :class:`Sample`, :class:`Variable`, :class:`Cut` and
:class:`Style` objects with the same :func:`plot` function. Lower layers are
exposed as sub-packages: :mod:`rootfig.io`, :mod:`rootfig.expressions`,
:mod:`rootfig.selection`, :mod:`rootfig.histograms`, :mod:`rootfig.plotting`.
"""

from rootfig.api import (
    SummaryTable,
    correlation,
    cutflow,
    efficiency,
    histogram,
    histograms,
    load,
    plot,
    plot2d,
    plot_histograms,
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
from rootfig.model import Cut, Sample, Style, Systematic, Variable, log_bins
from rootfig.plotting import Plot, dark_theme, use_style

__version__ = "0.5.0"

__all__ = [
    "BinningError",
    "Cut",
    "Cutflow",
    "CutflowTable",
    "Efficiency",
    "ExpressionError",
    "Histogram",
    "IncompatibleWeightError",
    "LuminosityError",
    "MissingBranchError",
    "Plot",
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
    "efficiency",
    "evaluate",
    "histogram",
    "histograms",
    "load",
    "log_bins",
    "plot",
    "plot2d",
    "plot_histograms",
    "profile",
    "ratio",
    "significance",
    "summarize",
    "use_style",
]
