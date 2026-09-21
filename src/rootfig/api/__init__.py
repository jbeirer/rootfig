"""The high-level API: :func:`plot`, :func:`histogram`, :func:`load` and friends.

Everything here is a thin orchestration of the lower layers
(:mod:`rootfig.io`, :mod:`rootfig.expressions`, :mod:`rootfig.selection`,
:mod:`rootfig.histograms`, :mod:`rootfig.plotting`), which remain usable on
their own.
"""

from rootfig.api.batch import ALL, PlotBook, PlotTask, discover_variables
from rootfig.api.data import histogram, histograms, load
from rootfig.api.measures import efficiency, profile
from rootfig.api.plots1d import plot
from rootfig.api.plots2d import correlation, plot2d
from rootfig.api.tables import SummaryTable, cutflow, summarize

__all__ = [
    "ALL",
    "PlotBook",
    "PlotTask",
    "SummaryTable",
    "correlation",
    "cutflow",
    "discover_variables",
    "efficiency",
    "histogram",
    "histograms",
    "load",
    "plot",
    "plot2d",
    "profile",
    "summarize",
]
