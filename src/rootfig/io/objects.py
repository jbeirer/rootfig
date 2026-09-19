"""Objects other than trees in ROOT files: listing them and reading stored histograms.

:class:`~rootfig.io.FileSource` delegates here. The functions take file paths so
they stay independent of the source classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import uproot

from rootfig._storage import add_into, as_weight_storage, same_binning
from rootfig._typing import Hist
from rootfig.errors import BinningError, SourceError

__all__ = [
    "RNTUPLE_MARKER",
    "histogram_names",
    "is_histogram_class",
    "is_tree_class",
    "object_classes",
    "read_histogram",
    "read_histograms",
]

_TREE_CLASSNAMES = ("TTree", "TNtuple", "TNtupleD", "TChain")
RNTUPLE_MARKER = "RNTuple"
"""Substring identifying uproot's ``RNTuple`` classes and models."""
_HISTOGRAM_CLASSNAMES = ("TH1", "TH2")
_UNSUPPORTED_HISTOGRAM_CLASSNAMES = ("TH3", "TProfile", "THn")


def is_tree_class(classname: str) -> bool:
    """Return True for ``TTree``-like classes and ``RNTuple``."""
    return classname.startswith(_TREE_CLASSNAMES) or RNTUPLE_MARKER in classname


def is_histogram_class(classname: str) -> bool:
    """Return True for the 1D and 2D ROOT histogram classes rootfig draws (``TH1*``, ``TH2*``)."""
    return classname.startswith(_HISTOGRAM_CLASSNAMES) and not classname.startswith(
        _UNSUPPORTED_HISTOGRAM_CLASSNAMES
    )


def strip_cycle(key: str) -> str:
    """Remove ROOT's ``;cycle`` suffix from an object key."""
    return key.rsplit(";", 1)[0]


def object_classes(path: str) -> dict[str, str]:
    """Map every object in ``path`` to its class name.

    Directories are descended (``"dir/name"``) and cycle numbers are stripped.
    """
    with uproot.open(path) as file:
        classnames: dict[str, str] = file.classnames(recursive=True)
    return {strip_cycle(key): cls for key, cls in classnames.items()}


def histogram_names(path: str) -> list[str]:
    """Return the names of the 1D and 2D histograms in ``path``."""
    return sorted(k for k, cls in object_classes(path).items() if is_histogram_class(cls))


def read_histogram(files: Sequence[str], name: str, *, assume_poisson: bool = False) -> Hist:
    """Read the histogram stored as ``name`` in every file and return their sum.

    :func:`read_histograms` for one name; see there for how the files are read and
    summed and for the errors.
    """
    return read_histograms(files, [name], assume_poisson=assume_poisson)[name]


def read_histograms(
    files: Sequence[str], names: Sequence[str], *, assume_poisson: bool = False
) -> dict[str, Hist]:
    """Read the histograms stored as ``names`` in every file and return their sums, by name.

    Each file is opened once and every name read from it, so a batch of
    histograms costs one open per file. ROOT's ``TH1`` and ``TH2`` classes are
    converted with uproot: axis titles become axis labels, labelled bins a
    category axis, and the under/overflow contents land in the flow bins. Every
    histogram is brought to ``Weight`` storage first
    (:func:`rootfig._storage.as_weight_storage`): one written with ``Sumw2``
    keeps its uncertainties, one without it takes its counts as variances, and
    negative contents without ``Sumw2`` are refused unless ``assume_poisson``.
    Each file is added into the running total of its name as soon as it is
    read, so two histograms per name are held at a time however many files
    there are; the first file's axis names and titles are kept.

    Raises
    ------
    SourceError
        If a file lacks an object, it is not a 1D/2D histogram (the message
        lists the histograms that file does hold), its variances are unusable,
        or there are no files.
    BinningError
        If the files disagree on the binning of a histogram.
    """
    totals: dict[str, Hist] = {}
    for path in files:
        with uproot.open(path) as file:
            for name in dict.fromkeys(names):
                raw = _read_one(file, path, name)
                try:
                    current = as_weight_storage(raw, assume_poisson=assume_poisson)
                except ValueError as exc:
                    msg = f"histogram {name!r} in {path!r}: {exc}"
                    raise SourceError(msg) from exc
                total = totals.get(name)
                if total is None:
                    totals[name] = current  # a fresh object from uproot, safe to accumulate into
                elif not same_binning(total, current):
                    msg = (
                        f"histogram {name!r} has different binnings in {files[0]!r} and {path!r}; "
                        "files summed into one sample must agree"
                    )
                    raise BinningError(msg)
                else:
                    add_into(total, current)
    if not files:
        msg = "no files to read a histogram from"
        raise SourceError(msg)
    return totals


def _read_one(file: Any, path: str, name: str) -> Hist:
    """Read ``name`` from the open uproot ``file`` at ``path`` (named in messages) as a ``Hist``."""
    try:
        obj = file[name]
    except uproot.KeyInFileError as exc:
        msg = (
            f"histogram {name!r} not found in {path!r}; histograms present: {histogram_names(path)}"
        )
        raise SourceError(msg) from exc
    classname = str(getattr(obj, "classname", type(obj).__name__))
    if not is_histogram_class(classname) or not hasattr(obj, "to_hist"):
        msg = (
            f"object {name!r} in {path!r} is a {classname}, not a 1D or 2D histogram; "
            f"histograms present: {histogram_names(path)}"
        )
        raise SourceError(msg)
    result: Hist = obj.to_hist()
    return result
