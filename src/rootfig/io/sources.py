"""Data sources: where branch arrays come from.

Two implementations of the :class:`Source` protocol are provided:

* :class:`FileSource` reads ROOT files with uproot. It accepts single paths,
  glob patterns, lists of either, remote URLs, and ``"file.root:tree"``
  shorthand, and works with both ``TTree`` and ``RNTuple`` objects.
* :class:`ArraySource` wraps data that is already in memory (a mapping of
  arrays, an Awkward record array, or a NumPy structured array).

Everything downstream of this module (expressions, selection, histogramming,
plotting) only sees the protocol, so it is independent of file I/O.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import awkward as ak
import numpy as np
import uproot

from rootfig.errors import SourceError

__all__ = ["ArraySource", "FileSource", "FilesLike", "Source", "as_source", "resolve_files"]

FilesLike = str | PathLike[str] | Sequence[str | PathLike[str]]
"""A path, glob pattern, URL, ``"path:tree"`` string, or a sequence of those."""

_REMOTE_PREFIXES = ("root://", "http://", "https://", "s3://", "gs://", "xrootd://")
_TREE_CLASSNAMES = ("TTree", "TNtuple", "TNtupleD", "TChain")
_RNTUPLE_MARKER = "RNTuple"


@runtime_checkable
class Source(Protocol):
    """Anything that can list branch names and return branch arrays.

    Implementations must be cheap to construct; opening files or reading data
    should happen inside :meth:`branches` and :meth:`arrays`.
    """

    def branches(self) -> list[str]:
        """Return the names of all available branches."""
        ...

    def arrays(self, branches: Sequence[str]) -> dict[str, ak.Array]:
        """Read the given branches and return them as Awkward arrays of equal length."""
        ...

    def describe(self) -> str:
        """Return a short human-readable description used in labels and messages."""
        ...

    # Implementations should also provide ``num_entries() -> int`` (the number of
    # events); rootfig falls back to reading one branch when it is absent.


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------


def _is_remote(path: str) -> bool:
    return path.startswith(_REMOTE_PREFIXES)


def _split_tree(path: str) -> tuple[str, str | None]:
    """Split ``"file.root:tree"`` into ``("file.root", "tree")``.

    Only the last colon is considered and only when the part before it ends
    with ``.root`` (case-insensitive), so URLs such as ``root://host//file.root``
    are left intact. The tree part may name a tree inside a directory
    (``"file.root:dir/events"``).
    """
    head, sep, tail = path.rpartition(":")
    if sep and head.lower().endswith(".root") and tail:
        return head, tail
    return path, None


def resolve_files(files: FilesLike) -> tuple[tuple[str, ...], str | None]:
    """Expand ``files`` into a sorted tuple of concrete paths and an optional tree name.

    Parameters
    ----------
    files
        A path, glob pattern, remote URL, ``"path:tree"`` string, or a sequence
        of any of these. Local glob patterns are expanded and sorted so results
        are deterministic regardless of filesystem order.

    Returns
    -------
    tuple
        ``(paths, tree)`` where ``tree`` is the tree name found in ``"path:tree"``
        strings (all must agree) or ``None``.

    Raises
    ------
    SourceError
        If a pattern matches nothing, a local file does not exist, or the
        embedded tree names disagree.
    """
    if isinstance(files, str | PathLike):
        items: list[str] = [os.fspath(files)]
    else:
        items = [os.fspath(f) for f in files]
    if not items:
        msg = "no input files given"
        raise SourceError(msg)

    paths: list[str] = []
    trees: set[str] = set()
    for item in items:
        path, tree = _split_tree(item)
        if tree is not None:
            trees.add(tree)
        if _is_remote(path):
            paths.append(path)
            continue
        expanded = os.path.expanduser(path)
        if glob.has_magic(expanded):
            matches = sorted(glob.glob(expanded))
            if not matches:
                msg = f"no files match pattern {path!r}"
                raise SourceError(msg)
            paths.extend(matches)
        else:
            if not Path(expanded).is_file():
                msg = f"file not found: {path!r}"
                raise SourceError(msg)
            paths.append(expanded)

    if len(trees) > 1:
        msg = f"conflicting tree names in file specification: {sorted(trees)}"
        raise SourceError(msg)
    return tuple(paths), (trees.pop() if trees else None)


def _strip_cycle(key: str) -> str:
    return key.rsplit(";", 1)[0]


def _leaf_names(keys: Sequence[str]) -> list[str]:
    """Branch names as users write them: sub-branches by their own (dotted) name, no parents.

    uproot lists split object branches of a ``TTree`` as ``Parent/Parent.field``
    plus the parent itself, and the nested fields of an ``RNTuple`` as
    ``Parent.field`` plus ``Parent``; only the leaves hold arrays that can be
    histogrammed.
    """
    parents = {key.rsplit("/", 1)[0] for key in keys if "/" in key}
    leaves = [key.rsplit("/", 1)[-1] for key in keys if key not in parents]
    records = {leaf for leaf in leaves if any(other.startswith(f"{leaf}.") for other in leaves)}
    return [leaf for leaf in leaves if leaf not in records]


def _extract_nested(data: Mapping[str, Any], name: str) -> ak.Array | None:
    """Return the field ``Parent.field.sub`` from a record array read as ``Parent``."""
    parts = name.split(".")
    if parts[0] not in data:
        return None
    array = data[parts[0]]
    for part in parts[1:]:
        if part not in getattr(array, "fields", ()):
            return None
        array = array[part]
    return ak.Array(array)


def _is_tree_class(classname: str) -> bool:
    return classname.startswith(_TREE_CLASSNAMES) or _RNTUPLE_MARKER in classname


def _detect_tree(path: str) -> str:
    """Return the name of the only tree-like object in ``path`` or raise."""
    with uproot.open(path) as file:
        classnames: dict[str, str] = file.classnames(recursive=True)
    trees = sorted({_strip_cycle(k) for k, cls in classnames.items() if _is_tree_class(cls)})
    if len(trees) == 1:
        return trees[0]
    if not trees:
        found = sorted({f"{_strip_cycle(k)} ({cls})" for k, cls in classnames.items()})
        msg = f"no TTree or RNTuple found in {path!r}; objects present: {found}"
        raise SourceError(msg)
    msg = f"several trees found in {path!r}: {trees}. Pass tree=... to choose one"
    raise SourceError(msg)


@dataclass(frozen=True)
class FileSource:
    """Branch arrays read from one or more ROOT files with uproot.

    Parameters
    ----------
    files
        See :data:`FilesLike`. Globs are expanded and sorted at construction.
    tree
        Name of the ``TTree``/``RNTuple``. If omitted, the files are inspected
        and the tree is auto-detected when there is exactly one.
    entry_start, entry_stop
        Restrict reading to a range of entries counted across all files, in the
        same way as :func:`uproot.concatenate`. Useful for quick looks at large
        inputs.
    """

    files: tuple[str, ...]
    tree: str | None = None
    entry_start: int | None = None
    entry_stop: int | None = None
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __init__(
        self,
        files: FilesLike,
        tree: str | None = None,
        *,
        entry_start: int | None = None,
        entry_stop: int | None = None,
    ) -> None:
        paths, embedded_tree = resolve_files(files)
        if tree is not None and embedded_tree is not None and tree != embedded_tree:
            msg = f"tree {tree!r} conflicts with {embedded_tree!r} given in the file path"
            raise SourceError(msg)
        object.__setattr__(self, "files", paths)
        object.__setattr__(self, "tree", tree if tree is not None else embedded_tree)
        object.__setattr__(self, "entry_start", entry_start)
        object.__setattr__(self, "entry_stop", entry_stop)
        object.__setattr__(self, "_cache", {})

    # -- Source protocol --------------------------------------------------------------

    def resolved_tree(self) -> str:
        """Return the tree name, auto-detecting it from the first file if needed."""
        if self.tree is not None:
            return self.tree
        if "tree" not in self._cache:
            self._cache["tree"] = _detect_tree(self.files[0])
        return str(self._cache["tree"])

    def branches(self) -> list[str]:
        """Return the branch (or RNTuple field) names of the tree in the first file.

        The result is cached on the instance; the file is opened only once.
        """
        if "branches" in self._cache:
            return list(self._cache["branches"])
        tree = self.resolved_tree()
        try:
            with uproot.open(self.files[0]) as file:
                obj = file[tree]
                keys = list(obj.keys())
                self._cache["rntuple"] = _RNTUPLE_MARKER in type(obj).__name__
        except uproot.KeyInFileError as exc:
            msg = f"tree {tree!r} not found in {self.files[0]!r}"
            raise SourceError(msg) from exc
        names = _leaf_names(keys)
        self._cache["branches"] = names
        return list(names)

    def _is_rntuple(self) -> bool:
        if "rntuple" not in self._cache:
            self.branches()
        return bool(self._cache["rntuple"])

    def num_entries(self) -> int:
        """Return the number of entries (all files, honouring the entry range)."""
        tree = self.resolved_tree()
        if "num_entries" not in self._cache:
            total = 0
            for path in self.files:
                with uproot.open(path) as file:
                    total += int(file[tree].num_entries)
            self._cache["num_entries"] = total
        total = int(self._cache["num_entries"])
        start, stop, _ = slice(self.entry_start, self.entry_stop).indices(total)
        return max(stop - start, 0)

    def read_scalar(self, key: str) -> float:
        """Sum a number stored under ``key`` in every file.

        Accepts histograms (their total content including flow bins, e.g. a
        sum-of-weights histogram) and ``TParameter`` objects (their value).
        Typical use: the number of generated events written by a framework.
        """
        total = 0.0
        for path in self.files:
            with uproot.open(path) as file:
                try:
                    obj = file[key]
                except uproot.KeyInFileError as exc:
                    msg = f"object {key!r} not found in {path!r}"
                    raise SourceError(msg) from exc
                if hasattr(obj, "axes") and hasattr(obj, "values"):  # a histogram
                    total += float(np.sum(obj.values(flow=True)))
                elif hasattr(obj, "member") and "fVal" in getattr(obj, "members", {}):
                    total += float(obj.member("fVal"))  # a TParameter
                else:
                    msg = f"object {key!r} in {path!r} ({type(obj).__name__}) holds no number"
                    raise SourceError(msg)
        return total

    def arrays(self, branches: Sequence[str]) -> dict[str, ak.Array]:
        """Read ``branches`` from all files and concatenate them."""
        tree = self.resolved_tree()
        if not branches:
            return {}
        wanted = set(branches)
        # A TTree presents split object branches by their full dotted leaf name; an
        # RNTuple presents nested fields by their short name, one level at a time, and
        # returns the parent as a record array. For RNTuples let every level through and
        # pick the nested field out of the record afterwards.
        parts = (
            {part for name in wanted for part in name.split(".")} if self._is_rntuple() else set()
        )
        try:
            # Select by name rather than by expression so that dotted sub-branch names
            # (``Collection.field.x`` in EDM4hep/podio files) and odd characters work.
            data = uproot.concatenate(
                [{path: tree} for path in self.files],
                filter_name=lambda name: name in wanted or name in parts,
                entry_start=self.entry_start,
                entry_stop=self.entry_stop,
                library="ak",
                how=dict,
            )
        except uproot.KeyInFileError as exc:
            msg = f"could not read {tree!r} from {self.files}: {exc}"
            raise SourceError(msg) from exc
        result: dict[str, ak.Array] = {}
        missing: list[str] = []
        for name in branches:
            array = ak.Array(data[name]) if name in data else _extract_nested(data, name)
            if array is None:
                missing.append(name)
            else:
                result[name] = array
        if missing:
            # RNTuple reads silently drop unknown fields; report them explicitly.
            msg = f"branches {missing} not found in tree {tree!r}"
            raise SourceError(msg)
        return result

    def describe(self) -> str:
        """Return e.g. ``'events.root'`` or ``'3 files (run*.root)'``."""
        if len(self.files) == 1:
            return Path(self.files[0]).name
        return f"{len(self.files)} files ({Path(self.files[0]).name}, ...)"

    @property
    def default_label(self) -> str:
        """A legend label derived from the first file's stem."""
        return Path(self.files[0]).stem


# --------------------------------------------------------------------------------------
# In-memory arrays
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArraySource:
    """Branch arrays already in memory.

    Parameters
    ----------
    data
        A mapping from branch name to array-like, an Awkward record array, or a
        NumPy structured array. All columns must have the same length.
    entry_start, entry_stop
        Keep only this range of entries (Python slice semantics).
    """

    data: Mapping[str, ak.Array]

    def __init__(
        self,
        data: Mapping[str, Any] | ak.Array | np.ndarray,
        *,
        entry_start: int | None = None,
        entry_stop: int | None = None,
    ) -> None:
        columns = _to_columns(data)
        lengths = {name: len(array) for name, array in columns.items()}
        if len(set(lengths.values())) > 1:
            msg = f"all arrays must have the same length, got {lengths}"
            raise SourceError(msg)
        if entry_start is not None or entry_stop is not None:
            columns = {name: array[entry_start:entry_stop] for name, array in columns.items()}
        object.__setattr__(self, "data", columns)

    def branches(self) -> list[str]:
        """Return the column names."""
        return list(self.data)

    def arrays(self, branches: Sequence[str]) -> dict[str, ak.Array]:
        """Return the requested columns."""
        missing = [b for b in branches if b not in self.data]
        if missing:
            msg = f"columns {missing} not found; available: {sorted(self.data)}"
            raise SourceError(msg)
        return {name: self.data[name] for name in branches}

    def num_entries(self) -> int:
        """Return the number of events (length of the arrays)."""
        return len(next(iter(self.data.values()))) if self.data else 0

    def describe(self) -> str:
        """Return a short description of the in-memory data."""
        n = len(next(iter(self.data.values()))) if self.data else 0
        return f"in-memory arrays ({n} events, {len(self.data)} columns)"

    @property
    def default_label(self) -> str:
        """A generic legend label for in-memory data."""
        return "arrays"


def _to_columns(data: object) -> dict[str, ak.Array]:
    if isinstance(data, Mapping):
        if not data:
            msg = "empty mapping given as data source"
            raise SourceError(msg)
        return {str(k): _as_ak(v) for k, v in data.items()}
    if isinstance(data, ak.Array):
        if not data.fields:
            msg = "Awkward array must be a record array with named fields"
            raise SourceError(msg)
        return {name: data[name] for name in data.fields}
    if isinstance(data, np.ndarray):
        if data.dtype.names is None:
            msg = "NumPy array must be a structured array with named fields"
            raise SourceError(msg)
        return {name: ak.Array(data[name]) for name in data.dtype.names}
    msg = f"unsupported in-memory data of type {type(data).__name__}"
    raise SourceError(msg)


def _as_ak(value: Any) -> ak.Array:
    if isinstance(value, ak.Array):
        return value
    try:
        return ak.Array(value)
    except Exception as exc:
        msg = f"cannot convert {type(value).__name__} to an Awkward array"
        raise SourceError(msg) from exc


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def as_source(
    data: Any,
    *,
    tree: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> Source:
    """Turn user input into a :class:`Source`.

    Strings, paths, and sequences of them become a :class:`FileSource`;
    mappings, Awkward record arrays and NumPy structured arrays become an
    :class:`ArraySource`; existing sources are returned unchanged. An entry
    range (or, for a :class:`FileSource`, a different ``tree``) cannot be
    applied to an existing source afterwards and raises :class:`SourceError`;
    give them to the source when constructing it.
    """
    if isinstance(data, FileSource | ArraySource):
        _reject_overrides(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
        return data
    if isinstance(data, str | PathLike):
        return FileSource(data, tree, entry_start=entry_start, entry_stop=entry_stop)
    if isinstance(data, Mapping | ak.Array | np.ndarray):
        return ArraySource(data, entry_start=entry_start, entry_stop=entry_stop)
    if isinstance(data, Sequence) and all(isinstance(f, str | PathLike) for f in data):
        return FileSource(data, tree, entry_start=entry_start, entry_stop=entry_stop)
    if isinstance(data, Source):
        _reject_overrides(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
        return data
    msg = (
        f"cannot interpret {type(data).__name__} as a data source; expected file path(s), "
        "a mapping of arrays, an Awkward record array, or a rootfig Source"
    )
    raise SourceError(msg)


def _reject_overrides(
    source: Source, *, tree: str | None, entry_start: int | None, entry_stop: int | None
) -> None:
    """Refuse options that cannot be applied to an already constructed source."""
    given = [
        f"{name}={value!r}"
        for name, value in (("entry_start", entry_start), ("entry_stop", entry_stop))
        if value is not None
    ]
    if tree is not None and isinstance(source, FileSource) and tree != source.resolved_tree():
        given.insert(0, f"tree={tree!r}")
    if given:
        msg = (
            f"{', '.join(given)} cannot be applied to an existing {type(source).__name__}; "
            "pass them when constructing it, e.g. "
            "FileSource(files, tree=..., entry_start=..., entry_stop=...)"
        )
        raise SourceError(msg)
