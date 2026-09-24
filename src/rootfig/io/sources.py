"""Data sources: where branch arrays come from.

Two implementations of the :class:`Source` protocol are provided:

* :class:`FileSource` reads ROOT files with uproot. It accepts single paths,
  glob patterns, lists of either, remote URLs, and ``"file.root:tree"``
  shorthand, and works with both ``TTree`` and ``RNTuple`` objects. It also
  reads histograms stored in the files (:meth:`FileSource.read_histograms`).
* :class:`ArraySource` wraps data that is already in memory (a mapping of
  arrays, an Awkward record array, or a NumPy structured array).

Everything downstream of this module (expressions, selection, histogramming,
plotting) only sees the protocol, so it is independent of file I/O.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import awkward as ak
import numpy as np
import uproot
from uproot.interpretation.identify import UnknownInterpretation
from uproot.interpretation.objects import CannotBeAwkward

from rootfig._threads import worker_pool
from rootfig._typing import Hist
from rootfig.errors import SourceError
from rootfig.io import compat, objects
from rootfig.io.schema import record_fields, select_field

__all__ = [
    "CHUNK_BYTES",
    "ArraySource",
    "FileSource",
    "FilesLike",
    "Source",
    "as_source",
    "resolve_files",
]

FilesLike = str | PathLike[str] | Sequence[str | PathLike[str]]
"""A path, glob pattern, URL, ``"path:tree"`` string, or a sequence of those."""

_REMOTE_PREFIXES = ("root://", "http://", "https://", "s3://", "gs://", "xrootd://")

CHUNK_BYTES = 32_000_000
"""About how many bytes of branch arrays :meth:`FileSource.iterate` reads at a time.

Large enough that a chunk costs little more than its share of a whole read, small
enough that several chunks being prepared at once stay far below what a large
input takes in memory.
"""


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
    # events); rootfig falls back to reading one branch when it is absent. They may
    # provide ``branch_forms() -> dict[str, awkward.forms.Form]``, the type of every
    # branch without its values, which lets ``PlotBook`` discover variables (``rf.ALL``);
    # without it that discovery refuses the source rather than reading its arrays.


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


def _leaf_keys(keys: Sequence[str]) -> dict[str, str]:
    """Map branch names as users write them to the keys uproot lists them under.

    uproot lists split object branches of a ``TTree`` as ``Parent/Parent.field``
    plus the parent itself, and the nested fields of an ``RNTuple`` as
    ``Parent.field`` plus ``Parent``; only the leaves, named by their own
    (dotted) name without the parents, hold arrays that can be histogrammed.
    """
    parents = {key.rsplit("/", 1)[0] for key in keys if "/" in key}
    leaves = {key.rsplit("/", 1)[-1]: key for key in keys if key not in parents}
    # a leaf that some other leaf continues with a dot is a record, not an array
    records = {leaf[:i] for leaf in leaves for i, char in enumerate(leaf) if char == "."}
    return {leaf: key for leaf, key in leaves.items() if leaf not in records}


def _leaf_names(keys: Sequence[str]) -> list[str]:
    """Branch names as users write them; see :func:`_leaf_keys`."""
    return list(_leaf_keys(keys))


def _dotted_spans(names: Iterable[str]) -> set[str]:
    """Return every run of dot-separated parts of ``names``: what a level of them may be called."""
    spans: set[str] = set()
    for name in names:
        parts = name.split(".")
        spans.update(
            ".".join(parts[start:stop])
            for start in range(len(parts))
            for stop in range(start + 1, len(parts) + 1)
        )
    return spans


def _form_paths(form: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield each field's original path and form, including record parents, in schema order."""
    for field_name in record_fields(form):
        child = select_field(form, field_name)
        child_path = (*path, field_name)
        yield child_path, child
        yield from _form_paths(child, child_path)


def _leaf_forms(form: Any) -> dict[str, Any]:
    """Map the leaf fields of an ``RNTuple`` schema form to their forms, by dotted path.

    A record (also inside a collection or an option) contributes its fields,
    named ``Parent.field``; anything else, a number, a list of numbers, a
    string, is a leaf. The schema, not the spelling of the names, decides what
    is nested, so a field whose own name holds a dot keeps it. When paths have
    the same dotted spelling, prefer the longest field name at the first level
    where they differ, as :func:`_extract_nested` does. Record parents take part
    in this choice too: a literal record ``a.b`` shadows a numeric ``a -> b``.
    """
    resolved: dict[str, Any] = {}
    precedence: dict[str, tuple[int, ...]] = {}
    for path, field_form in _form_paths(form):
        name = ".".join(path)
        rank = tuple(len(part) for part in path)
        if name not in precedence or rank > precedence[name]:
            precedence[name] = rank
            resolved[name] = field_form
    return {
        name: field_form for name, field_form in resolved.items() if not record_fields(field_form)
    }


def _extract_nested(data: Mapping[str, Any] | ak.Array, name: str) -> ak.Array | None:
    """Resolve a dotted name, preferring the longest field that resolves the complete path."""
    fields = data.keys() if isinstance(data, Mapping) else data.fields
    matches = (field for field in fields if name == field or name.startswith(f"{field}."))
    for field_name in sorted(matches, key=len, reverse=True):
        array = ak.Array(data[field_name])
        if name == field_name:
            return array
        nested = _extract_nested(array, name[len(field_name) + 1 :])
        if nested is not None:
            return nested
    return None


def _picked(data: Mapping[str, Any], branches: Sequence[str], tree: str) -> dict[str, ak.Array]:
    """Return ``branches`` from what uproot read, nested fields picked out of their records.

    RNTuple reads silently drop unknown fields, so a branch that is not there is
    reported explicitly.
    """
    result: dict[str, ak.Array] = {}
    missing: list[str] = []
    for name in branches:
        array = ak.Array(data[name]) if name in data else _extract_nested(data, name)
        if array is None:
            missing.append(name)
        else:
            result[name] = array
    if missing:
        msg = f"branches {missing} not found in tree {tree!r}"
        raise SourceError(msg)
    return result


def _joined(pieces: Sequence[dict[str, ak.Array]]) -> dict[str, ak.Array]:
    """Concatenate consecutive pieces of the same branches, as one read of them returns them."""
    if len(pieces) == 1:
        return pieces[0]
    return {name: ak.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}


def _read_range(
    obj: Any, first: int, last: int, *, name_filter: Any, chunk_bytes: int, pool: Any
) -> Iterator[Mapping[str, Any]]:
    """Read entries ``first`` to ``last`` of one file's tree, about ``chunk_bytes`` at a time."""
    options: dict[str, Any] = {
        "filter_name": name_filter,
        "library": "ak",
        "how": dict,
        "decompression_executor": pool,
        "interpretation_executor": pool,
    }
    if objects.RNTUPLE_MARKER in type(obj).__name__:
        yield from _cluster_runs(obj, first, last, chunk_bytes, options)
        return
    # with uproot>=5.7.3 (io/compat.py goes): step_size=f"{chunk_bytes} B"
    step = compat.tree_step(obj, first, last, chunk_bytes, name_filter)
    yield from obj.iterate(entry_start=first, entry_stop=last, step_size=step, **options)


def _cluster_runs(
    ntuple: Any, first: int, last: int, chunk_bytes: int, options: Mapping[str, Any]
) -> Iterator[Mapping[str, Any]]:
    """Read entries ``first`` to ``last`` of an RNTuple in runs of whole clusters.

    uproot decodes every cluster a read touches in full (and the one starting where
    the read stops), so a read of part of a cluster holds all of it, and reading a
    cluster in parts decodes it once per part (``RNTuple.iterate`` steps without
    regard to clusters). A run is therefore as many consecutive clusters as fit in
    ``chunk_bytes`` at the bytes the previous run held per entry, at least one, cut
    to the range at its ends; the first run is the first cluster.
    """
    clusters = [
        (c.num_first_entry, c.num_first_entry + c.num_entries)
        for c in ntuple.cluster_summaries
        if c.num_first_entry < last and first < c.num_first_entry + c.num_entries
    ]
    per_entry = None
    at = 0
    while at < len(clusters):
        end = at + 1
        if per_entry is not None:
            while (
                end < len(clusters)
                and (clusters[end][1] - clusters[at][0]) * per_entry <= chunk_bytes
            ):
                end += 1
        start, stop = max(clusters[at][0], first), min(clusters[end - 1][1], last)
        data = ntuple.arrays(entry_start=start, entry_stop=stop, **options)
        yield data
        size = sum(array.nbytes for array in data.values())
        if size:
            per_entry = size / (stop - start)
        at = end


def _tree_in(file: Any, tree: str, files: Sequence[str]) -> Any:
    """Return ``tree`` from the open ``file``, one of ``files``, or raise a SourceError."""
    try:
        return file[tree]
    except uproot.KeyInFileError as exc:
        msg = f"could not read {tree!r} from {files}: {exc}"
        raise SourceError(msg) from exc


def _tree_names(classnames: Mapping[str, str]) -> list[str]:
    return sorted(k for k, cls in classnames.items() if objects.is_tree_class(cls))


def _detect_tree(path: str, classnames: Mapping[str, str]) -> str:
    """Return the only tree-like object among ``classnames`` (the objects of ``path``) or raise."""
    trees = _tree_names(classnames)
    if len(trees) == 1:
        return trees[0]
    if not trees:
        found = sorted({f"{k} ({cls})" for k, cls in classnames.items()})
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
            # the object map is read once and shared with the stored-histogram lookup
            self._cache["tree"] = _detect_tree(self.files[0], self.objects())
        return str(self._cache["tree"])

    @contextmanager
    def _tree_object(self) -> Iterator[Any]:
        """Open the tree in the first file for reading its metadata."""
        tree = self.resolved_tree()
        with uproot.open(self.files[0]) as file:
            try:
                obj = file[tree]
            except uproot.KeyInFileError as exc:
                msg = f"tree {tree!r} not found in {self.files[0]!r}"
                raise SourceError(msg) from exc
            self._cache["rntuple"] = objects.RNTUPLE_MARKER in type(obj).__name__
            yield obj

    def branches(self) -> list[str]:
        """Return the branch (or RNTuple field) names of the tree in the first file.

        The leaves only, a nested field by its dotted path (``Muon.pt``): the
        branches a ``TTree`` lists, the fields of an ``RNTuple``'s schema. The
        result is cached on the instance; the file is opened only once.
        """
        if "branches" not in self._cache:
            with self._tree_object() as obj:
                if self._cache["rntuple"]:
                    self._cache["forms"] = _leaf_forms(obj.to_akform()[0])
                    self._cache["branches"] = list(self._cache["forms"])
                else:
                    self._cache["branches"] = _leaf_names(list(obj.keys()))
        return list(self._cache["branches"])

    def branch_forms(self) -> dict[str, Any]:
        """Return the Awkward form of every branch of :meth:`branches`, as :meth:`arrays` reads it.

        The forms come from the tree's metadata alone (the interpretation of each
        ``TTree`` branch, the field schema of an ``RNTuple``), so no entry is
        read; they tell the type and structure of a branch, which
        :mod:`rootfig.io.schema` classifies. A ``TTree`` branch uproot has no
        interpretation for, or whose interpretation has no Awkward form, is left
        out: :meth:`arrays` could not read it either. Cached on the instance like
        :meth:`branches`.
        """
        if "forms" not in self._cache:
            with self._tree_object() as obj:
                forms: dict[str, Any] = {}
                if self._cache["rntuple"]:
                    forms = _leaf_forms(obj.to_akform()[0])
                else:
                    leaves = _leaf_keys(list(obj.keys()))
                    for name, key in leaves.items():
                        try:
                            forms[name] = obj[key].interpretation.awkward_form(obj.file)
                        except (UnknownInterpretation, CannotBeAwkward):
                            continue
                    self._cache.setdefault("branches", list(leaves))
                self._cache.setdefault("branches", list(forms))
                self._cache["forms"] = forms
        return dict(self._cache["forms"])

    def _is_rntuple(self) -> bool:
        if "rntuple" not in self._cache:
            self.branches()
        return bool(self._cache["rntuple"])

    def num_entries(self) -> int:
        """Return the number of entries (all files, honouring the entry range)."""
        start, stop, _ = slice(self.entry_start, self.entry_stop).indices(self._total_entries())
        return max(stop - start, 0)

    def _total_entries(self) -> int:
        """Count the entries of every file, before the entry range; once per instance."""
        tree = self.resolved_tree()
        if "num_entries" not in self._cache:
            total = 0
            for path in self.files:
                with uproot.open(path) as file:
                    total += int(_tree_in(file, tree, self.files).num_entries)
            self._cache["num_entries"] = total
        return int(self._cache["num_entries"])

    def read_scalar(self, key: str) -> float:
        """Sum a number stored under ``key`` in every file.

        Accepts histograms (their total content including flow bins, e.g. a
        sum-of-weights histogram) and ``TParameter`` objects (their value).
        Typical use: the number of generated events written by a framework. The
        result is cached on the instance, like :meth:`branches`, so scaling many
        histograms to a luminosity opens the files once.
        """
        scalars: dict[str, float] = self._cache.setdefault("scalars", {})
        if key in scalars:
            return scalars[key]
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
        scalars[key] = total
        return total

    # -- stored histograms ------------------------------------------------------------

    def objects(self) -> dict[str, str]:
        """Return ``{name: class}`` for every object in the first file (cycle numbers stripped).

        Objects inside directories are listed as ``"dir/name"``. The result is
        cached on the instance.
        """
        if "objects" not in self._cache:
            self._cache["objects"] = objects.object_classes(self.files[0])
        return dict(self._cache["objects"])

    def trees(self) -> list[str]:
        """Return the names of the ``TTree``/``RNTuple`` objects in the first file."""
        return _tree_names(self.objects())

    def histograms(self, ndim: int | None = None) -> list[str]:
        """Return the names of the histograms (``TH1*``, ``TH2*``) in the first file, sorted.

        ``ndim`` keeps the 1D or the 2D ones only; a histogram inside a directory
        is listed as ``"dir/name"``.
        """
        return sorted(
            k for k, cls in self.objects().items() if objects.is_histogram_class(cls, ndim)
        )

    def read_histogram(self, name: str, *, assume_poisson: bool = False) -> Hist:
        """Read the histogram stored as ``name`` in every file and return their sum.

        The result has ``Weight`` storage; see :func:`rootfig.io.objects.read_histograms`
        for how uncertainties are treated and what ``assume_poisson`` accepts.
        ``name`` may address an object inside a directory (``"dir/name"``).
        """
        return self.read_histograms([name], assume_poisson=assume_poisson)[name]

    def read_histograms(
        self, names: Sequence[str], *, assume_poisson: bool = False
    ) -> dict[str, Hist]:
        """Read the histograms stored as ``names`` in every file and return their sums, by name.

        Each file is opened once for all of them; otherwise as :meth:`read_histogram`.
        """
        return objects.read_histograms(self.files, names, assume_poisson=assume_poisson)

    def arrays(self, branches: Sequence[str]) -> dict[str, ak.Array]:
        """Read ``branches`` from all files and concatenate them."""
        tree = self.resolved_tree()
        if not branches:
            return {}
        # one pool decompresses and interprets: uproot's reading thread hands out every
        # task, and none waits on another
        with worker_pool() as pool:
            try:
                data = uproot.concatenate(
                    [{path: tree} for path in self.files],
                    filter_name=self._name_filter(branches),
                    entry_start=self.entry_start,
                    entry_stop=self.entry_stop,
                    library="ak",
                    how=dict,
                    decompression_executor=pool,
                    interpretation_executor=pool,
                )
            except uproot.KeyInFileError as exc:
                msg = f"could not read {tree!r} from {self.files}: {exc}"
                raise SourceError(msg) from exc
        return _picked(data, branches, tree)

    def iterate(
        self, branches: Sequence[str], chunk_bytes: int | None = None
    ) -> Iterator[dict[str, ak.Array]]:
        """Read ``branches`` a chunk of entries at a time, in order.

        A chunk holds about ``chunk_bytes`` of arrays, :data:`CHUNK_BYTES` unless given.

        The chunks, concatenated, are what :meth:`arrays` returns, entry range
        included, while only one of them is held here at a time; the end of a file
        and small files are joined with what follows as long as together they fit in
        a chunk. At least one chunk is yielded, an empty one when no entry is in
        range, so the types of the branches are always known.
        """
        chunk_bytes = CHUNK_BYTES if chunk_bytes is None else chunk_bytes
        held: list[dict[str, ak.Array]] = []
        size = 0
        yielded = False
        for piece in self._pieces(branches, chunk_bytes):
            piece_size = sum(array.nbytes for array in piece.values())
            if held and size + piece_size > chunk_bytes:
                yield _joined(held)
                held, size, yielded = [], 0, True
            held.append(piece)
            size += piece_size
            if size >= chunk_bytes:
                yield _joined(held)
                held, size, yielded = [], 0, True
        if held or not yielded:
            yield _joined(held) if held else self.arrays(branches)

    def _pieces(self, branches: Sequence[str], chunk_bytes: int) -> Iterator[dict[str, ak.Array]]:
        """Read ``branches`` from each file in turn, within the entry range."""
        tree = self.resolved_tree()
        start, stop = 0, None
        if self.entry_start is not None or self.entry_stop is not None:
            start, stop, _ = slice(self.entry_start, self.entry_stop).indices(self._total_entries())
        name_filter = self._name_filter(branches)
        offset = 0
        # the pool of the preparation these pieces are read for, if any, which shares it
        with worker_pool() as pool:
            for path in self.files:
                if stop is not None and offset >= stop:
                    return
                with uproot.open(path) as file:
                    obj = _tree_in(file, tree, self.files)
                    entries = int(obj.num_entries)
                    first = max(start - offset, 0)
                    last = entries if stop is None else min(stop - offset, entries)
                    offset += entries
                    if first >= last:
                        continue
                    for data in _read_range(
                        obj,
                        first,
                        last,
                        name_filter=name_filter,
                        chunk_bytes=chunk_bytes,
                        pool=pool,
                    ):
                        yield _picked(data, branches, tree)

    def _name_filter(self, branches: Sequence[str]) -> Any:
        """Return the ``filter_name`` that reads ``branches``: by name, so dots work.

        Selecting by name keeps dotted sub-branch names (``Collection.field.x`` in
        EDM4hep/podio files) and odd characters readable. A TTree presents split
        object branches by their full dotted leaf name; an RNTuple presents nested
        fields by their short name, one level at a time, and returns the parent as
        a record array. For RNTuples every level is let through (a level's name
        may itself hold dots) and the nested field is picked out afterwards.
        """
        wanted = set(branches)
        parts = _dotted_spans(wanted) if self._is_rntuple() else set()
        return lambda name: name in wanted or name in parts

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
    entry_start: int | None = None
    entry_stop: int | None = None

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
        object.__setattr__(self, "entry_start", entry_start)
        object.__setattr__(self, "entry_stop", entry_stop)

    def branches(self) -> list[str]:
        """Return the column names."""
        return list(self.data)

    def branch_forms(self) -> dict[str, Any]:
        """Return the Awkward form of every column: its type, read off the array, not its values."""
        return {name: ak.to_layout(array).form for name, array in self.data.items()}

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
