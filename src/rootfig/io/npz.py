"""Branch arrays and binned histograms read from NumPy ``.npz`` archives (:func:`numpy.savez`)."""

from __future__ import annotations

import io
import struct
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, TypeAlias

import awkward as ak
import hist
import numpy as np

from rootfig._storage import add_into, as_weight_storage, same_binning
from rootfig._typing import Hist
from rootfig.errors import BinningError, RootfigWarning, SourceError
from rootfig.io.sources import FilesLike, _is_remote, resolve_files

__all__ = ["NpzSource", "with_histogram_edges"]

_Header: TypeAlias = tuple[tuple[int, ...], np.dtype[Any]]

HistogramEdges: TypeAlias = Mapping[str, Sequence[str]]
"""Explicit histograms of an archive: counts array name to its edge array names, one per axis."""

_EDGE_SUFFIXES = {1: ("_edges",), 2: ("_xedges", "_yedges")}
"""The edge arrays that make ``NAME`` a 1D or 2D histogram (``NAME_edges``, ``NAME_x/yedges``)."""
_VARIANCES_SUFFIX = "_variances"
_PLACEHOLDER_AXES = ("xaxis", "yaxis")
"""Axis names (ROOT's default titles) for histograms whose edges carry no quantity name."""


@dataclass(frozen=True)
class _HistogramSpec:
    """Where the counts, edges and (optional) variances of one histogram live in an archive."""

    counts: str
    edges: tuple[str, ...]
    variances: str | None
    titled: bool
    """Whether the edge names describe the axes (explicit edges) or are the convention's."""

    def members(self) -> tuple[str, ...]:
        return (self.counts, *self.edges, *((self.variances,) if self.variances else ()))


@dataclass(frozen=True)
class _Layout:
    """The arrays of one archive, split into event branches and binned histograms."""

    branches: dict[str, _Header]
    histograms: dict[str, _HistogramSpec]
    near_misses: dict[str, str] = field(default_factory=dict)
    """Branches saved beside edge arrays named for them that do not fit, and why."""


@dataclass(frozen=True)
class NpzSource:
    """Branch arrays and binned histograms read from one or more ``.npz`` files.

    Every array with at least one dimension is a branch with one entry per event:
    a flat array is per-event, an ``(events, n)`` array per-object, like any
    fixed-size branch. Arrays of no dimension (a scalar saved beside the events)
    are not branches. Object arrays (ragged lists saved by NumPy) are refused,
    since NumPy can only load them by unpickling, which runs code from the file.

    An archive may also hold histograms that are already binned, which are read
    like ``TH1``/``TH2`` objects stored in a ROOT file (:meth:`read_histogram`),
    and are not branches. A 1D or 2D array ``NAME`` is such a histogram when the
    archive also holds its bin edges, as :func:`numpy.histogram` and
    :func:`numpy.histogram2d` return them:

    * ``NAME_edges`` (``n + 1`` edges) for a 1D ``NAME`` of ``n`` bins;
    * ``NAME_xedges`` and ``NAME_yedges`` for a 2D ``NAME`` of ``(nx, ny)`` bins;
    * optionally ``NAME_variances``, the sums of squared weights in the shape of
      ``NAME``; without it the counts are their own variances (Poisson). One of
      another shape is refused rather than ignored.

    Edges named otherwise are given with ``histograms``. Arrays named like a
    histogram whose edges do not fit (``n`` edges for ``n`` bins, say) are not
    one: they stay branches when every branch has the same length, with a
    :class:`~rootfig.errors.RootfigWarning` when such a branch is read, and the
    error for unequal lengths says which edges do not fit.

    Parameters
    ----------
    files
        A local path, glob pattern, or a sequence of them; see :data:`FilesLike`.
        The files are joined in the order given, the matches of a glob sorted,
        and must hold the same branches.
    entry_start, entry_stop
        Keep only this range of entries, counted across all files (Python slice
        semantics). Only the files the range overlaps are read.
    histograms
        Histograms whose edges do not follow the convention: the name of the
        counts array to the names of its edge arrays, one per axis in the order
        of the counts' dimensions, e.g. ``{"hist": ["g_edges", "B2_edges"]}``.
        The edge names, without an ``_edges`` suffix, become the axis titles
        (``g``, ``B2``). ``NAME_variances`` is picked up as above.
    """

    files: tuple[str, ...]
    entry_start: int | None = None
    entry_stop: int | None = None
    histogram_edges: tuple[tuple[str, tuple[str, ...]], ...] = ()
    _layouts: dict[str, _Layout] = field(default_factory=dict, repr=False, compare=False)

    def __init__(
        self,
        files: FilesLike,
        *,
        entry_start: int | None = None,
        entry_stop: int | None = None,
        histograms: HistogramEdges | None = None,
    ) -> None:
        paths, _tree = resolve_files(files)
        remote = [path for path in paths if _is_remote(path)]
        if remote:
            msg = f".npz files are read from local paths only, got {remote}"
            raise SourceError(msg)
        object.__setattr__(self, "files", paths)
        object.__setattr__(self, "entry_start", entry_start)
        object.__setattr__(self, "entry_stop", entry_stop)
        object.__setattr__(self, "histogram_edges", _explicit_edges(histograms))
        object.__setattr__(self, "_layouts", {})

    def branches(self) -> list[str]:
        """Return the names of the arrays in the first file, scalars and histograms left out."""
        return list(self._headers(self.files[0]))

    def branch_forms(self) -> dict[str, Any]:
        """Return the Awkward form of every branch, read from the array headers, not the values."""
        return {
            name: ak.to_layout(_empty(header)).form
            for name, header in self._headers(self.files[0]).items()
            if not header[1].hasobject
        }

    def arrays(self, branches: Sequence[str]) -> dict[str, ak.Array]:
        """Read ``branches`` within the entry range, joined in file order.

        Each file's arrays are cut to its part of the range before they are joined,
        so a short range of many large files holds only that part. A branch saved
        beside edge arrays named for it that do not fit (a histogram saved with
        the wrong edges, or a branch that merely shares the naming) is read with
        a :class:`~rootfig.errors.RootfigWarning`.
        """
        near_misses = self._layout(self.files[0]).near_misses
        for name in dict.fromkeys(branches):
            if name in near_misses:
                warnings.warn(
                    f"{name!r} in {self.files[0]!r} is read as an event branch, not as a binned "
                    f"histogram: {near_misses[name]}. If it is a histogram, save n + 1 edges "
                    "for n bins",
                    RootfigWarning,
                    stacklevel=2,
                )
        pieces = [self._read(path, branches, start, stop) for path, start, stop in self._spans()]
        if len(pieces) == 1:
            return pieces[0]
        return {name: ak.concatenate([p[name] for p in pieces]) for name in branches}

    def num_entries(self) -> int:
        """Return the number of events within the entry range, from the array headers."""
        return sum(stop - start for _path, start, stop in self._spans())

    def describe(self) -> str:
        """Return e.g. ``'events.npz'`` or ``'3 files (run1.npz, ...)'``."""
        if len(self.files) == 1:
            return Path(self.files[0]).name
        return f"{len(self.files)} files ({Path(self.files[0]).name}, ...)"

    @property
    def default_label(self) -> str:
        """A legend label derived from the first file's stem."""
        return Path(self.files[0]).stem

    # -- binned histograms --------------------------------------------------------------

    def histograms(self, ndim: int | None = None) -> list[str]:
        """Return the names of the binned histograms in the first file, sorted.

        ``ndim`` keeps the 1D or the 2D ones only.
        """
        specs = self._layout(self.files[0]).histograms
        return sorted(
            name for name, spec in specs.items() if ndim is None or len(spec.edges) == ndim
        )

    def read_histogram(self, name: str, *, variances_from_contents: bool = False) -> Hist:
        """Read the binned histogram ``name`` from every file and return their sum.

        The result has ``Weight`` storage and ``Variable`` axes without content in
        their flow bins. Counts without ``NAME_variances`` are their own
        variances; negative ones then leave no usable variances and are refused
        unless ``variances_from_contents`` (see
        :func:`rootfig._storage.as_weight_storage`).

        Raises
        ------
        SourceError
            If a file lacks the histogram or its arrays cannot make one.
        BinningError
            If the files disagree on its binning.
        """
        return self.read_histograms([name], variances_from_contents=variances_from_contents)[name]

    def read_histograms(
        self, names: Sequence[str], *, variances_from_contents: bool = False
    ) -> dict[str, Hist]:
        """Read the binned histograms ``names`` from every file and return their sums, by name.

        Each file is opened once for all of them; otherwise as :meth:`read_histogram`.
        """
        totals: dict[str, Hist] = {}
        for path in self.files:
            specs = self._layout(path).histograms
            missing = [name for name in names if name not in specs]
            if missing:
                msg = (
                    f"histograms {missing} not found in {path!r}; histograms present: "
                    f"{sorted(specs)}. A histogram NAME needs its edges as NAME_edges (1D) or "
                    "NAME_xedges and NAME_yedges (2D), or NpzSource(..., histograms={NAME: "
                    "[edge arrays]})"
                )
                raise SourceError(msg)
            try:
                with np.load(path, allow_pickle=False) as archive:
                    read = {
                        name: _histogram(archive, specs[name], variances_from_contents)
                        for name in dict.fromkeys(names)
                    }
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                msg = f"could not read histograms {list(names)} from {path!r}: {exc}"
                raise SourceError(msg) from exc
            for name, current in read.items():
                total = totals.get(name)
                if total is None:
                    totals[name] = current
                elif not same_binning(total, current):
                    msg = (
                        f"histogram {name!r} has different binnings in {self.files[0]!r} and "
                        f"{path!r}; files summed into one sample must agree"
                    )
                    raise BinningError(msg)
                else:
                    add_into(total, current)
        return totals

    def _spans(self) -> list[tuple[str, int, int]]:
        """Return ``(path, start, stop)`` for each file the entry range overlaps, in file order.

        A range without entries gives the first file with an empty span, so the
        branches are still checked and typed.
        """
        lengths = [self._length(path) for path in self.files]
        window = range(sum(lengths))[self.entry_start : self.entry_stop]
        spans: list[tuple[str, int, int]] = []
        offset = 0
        for path, length in zip(self.files, lengths, strict=True):
            start, stop = max(window.start - offset, 0), min(window.stop - offset, length)
            if start < stop:
                spans.append((path, start, stop))
            offset += length
        return spans or [(self.files[0], 0, 0)]

    def _read(
        self, path: str, branches: Sequence[str], start: int, stop: int
    ) -> dict[str, ak.Array]:
        headers = self._headers(path)
        missing = [name for name in branches if name not in headers]
        if missing:
            msg = f"arrays {missing} not found in {path!r}; available: {sorted(headers)}"
            raise SourceError(msg)
        ragged = [name for name in branches if headers[name][1].hasobject]
        if ragged:
            msg = (
                f"{path!r} holds {ragged} as object arrays, which NumPy loads only by "
                "unpickling, and rootfig does not unpickle files. Save ragged data in a ROOT "
                "file, or, for a file you trust, load it yourself (np.load(path, "
                "allow_pickle=True)) and pass {name: awkward.from_iter(values)}"
            )
            raise SourceError(msg)
        if start == stop:
            return {name: ak.from_numpy(_empty(headers[name])) for name in branches}
        try:
            with np.load(path, allow_pickle=False) as archive:
                # by member name: NumPy resolves ``archive["x.npy"]`` to the array ``x`` first
                return {
                    name: ak.from_numpy(_cut(archive[f"{name}.npy"], start, stop))
                    for name in branches
                }
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            msg = f"could not read {list(branches)} from {path!r}: {exc}"
            raise SourceError(msg) from exc

    def _length(self, path: str) -> int:
        headers = self._headers(path)
        return next(iter(headers.values()))[0][0] if headers else 0

    def _headers(self, path: str) -> dict[str, _Header]:
        """Return the shape and type of every branch in ``path``, read once per file."""
        return self._layout(path).branches

    def _layout(self, path: str) -> _Layout:
        """Return the branches and histograms of ``path``, read once per file."""
        if path not in self._layouts:
            self._layouts[path] = _read_layout(path, dict(self.histogram_edges))
        return self._layouts[path]


def _empty(header: _Header) -> np.ndarray[Any, Any]:
    """Return an array without entries of the shape and type ``header`` gives."""
    shape, dtype = header
    return np.empty((0, *shape[1:]), dtype)


def _cut(values: np.ndarray[Any, Any], start: int, stop: int) -> np.ndarray[Any, Any]:
    """Return entries ``start:stop`` of ``values``, a copy when part, so the rest is released."""
    return values if (start, stop) == (0, len(values)) else values[start:stop].copy()


def _array_header(stream: IO[bytes]) -> _Header:
    """Read the shape and type from the header of an ``.npy`` member, leaving its data unread."""
    version = np.lib.format.read_magic(stream)
    if version == (3, 0):
        # 3.0 is the 2.0 layout with a UTF-8 header, which NumPy's public readers decode as
        # latin-1; the header is a Python literal, so ASCII escapes read back as the same text
        (length,) = struct.unpack("<I", stream.read(4))
        text = stream.read(length).decode("utf-8").encode("ascii", "backslashreplace")
        stream, version = io.BytesIO(struct.pack("<I", len(text)) + text), (2, 0)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        msg = f".npy format version {version[0]}.{version[1]} is not supported"
        raise ValueError(msg)
    return shape, dtype


def _read_headers(path: str) -> dict[str, _Header]:
    """Read the ``.npy`` header of every array of at least one dimension in an archive."""
    headers: dict[str, _Header] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".npy"):
                    continue
                with archive.open(member) as stream:
                    header = _array_header(stream)
                if header[0]:
                    headers[member.removesuffix(".npy")] = header
    except (OSError, ValueError, struct.error, zipfile.BadZipFile) as exc:
        msg = f"could not read {path!r} as a NumPy .npz archive: {exc}"
        raise SourceError(msg) from exc
    return headers


def _read_layout(path: str, explicit: Mapping[str, tuple[str, ...]]) -> _Layout:
    """Split the arrays of ``path`` into histograms and branches; branches must agree in length."""
    headers = _read_headers(path)
    specs = {name: _explicit_spec(path, headers, name, edges) for name, edges in explicit.items()}
    used = {member for spec in specs.values() for member in spec.members()}
    for name in sorted(headers):
        if name in used or name in specs:
            continue
        spec = _conventional_spec(headers, name)
        if spec is not None and not used.intersection(spec.members()):
            specs[name] = spec
            used.update(spec.members())
    for spec in specs.values():
        _check_variances(path, headers, spec.counts)
    branches = {name: header for name, header in headers.items() if name not in used}
    near_misses = {
        name: reason
        for name in branches
        if (reason := _near_miss(headers, name, branches)) is not None
    }
    lengths = {name: shape[0] for name, (shape, _dtype) in branches.items()}
    if len(set(lengths.values())) > 1:
        detail = "".join(f"; {name!r}: {reason}" for name, reason in near_misses.items())
        msg = (
            f"the arrays of {path!r} have different lengths {lengths}; every array of a "
            ".npz file is a branch with one entry per event, so save other arrays apart. "
            "A histogram that is already binned is recognised by its edges: NAME with "
            "NAME_edges (1D), or NAME_xedges and NAME_yedges (2D), n + 1 edges for n bins; "
            "for edges named otherwise pass NpzSource(path, histograms={NAME: [edge arrays, "
            f"one per axis]}}){detail}"
        )
        raise SourceError(msg)
    return _Layout(branches, specs, near_misses)


def _near_miss(
    headers: Mapping[str, _Header], name: str, branches: Mapping[str, _Header]
) -> str | None:
    """Say why ``name`` with the edge arrays named for it is no histogram; ``None`` if it has none.

    Only edge arrays that are branches themselves count: one taken by another
    histogram describes that one.
    """
    shape = headers[name][0]
    suffixes = _EDGE_SUFFIXES.get(len(shape))
    if suffixes is None:
        return None
    edges = [f"{name}{suffix}" for suffix in suffixes]
    if not all(edge in branches for edge in edges):
        return None
    wrong = [
        f"{edge!r} has shape {headers[edge][0]}, not ({bins + 1},)"
        for edge, bins in zip(edges, shape, strict=True)
        if not _fits(headers, edge, bins)
    ]
    return f"its shape is {shape}, but {', '.join(wrong)}" if wrong else None


def _check_variances(path: str, headers: Mapping[str, _Header], name: str) -> None:
    """Refuse a ``NAME_variances`` array that does not have the shape of the counts ``name``."""
    candidate = f"{name}{_VARIANCES_SUFFIX}"
    header = headers.get(candidate)
    if header is not None and header[0] != headers[name][0]:
        msg = (
            f"{candidate!r} in {path!r} has shape {header[0]}, but the histogram {name!r} has "
            f"{headers[name][0]}; the variances must have the shape of the counts"
        )
        raise SourceError(msg)


def _explicit_edges(histograms: HistogramEdges | None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return ``histograms`` as a hashable tuple, checking its shape."""
    if histograms is None:
        return ()
    result = []
    for name, edges in histograms.items():
        names = (edges,) if isinstance(edges, str) else tuple(edges)
        if not 1 <= len(names) <= 2 or not all(isinstance(e, str) for e in names):
            msg = (
                f"histograms[{name!r}] must name one edge array per axis (one or two), "
                f"got {edges!r}"
            )
            raise SourceError(msg)
        result.append((str(name), names))
    return tuple(result)


def _variances_name(headers: Mapping[str, _Header], name: str) -> str | None:
    """Return ``NAME_variances`` if the archive holds it (its shape is checked by the layout)."""
    candidate = f"{name}{_VARIANCES_SUFFIX}"
    return candidate if candidate in headers else None


def _fits(headers: Mapping[str, _Header], edges: str, bins: int) -> bool:
    header = headers.get(edges)
    return header is not None and header[0] == (bins + 1,)


def _conventional_spec(headers: Mapping[str, _Header], name: str) -> _HistogramSpec | None:
    """Return the histogram ``name`` makes with its conventionally named edges, if they fit."""
    shape = headers[name][0]
    suffixes = _EDGE_SUFFIXES.get(len(shape))
    if suffixes is None:
        return None
    edges = tuple(f"{name}{suffix}" for suffix in suffixes)
    if not all(_fits(headers, e, n) for e, n in zip(edges, shape, strict=True)):
        return None
    return _HistogramSpec(name, edges, _variances_name(headers, name), titled=False)


def _explicit_spec(
    path: str, headers: Mapping[str, _Header], name: str, edges: tuple[str, ...]
) -> _HistogramSpec:
    """Return the histogram ``histograms=`` declares, checking the arrays against each other."""
    available = sorted(headers)
    absent = [member for member in (name, *edges) if member not in headers]
    if absent:
        msg = f"histograms=: arrays {absent} not found in {path!r}; available: {available}"
        raise SourceError(msg)
    shape = headers[name][0]
    if len(shape) != len(edges):
        msg = (
            f"histograms=: {name!r} in {path!r} has {len(shape)} dimension(s) {shape}, "
            f"so it needs as many edge arrays, got {list(edges)}"
        )
        raise SourceError(msg)
    for edge, bins in zip(edges, shape, strict=True):
        if not _fits(headers, edge, bins):
            msg = (
                f"histograms=: {edge!r} in {path!r} has shape {headers[edge][0]}, but the "
                f"matching axis of {name!r} {shape} has {bins} bins, so it needs {bins + 1} edges"
            )
            raise SourceError(msg)
    return _HistogramSpec(name, edges, _variances_name(headers, name), titled=True)


def _numeric(values: np.ndarray[Any, Any], what: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    if values.dtype.kind not in "biuf":
        msg = f"{what} must hold numbers, not {values.dtype}"
        raise ValueError(msg)
    return np.asarray(values, dtype=float)


def _axis_name(spec: _HistogramSpec, index: int) -> str:
    if not spec.titled:
        return _PLACEHOLDER_AXES[index]
    edge = spec.edges[index]
    return edge.removesuffix("_edges") or edge


def _histogram(archive: Any, spec: _HistogramSpec, variances_from_contents: bool) -> Hist:
    """Build the ``Weight``-storage histogram ``spec`` describes from the open ``archive``."""
    counts = _numeric(archive[f"{spec.counts}.npy"], f"histogram {spec.counts!r}")
    axes = []
    names = [_axis_name(spec, index) for index in range(len(spec.edges))]
    if len(names) == len(set(names)) + 1:  # two edge arrays that read as one quantity
        names[1] = f"{names[1]}_y"
    for edge, name in zip(spec.edges, names, strict=True):
        values = _numeric(archive[f"{edge}.npy"], f"edges {edge!r}")
        if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
            msg = f"edges {edge!r} of histogram {spec.counts!r} must be finite and increasing"
            raise ValueError(msg)
        axes.append(hist.axis.Variable(values, name=name, label=name))
    if spec.variances is None:
        counted = hist.Hist(*axes, storage=hist.storage.Double())
        counted.view()[...] = counts
        try:
            return as_weight_storage(counted, variances_from_contents=variances_from_contents)
        except ValueError as exc:
            msg = (
                f"histogram {spec.counts!r}: {exc}. Save the sums of squared weights as "
                f"{spec.counts}{_VARIANCES_SUFFIX} to give its uncertainties"
            )
            raise ValueError(msg) from exc
    variances = _numeric(archive[f"{spec.variances}.npy"], f"variances {spec.variances!r}")
    if np.any(variances < 0):
        msg = f"variances {spec.variances!r} of histogram {spec.counts!r} are negative"
        raise ValueError(msg)
    result = hist.Hist(*axes, storage=hist.storage.Weight())
    view: Any = result.view()
    view.value = counts
    view.variance = variances
    return result


def with_histogram_edges(source: Any, nominal: Any) -> Any:
    """Return ``source`` reading the histograms ``nominal`` names explicitly, if both are ``.npz``.

    A ``Systematic.samples`` variation given as paths holds the same arrays as the
    nominal files, so the edges mapped with ``NpzSource(histograms=...)`` apply
    to it too. Any other pair of sources, or a ``source`` with a mapping of its
    own, is returned unchanged.
    """
    if (
        isinstance(source, NpzSource)
        and isinstance(nominal, NpzSource)
        and nominal.histogram_edges
        and not source.histogram_edges
    ):
        return NpzSource(
            source.files,
            entry_start=source.entry_start,
            entry_stop=source.entry_stop,
            histograms=dict(nominal.histogram_edges),
        )
    return source
