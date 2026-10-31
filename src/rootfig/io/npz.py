"""Branch arrays read from NumPy ``.npz`` archives (:func:`numpy.savez`)."""

from __future__ import annotations

import io
import struct
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, TypeAlias

import awkward as ak
import numpy as np

from rootfig.errors import SourceError
from rootfig.io.sources import FilesLike, _is_remote, resolve_files

__all__ = ["NpzSource"]

_Header: TypeAlias = tuple[tuple[int, ...], np.dtype[Any]]


@dataclass(frozen=True)
class NpzSource:
    """Branch arrays read from one or more ``.npz`` files, one array per branch.

    Every array with at least one dimension is a branch with one entry per event:
    a flat array is per-event, an ``(events, n)`` array per-object, like any
    fixed-size branch. Arrays of no dimension (a scalar saved beside the events)
    are not branches. Object arrays (ragged lists saved by NumPy) are refused,
    since NumPy can only load them by unpickling, which runs code from the file.

    Parameters
    ----------
    files
        A local path, glob pattern, or a sequence of them; see :data:`FilesLike`.
        The files are joined in the order given, the matches of a glob sorted,
        and must hold the same branches.
    entry_start, entry_stop
        Keep only this range of entries, counted across all files (Python slice
        semantics). Only the files the range overlaps are read.
    """

    files: tuple[str, ...]
    entry_start: int | None = None
    entry_stop: int | None = None
    _headers_by_file: dict[str, dict[str, _Header]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __init__(
        self,
        files: FilesLike,
        *,
        entry_start: int | None = None,
        entry_stop: int | None = None,
    ) -> None:
        paths, _tree = resolve_files(files)
        remote = [path for path in paths if _is_remote(path)]
        if remote:
            msg = f".npz files are read from local paths only, got {remote}"
            raise SourceError(msg)
        object.__setattr__(self, "files", paths)
        object.__setattr__(self, "entry_start", entry_start)
        object.__setattr__(self, "entry_stop", entry_stop)
        object.__setattr__(self, "_headers_by_file", {})

    def branches(self) -> list[str]:
        """Return the names of the arrays in the first file, scalars left out."""
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
        so a short range of many large files holds only that part.
        """
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
                return {name: ak.from_numpy(_cut(archive[name], start, stop)) for name in branches}
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            msg = f"could not read {list(branches)} from {path!r}: {exc}"
            raise SourceError(msg) from exc

    def _length(self, path: str) -> int:
        headers = self._headers(path)
        return next(iter(headers.values()))[0][0] if headers else 0

    def _headers(self, path: str) -> dict[str, _Header]:
        """Return the shape and type of every branch in ``path``, read once per file."""
        if path not in self._headers_by_file:
            self._headers_by_file[path] = _read_headers(path)
        return self._headers_by_file[path]


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
    """Read the ``.npy`` header of every array in an archive, refusing unequal lengths."""
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
    lengths = {name: shape[0] for name, (shape, _dtype) in headers.items()}
    if len(set(lengths.values())) > 1:
        msg = (
            f"the arrays of {path!r} have different lengths {lengths}; every array of a "
            ".npz file is a branch with one entry per event, so save other arrays apart"
        )
        raise SourceError(msg)
    return headers
