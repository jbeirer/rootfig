"""Branch arrays read from NumPy ``.npz`` archives (:func:`numpy.savez`)."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

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
        The files of a sample are joined in sorted order and must hold the same
        branches.
    entry_start, entry_stop
        Keep only this range of entries, counted across all files (Python slice
        semantics).
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
            name: ak.to_layout(np.empty((0, *shape[1:]), dtype)).form
            for name, (shape, dtype) in self._headers(self.files[0]).items()
            if not dtype.hasobject
        }

    def arrays(self, branches: Sequence[str]) -> dict[str, ak.Array]:
        """Read ``branches`` from every file, joined in file order, within the entry range."""
        pieces = [self._read(path, branches) for path in self.files]
        if len(pieces) == 1:
            columns = pieces[0]
        else:
            columns = {name: ak.concatenate([p[name] for p in pieces]) for name in branches}
        if self.entry_start is not None or self.entry_stop is not None:
            columns = {name: a[self.entry_start : self.entry_stop] for name, a in columns.items()}
        return columns

    def num_entries(self) -> int:
        """Return the number of events within the entry range, from the array headers."""
        total = sum(self._length(path) for path in self.files)
        return len(range(total)[self.entry_start : self.entry_stop])

    def describe(self) -> str:
        """Return e.g. ``'events.npz'`` or ``'3 files (run1.npz, ...)'``."""
        if len(self.files) == 1:
            return Path(self.files[0]).name
        return f"{len(self.files)} files ({Path(self.files[0]).name}, ...)"

    @property
    def default_label(self) -> str:
        """A legend label derived from the first file's stem."""
        return Path(self.files[0]).stem

    def _read(self, path: str, branches: Sequence[str]) -> dict[str, ak.Array]:
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
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {name: ak.from_numpy(archive[name]) for name in branches}
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


def _read_headers(path: str) -> dict[str, _Header]:
    """Read the ``.npy`` header of every array in an archive, refusing unequal lengths."""
    headers: dict[str, _Header] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".npy"):
                    continue
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    # format 3.0 has the 2.0 layout, its header only being UTF-8
                    read = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, _fortran, dtype = read(stream)
                if shape:
                    headers[member.removesuffix(".npy")] = (shape, dtype)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
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
