"""A read cache for one batch of histograms: branch arrays and stored histograms read once."""

from __future__ import annotations

from collections.abc import Sequence

import awkward as ak

from rootfig._typing import Hist
from rootfig.io.sources import FileSource

__all__ = ["ReadCache"]


class ReadCache:
    """Branch arrays and stored histograms read once, served to every histogram built from them.

    Keyed by a :class:`FileSource`'s value (files, tree, entry range), so two
    samples over the same files, or the fresh source a systematic variation
    builds over them, share one read. :meth:`arrays` reads only the names it
    does not hold yet and keeps them, so the cache can be warmed with the union
    of what a batch will need (:func:`rootfig.histograms.prefetch`) or simply
    fill as histograms are built. :meth:`source` hands out one instance per
    value, so what a source learns about its files is learnt once too.
    In-memory and third-party sources are not cached; callers read those
    directly. Drop the cache to release what it holds.
    """

    def __init__(self) -> None:
        self._sources: dict[FileSource, FileSource] = {}
        self._arrays: dict[FileSource, dict[str, ak.Array]] = {}
        self._histograms: dict[tuple[FileSource, str, bool], Hist] = {}

    def source(self, source: FileSource) -> FileSource:
        """Return the instance held for ``source``'s value, adopting ``source`` if it is the first.

        A :class:`FileSource` keeps what it learns about its files (branches,
        objects, tree, entry count, scalars) on the instance. Sources rebuilt
        from the same paths for every histogram, such as raw paths given as
        ``data`` or the files of a ``Systematic.samples`` variation, read through
        the first instance and so learn it once. Reading through the cache
        applies this (:func:`~rootfig.histograms.sources.shared_source`), so
        callers only need it themselves for what they read from a source
        directly.
        """
        return self._sources.setdefault(source, source)

    def arrays(self, source: FileSource, branches: Sequence[str]) -> dict[str, ak.Array]:
        """Return ``branches`` of ``source``, reading the ones not held yet.

        A fresh mapping holding exactly the requested names in their order, as
        :meth:`FileSource.arrays` returns them.
        """
        held = self._arrays.setdefault(source, {})
        missing = [name for name in dict.fromkeys(branches) if name not in held]
        if missing:
            held.update(source.arrays(missing))
        return {name: held[name] for name in branches}

    def histogram(
        self, source: FileSource, name: str, *, variances_from_contents: bool = False
    ) -> Hist:
        """Return the stored histogram ``name`` of ``source``, summed over its files, read once."""
        key = (source, name, variances_from_contents)
        if key not in self._histograms:
            self._histograms[key] = source.read_histogram(
                name, variances_from_contents=variances_from_contents
            )
        return self._histograms[key]

    def histograms(
        self, source: FileSource, names: Sequence[str], *, variances_from_contents: bool = False
    ) -> None:
        """Read the stored histograms ``names`` of ``source`` not held yet, one open per file."""
        missing = [
            name
            for name in dict.fromkeys(names)
            if (source, name, variances_from_contents) not in self._histograms
        ]
        if not missing:
            return
        read = source.read_histograms(missing, variances_from_contents=variances_from_contents)
        for name, histogram in read.items():
            self._histograms[(source, name, variances_from_contents)] = histogram
