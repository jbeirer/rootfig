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
    fill as histograms are built. In-memory and third-party sources are not
    cached; callers read those directly. Drop the cache to release what it holds.
    """

    def __init__(self) -> None:
        self._arrays: dict[FileSource, dict[str, ak.Array]] = {}
        self._histograms: dict[tuple[FileSource, str, bool], Hist] = {}

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

    def histogram(self, source: FileSource, name: str, *, assume_poisson: bool = False) -> Hist:
        """Return the stored histogram ``name`` of ``source``, summed over its files, read once."""
        key = (source, name, assume_poisson)
        if key not in self._histograms:
            self._histograms[key] = source.read_histogram(name, assume_poisson=assume_poisson)
        return self._histograms[key]

    def histograms(
        self, source: FileSource, names: Sequence[str], *, assume_poisson: bool = False
    ) -> None:
        """Read the stored histograms ``names`` of ``source`` not held yet, one open per file."""
        missing = [
            name
            for name in dict.fromkeys(names)
            if (source, name, assume_poisson) not in self._histograms
        ]
        if not missing:
            return
        read = source.read_histograms(missing, assume_poisson=assume_poisson)
        for name, histogram in read.items():
            self._histograms[(source, name, assume_poisson)] = histogram
