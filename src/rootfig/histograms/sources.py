"""Samples reading through the source instances a :class:`~rootfig.io.ReadCache` holds."""

from __future__ import annotations

from rootfig.io import FileSource, ReadCache
from rootfig.model.samples import Sample

__all__ = ["shared_source"]


def shared_source(sample: Sample, cache: ReadCache | None) -> Sample:
    """Return ``sample`` reading through the source instance ``cache`` holds for its files.

    A :class:`~rootfig.io.FileSource` keeps what it learns about its files on the
    instance (see :meth:`~rootfig.io.ReadCache.source`), so a sample whose
    source was rebuilt from the same paths (raw paths given as ``data``, the
    files of a ``Systematic.samples`` variation) is given the instance seen
    first; the sample is returned as it is when that is its own or without a
    cache. The result equals ``sample``: only the source instance may differ.
    """
    if cache is None or not isinstance(sample.source, FileSource):
        return sample
    source = cache.source(sample.source)
    return sample if source is sample.source else sample.replace(source=source)
