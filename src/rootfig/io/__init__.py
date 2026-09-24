"""Reading branch arrays from ROOT files (uproot) or from memory."""

from rootfig.io.cache import ReadCache
from rootfig.io.sources import (
    CHUNK_BYTES,
    ArraySource,
    FilesLike,
    FileSource,
    Source,
    as_source,
    resolve_files,
)

__all__ = [
    "CHUNK_BYTES",
    "ArraySource",
    "FileSource",
    "FilesLike",
    "ReadCache",
    "Source",
    "as_source",
    "resolve_files",
]
