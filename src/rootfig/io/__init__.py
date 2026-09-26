"""Reading branch arrays from ROOT files (uproot) or from memory."""

from rootfig.io.cache import ReadCache
from rootfig.io.objects import ErrorOption, stored_error_option
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
    "ErrorOption",
    "FileSource",
    "FilesLike",
    "ReadCache",
    "Source",
    "as_source",
    "resolve_files",
    "stored_error_option",
]
