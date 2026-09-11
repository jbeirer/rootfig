"""Reading branch arrays from ROOT files (uproot) or from memory."""

from rootfig.io.sources import (
    ArraySource,
    FilesLike,
    FileSource,
    Source,
    as_source,
    resolve_files,
)

__all__ = ["ArraySource", "FileSource", "FilesLike", "Source", "as_source", "resolve_files"]
