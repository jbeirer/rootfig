"""Exception and warning types raised by rootfig.

All exceptions derive from :class:`RootfigError` so callers can catch
everything from the library with a single ``except`` clause. The more
specific subclasses also inherit from the matching built-in exception
(``ValueError``, ``KeyError``-like ``LookupError``) so generic handlers
keep working.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class RootfigError(Exception):
    """Base class for all errors raised by rootfig."""


class ExpressionError(RootfigError, ValueError):
    """An expression string could not be parsed, is not allowed, or failed to evaluate."""


class MissingBranchError(ExpressionError, LookupError):
    """An expression refers to a name that is not a branch, alias, function, or constant."""

    def __init__(
        self,
        name: str,
        *,
        available: Sequence[str] = (),
        suggestions: Sequence[str] = (),
        context: str | None = None,
    ) -> None:
        self.name = name
        self.available = tuple(available)
        self.suggestions = tuple(suggestions)
        message = f"unknown name {name!r}"
        if context:
            message += f" in {context}"
        if self.suggestions:
            message += ". Did you mean " + ", ".join(repr(s) for s in self.suggestions) + "?"
        elif self.available:
            shown = ", ".join(repr(a) for a in self.available[:12])
            more = "" if len(self.available) <= 12 else f", ... ({len(self.available)} total)"
            message += f". Available names: {shown}{more}"
        super().__init__(message)


class SelectionError(RootfigError, ValueError):
    """Variable, selection, and weight arrays have incompatible structures."""


class IncompatibleWeightError(SelectionError):
    """A weight array cannot be broadcast to the variable being histogrammed."""


class BinningError(RootfigError, ValueError):
    """A binning or range specification is invalid or cannot be inferred."""


class SourceError(RootfigError):
    """A file, tree, or in-memory data source cannot be opened or interpreted."""


class LuminosityError(RootfigError, ValueError):
    """A sample cannot be scaled to a luminosity (no luminosity, cross section or event count)."""


class SystematicError(RootfigError, ValueError):
    """A systematic variation is malformed or cannot be applied to a sample or histogram."""


class RootfigWarning(UserWarning):
    """Base class for warnings emitted by rootfig (dropped values, empty selections, ...)."""


@contextmanager
def annotate(note: str) -> Iterator[None]:
    """Attach ``note`` to any :class:`RootfigError` raised inside the block.

    The note is shown below the message (PEP 678). rootfig uses it to name the
    systematic variation an error was raised for.
    """
    try:
        yield
    except RootfigError as exc:
        exc.add_note(note)
        raise
