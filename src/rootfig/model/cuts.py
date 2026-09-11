"""Composable selection expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from rootfig.expressions import Expression, parse

__all__ = ["Cut", "CutLike", "as_cut"]

# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cut:
    """A selection expression that composes with ``&``, ``|`` and ``~``.

    Plain strings are accepted wherever a ``Cut`` is; the class exists so
    selections can be built up and named in analysis scripts::

        base = rf.Cut("nMuon >= 2", label="2 muons")
        signal_region = base & "abs(Muon_eta) < 2.4" & ~rf.Cut("has_bjet")

    Parameters
    ----------
    expression
        A rootfig expression producing booleans (see :mod:`rootfig.expressions`).
    label
        Optional short description for legends or logging.
    """

    expression: str
    label: str | None = None

    def __post_init__(self) -> None:
        parse(self.expression)  # validate early

    def __str__(self) -> str:
        return self.expression

    def parsed(self) -> Expression:
        """Return the parsed :class:`~rootfig.expressions.Expression`."""
        return parse(self.expression)

    def _combine(self, other: CutLike, op: str) -> Cut:
        rhs = Cut(other) if isinstance(other, str) else other
        label = None
        if self.label and rhs.label:
            label = f"{self.label} {op} {rhs.label}"
        return Cut(f"({self.expression}) {op} ({rhs.expression})", label=label)

    def __and__(self, other: CutLike) -> Cut:
        return self._combine(other, "&")

    def __rand__(self, other: CutLike) -> Cut:
        return (
            Cut(other)._combine(self, "&") if isinstance(other, str) else other._combine(self, "&")
        )

    def __or__(self, other: CutLike) -> Cut:
        return self._combine(other, "|")

    def __ror__(self, other: CutLike) -> Cut:
        return (
            Cut(other)._combine(self, "|") if isinstance(other, str) else other._combine(self, "|")
        )

    def __invert__(self) -> Cut:
        label = f"not {self.label}" if self.label else None
        return Cut(f"~({self.expression})", label=label)


CutLike: TypeAlias = str | Cut
"""Anything accepted as a selection."""


def as_cut(selection: CutLike | None) -> Cut | None:
    """Coerce a string or ``Cut`` to a ``Cut`` (``None`` passes through)."""
    if selection is None or isinstance(selection, Cut):
        return selection
    if isinstance(selection, str):
        return Cut(selection)
    msg = f"selection must be a string or Cut, got {type(selection).__name__}"  # type: ignore[unreachable]
    raise TypeError(msg)
