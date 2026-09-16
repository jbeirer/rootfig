"""The :class:`Variable` description of what to histogram."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any

from rootfig.expressions import Expression, parse
from rootfig.model.binning import DEFAULT_RANGE, Bins, RangeSpec, validate_bins

__all__ = ["Variable", "as_variable"]

# --------------------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")


@dataclass(frozen=True)
class Variable:
    """What to histogram and how to present it.

    Parameters
    ----------
    expression
        A rootfig expression (a branch name or a formula, see
        :mod:`rootfig.expressions`).
    bins
        Binning specification, see :data:`Bins`. ``None`` (the default) means no
        preference: :data:`DEFAULT_BINS` bins over a range inferred from the
        data when filling from a tree, and the stored binning when the variable
        names a histogram stored in a file. An integer asks for that many bins;
        a stored histogram is then rebinned to that count.
    range
        Range used when ``bins`` is an integer, see :data:`RangeSpec`. Defaults
        to :data:`DEFAULT_RANGE` (``"robust"``), which ignores far outliers such
        as ``-999`` sentinels and cuts the thin end of a tail; ``"auto"`` uses
        the full finite minimum and maximum instead.
    label
        Axis label; may contain matplotlib math text. Defaults to the
        expression.
    unit
        Physical unit appended to the axis label as ``[unit]`` and used in the
        automatic y-axis label (``Events / 2 GeV``).
    log
        Draw the x axis with a logarithmic scale.
    name
        Short identifier used for file names (:meth:`Plot.save` with a
        directory). Defaults to a sanitised version of the expression. An
        explicit name must be a plain file stem: it cannot contain path
        separators or be ``"."``/``".."``.
    """

    expression: str
    bins: Bins | None = None
    range: RangeSpec = DEFAULT_RANGE
    label: str | None = None
    unit: str | None = None
    log: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        parse(self.expression)
        validate_bins(self.bins, self.range)
        if self.name is not None and (
            self.name in (".", "..") or any(sep in self.name for sep in {"/", "\\", os.sep})
        ):
            suggestion = _SAFE_NAME_RE.sub("_", self.name).strip("_") or "variable"
            msg = (
                f"Variable name {self.name!r} must be a plain file stem (it names the file "
                "written by Plot.save(directory)) and cannot contain path separators or be "
                f"'.' or '..'; use e.g. name={suggestion!r}"
            )
            raise ValueError(msg)

    def __str__(self) -> str:
        return self.expression

    def parsed(self) -> Expression:
        """Return the parsed :class:`~rootfig.expressions.Expression`."""
        return parse(self.expression)

    @property
    def safe_name(self) -> str:
        """A file-system friendly identifier for this variable."""
        if self.name:
            return self.name
        return _SAFE_NAME_RE.sub("_", self.expression).strip("_") or "variable"

    @property
    def axis_label(self) -> str:
        """The x-axis label including the unit, e.g. ``'$p_T$ [GeV]'``."""
        base = self.label if self.label is not None else self.expression
        return f"{base} [{self.unit}]" if self.unit else base

    def replace(self, **changes: Any) -> Variable:
        """Return a copy with the given fields changed, e.g. ``var.replace(bins=20)``."""
        return replace(self, **changes)


def as_variable(variable: str | Variable, **overrides: Any) -> Variable:
    """Coerce a string or ``Variable`` to a ``Variable``, applying explicit overrides.

    Overrides whose value is ``None`` are ignored so callers can forward
    keyword arguments straight from a ``plot(...)`` signature.
    """
    effective = {k: v for k, v in overrides.items() if v is not None}
    if isinstance(variable, Variable):
        return replace(variable, **effective) if effective else variable
    if isinstance(variable, str):
        return Variable(variable, **effective)
    msg = f"variable must be a string or Variable, got {type(variable).__name__}"  # type: ignore[unreachable]
    raise TypeError(msg)
