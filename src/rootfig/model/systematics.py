"""The :class:`Systematic` description of an uncertainty source."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from rootfig._mapping import FrozenMapping
from rootfig.errors import ExpressionError, SystematicError
from rootfig.expressions import parse

__all__ = ["Systematic", "SystematicKind", "SystematicLike", "as_systematics"]

SystematicKind: TypeAlias = Literal["weight", "replace", "samples", "norm"]
"""How a variation is obtained: another weight, other branches, other data, or a factor."""


@dataclass(frozen=True, init=False)
class Systematic:
    """One source of systematic uncertainty, as an up and an optional down variation.

    ``Sample(systematics=...)`` and ``rf.plot(systematics=...)`` take a mapping
    from source name to one of these forms; only variations from other data
    need a ``Systematic`` written out (:meth:`samples`):

    =====================================  =============================================
    form                                   variation
    =====================================  =============================================
    ``"w_up"``, ``("w_up", "w_down")``     weight expression(s) replacing ``Sample.weight``
    ``0.05``                               the nominal histogram scaled by ``1 +- 0.05``; 0 <= u < 1
    ``(1.10, 0.95)``                       the nominal histogram scaled by these factors
    ``{"pt": ("pt_up", "pt_down")}``       branches replaced in variable, selection and weight
    ``Systematic.samples(up, down)``       other files or arrays (see :meth:`samples`)
    =====================================  =============================================

    Every pair is ``(up, down)``. A variation without a down direction is
    symmetrised: the down variation shifts every bin by the opposite of the up
    shift. The plot-level ``weight=``, the sample's ``scale`` and the luminosity
    scaling multiply every weight variation.

    Attributes
    ----------
    kind
        See :data:`SystematicKind`.
    up, down
        The up and down variation: a weight expression (``"weight"``), a mapping
        of branch names (``"replace"``), a dataset (``"samples"``) or a factor
        (``"norm"``). ``down`` is ``None`` for a symmetrised variation.
        Branch replacement mappings are copied and made read-only.
    """

    kind: SystematicKind
    up: Any
    down: Any = None

    def __init__(self, kind: SystematicKind, up: Any, down: Any = None) -> None:
        if kind not in ("weight", "replace", "samples", "norm"):
            msg = f"systematic kind must be 'weight', 'replace', 'samples' or 'norm', got {kind!r}"
            raise SystematicError(msg)
        if up is None:
            msg = f"a {kind!r} systematic needs an up variation"
            raise SystematicError(msg)
        if kind == "weight":
            up = _check_expression(up)
            down = None if down is None else _check_expression(down)
        elif kind == "norm":
            up = _check_factor(up)
            down = None if down is None else _check_factor(down)
        elif kind == "replace":
            up = _check_replacements(up)
            down = None if down is None else _check_replacements(down)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "down", down)

    @property
    def symmetric(self) -> bool:
        """True if the down variation is derived by mirroring the up variation."""
        return self.down is None

    @classmethod
    def samples(cls, up: Any, down: Any = None) -> Systematic:
        """Take the variation from other data: file path(s), in-memory arrays or a ``Sample``.

        Anything that is not a ``Sample`` replaces only the nominal sample's data;
        selection, weight, cross section, tree name and entry range are kept. A
        ``Sample`` is used exactly as given.
        """
        return cls("samples", up, down)


SystematicLike: TypeAlias = (
    Systematic
    | str
    | float
    | tuple[str, str]
    | tuple[float, float]
    | Mapping[str, str | tuple[str, str]]
)
"""One of the forms listed in :class:`Systematic`."""


def as_systematics(
    systematics: Mapping[str, SystematicLike] | None, context: str = ""
) -> dict[str, Systematic]:
    """Normalise a ``{name: variation}`` mapping into :class:`Systematic` objects.

    Raises
    ------
    SystematicError
        If a name is not a non-empty string or a value is not a recognised form.
    """
    if systematics is None:
        return {}
    where = f"{context}: " if context else ""
    if not isinstance(systematics, Mapping):  # runtime guard for untyped callers
        msg = (  # type: ignore[unreachable]
            f"{where}systematics must be a mapping {{name: variation}}, "
            f"got {type(systematics).__name__}"
        )
        raise SystematicError(msg)
    result: dict[str, Systematic] = {}
    for name, value in systematics.items():
        if not isinstance(name, str) or not name.strip():
            msg = f"{where}systematic names must be non-empty strings, got {name!r}"
            raise SystematicError(msg)
        try:
            result[name] = _as_systematic(value)
        except SystematicError as exc:
            msg = f"{where}systematic {name!r}: {exc}"
            raise SystematicError(msg) from exc
    return result


def _as_systematic(value: Any) -> Systematic:
    if isinstance(value, Systematic):
        return value
    if isinstance(value, str):
        return Systematic("weight", value)
    if isinstance(value, Mapping):
        return _branch_variation(value)
    if _is_number(value):
        size = _check_number(value)
        if not 0.0 <= size < 1.0:
            msg = (
                f"a relative normalisation uncertainty is a magnitude from 0 to below 1, got "
                f"{size!r}; give (up, down) factors such as (0.95, 1.05) for a directed variation"
            )
            raise SystematicError(msg)
        return Systematic("norm", 1.0 + size, 1.0 - size)
    if isinstance(value, tuple | list) and len(value) == 2:
        if all(isinstance(v, str) for v in value):
            return Systematic("weight", value[0], value[1])
        if all(_is_number(v) for v in value):
            return Systematic("norm", value[0], value[1])
    msg = (
        f"cannot interpret {value!r}; use a weight expression or an (up, down) pair of them, "
        "a relative normalisation uncertainty (0.05), an (up, down) pair of normalisation "
        "factors ((1.1, 0.95)), a mapping {branch: (up_branch, down_branch)}, or "
        "rf.Systematic.samples(...) for other files"
    )
    raise SystematicError(msg)


def _branch_variation(branches: Mapping[Any, Any]) -> Systematic:
    """Return the variation replacing branches: ``{branch: (up, down)}`` or ``{branch: up}``."""
    if not branches:
        msg = "a branch variation needs a non-empty mapping {branch: (up, down)}"
        raise SystematicError(msg)
    up: dict[str, str] = {}
    down: dict[str, str] = {}
    for name, target in branches.items():
        if isinstance(target, str):
            up[_check_name(name)] = _check_name(target)
        elif isinstance(target, tuple | list) and len(target) == 2:
            up[_check_name(name)] = _check_name(target[0])
            down[_check_name(name)] = _check_name(target[1])
        else:
            msg = f"branch {name!r} must map to a branch name or an (up, down) pair of them"
            raise SystematicError(msg)
    if down and len(down) != len(up):
        msg = (
            "give every branch an (up, down) pair, or every branch a single name for a "
            "symmetrised variation"
        )
        raise SystematicError(msg)
    return Systematic("replace", up, down or None)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_number(value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value):
        msg = f"normalisation uncertainties must be finite numbers, got {value!r}"
        raise SystematicError(msg)
    return float(value)


def _check_factor(value: Any) -> float:
    factor = _check_number(value)
    if factor <= 0:
        msg = (
            f"normalisation factors must be positive (a relative uncertainty below 1), "
            f"got {factor!r}"
        )
        raise SystematicError(msg)
    return factor


def _check_expression(expression: Any) -> str:
    if not isinstance(expression, str):
        msg = f"the weight must be an expression string, got {expression!r}"
        raise SystematicError(msg)
    try:
        parse(expression)
    except ExpressionError as exc:
        msg = f"the weight {expression!r} is not a valid expression: {exc}"
        raise SystematicError(msg) from exc
    return expression


def _check_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        msg = f"branch names must be non-empty strings, got {name!r}"
        raise SystematicError(msg)
    if "`" in name:
        msg = f"branch names cannot contain backticks, got {name!r}"
        raise SystematicError(msg)
    return name


def _check_replacements(branches: Any) -> Mapping[str, str]:
    if not isinstance(branches, Mapping) or not branches:
        msg = "a branch variation needs a non-empty mapping of branch names"
        raise SystematicError(msg)
    return FrozenMapping(
        {_check_name(name): _check_name(target) for name, target in branches.items()}
    )
