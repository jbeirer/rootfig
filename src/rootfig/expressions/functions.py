"""Functions and constants available inside expressions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Final

import awkward as ak
import numpy as np

from rootfig.errors import ExpressionError

__all__ = ["CONSTANTS", "FUNCTIONS"]

# --------------------------------------------------------------------------------------


def _require_jagged(name: str, array: Any) -> Any:
    if not isinstance(array, ak.Array):
        array = ak.Array(array)
    if array.layout.purelist_depth < 2:
        msg = (
            f"{name}() reduces over the objects in each event and needs a jagged "
            f"(per-object) array, but got a flat (per-event) array"
        )
        raise ExpressionError(msg)
    return array


def _reduction(name: str, fn: Callable[..., Any]) -> Callable[[Any], Any]:
    def reduce(array: Any) -> Any:
        return fn(_require_jagged(name, array), axis=-1)

    reduce.__name__ = name
    return reduce


def _clip(x: Any, lo: Any, hi: Any) -> Any:
    return np.minimum(np.maximum(x, lo), hi)


def _first(array: Any) -> Any:
    return ak.firsts(_require_jagged("first", array), axis=-1)


# -- kinematics from Cartesian components (EDM4hep stores momentum.x/y/z and energy) ----


def _pt(px: Any, py: Any) -> Any:
    return np.hypot(px, py)


def _p(px: Any, py: Any, pz: Any) -> Any:
    return np.sqrt(px * px + py * py + pz * pz)


def _theta(px: Any, py: Any, pz: Any) -> Any:
    return np.arctan2(np.hypot(px, py), pz)


def _costheta(px: Any, py: Any, pz: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):  # nan for a zero momentum
        return pz / _p(px, py, pz)


def _eta(px: Any, py: Any, pz: Any) -> Any:
    with np.errstate(divide="ignore"):  # +-inf along the beam axis
        return np.arcsinh(pz / np.hypot(px, py))


def _phi(px: Any, py: Any) -> Any:
    return np.arctan2(py, px)


def _mass(energy: Any, px: Any, py: Any, pz: Any) -> Any:
    return np.sqrt(np.maximum(energy * energy - (px * px + py * py + pz * pz), 0.0))


FUNCTIONS: Final[Mapping[str, Callable[..., Any]]] = MappingProxyType(
    {
        # element-wise (NumPy ufuncs dispatch to Awkward automatically)
        "abs": np.abs,
        "sqrt": np.sqrt,
        "cbrt": np.cbrt,
        "exp": np.exp,
        "expm1": np.expm1,
        "log": np.log,
        "log10": np.log10,
        "log2": np.log2,
        "log1p": np.log1p,
        "power": np.power,
        "hypot": np.hypot,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "arctan2": np.arctan2,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsinh": np.arcsinh,
        "arccosh": np.arccosh,
        "arctanh": np.arctanh,
        "deg2rad": np.deg2rad,
        "rad2deg": np.rad2deg,
        "floor": np.floor,
        "ceil": np.ceil,
        "round": np.rint,
        "trunc": np.trunc,
        "sign": np.sign,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "clip": _clip,
        "isnan": np.isnan,
        "isinf": np.isinf,
        "isfinite": np.isfinite,
        "where": ak.where,
        # per-event reductions over jagged branches
        "count": _reduction("count", ak.num),
        "len": _reduction("len", ak.num),
        "sum": _reduction("sum", ak.sum),
        "prod": _reduction("prod", ak.prod),
        "min": _reduction("min", ak.min),
        "max": _reduction("max", ak.max),
        "mean": _reduction("mean", ak.mean),
        "std": _reduction("std", ak.std),
        "any": _reduction("any", ak.any),
        "all": _reduction("all", ak.all),
        "argmin": _reduction("argmin", ak.argmin),
        "argmax": _reduction("argmax", ak.argmax),
        "first": _first,
        # kinematics from Cartesian components
        "pt": _pt,
        "p": _p,
        "theta": _theta,
        "costheta": _costheta,
        "eta": _eta,
        "phi": _phi,
        "mass": _mass,
    }
)
"""Functions callable from expressions, keyed by the name used in the expression."""

CONSTANTS: Final[Mapping[str, float]] = MappingProxyType(
    {"pi": math.pi, "e": math.e, "inf": math.inf, "nan": math.nan}
)
"""Named constants available in expressions when no branch of that name exists."""
