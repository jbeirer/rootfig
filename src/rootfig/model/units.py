"""Numbers with units: luminosities, cross sections and centre-of-mass energies.

Used for label text (``"10.8 ab^-1"``, ``"240 GeV"``) and for scaling
simulated samples to a luminosity (``Sample(xsec=...)`` with ``lumi=...``).
Numbers without a unit are interpreted in the caller's default unit; strings
may carry any of the usual spellings of a unit.
"""

from __future__ import annotations

import math
import re

from rootfig.errors import LuminosityError

__all__ = ["cross_section_pb", "luminosity_fb", "split_quantity"]

_QUANTITY_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")

_AREA_IN_PB: dict[str, float] = {
    "b": 1e12,
    "mb": 1e9,
    "ub": 1e6,
    "μb": 1e6,
    "µb": 1e6,
    "nb": 1e3,
    "pb": 1.0,
    "fb": 1e-3,
    "ab": 1e-6,
    "zb": 1e-9,
}


def split_quantity(
    value: float | str | None, default_unit: str, *, inverse: bool = False
) -> tuple[str, str] | None:
    """Split ``value`` into ``(number, unit)`` text, the unit in mathtext form.

    Numbers use ``default_unit``; strings may carry their own unit in any of the
    common spellings (``"10.8 ab^-1"``, ``"10.8 ab⁻¹"``, ``"10.8 ab$^{-1}$"``,
    ``"240 GeV"``). With ``inverse=True`` a bare area unit such as ``"ab"`` is
    read as ``ab^{-1}`` (luminosities are always inverse areas). Text that does
    not start with a number (``"Run 2"``) is returned as-is with an empty unit.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return (f"{value:g}", default_unit)
    match = _QUANTITY_RE.match(value)
    if match is None or not match.group(1):
        return (value.strip(), "")
    number, unit = match.group(1), match.group(2)
    if not unit:
        return (number, default_unit)
    unit = unit.replace("$", "").replace("\\mathrm{", "").replace("}", "").replace("{", "")
    unit = unit.replace("⁻¹", "^-1").replace("^-1", "^{-1}").replace("-1", "^{-1}")
    unit = unit.replace("^{^{-1}}", "^{-1}")
    if inverse and not unit.endswith("^{-1}"):
        unit += "^{-1}"
    return (number, unit)


def _area_factor(unit: str, what: str) -> float:
    base = unit.removesuffix("^{-1}").strip()
    try:
        return _AREA_IN_PB[base]
    except KeyError:
        known = ", ".join(u for u in _AREA_IN_PB if u.isascii())
        msg = f"unknown unit {unit!r} for {what}; use one of {known}"
        raise LuminosityError(msg) from None


def _magnitude(
    value: float | str, default_unit: str, what: str, *, inverse: bool
) -> tuple[float, str]:
    """Split ``value`` into a validated (finite, non-negative) number and its unit."""
    if not isinstance(value, str):
        magnitude, unit = float(value), default_unit  # no round trip through label text
    else:
        parts = split_quantity(value, default_unit, inverse=inverse)
        assert parts is not None
        number, unit = parts
        try:
            magnitude = float(number)
        except ValueError:
            msg = f"cannot read {what} from {value!r}"
            raise LuminosityError(msg) from None
    if not (math.isfinite(magnitude) and magnitude >= 0):
        msg = f"{what} must be a finite, non-negative number, got {value!r}"
        raise LuminosityError(msg)
    return magnitude, unit


def cross_section_pb(value: float | str) -> float:
    """Convert to pb: numbers are taken as pb, strings may carry a unit (``"1.2 fb"``)."""
    magnitude, unit = _magnitude(value, "pb", "a cross section", inverse=False)
    if unit.endswith("^{-1}"):
        msg = f"a cross section cannot be an inverse area: {value!r}"
        raise LuminosityError(msg)
    return magnitude * _area_factor(unit, "a cross section")


def luminosity_fb(value: float | str) -> float:
    """Convert to fb^-1: numbers are taken as fb^-1, strings may carry a unit (``"10.8 ab^-1"``)."""
    magnitude, unit = _magnitude(value, "fb^{-1}", "a luminosity", inverse=True)
    # 1 ab^-1 = 1000 fb^-1: inverse areas scale inversely to areas
    return magnitude * _AREA_IN_PB["fb"] / _area_factor(unit, "a luminosity")
