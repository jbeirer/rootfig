"""Turn variable, selection, and weight arrays into flat columns for histogram filling.

This module defines the semantics of combining event-level (flat) and
object-level (jagged) arrays. The rules, with ``depth`` meaning the number of
nested list levels (1 = one value per event, 2 = a list of objects per event):

==========  ==========  ============================================================
variable    selection   result
==========  ==========  ============================================================
event       event       events where the selection is true
object      event       all objects of the selected events, flattened
object      object      objects where the selection is true (same structure required)
event       object      :class:`~rootfig.errors.SelectionError`; reduce the selection
                        with ``any()``, ``all()`` or ``count()`` first
==========  ==========  ============================================================

Weights follow the variable: an event weight is broadcast onto every object
of the event, an object weight must share the variable's structure, and an
object weight cannot be used with an event-level variable
(:class:`~rootfig.errors.IncompatibleWeightError`).

Fixed-size collections (``float x[3]`` branches, two-dimensional NumPy arrays)
are treated exactly like variable-length lists: one entry per object, event
weights broadcast onto the objects.

Missing values (``None``, produced e.g. by ``max()`` of an empty list) and
non-finite values (``nan``, ``inf``) never enter a histogram; they are
dropped and counted. A missing *collection* (a whole list that is ``None``)
or a missing event weight drops the event: the objects it would have
contributed are counted as missing (at least one per missing list).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import awkward as ak
import numpy as np

from rootfig.errors import IncompatibleWeightError, RootfigWarning, SelectionError
from rootfig.expressions import Expression, ExpressionLike, parse

__all__ = [
    "Columns",
    "NonFinitePolicy",
    "boolean_mask",
    "depth_of",
    "event_mask",
    "event_weights",
    "prepare",
    "same_structure",
]

NonFinitePolicy = Literal["drop", "error"]
"""What to do with ``nan``/``inf`` values: silently drop them (with a warning) or raise."""


@dataclass(frozen=True)
class Columns:
    """Flat, finite, aligned columns ready to fill a histogram.

    Attributes
    ----------
    arrays
        One flat ``float64`` array per requested variable, all the same length.
    weights
        Weights aligned with ``arrays``, or ``None`` for unweighted data.
    n_events
        Number of events in the input arrays.
    n_selected_events
        Number of events contributing at least one entry after selection.
    n_missing
        Entries dropped because a value (or weight) was ``None``, including the
        objects of events whose collection or event weight was missing.
    n_nonfinite
        Entries dropped because a value (or weight) was ``nan`` or ``inf``
        (entries that were also missing count as missing only).
    per_object
        True if the variable was jagged (one entry per object rather than per event).
    """

    arrays: tuple[np.ndarray, ...]
    weights: np.ndarray | None
    n_events: int
    n_selected_events: int
    n_missing: int = 0
    n_nonfinite: int = 0
    per_object: bool = False

    @property
    def values(self) -> np.ndarray:
        """The first (usually only) variable column."""
        return self.arrays[0]

    @property
    def n_entries(self) -> int:
        """Number of entries that will be histogrammed."""
        return int(self.values.size)

    @property
    def sum_weights(self) -> float:
        """Sum of weights (or number of entries when unweighted)."""
        if self.weights is None:
            return float(self.n_entries)
        return float(self.weights.sum())

    def effective_weights(self) -> np.ndarray:
        """Weights as an array, ``1.0`` everywhere when unweighted."""
        if self.weights is None:
            return np.ones(self.n_entries, dtype=np.float64)
        return self.weights


# --------------------------------------------------------------------------------------
# Structure helpers
# --------------------------------------------------------------------------------------


def depth_of(array: ak.Array) -> int:
    """Return the list depth of ``array``: 1 for flat, 2 for jagged, and so on."""
    layout = ak.to_layout(array)
    if layout.parameter("__array__") in ("string", "bytestring"):
        msg = "string branches cannot be histogrammed or used in selections"
        raise SelectionError(msg)
    return int(layout.purelist_depth)


def same_structure(a: ak.Array, b: ak.Array) -> bool:
    """Return True if ``a`` and ``b`` have identical list lengths at every level."""
    depth_a, depth_b = depth_of(a), depth_of(b)
    if depth_a != depth_b or len(a) != len(b):
        return False
    return all(
        bool(ak.all(ak.num(a, axis=axis) == ak.num(b, axis=axis))) for axis in range(1, depth_a)
    )


def _kind(array: ak.Array) -> str:
    return "flat" if depth_of(array) == 1 else "jagged"


def _as_jagged(array: ak.Array) -> ak.Array:
    """Turn fixed-size (regular) dimensions into variable-length lists.

    Regular arrays broadcast and index like NumPy matrices (an event weight would
    be spread along the *object* axis, a boolean mask would flatten the result);
    as lists they follow the per-event/per-object rules of this module.
    """
    if depth_of(array) == 1:
        return array
    return ak.Array(ak.from_regular(array, axis=None))


def _leaf_counts(array: ak.Array, axis: int) -> ak.Array:
    """Count the leaf entries below each position at list level ``axis`` (``None`` -> 0)."""
    flat = array
    for level in range(depth_of(array) - 1, axis + 1, -1):
        flat = ak.flatten(flat, axis=level)
    return ak.Array(ak.fill_none(ak.num(flat, axis=axis + 1), 0))


def _drop_missing_lists(arrays: list[ak.Array], depth: int) -> tuple[list[ak.Array], int]:
    """Remove positions that are ``None`` at a list level from all ``arrays`` alike.

    ``arrays`` share their structure. A missing list (a collection that could not
    be evaluated for an event) or a missing event weight removes the event from
    every array; the entries lost are counted, at least one per missing list.
    """
    n_missing = 0
    for axis in range(depth - 1):
        missing = ak.is_none(arrays[0], axis=axis)
        for array in arrays[1:]:
            missing = missing | ak.is_none(array, axis=axis)
        missing = ak.fill_none(missing, True)
        if not ak.any(missing):
            continue
        lost = ak.fill_none(_leaf_counts(arrays[0], axis), 0)
        n_missing += int(ak.sum(np.maximum(lost[missing], 1)))
        keep = ~missing
        arrays = [array[keep] for array in arrays]
    return arrays, n_missing


def _flatten_to_numpy(array: ak.Array, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """Flatten completely; return the values (``None`` -> ``fill``) and the missing mask."""
    missing = ak.is_none(array, axis=-1)
    filled = ak.fill_none(array, fill, axis=-1)
    if depth_of(filled) > 1:
        filled = ak.flatten(filled, axis=None)
        missing = ak.flatten(missing, axis=None)
    result = ak.to_numpy(filled)
    if np.ma.isMaskedArray(result):
        result = result.filled(fill)
    return np.asarray(result), np.asarray(ak.to_numpy(missing), dtype=bool)


def _check_nonfinite_policy(nonfinite: str) -> None:
    """Reject unsupported ``nonfinite`` policies instead of silently dropping values."""
    if nonfinite not in ("drop", "error"):
        msg = f"nonfinite must be 'drop' or 'error', got {nonfinite!r}"
        raise SelectionError(msg)


def _to_float(values: np.ndarray, what: str) -> np.ndarray:
    if values.dtype.kind in "biuf":
        return values.astype(np.float64, copy=False)
    msg = f"{what} must be numeric or boolean, got dtype {values.dtype}"
    raise SelectionError(msg)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def prepare(
    arrays: Mapping[str, Any] | ak.Array,
    variables: ExpressionLike | Sequence[ExpressionLike],
    *,
    selection: ExpressionLike | None = None,
    weight: ExpressionLike | None = None,
    scale: float = 1.0,
    nonfinite: NonFinitePolicy = "drop",
    context: str = "",
    n_events: int | None = None,
) -> Columns:
    """Evaluate expressions and apply the selection/weight rules.

    Parameters
    ----------
    arrays
        Branch arrays (mapping or record array) covering every name used.
    variables
        One expression, or several that must share the same structure (used
        for 2D histograms and correlations).
    selection
        Boolean expression, or ``None`` to keep everything.
    weight
        Weight expression, or ``None`` for unit weights.
    scale
        Constant factor multiplied into the weights.
    nonfinite
        ``"drop"`` (default) removes ``nan``/``inf`` entries with a warning;
        ``"error"`` raises :class:`~rootfig.errors.SelectionError` instead.
    context
        Short text (e.g. the sample label) prepended to warnings.
    n_events
        Number of events, needed only when every expression is a constant
        (``"1"``) so that nothing in ``arrays`` gives the length.

    Returns
    -------
    Columns
        Flat aligned values and weights.
    """
    _check_nonfinite_policy(nonfinite)
    items = [variables] if isinstance(variables, str | Expression) else list(variables)
    exprs = [parse(v) for v in items]
    if not exprs:
        msg = "at least one variable expression is required"
        raise SelectionError(msg)

    if not math.isfinite(scale):
        msg = f"scale must be finite, got {scale!r}"
        raise IncompatibleWeightError(msg)
    values = [_as_jagged(expr.evaluate(arrays, length=n_events)) for expr in exprs]
    lead = values[0]
    n_events = len(lead)
    depth = depth_of(lead)
    for expr, value in zip(exprs[1:], values[1:], strict=True):
        if not same_structure(lead, value):
            msg = (
                f"variables {exprs[0].text!r} ({_kind(lead)}) and {expr.text!r} "
                f"({_kind(value)}) have different structures and cannot be combined"
            )
            raise SelectionError(msg)

    # -- weights: bring to the variable's structure -----------------------------------
    weights: ak.Array | None = None
    if weight is not None:
        weight_expr = parse(weight)
        weights = _as_jagged(weight_expr.evaluate(arrays, length=n_events))
        w_depth = depth_of(weights)
        if w_depth > depth:
            msg = (
                f"weight {weight_expr.text!r} is per-object ({_kind(weights)}) but the variable "
                f"{exprs[0].text!r} is per-event; use a per-event weight or reduce it, e.g. "
                f"'sum({weight_expr.text})'"
            )
            raise IncompatibleWeightError(msg)
        if w_depth == depth:
            if not same_structure(lead, weights):
                msg = (
                    f"weight {weight_expr.text!r} has a different structure from the variable "
                    f"{exprs[0].text!r}; per-object weights must have one value per object"
                )
                raise IncompatibleWeightError(msg)
        elif w_depth == 1:
            weights = ak.broadcast_arrays(weights, lead)[0]
        else:
            msg = (
                f"weight {weight_expr.text!r} has depth {w_depth} and cannot be broadcast to "
                f"the variable {exprs[0].text!r} of depth {depth}"
            )
            raise IncompatibleWeightError(msg)

    # -- selection ----------------------------------------------------------------------
    if selection is not None:
        mask = boolean_mask(parse(selection), arrays, length=n_events)
        s_depth = depth_of(mask)
        if s_depth == 1 and len(mask) != n_events:
            msg = "selection and variable have different numbers of events"
            raise SelectionError(msg)
        if s_depth > depth:
            sel_text = parse(selection).text
            msg = (
                f"selection {sel_text!r} is per-object ({_kind(mask)}) but the variable "
                f"{exprs[0].text!r} is per-event; reduce the selection to one value per event "
                f"with any(...), all(...) or count(...), e.g. 'any({sel_text})'"
            )
            raise SelectionError(msg)
        if s_depth == depth and depth > 1 and not same_structure(lead, mask):
            msg = (
                f"selection {parse(selection).text!r} and variable {exprs[0].text!r} are both "
                "per-object but have different structures (different collections?)"
            )
            raise SelectionError(msg)
        if 1 < s_depth < depth:
            msg = f"selection of depth {s_depth} cannot be applied to a variable of depth {depth}"
            raise SelectionError(msg)
        values = [v[mask] for v in values]
        if weights is not None:
            weights = weights[mask]
        lead = values[0]

    # -- missing collections / event weights drop the whole event -----------------------
    n_missing = 0
    if depth > 1:
        columns = [*values, *([weights] if weights is not None else [])]
        columns, n_missing = _drop_missing_lists(columns, depth)
        values, weights = columns[: len(values)], (columns[-1] if weights is not None else None)
        lead = values[0]

    # -- flatten, drop missing and non-finite -------------------------------------------
    flat_values: list[np.ndarray] = []
    missing = np.zeros(0, dtype=bool)
    for value, expr in zip(values, exprs, strict=True):
        column, column_missing = _flatten_to_numpy(value, np.nan)
        flat_values.append(_to_float(column, f"variable {expr.text!r}"))
        missing = column_missing if missing.size == 0 else missing | column_missing
    flat_weights: np.ndarray | None = None
    if weights is not None:
        column, column_missing = _flatten_to_numpy(weights, np.nan)
        flat_weights = _to_float(column, "weight")
        if flat_weights.size != flat_values[0].size:  # pragma: no cover - guarded above
            msg = "internal error: weights and values misaligned after selection"
            raise IncompatibleWeightError(msg)
        missing |= column_missing

    keep = np.ones(flat_values[0].size, dtype=bool)
    for column in flat_values:
        keep &= np.isfinite(column)
    if flat_weights is not None:
        keep &= np.isfinite(flat_weights)

    # -- selected event count: events with at least one entry that survives ---------------
    if depth == 1:
        n_selected = int(np.count_nonzero(keep))
    else:
        # After _drop_missing_lists only leaf-level None remain; they were flattened to nan
        # above and are excluded by ``keep``, so the leaf counts add up to the flat size.
        counts = np.asarray(ak.to_numpy(_leaf_counts(lead, 0)), dtype=np.int64)
        event_index = np.repeat(np.arange(len(lead)), counts)
        n_selected = int(np.unique(event_index[keep]).size)

    n_missing += int(np.count_nonzero(missing))
    n_nonfinite = int(np.count_nonzero(~keep & ~missing))
    if n_nonfinite:
        prefix = f"{context}: " if context else ""
        message = (
            f"{prefix}dropped {n_nonfinite} non-finite (nan/inf) value(s) for {exprs[0].text!r}"
        )
        if nonfinite == "error":
            raise SelectionError(message)
        warnings.warn(message, RootfigWarning, stacklevel=3)
    if not keep.all():
        flat_values = [column[keep] for column in flat_values]
        if flat_weights is not None:
            flat_weights = flat_weights[keep]

    if scale != 1.0:
        flat_weights = (
            flat_weights if flat_weights is not None else np.ones_like(flat_values[0])
        ) * scale

    return Columns(
        arrays=tuple(flat_values),
        weights=flat_weights,
        n_events=n_events,
        n_selected_events=n_selected,
        n_missing=n_missing,
        n_nonfinite=n_nonfinite,
        per_object=depth > 1,
    )


def boolean_mask(
    selection: ExpressionLike, arrays: Mapping[str, Any] | ak.Array, *, length: int | None = None
) -> ak.Array:
    """Evaluate ``selection`` and check that it is boolean; missing values become ``False``.

    Fixed-size dimensions are turned into lists (see :mod:`rootfig.selection`).

    Raises
    ------
    SelectionError
        If the selection evaluates to numbers rather than booleans, so an
        integer flag is never mistaken for an index array.
    """
    expr = parse(selection)
    mask = _as_jagged(expr.evaluate(arrays, length=length))
    mask = ak.fill_none(mask, False, axis=None)
    flat = ak.flatten(mask, axis=None) if depth_of(mask) > 1 else mask
    dtype = ak.to_numpy(flat).dtype if len(flat) else np.dtype(bool)
    if dtype.kind != "b":
        msg = (
            f"selection {expr.text!r} must be boolean but evaluates to {dtype}; compare "
            f"explicitly, e.g. '({expr.text}) != 0'"
        )
        raise SelectionError(msg)
    return ak.Array(mask)


def event_mask(
    selection: ExpressionLike, arrays: Mapping[str, Any] | ak.Array, *, length: int | None = None
) -> np.ndarray:
    """Evaluate ``selection`` to one boolean per event.

    Per-object selections count an event as passing when *any* of its objects
    passes (the cut-flow convention); missing values count as ``False``.
    ``length`` sizes constant selections when ``arrays`` is empty.
    """
    expr = parse(selection)
    mask = boolean_mask(expr, arrays, length=length)
    depth = depth_of(mask)
    if depth > 2:
        msg = (
            f"selection {expr.text!r} has depth {depth}; cut flows need per-event or "
            "per-object cuts"
        )
        raise SelectionError(msg)
    if depth == 2:
        mask = ak.any(mask, axis=1)
    return np.asarray(ak.fill_none(mask, False), dtype=bool)


def event_weights(
    weight: ExpressionLike | None,
    arrays: Mapping[str, Any] | ak.Array,
    n_events: int,
    *,
    nonfinite: NonFinitePolicy = "drop",
    context: str = "",
) -> np.ndarray:
    """Evaluate a per-event ``weight`` expression (``None`` gives unit weights).

    Missing (``None``) and non-finite weights are returned as ``nan`` so the
    caller can exclude those events; non-finite ones are reported following
    ``nonfinite`` (a :class:`~rootfig.errors.RootfigWarning`, or a
    :class:`~rootfig.errors.SelectionError` for ``"error"``).
    """
    _check_nonfinite_policy(nonfinite)
    if weight is None:
        return np.ones(n_events, dtype=float)
    expr = parse(weight)
    values = _as_jagged(expr.evaluate(arrays, length=n_events))
    if depth_of(values) != 1:
        msg = (
            f"weight {expr.text!r} is per-object ({_kind(values)}); cut flows need one weight "
            f"per event, e.g. 'sum({expr.text})'"
        )
        raise IncompatibleWeightError(msg)
    column, missing = _flatten_to_numpy(values, np.nan)
    flat = _to_float(column, "weight")
    n_nonfinite = int(np.count_nonzero(~np.isfinite(flat) & ~missing))
    if n_nonfinite:
        prefix = f"{context}: " if context else ""
        message = f"{prefix}dropped {n_nonfinite} event(s) with a non-finite weight {expr.text!r}"
        if nonfinite == "error":
            raise SelectionError(message)
        warnings.warn(message, RootfigWarning, stacklevel=3)
    return np.where(np.isfinite(flat), flat, np.nan).astype(np.float64)
