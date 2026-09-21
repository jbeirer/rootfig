"""What a source's branches hold, read from their Awkward forms rather than from the data.

A form describes the type and structure of an array without its values: uproot
derives one from a ``TTree`` branch's interpretation or an ``RNTuple``'s field
schema, and an in-memory array carries its own. :func:`plottable_names` tells
which branches the 1D histogram path can fill from, which is how
:class:`~rootfig.PlotBook` discovers variables (``rf.ALL``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import awkward as ak
import numpy as np

__all__ = ["leaf_dtype", "plottable", "plottable_names", "record_fields", "select_field"]

_TEXT_ARRAYS = frozenset({"string", "bytestring", "char", "byte"})
"""Awkward's ``__array__`` parameters marking lists of characters, which are text, not numbers."""

_PLOTTABLE_KINDS = "biuf"
"""NumPy dtype kinds the selection layer converts to float64: bool, signed, unsigned, float."""


def leaf_dtype(form: Any) -> np.dtype[Any] | None:
    """Return the NumPy dtype of the values at the bottom of ``form``, or ``None``.

    Lists, fixed-size dimensions, missing-value wrappers and indirections are
    looked through, since they hold no values of their own. A record, a union, a
    string or bytestring, and anything else that is not a plain array of one
    numeric type gives ``None``. An array without a type (every list empty) is
    ``float64``, which is what Awkward converts it to.
    """
    while True:
        if form.parameter("__array__") in _TEXT_ARRAYS:
            return None
        if isinstance(form, ak.forms.NumpyForm):
            dtype: np.dtype[Any] = np.dtype(form.primitive)
            return dtype
        if isinstance(form, ak.forms.EmptyForm):
            return np.dtype(np.float64)
        content = getattr(form, "content", None)
        if content is None or isinstance(form, ak.forms.RecordForm | ak.forms.UnionForm):
            return None
        form = content


def plottable(form: Any) -> bool:
    """Whether a branch of this form can be histogrammed: numeric or boolean leaves.

    Lists of them, fixed-size arrays and missing values included; the same
    values the selection layer accepts when it prepares the columns of a plot.
    """
    dtype = leaf_dtype(form)
    return dtype is not None and dtype.kind in _PLOTTABLE_KINDS


def plottable_names(forms: Mapping[str, Any]) -> list[str]:
    """Return the names of ``forms`` (branch name to form) that :func:`plottable` accepts."""
    return [name for name, form in forms.items() if plottable(form)]


def record_fields(form: Any) -> tuple[str, ...]:
    """Return the field names of the records ``form`` describes, through any list or option wrapper.

    Empty when ``form`` describes no records (plain numbers, strings, unions).
    """
    while not isinstance(form, ak.forms.RecordForm):
        content = getattr(form, "content", None)
        if content is None or isinstance(form, ak.forms.UnionForm):
            return ()
        form = content
    return tuple(form.fields)


def select_field(form: Any, field: str) -> Any:
    """Return the form of ``field`` of the records ``form`` describes.

    A record gives the form of that field; a list, fixed-size dimension or
    missing-value wrapper around records gives the field inside the same
    wrapper, so the field of a collection stays a list per event, as reading
    ``Collection.field`` returns it.

    Raises
    ------
    KeyError
        If ``form`` holds no such field.
    """
    if isinstance(form, ak.forms.RecordForm):
        if field not in form.fields:
            raise KeyError(field)
        return form.content(field)
    content = getattr(form, "content", None)
    if content is None or isinstance(form, ak.forms.UnionForm):
        raise KeyError(field)
    return form.copy(content=select_field(content, field))
