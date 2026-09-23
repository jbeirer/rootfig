"""Selection semantics: combine variable, cut and weight arrays into flat columns."""

from rootfig.selection.chunks import Request, prepare_chunks
from rootfig.selection.columns import (
    Columns,
    NonFinitePolicy,
    boolean_mask,
    depth_of,
    event_mask,
    event_weights,
    prepare,
)

__all__ = [
    "Columns",
    "NonFinitePolicy",
    "Request",
    "boolean_mask",
    "depth_of",
    "event_mask",
    "event_weights",
    "prepare",
    "prepare_chunks",
]
