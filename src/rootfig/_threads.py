"""How many threads rootfig reads and prepares event data in."""

from __future__ import annotations

import os

__all__ = ["THREADS"]


def _threads() -> int:
    given = os.environ.get("ROOTFIG_THREADS")
    if given is not None:
        try:
            return max(1, int(given))
        except ValueError:
            msg = f"ROOTFIG_THREADS must be a whole number of threads, got {given!r}"
            raise ValueError(msg) from None
    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    return min(8, available)


THREADS: int = _threads()
"""Threads for decompressing baskets and for preparing chunks of events, each.

Both release the GIL for most of their work, so a few threads make reading and
preparing several times faster; more add little and would crowd a shared
machine. ``ROOTFIG_THREADS`` sets the number (``1`` runs everything in the
calling thread), for instance to match the cores a batch job was given.
"""
