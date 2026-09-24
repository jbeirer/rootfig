"""How many threads rootfig reads and prepares event data in, and the pool they share."""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = ["THREADS", "worker_pool"]


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
"""Worker threads that decompress baskets and prepare chunks of events, together.

Both release the GIL for most of their work, so a few threads make reading and
preparing several times faster; more add little and would crowd a shared
machine. ``ROOTFIG_THREADS`` sets the number (``1`` runs everything in the
calling thread), for instance to match the cores a batch job was given.
"""

_LENT: ContextVar[ThreadPoolExecutor | None] = ContextVar("rootfig_worker_pool", default=None)


@contextmanager
def worker_pool(*, lend: bool = False) -> Iterator[ThreadPoolExecutor | None]:
    """Give the block :data:`THREADS` worker threads; ``None`` for one (the calling thread).

    With ``lend``, every block inside takes this pool instead of starting its own:
    chunks are read while earlier ones are prepared, and one pool keeps both to
    :data:`THREADS` threads together. A pool lives for its block only, so no thread
    outlives it (and none is inherited, dead, by a forked process). Only a plain
    function may lend: a generator would lend to its caller between its yields.
    """
    if THREADS == 1:
        yield None
        return
    lent = _LENT.get()
    if lent is not None:
        yield lent
        return
    pool = ThreadPoolExecutor(max_workers=THREADS)
    token = _LENT.set(pool) if lend else None
    try:
        yield pool
    finally:
        if token is not None:
            _LENT.reset(token)
        pool.shutdown(cancel_futures=True)
