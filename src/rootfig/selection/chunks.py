"""Columns prepared a chunk of events at a time, several chunks at once in threads.

Events are independent under the selection rules, so :func:`~rootfig.selection.prepare`
gives the same columns for chunks of events, joined in order, as for all of them:
every chunk holds whole events, and the counts add up. Preparing chunk by chunk
bounds what is held at a time to the flat columns plus a few chunks, and the chunks
are prepared in parallel, since Awkward and NumPy release the GIL for most of it.
"""

from __future__ import annotations

import contextvars
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import numpy as np

from rootfig._threads import THREADS
from rootfig.errors import annotate
from rootfig.selection.columns import Columns, NonFinitePolicy, prepare, report_nonfinite

__all__ = ["Request", "prepare_chunks"]

Chunk = tuple[Mapping[str, Any], int]
"""Branch arrays of some events, and how many events they hold."""


@dataclass(frozen=True)
class Request:
    """The arguments of one :func:`~rootfig.selection.prepare` call, made for every chunk.

    Attributes
    ----------
    variables, selection, weight, scale, nonfinite, context
        As :func:`~rootfig.selection.prepare` takes them.
    substitutes
        Branches evaluated in place of others, ``{branch: replacement}``, as a
        ``replace`` systematic variation shifts them.
    note
        Added to any rootfig error the request raises, e.g. naming the systematic
        variation it evaluates.
    """

    variables: tuple[str, ...]
    selection: str | None = None
    weight: str | None = None
    scale: float = 1.0
    nonfinite: NonFinitePolicy = "drop"
    context: str = ""
    substitutes: Mapping[str, str] = field(default_factory=dict)
    note: str | None = None

    def noted(self) -> AbstractContextManager[None]:
        """Add :attr:`note` to a rootfig error raised inside."""
        return nullcontext() if self.note is None else annotate(self.note)

    def prepare(self, arrays: Mapping[str, Any], n_events: int) -> Columns:
        """Prepare one chunk, leaving its dropped non-finite values to be reported for all."""
        if self.substitutes:
            arrays = {**arrays, **{old: arrays[new] for old, new in self.substitutes.items()}}
        with self.noted():
            return prepare(
                arrays,
                list(self.variables),
                selection=self.selection,
                weight=self.weight,
                scale=self.scale,
                nonfinite=self.nonfinite,
                context=self.context,
                n_events=n_events,
                report=False,
            )


def prepare_chunks(chunks: Iterable[Chunk], requests: Sequence[Request]) -> list[Columns]:
    """Run every request on every chunk and return each request's columns for all of them.

    The columns are exactly what :func:`~rootfig.selection.prepare` returns for
    the chunks joined into one, dropped non-finite values reported once per
    request with their total. Chunks are prepared in up to
    :data:`~rootfig._threads.THREADS` threads, and no more of them are taken from
    ``chunks`` than are being prepared, so a lazily read ``chunks`` is held a few
    chunks at a time; a single chunk is prepared in the calling thread. Errors
    come from the first chunk that raises one, as they would for all at once.
    """
    parts: list[list[Columns]] = [[] for _ in requests]

    def run(chunk: Chunk) -> list[Columns]:
        return [request.prepare(*chunk) for request in requests]

    def keep(prepared: list[Columns]) -> None:
        for held, columns in zip(parts, prepared, strict=True):
            held.append(columns)

    iterator = iter(chunks)
    try:
        first = next(iterator, None)
        if first is None:
            msg = "no chunk of events to prepare; an empty read is one empty chunk"
            raise ValueError(msg)
        second = next(iterator, None)
        if second is None or THREADS == 1:
            for chunk in chain([first], [] if second is None else [second], iterator):
                keep(run(chunk))
        else:
            # the caller's floating-point error handling, which NumPy 1.x keeps per thread
            errors: dict[str, Any] = dict(np.geterr())
            if np.geterrcall() is not None:
                errors["call"] = np.geterrcall()

            def work(chunk: Chunk) -> list[Columns]:
                with np.errstate(**errors):
                    return run(chunk)

            pool = ThreadPoolExecutor(max_workers=THREADS)
            pending: deque[Future[list[Columns]]] = deque()
            try:
                for chunk in chain([first, second], iterator):
                    # in a copy of the caller's context, so that its np.errstate (a context
                    # variable from NumPy 2) and, from Python 3.14, its warning filters apply
                    pending.append(pool.submit(contextvars.copy_context().run, work, chunk))
                    if len(pending) >= THREADS:
                        keep(pending.popleft().result())
                while pending:
                    keep(pending.popleft().result())
            finally:
                pool.shutdown(cancel_futures=True)
    finally:
        # a lazily read iterator releases its file and reading threads now, also when a
        # chunk raised and the error (holding this frame) outlives the call
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    joined = [Columns.concatenate(held) for held in parts]
    for request, columns in zip(requests, joined, strict=True):
        with request.noted():
            report_nonfinite(
                columns.n_nonfinite,
                request.variables[0],
                nonfinite=request.nonfinite,
                context=request.context,
                stacklevel=3,
            )
    return joined
