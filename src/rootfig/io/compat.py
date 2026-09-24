"""Workarounds for uproot 5.7.1 to 5.7.4, all of which go once rootfig requires 5.7.5.

uproot 5.7.1 is the oldest release rootfig supports, since the Key4hep stacks ship
it. Before 5.7.3, uproot converts a step given in bytes from the compressed sizes,
which well-compressed data decodes to several times over; before 5.7.5,
``RNTuple.iterate`` reads outside the entry range (5.7.1 from entry 0, 5.7.2 to
5.7.4 past its end). With ``uproot>=5.7.5`` this module is deleted, and its one
caller, ``_read_range`` in :mod:`rootfig.io.sources`, reads both formats with::

    obj.iterate(entry_start=first, entry_stop=last, step_size=f"{chunk_bytes} B", **options)

The tests of these workarounds, ``tests/test_io.py::TestUprootCompat``, go with it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from rootfig.io import objects

__all__ = ["iterate"]

RNTUPLE_PROBE = 1_000
"""Entries in the first piece of an RNTuple, whose size sets the length of the next."""


def iterate(
    obj: Any, first: int, last: int, chunk_bytes: int, **options: Any
) -> Iterator[Mapping[str, Any]]:
    """Read entries ``first`` to ``last`` of ``obj``, about ``chunk_bytes`` of arrays at a time.

    The steps are counted here, in entries. A ``TTree`` states the uncompressed
    bytes of each branch, and uproot iterates it, keeping a basket that spans two
    steps for the next. An ``RNTuple`` is read in explicit ranges, the first of at
    most :data:`RNTUPLE_PROBE` entries and each later one sized by what was read.
    """
    name_filter = options["filter_name"]
    if objects.RNTUPLE_MARKER not in type(obj).__name__:
        step = _tree_step(obj, name_filter, chunk_bytes) or last - first
        yield from obj.iterate(entry_start=first, entry_stop=last, step_size=step, **options)
        return
    step = obj.num_entries_for(
        f"{chunk_bytes} B", filter_name=name_filter, entry_start=first, entry_stop=last
    )
    step = min(step or last - first, RNTUPLE_PROBE)  # None when no field is selected
    start = first
    while start < last:
        stop = min(start + step, last)
        data = obj.arrays(entry_start=start, entry_stop=stop, **options)
        yield data
        size = sum(array.nbytes for array in data.values())
        if size:
            step = max(1, chunk_bytes * (stop - start) // size)
        start = stop


def _tree_step(tree: Any, name_filter: Any, chunk_bytes: int) -> int | None:
    """Return how many entries of ``tree`` hold about ``chunk_bytes`` of what is read, if any.

    ``name_filter`` selects branches as ``arrays`` does; a selected parent is read
    with its subbranches, so theirs count too.
    """
    read = {}
    for branch in tree.itervalues(filter_name=name_filter):
        for each in (branch, *branch.itervalues()):
            read[each.cache_key] = each
    size = sum(branch.uncompressed_bytes for branch in read.values())
    return max(1, chunk_bytes * tree.num_entries // size) if size else None
