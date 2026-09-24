"""Workarounds for uproot 5.7.1 and 5.7.2, which go once rootfig requires 5.7.3.

uproot 5.7.1 is the oldest release rootfig supports, since the Key4hep stacks ship
it. Before 5.7.3, uproot converts a ``TTree`` step given in bytes from the
compressed basket sizes, which well-compressed data decodes to several times over.
With ``uproot>=5.7.3`` this module is deleted, and its one caller, ``_read_range``
in :mod:`rootfig.io.sources`, passes ``step_size=f"{chunk_bytes} B"`` instead of
:func:`tree_step`.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

__all__ = ["tree_step"]


def tree_step(tree: Any, first: int, last: int, chunk_bytes: int, name_filter: Any) -> int:
    """Return how many entries of ``tree`` from ``first`` to ``last`` hold about ``chunk_bytes``.

    As uproot 5.7.3 and later count a step given in bytes: the uncompressed bytes of
    the baskets that overlap the range, read from their keys, over the entries of
    the range. ``name_filter`` selects branches as ``arrays`` does; a selected
    parent is read with its subbranches, which count too.
    """
    branches = {}
    for branch in tree.itervalues(filter_name=name_filter):
        for each in (branch, *branch.itervalues()):
            branches[each.cache_key] = each
    size = 0
    for branch in branches.values():
        for basket, (start, stop) in enumerate(pairwise(branch.entry_offsets)):
            if start < last and first < stop:
                size += _uncompressed_bytes(branch, basket)
    return max(1, round(chunk_bytes * (last - first) / size)) if size else max(1, last - first)


def _uncompressed_bytes(branch: Any, basket: int) -> int:
    """Return the uncompressed bytes of a basket, header included, without decompressing it.

    A basket written separately has them in its key (``basket_uncompressed_bytes``
    of uproot 5.7.1 decompresses the basket instead); one kept in the branch
    (embedded), which has no key, is already in memory.
    """
    try:
        key = branch.basket_key(basket)
    except ValueError:  # embedded
        return int(branch.basket(basket).uncompressed_bytes)
    return int(key.data_uncompressed_bytes + key.fKeylen)
