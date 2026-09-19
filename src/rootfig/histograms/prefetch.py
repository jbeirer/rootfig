"""Warming a :class:`~rootfig.io.ReadCache` with what a batch of histograms will read."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress

from rootfig.errors import RootfigError
from rootfig.histograms.pipeline import _read_plan, _sample_systematics, branch_names
from rootfig.histograms.stored import _stored_name, _stored_source, stored_mode
from rootfig.io import FileSource, ReadCache
from rootfig.model.cuts import CutLike
from rootfig.model.inputs import PlotItem, leaf_samples
from rootfig.model.systematics import SystematicLike, as_systematics
from rootfig.model.variables import Variable

__all__ = ["prefetch"]


def prefetch(
    cache: ReadCache,
    items: Sequence[PlotItem],
    variables: Sequence[Variable],
    *,
    selections: Sequence[CutLike | None] = (None,),
    weight: str | None = None,
    systematics: Mapping[str, SystematicLike] | None = None,
    assume_poisson: bool = False,
) -> None:
    """Read into ``cache`` what :func:`build_histograms` reads for ``variables`` and ``selections``.

    One read per file source with the union of the branches the variables, the
    selections, the weight and the samples' systematics need, and one pass over
    each source's files for the stored histograms among the variables, instead
    of one read per variable and selection. Best effort, so it never raises a
    :class:`~rootfig.errors.RootfigError`: a variable, sample or selection that
    :func:`build_histograms` will refuse (an unknown name, a stored histogram
    some samples lack, a sample whose own selection forbids reading one) is
    skipped here and refused there, with the error naming it. A read that fails
    leaves its source to :func:`build_histograms` too, which reads and reports
    it as it would without a cache. The variations of ``Systematic.samples``
    read their own files and are not planned; the cache keeps what they read.
    """
    samples = leaf_samples(items)
    try:
        plot_level = as_systematics(systematics, "plot")
    except RootfigError:
        return  # refused by build_histograms for every task
    branches: dict[FileSource, list[str]] = {}
    stored: dict[FileSource, list[str]] = {}
    for variable in variables:
        try:
            is_stored = stored_mode(samples, [variable])
        except RootfigError:
            continue
        if is_stored:
            name = _stored_name([variable])
            assert name is not None  # stored_mode is only True for a bare name
            for sample in samples:
                try:
                    files = _stored_source(sample, name)
                except RootfigError:
                    continue
                _extend(stored, files, [name])
            continue
        for sample in samples:
            source = sample.source
            if not isinstance(source, FileSource):
                continue
            for selection in selections:
                try:
                    expressions, _ = _read_plan(
                        sample,
                        variable,
                        _sample_systematics(sample, plot_level),
                        selection=selection,
                        weight=weight,
                    )
                    names = branch_names(source, expressions)
                except RootfigError:
                    continue
                _extend(branches, source, names)
    # A failing read cannot be attributed to one variable here; build_histograms reads the
    # source again for each task and raises for the task that needs what cannot be read.
    for source, names in branches.items():
        with suppress(Exception):
            cache.arrays(source, names)
    for source, names in stored.items():
        with suppress(Exception):
            cache.histograms(source, names, assume_poisson=assume_poisson)


def _extend(plan: dict[FileSource, list[str]], source: FileSource, names: Sequence[str]) -> None:
    """Add ``names`` to the ordered union planned for ``source``."""
    held = plan.setdefault(source, [])
    held.extend(name for name in names if name not in held)
