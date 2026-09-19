"""Warming a :class:`~rootfig.io.ReadCache` with what a batch of histograms will read."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress

from rootfig.errors import RootfigError
from rootfig.histograms.pipeline import _read_plan, _sample_systematics, branch_names
from rootfig.histograms.sources import shared_source
from rootfig.histograms.stored import _stored_name, _stored_source, stored_mode
from rootfig.io import FileSource, ReadCache
from rootfig.model.cuts import CutLike
from rootfig.model.inputs import PlotItem, leaf_samples
from rootfig.model.samples import Sample
from rootfig.model.systematics import SystematicLike, as_systematics
from rootfig.model.variables import Variable

__all__ = ["ReadPlan", "prefetch"]


class ReadPlan:
    """What a batch of histograms will read, gathered before anything is read.

    :meth:`add` plans one configuration (the items, variables and selections of
    :func:`build_histograms`, with its weight and systematics) and :meth:`read`
    reads every source once for all of them, so configurations that need
    different branches, such as two variants with different weights, still cost
    one pass over each file. :func:`prefetch` is the two together for one
    configuration.
    """

    def __init__(self, cache: ReadCache) -> None:
        self.cache = cache
        self._branches: dict[FileSource, list[str]] = {}
        self._stored: dict[tuple[FileSource, bool], list[str]] = {}

    def add(
        self,
        items: Sequence[PlotItem],
        variables: Sequence[Variable],
        *,
        selections: Sequence[CutLike | None] = (None,),
        weight: str | None = None,
        systematics: Mapping[str, SystematicLike] | None = None,
        assume_poisson: bool = False,
    ) -> None:
        """Plan what :func:`build_histograms` reads for ``variables`` and ``selections``.

        Never raises a :class:`~rootfig.errors.RootfigError`: a variable, sample
        or selection that :func:`build_histograms` will refuse (an unknown name,
        a stored histogram some samples lack, a sample whose own selection
        forbids reading one) is left out here and refused there, with the error
        naming it. The variations of ``Systematic.samples`` read their own files
        and are not planned; the cache keeps what they read.
        """
        samples = [shared_source(sample, self.cache) for sample in leaf_samples(items)]
        try:
            plot_level = as_systematics(systematics, "plot")
        except RootfigError:
            return  # refused by build_histograms for every task
        for variable in variables:
            try:
                is_stored = stored_mode(samples, [variable])
            except RootfigError:
                continue
            if is_stored:
                self._add_stored(samples, variable, assume_poisson=assume_poisson)
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
                    _extend(self._branches, source, names)

    def _add_stored(
        self, samples: Sequence[Sample], variable: Variable, *, assume_poisson: bool
    ) -> None:
        """Plan the stored histogram ``variable`` names, for the samples that provide it."""
        name = _stored_name([variable])
        assert name is not None  # stored_mode is only True for a bare name
        for sample in samples:
            try:
                files = _stored_source(sample, name)
            except RootfigError:
                continue
            _extend(self._stored, (files, assume_poisson), [name])

    def read(self) -> None:
        """Read what was planned into the cache, once per source.

        A read that fails is left to :func:`build_histograms`, which reads the
        source again for each task and raises for the task that needs what
        cannot be read; a failure here belongs to no single task.
        """
        for source, names in self._branches.items():
            with suppress(Exception):
                self.cache.arrays(source, names)
        for (source, assume_poisson), names in self._stored.items():
            with suppress(Exception):
                self.cache.histograms(source, names, assume_poisson=assume_poisson)


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
    of one read per variable and selection. A :class:`ReadPlan` planned and
    read; see there for what is left out and for planning several
    configurations together.
    """
    plan = ReadPlan(cache)
    plan.add(
        items,
        variables,
        selections=selections,
        weight=weight,
        systematics=systematics,
        assume_poisson=assume_poisson,
    )
    plan.read()


def _extend[K](plan: dict[K, list[str]], key: K, names: Sequence[str]) -> None:
    """Add ``names`` to the ordered union planned for ``key``."""
    held = plan.setdefault(key, [])
    held.extend(name for name in names if name not in held)
