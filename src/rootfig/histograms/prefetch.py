"""Warming a :class:`~rootfig.io.ReadCache` with what a batch of histograms will read."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress

from rootfig.errors import RootfigError
from rootfig.histograms.pipeline import (
    _read_plan,
    _sample_systematics,
    branch_names,
)
from rootfig.histograms.pipeline import _variant_sample as _varied_sample
from rootfig.histograms.sources import shared_source
from rootfig.histograms.stored import _stored_name, _stored_source, stored_mode
from rootfig.histograms.stored import _variant_sample as _varied_stored_sample
from rootfig.io import FileSource, ReadCache
from rootfig.model.cuts import CutLike
from rootfig.model.inputs import PlotItem, leaf_samples
from rootfig.model.samples import Sample
from rootfig.model.systematics import Systematic, SystematicLike, as_systematics
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
        variances_from_contents: bool = False,
    ) -> None:
        """Plan what :func:`build_histograms` reads for ``variables`` and ``selections``.

        The files a ``Systematic.samples`` variation reads are planned like the
        samples' own, so an alternative generator or calibration is read once
        for the whole batch too.

        Never raises a :class:`~rootfig.errors.RootfigError`: a variable, sample
        or selection that :func:`build_histograms` will refuse (an unknown name,
        a stored histogram some samples lack, a sample whose own selection
        forbids reading one) is left out here and refused there, with the error
        naming it.
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
                self._add_stored(
                    samples, variable, plot_level, variances_from_contents=variances_from_contents
                )
                continue
            for sample in samples:
                systematics_of = _sample_systematics(sample, plot_level)
                for reading, own in self._reading_samples(sample, systematics_of):
                    source = reading.source
                    if not isinstance(source, FileSource):
                        continue
                    for selection in selections:
                        try:
                            expressions, _ = _read_plan(
                                reading, variable, own, selection=selection, weight=weight
                            )
                            names = branch_names(source, expressions)
                        except RootfigError:
                            continue
                        _extend(self._branches, source, names)

    def _reading_samples(
        self, sample: Sample, systematics: Mapping[str, Systematic]
    ) -> list[tuple[Sample, Mapping[str, Systematic]]]:
        """``sample`` and the samples its ``Systematic.samples`` variations fill from.

        Each with the systematics that decide what it reads: the sample's own
        (their weight expressions and replacement branches join its read), and
        none for a variation, which is filled like a plain sample.
        """
        reading = [(sample, systematics)]
        for name, syst in systematics.items():
            if syst.kind != "samples":
                continue
            for direction, spec in (("up", syst.up), ("down", syst.down)):
                if spec is None:
                    continue  # mirrored from the up shift: nothing of its own to read
                context = f"{sample.label} [{name} {direction}]"
                try:
                    variant = _varied_sample(sample, spec, context, cache=self.cache)
                except RootfigError:
                    continue
                reading.append((variant, {}))
        return reading

    def _add_stored(
        self,
        samples: Sequence[Sample],
        variable: Variable,
        plot_level: Mapping[str, Systematic] | None,
        *,
        variances_from_contents: bool,
    ) -> None:
        """Plan the stored histogram ``variable`` names, for the samples that provide it.

        The files of a ``Systematic.samples`` variation hold the same name and
        are planned with them, so a batch reads each of those files once too.
        """
        name = _stored_name([variable])
        assert name is not None  # stored_mode is only True for a bare name
        for sample in samples:
            systematics_of = _sample_systematics(sample, plot_level)
            for files in self._stored_sources(sample, systematics_of, name):
                _extend(self._stored, (files, variances_from_contents), [name])

    def _stored_sources(
        self, sample: Sample, systematics: Mapping[str, Systematic], name: str
    ) -> list[FileSource]:
        """Return the files ``name`` is read from: ``sample``'s own and its variations'."""
        sources = []
        try:
            sources.append(_stored_source(sample, name))
        except RootfigError:
            return []  # refused for the sample itself, so its variations are never reached
        for syst_name, syst in systematics.items():
            if syst.kind != "samples":
                continue
            for direction, spec in (("up", syst.up), ("down", syst.down)):
                if spec is None:
                    continue
                context = f"{sample.label} [{syst_name} {direction}]"
                try:
                    variant = _varied_stored_sample(sample, spec, context, cache=self.cache)
                    sources.append(_stored_source(variant, name, variation=True))
                except RootfigError:
                    continue
        return sources

    def read(self) -> None:
        """Read what was planned into the cache, once per source.

        A read that fails is left to :func:`build_histograms`, which reads the
        source again for each task and raises for the task that needs what
        cannot be read; a failure here belongs to no single task.
        """
        for source, names in self._branches.items():
            with suppress(Exception):
                self.cache.arrays(source, names)
        for (source, variances_from_contents), names in self._stored.items():
            with suppress(Exception):
                self.cache.histograms(
                    source, names, variances_from_contents=variances_from_contents
                )


def prefetch(
    cache: ReadCache,
    items: Sequence[PlotItem],
    variables: Sequence[Variable],
    *,
    selections: Sequence[CutLike | None] = (None,),
    weight: str | None = None,
    systematics: Mapping[str, SystematicLike] | None = None,
    variances_from_contents: bool = False,
) -> None:
    """Read into ``cache`` what :func:`build_histograms` reads for ``variables`` and ``selections``.

    One read per file source with the union of the branches the variables, the
    selections, the weight and the samples' systematics need, and one pass over
    each source's files for the stored histograms among the variables, instead
    of one read per variable and selection. The files of ``Systematic.samples``
    variations are read once for the batch alike. A :class:`ReadPlan` planned
    and read; see there for what is left out and for planning several
    configurations together.
    """
    plan = ReadPlan(cache)
    plan.add(
        items,
        variables,
        selections=selections,
        weight=weight,
        systematics=systematics,
        variances_from_contents=variances_from_contents,
    )
    plan.read()


def _extend[K](plan: dict[K, list[str]], key: K, names: Sequence[str]) -> None:
    """Add ``names`` to the ordered union planned for ``key``."""
    held = plan.setdefault(key, [])
    held.extend(name for name in names if name not in held)
