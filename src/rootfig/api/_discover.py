"""Discovering the variables of a :class:`~rootfig.PlotBook` from what its inputs hold.

``PlotBook(data, rf.ALL)`` lists every quantity the tasks of the book can plot,
from metadata alone: the types of the branches of a tree
(:meth:`~rootfig.io.FileSource.branch_forms`), the classes of the objects stored
in a file (:meth:`~rootfig.io.FileSource.histograms`), the types of in-memory
columns. No entry and no bin content is read. The result is an ordinary tuple of
:class:`~rootfig.model.Variable` objects, which the book then runs exactly like an
explicit list.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Literal

from rootfig.api._hists import histogram_objects
from rootfig.api.plots1d import plot_items
from rootfig.errors import RootfigError, SourceError
from rootfig.expressions import quote_name
from rootfig.histograms import stored_names
from rootfig.histograms.pipeline import _sample_systematics
from rootfig.histograms.pipeline import _variant_sample as _varied_sample
from rootfig.histograms.sources import shared_source
from rootfig.histograms.stored import _variant_sample as _varied_stored_sample
from rootfig.io import FileSource, ReadCache, Source
from rootfig.io.objects import histogram_dimension, is_tree_class
from rootfig.io.schema import plottable
from rootfig.model import (
    CutLike,
    PlotItem,
    Sample,
    Systematic,
    Variable,
    as_cut,
    as_systematics,
    leaf_samples,
)

__all__ = ["discover"]

_EVENT_DATA_KINDS = frozenset({"weight", "replace"})
"""Kinds of systematic variation that need event data, which a stored histogram no longer has."""

_Mode = Literal["branch", "stored"]
"""How a name is plotted: filled from a branch of the tree, or read as a stored histogram."""

_SHOWN = 12
"""How many names a message lists before it abbreviates."""


def discover(
    data: Any,
    *,
    selections: Sequence[CutLike | None],
    options: Sequence[Mapping[str, Any]],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[Variable, ...]:
    """Return a :class:`~rootfig.model.Variable` for every quantity all tasks of a book can plot.

    ``options`` are the effective :func:`~rootfig.plot` keyword sets of the
    tasks, one per variant (the book's own keywords with the variant's on top):
    ``tree`` and ``observed`` decide which samples take part; ``weight``,
    ``nonfinite``, ``systematics``, ``stats``, ``range`` and ``bins`` whether
    stored histograms can. ``selections`` are the cuts of the book's selection
    axis. A name is kept when every leaf sample of every option set provides it
    the same way, as a branch the 1D histogram path can fill from (numeric or
    boolean leaves, lists and fixed-size arrays of them) or as a stored ``TH1``
    that :func:`~rootfig.histograms.stored_mode` reads, and when nothing is bound
    to refuse it that way: a selection, a weight, ``nonfinite="error"``, a stats
    box, a ``(low, high)`` range without bins, or a systematic varying event data
    that applies to the sample (the plot's, unless the sample's own source of
    that name replaces it; none for observed data) rules stored histograms out.
    The data a ``Systematic.samples`` variation fills or reads from is surveyed
    like a sample of its own and must provide the name in the sample's mode;
    one that cannot be built or surveyed is left to the task, which reports it
    whatever the variable. A branch that a ``replace`` systematic replaces is
    kept only when its replacement is a plottable branch of the sample, since
    that replacement is read whenever the branch is used.
    ``include`` keeps the names matching one of its shell patterns, ``exclude``
    drops those matching one of its; both match the source name,
    case-sensitively. The result is sorted by source name, a name that is not an
    identifier addressed in backticks (:func:`~rootfig.expressions.quote_name`).

    Only metadata is inspected (object classes, tree schemas, array types), the
    first file of every sample standing for all of them as everywhere else, and
    every distinct file once however many samples or variants read it.

    Raises
    ------
    TypeError
        For histogram objects as ``data``: they are drawn as they are and carry
        no names to discover, so the variable must be given explicitly.
    SourceError
        If the inputs offer nothing every task can plot (the message lists what
        each sample holds, what was left out and why, and what keeps the names
        from being common), if a source cannot tell what its branches hold, and
        for what the sources themselves refuse: a file with several trees and no
        ``tree=``, a tree that does not exist, an input
        :func:`~rootfig.model.as_plot_items` cannot read.
    ValueError
        If ``include`` and ``exclude`` leave nothing.
    """
    if histogram_objects(data) is not None:
        msg = (
            "variables=rf.ALL discovers variables from files or arrays; histogram objects are "
            "drawn as they are and carry no names to discover, so name the variable: "
            "PlotBook(hist, 'x')"
        )
        raise TypeError(msg)
    discovery = _Discovery()
    common: dict[str, _Mode] | None = None
    for option_set in options:
        items = plot_items(
            data, tree=option_set.get("tree"), label=None, observed=option_set.get("observed")
        )
        names = discovery.names(items, _Constraints.of(option_set, selections))
        common = names if common is None else _both(common, names)
    expressions: dict[str, str] = {}
    for name in sorted(common or {}):
        expression = quote_name(name)
        if expression is not None:  # a name no expression can address cannot be plotted
            expressions[name] = expression
    if not expressions:
        raise SourceError(discovery.nothing_found())
    kept = _matching(list(expressions), include, exclude)
    if not kept:
        filters = (("include", include), ("exclude", exclude))
        shown = " and ".join(
            f"{name}={list(value)}" for name, value in filters if value is not None
        )
        msg = (
            f"no plottable variables remain after {shown}; discovered: {_listed(list(expressions))}"
        )
        raise ValueError(msg)
    return tuple(Variable(expressions[name]) for name in kept)


@dataclass(frozen=True)
class _Constraints:
    """What one task configuration demands of every histogram it draws."""

    stored_blocker: str | None
    """Why stored histograms are out for every sample of the configuration; ``None`` if not."""

    systematics: Mapping[str, Systematic]
    """The plot-level systematics as the tasks parse them; empty when they are malformed,
    since every task then fails alike and the mode is not what decides."""

    @classmethod
    def of(cls, options: Mapping[str, Any], selections: Sequence[CutLike | None]) -> _Constraints:
        """Read the constraints off the effective ``plot()`` keywords and the selection axis.

        The structural reasons only, which hold for every stored histogram alike;
        a binning that does not merge the stored bins or a file lacking the
        histogram is found when the task runs.
        """
        blocker = None
        if any(as_cut(cut) is not None for cut in selections):
            blocker = "selections= needs event data"
        elif options.get("weight") is not None:
            blocker = "weight= needs event data"
        elif options.get("nonfinite", "drop") != "drop":
            blocker = f"nonfinite={options['nonfinite']!r} needs event data"
        elif options.get("stats"):
            blocker = "stats= needs the unbinned statistics collected while filling from event data"
        elif isinstance(options.get("range"), tuple) and options.get("bins") is None:
            blocker = "range=(low, high) without bins= cannot be applied to a stored histogram"
        try:
            systematics = as_systematics(options.get("systematics"), "plot") or {}
        except RootfigError:
            systematics = {}
        return cls(blocker, systematics)


@dataclass(frozen=True)
class _Inventory:
    """The names one source offers, before the plot's configuration is considered."""

    branches: frozenset[str]
    """Branches of the tree that the 1D histogram path can fill from."""

    stored: frozenset[str]
    """1D histograms stored in the file that a bare name reads (:func:`stored_names`)."""

    left_out: tuple[str, ...]
    """What else the source holds, each with its type: branches of other types, other objects."""


@dataclass(frozen=True)
class _Seen:
    """One leaf sample met while discovering, for the message when nothing is left."""

    sample: Sample
    inventory: _Inventory
    names: Mapping[str, _Mode]
    """What the sample offered under its configuration, and how."""
    stored_left_out: str | None
    """Why its stored histograms were not offered, when it has any."""


@dataclass(frozen=True)
class _Varied:
    """What the data of one ``Systematic.samples`` variation offers, in either mode."""

    branches: frozenset[str]
    """Branches it can be filled from."""

    stored: frozenset[str]
    """1D histograms read by name from its files; empty when it cannot read any."""

    seen: _Seen
    """Its entry for the message when nothing is left."""

    def offers(self, name: str, mode: _Mode) -> bool:
        """Whether the variation provides ``name`` the way the nominal sample does."""
        return name in (self.branches if mode == "branch" else self.stored)


class _Discovery:
    """Inventories of the sources of a book, each file inspected once however often it is met."""

    def __init__(self) -> None:
        # interns file sources by value, so one instance learns about the files
        self._cache = ReadCache()
        self._inventories: dict[FileSource, _Inventory] = {}
        self.seen: list[_Seen] = []

    def names(self, items: Sequence[PlotItem], constraints: _Constraints) -> dict[str, _Mode]:
        """Return the names every leaf sample of ``items`` provides, and how.

        The configuration's ``constraints`` rule stored histograms out for every
        sample or for none; a sample's own selection or weight, or a systematic
        varying event data that applies to it, rules them out for that sample.
        The data of every ``Systematic.samples`` variation that applies to the
        sample must provide a name in the sample's mode too, and a branch that a
        ``replace`` systematic replaces needs its replacement among the branches.
        """
        common: dict[str, _Mode] | None = None
        for sample in leaf_samples(items):
            inventory = self.inventory(sample)
            unreplaced = _missing_replacements(sample, constraints.systematics, inventory.branches)
            names: dict[str, _Mode] = dict.fromkeys(
                sorted(inventory.branches - set(unreplaced)), "branch"
            )
            left_out = constraints.stored_blocker
            if left_out is None:
                left_out = _needs_event_data(sample, constraints.systematics)
            if left_out is None:
                for name in sorted(inventory.stored - inventory.branches):
                    names[name] = "stored"
            shown = _Inventory(
                inventory.branches - set(unreplaced),
                inventory.stored,
                (*inventory.left_out, *unreplaced.values()),
            )
            self.seen.append(_Seen(sample, shown, names, left_out if inventory.stored else None))
            for varied in self._variations(sample, constraints.systematics):
                names = {name: mode for name, mode in names.items() if varied.offers(name, mode)}
                self.seen.append(varied.seen)
            common = names if common is None else _both(common, names)
        return common or {}

    def _variations(
        self, sample: Sample, plot_level: Mapping[str, Systematic]
    ) -> Iterator[_Varied]:
        """Survey the data of every ``Systematic.samples`` variation that applies to ``sample``.

        A variation is filled like a sample of its own or its histogram read by
        name from its files (:func:`~rootfig.histograms.build_histograms`,
        :func:`~rootfig.histograms.read_stored`), so it must provide the
        variable too. One whose data cannot be built or surveyed is left to the
        task, which reports it whatever the variable.
        """
        for name, systematic in _sample_systematics(sample, plot_level).items():
            if systematic.kind != "samples":
                continue
            for direction, spec in (("up", systematic.up), ("down", systematic.down)):
                if spec is None:
                    continue
                varied = self._variation(sample, spec, f"{sample.label} [{name} {direction}]")
                if varied is not None:
                    yield varied

    def _variation(self, sample: Sample, spec: Any, context: str) -> _Varied | None:
        """Survey one variation's data as it is filled from and as its histograms are read."""
        shown: Sample | None = None
        branches: frozenset[str] = frozenset()
        left_out: tuple[str, ...] = ()
        try:
            filled = _varied_sample(sample, spec, context, cache=self._cache)
            inventory = self.inventory(filled)
        except RootfigError:
            pass  # not to be built or surveyed: the task reports it, whatever the variable
        else:
            shown, branches, left_out = filled, inventory.branches, inventory.left_out
        stored: frozenset[str] = frozenset()
        stored_left_out: str | None = None
        try:
            read = _varied_stored_sample(sample, spec, context, cache=self._cache)
        except RootfigError:
            if shown is not None:  # arrays: to be filled from, but holding no stored histograms
                stored_left_out = "in-memory data holds no stored histograms"
        else:
            shown = shown or read
            stored_left_out = _needs_event_data(read, {})
            if stored_left_out is None:
                stored = frozenset(stored_names(read, variation=True))
        if shown is None:
            return None
        offered: dict[str, _Mode] = dict.fromkeys(sorted(branches), "branch")
        for name in sorted(stored - branches):
            offered[name] = "stored"
        seen = _Seen(shown, _Inventory(branches, stored, left_out), offered, stored_left_out)
        return _Varied(branches, stored, seen)

    def inventory(self, sample: Sample) -> _Inventory:
        """Return what the source of ``sample`` offers, inspecting each distinct file once."""
        sample = shared_source(sample, self._cache)
        source = sample.source
        if not isinstance(source, FileSource):
            branches, left_out = _plottable(_branch_forms(source))
            return _Inventory(frozenset(branches), frozenset(), tuple(left_out))
        if source not in self._inventories:
            self._inventories[source] = _file_inventory(source, sample)
        return self._inventories[source]

    def nothing_found(self) -> str:
        """Say what every sample met holds, what was left out and why, and what is not common."""
        entries = _diagnostic_entries(self.seen)
        lines = [
            "variables=rf.ALL found nothing that every task can plot: a variable must be "
            "present in every sample (observed= included) under every variant, the same way."
        ]
        for label, seen in entries:
            described = seen.sample.source.describe()
            inventory = seen.inventory
            held = [
                f"plottable branches {_listed(sorted(inventory.branches))}"
                if inventory.branches
                else "no plottable branches"
            ]
            if inventory.stored:
                stored = f"stored 1D histograms {_listed(sorted(inventory.stored))}"
                if seen.stored_left_out is not None:
                    stored += f" left out ({seen.stored_left_out})"
                held.append(stored)
            elif seen.stored_left_out is not None:
                held.append(f"no stored 1D histograms ({seen.stored_left_out})")
            if inventory.left_out:
                held.append(f"not plottable {_listed(inventory.left_out)}")
            lines.append(f"{label!r} ({described}): {'; '.join(held)}")
        if len(entries) > 1:
            lines.extend(_not_common({label: seen.names for label, seen in entries}))
        return "\n".join(lines)


def _diagnostic_entries(seen: Sequence[_Seen]) -> list[tuple[str, _Seen]]:
    """Deduplicate repeated source/configuration sightings and distinguish repeated labels."""
    unique: dict[
        tuple[FileSource | int, str, tuple[tuple[str, _Mode], ...], str | None], _Seen
    ] = {}
    for entry in seen:
        source = entry.sample.source
        source_key = source if isinstance(source, FileSource) else id(source)
        key = (source_key, entry.sample.label, tuple(entry.names.items()), entry.stored_left_out)
        unique.setdefault(key, entry)
    counts = Counter(entry.sample.label for entry in unique.values())
    used = set(counts)
    suffixes: Counter[str] = Counter()
    entries = []
    for entry in unique.values():
        label = entry.sample.label
        if counts[label] > 1:
            while True:
                suffixes[label] += 1
                candidate = f"{label} [{suffixes[label]}]"
                if candidate not in used:
                    used.add(candidate)
                    label = candidate
                    break
        entries.append((label, entry))
    return entries


def _not_common(offered: Mapping[str, Mapping[str, _Mode]]) -> list[str]:
    """Say, per group of names, which samples lack them or provide them another way."""
    by_missing: dict[tuple[str, ...], list[str]] = {}
    conflicts: list[str] = []
    for name in sorted(set().union(*(names.keys() for names in offered.values()))):
        modes = {label: names[name] for label, names in offered.items() if name in names}
        missing = tuple(label for label in offered if label not in modes)
        if missing:
            by_missing.setdefault(missing, []).append(name)
        elif len(set(modes.values())) > 1:
            branch = [label for label, mode in modes.items() if mode == "branch"]
            stored = [label for label, mode in modes.items() if mode == "stored"]
            conflicts.append(f"{name!r} is a branch of {branch} but a stored histogram of {stored}")
    lines = [
        f"{_listed(names)} missing from {list(missing)}"
        for missing, names in list(by_missing.items())[:_SHOWN]
    ]
    return lines + conflicts[:_SHOWN]


def _both(first: Mapping[str, _Mode], second: Mapping[str, _Mode]) -> dict[str, _Mode]:
    """Return the names both mappings hold with the same mode, in the order of ``first``."""
    return {name: mode for name, mode in first.items() if second.get(name) == mode}


def _branch_forms(source: Source) -> Mapping[str, Any]:
    """Ask a source for its branch metadata, refusing sources that would need a data read."""
    forms = getattr(source, "branch_forms", None)
    if not callable(forms):
        msg = (
            f"variables=rf.ALL needs the types of the branches of {source.describe()}, which "
            f"{type(source).__name__} does not provide (no branch_forms() method); list the "
            "variables explicitly"
        )
        raise SourceError(msg)
    result: Mapping[str, Any] = forms()
    return result


def _plottable(forms: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Split branch forms into plottable names and the rest, described with their types."""
    good: list[str] = []
    other: list[str] = []
    for name, form in forms.items():
        if plottable(form):
            good.append(name)
        else:
            other.append(f"{name} ({form.type})")
    return good, other


def _file_inventory(source: FileSource, sample: Sample) -> _Inventory:
    """Survey the first file of ``source``: its tree's branches and its stored objects."""
    branches: list[str] = []
    left_out: list[str] = []
    if source.tree is not None or source.trees():
        # several trees and no tree=: branches() asks for one, as it does for any plot
        forms = _branch_forms(source)
        branches, left_out = _plottable(forms)
        left_out.extend(
            f"{name} (no interpretation)" for name in source.branches() if name not in forms
        )
    left_out.extend(
        f"{name} ({classname})"
        for name, classname in sorted(source.objects().items())
        if histogram_dimension(classname) != 1
        and not is_tree_class(classname)
        and "Directory" not in classname
    )
    return _Inventory(frozenset(branches), frozenset(stored_names(sample)), tuple(left_out))


def _needs_event_data(sample: Sample, plot_level: Mapping[str, Systematic]) -> str | None:
    """Why ``sample`` cannot read stored histograms under these systematics; ``None`` when it can.

    Its own selection or weight, or a systematic varying event data among the
    ones that apply to it: the plot's with the sample's own on top, none for
    observed data, exactly as the histograms are built.
    """
    if sample.selection is not None:
        return f"sample {sample.label!r} has a selection of its own, which needs event data"
    if sample.weight is not None:
        return f"sample {sample.label!r} has a weight of its own, which needs event data"
    for name, systematic in _sample_systematics(sample, plot_level).items():
        if systematic.kind in _EVENT_DATA_KINDS:
            what = "the weight" if systematic.kind == "weight" else "branches"
            return (
                f"systematic {name!r} of sample {sample.label!r} varies {what}, which needs "
                "event data"
            )
    return None


def _missing_replacements(
    sample: Sample, plot_level: Mapping[str, Systematic], branches: frozenset[str]
) -> dict[str, str]:
    """Return the replaced branches whose replacement is not among ``branches``, described.

    A ``replace`` systematic that applies to ``sample`` reads the replacing
    branch whenever the replaced one is used, so a variable that is such a
    branch fails when the replacement is missing or cannot be histogrammed
    (:func:`~rootfig.histograms.pipeline._read_plan` refuses the read). A
    replaced branch of the selection or weight fails every variable alike and is
    left to the task.
    """
    missing: dict[str, str] = {}
    for name, systematic in _sample_systematics(sample, plot_level).items():
        if systematic.kind != "replace":
            continue
        for spec in (systematic.up, systematic.down):
            for old, new in (spec or {}).items():
                if new not in branches and old not in missing:
                    missing[old] = (
                        f"{old} (its replacement {new!r} under systematic {name!r} is not a "
                        "plottable branch)"
                    )
    return missing


def _matching(
    names: Sequence[str], include: Sequence[str] | None, exclude: Sequence[str] | None
) -> list[str]:
    """Return the ``names`` matching an ``include`` pattern (all when unset), minus ``exclude``."""
    kept = [
        name
        for name in names
        if include is None or any(fnmatchcase(name, pattern) for pattern in include)
    ]
    if exclude is not None:
        kept = [name for name in kept if not any(fnmatchcase(name, p) for p in exclude)]
    return kept


def _listed(names: Sequence[str]) -> str:
    """Show up to :data:`_SHOWN` names, then how many more there are."""
    shown = ", ".join(repr(name) for name in names[:_SHOWN])
    more = f", ... ({len(names) - _SHOWN} more)" if len(names) > _SHOWN else ""
    return f"[{shown}{more}]"
