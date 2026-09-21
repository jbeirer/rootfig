"""Batch plotting: :class:`PlotBook` repeats :func:`~rootfig.plot` over variables and selections.

Every task produces what ``plot(data, task.variable, selection=task.selection,
**task.kwargs)`` returns, through the two halves of that call: the variables
are taken :data:`_VARIABLE_BATCH_SIZE` at a time, a :class:`~rootfig.io.ReadCache`
warmed with the branches they, every selection and every preparation the batch
uses need serves its reads (:func:`~rootfig.api.plots1d.prefetch_plots`), each ``(variable,
selection)`` is prepared once for the variants that only change the drawing
(:func:`~rootfig.api.plots1d.prepare_plot`) and every variant is drawn from that
(:func:`~rootfig.api.plots1d.draw_plot`). The variables are listed by the caller
or, with :data:`ALL`, discovered from the metadata of the inputs when the book is
built (:mod:`rootfig.api._discover`); the book runs either the same way.
"""

from __future__ import annotations

import inspect
import itertools
import os
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Final

from rootfig._mapping import FrozenMapping
from rootfig.api._discover import discover
from rootfig.api.plots1d import PreparedPlot, draw_plot, plot, prefetch_plots, prepare_plot
from rootfig.io import ReadCache
from rootfig.model import Cut, CutLike, Variable, as_cut, as_variable, check_file_stem
from rootfig.plotting import Plot
from rootfig.plotting.figure import close_figures_since, open_figure_ids
from rootfig.plotting.result import normalize_formats

__all__ = ["ALL", "PlotBook", "PlotTask", "discover_variables"]


class _AllVariables:
    """The type of :data:`ALL`; nothing else is an instance of it."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "ALL"


ALL: Final = _AllVariables()
"""Discover the variables of a :class:`PlotBook` from its inputs: ``PlotBook(data, rf.ALL)``.

Every branch of the trees (``TTree`` or ``RNTuple``) holding numbers or booleans,
lists and fixed-size arrays of them included, and every 1D histogram stored in
files read without a tree, when every sample provides it under every variant;
``include=`` and ``exclude=`` narrow the set with shell patterns. Only metadata
is read. A distinct object rather than a string, so that a branch or histogram
named ``"all"`` stays an ordinary variable name.
"""

_DISCOVERED_REMEDY = (
    "with variables=rf.ALL, exclude= one of them or list the variables explicitly with "
    "distinct name= values"
)
"""How to resolve two discovered variables whose output files would collide."""

_RESERVED_PLOT_KWARGS = frozenset({"data", "variable", "selection", "save", "ax"})
"""Keywords of :func:`~rootfig.plot` that :class:`PlotBook` fills in itself."""

_PLOT_KEYWORDS = frozenset(inspect.signature(plot).parameters) - _RESERVED_PLOT_KWARGS
"""Keywords of :func:`~rootfig.plot` a book or variant may set.

Checked when the book is built: a misspelt keyword would otherwise surface only
once the batch is running, after earlier tasks have written their files.
"""

_PREPARE_KEYWORDS = frozenset(inspect.signature(prepare_plot).parameters) - {
    "data",
    "variable",
    "cache",
}
"""Keywords of :func:`~rootfig.plot` that decide what is read and how the histograms are filled.

A variant that overrides none of them draws the histograms prepared for the
book's own keywords; one that does is prepared on its own.
"""

_DRAW_KEYWORDS = frozenset(inspect.signature(draw_plot).parameters) - {"prepared"}
"""Keywords of :func:`~rootfig.plot` that only change the drawing."""

_VARIABLE_BATCH_SIZE = 32
"""Variables whose branches are read together.

Bounds what a batch holds in memory: the arrays of this many variables, plus
those of the selections and weights, for every sample of the book.
"""

_RESERVED_SAVEFIG_KWARGS = frozenset({"format", "fname"})
"""Keywords of ``savefig`` that would fight the file names :meth:`PlotBook.save` builds."""

_SEPARATOR = "__"
"""Joins the variable, selection and variant components of an output file name."""

DEFAULT_FORMATS = ("pdf",)
"""What :meth:`PlotBook.save` writes when no ``formats=`` is given."""

_IMPLICIT_SELECTION = "all"
_IMPLICIT_VARIANT = "default"
"""Display names of the single choice on an axis the book was built without."""


def _stem(variable: str, selection: str | None, variant: str | None) -> str:
    """Join the components of an output file name, leaving out the implicit axes."""
    return _SEPARATOR.join(part for part in (variable, selection, variant) if part is not None)


def _file_key(stem: str) -> str:
    """Fold a stem the way case- and normalisation-insensitive file systems compare names.

    Two stems with one key would be one file on such a file system, so the second
    save would overwrite the first; the book rejects them. NFKC plus ``casefold``
    is deliberately conservative: a false clash costs a rename, a missed one a plot.
    """
    return unicodedata.normalize("NFKC", stem).casefold()


def _task_options(
    plot_kwargs: Mapping[str, Any], variants: Mapping[str, Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    """Return the effective ``plot()`` keywords of a book's tasks, one set per variant.

    The common keywords with a variant's overrides on top, or the common ones
    alone without variants. Every variable runs under each of them, so what is
    discovered (:data:`ALL`) has to hold for all.
    """
    overrides = variants.values() if variants is not None else [{}]
    return [{**plot_kwargs, **own} for own in overrides]


def discover_variables(
    data: Any,
    *,
    selections: Mapping[str, CutLike | None] | None = None,
    variants: Mapping[str, Mapping[str, Any]] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    include: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
) -> tuple[Variable, ...]:
    """Return the variables ``PlotBook(data, rf.ALL, ...)`` would plot, without building the book.

    The same discovery from metadata alone, under the same keywords (see
    :class:`PlotBook`): every branch or stored ``TH1`` that every sample,
    ``observed=`` included, provides the same way under every variant and that
    nothing in the configuration is bound to refuse, filtered by ``include`` and
    ``exclude`` and sorted by source name. Unlike the book it does not check that
    the output file names are distinct: two variables that sanitise to one name
    (``a-b`` and ``a_b``) are both returned, to be told apart with
    :meth:`~rootfig.model.Variable.replace` before they are given to a book.

    Raises
    ------
    SourceError, TypeError, ValueError
        As :class:`PlotBook` with ``variables=rf.ALL``.

    Examples
    --------
    >>> variables = rf.discover_variables(samples, include="jet*")  # doctest: +SKIP
    >>> renamed = [  # doctest: +SKIP
    ...     v.replace(name="jet1_btag") if v.expression == "`jet1_b-tag`" else v for v in variables
    ... ]
    >>> book = rf.PlotBook(samples, renamed)  # doctest: +SKIP
    """
    book_selections = _selections(selections)
    kwargs = _plot_kwargs({} if plot_kwargs is None else plot_kwargs, where="plot_kwargs")
    return discover(
        data,
        selections=list(book_selections.values()) if book_selections is not None else [None],
        options=_task_options(kwargs, _variants(variants)),
        include=_patterns(include, what="include="),
        exclude=_patterns(exclude, what="exclude="),
    )


@dataclass(frozen=True, eq=False)
class PlotTask:
    """One plot of a :class:`PlotBook`: a variable, a selection and the keywords to draw it with.

    Tasks compare and hash by their :attr:`stem`, which is unique within a book, so
    they can be collected in a set or used as dictionary keys. The keyword values
    are not compared: valid ones such as ``bins=np.array(...)`` have no scalar
    equality.

    Attributes
    ----------
    variable
        The :class:`~rootfig.model.Variable` to histogram.
    selection_name
        The key of the selection in the book's ``selections`` mapping, or ``None``
        when the book has no selection axis (``selections=None``).
    selection
        The :class:`~rootfig.model.Cut` passed as ``selection=``, or ``None``.
    variant_name
        The key of the variant in the book's ``variants`` mapping, or ``None`` when
        the book has no variant axis (``variants=None``).
    kwargs
        The keywords passed to :func:`~rootfig.plot`: the book's common
        ``plot_kwargs`` with the variant's overrides on top.
    """

    variable: Variable
    selection_name: str | None
    selection: Cut | None
    variant_name: str | None
    kwargs: Mapping[str, Any]

    @property
    def selection_id(self) -> str:
        """The selection name, or ``"all"`` for the implicit no-selection choice (display only)."""
        return self.selection_name if self.selection_name is not None else _IMPLICIT_SELECTION

    @property
    def variant_id(self) -> str:
        """The variant name, or ``"default"`` for the implicit single variant (display only)."""
        return self.variant_name if self.variant_name is not None else _IMPLICIT_VARIANT

    @property
    def stem(self) -> str:
        """The output file name without suffix: ``<variable>[__<selection>][__<variant>]``.

        Only axes the book was given explicitly appear, so a book without
        ``selections=`` writes ``mass.pdf`` while ``selections={"all": ...}`` writes
        ``mass__all.pdf``: an explicit name is never dropped, whatever it is.
        """
        return _stem(self.variable.safe_name, self.selection_name, self.variant_name)

    def describe(self) -> str:
        """Name the task for messages: ``variable='mass', selection='sr', variant='log'``."""
        return (
            f"variable={self.variable.safe_name!r}, selection={self.selection_id!r}, "
            f"variant={self.variant_id!r}"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlotTask) and self.stem == other.stem

    def __hash__(self) -> int:
        return hash(self.stem)


def _identifier(value: object, *, what: str) -> str:
    """Return ``value`` if it can name an output file component on every platform, else raise."""
    if not isinstance(value, str):
        msg = f"{what} names must be non-empty strings, got {value!r}"
        raise TypeError(msg)
    return check_file_stem(value, what=f"{what} name")


def _as_mapping(values: object, *, what: str, holds: str) -> Mapping[Any, Any]:
    """Return ``values`` if it is a mapping, else raise naming what was expected.

    Keys are ``Any``: that they are strings is checked by the caller, with a message.
    """
    if not isinstance(values, Mapping):
        msg = f"{what} must map {holds}, got {type(values).__name__}"
        raise TypeError(msg)
    return values


def _variables(
    values: str | Variable | Sequence[str | Variable], *, remedy: str = "give one a distinct name="
) -> tuple[Variable, ...]:
    """Coerce the variables and reject an empty list or two variables with one identifier.

    ``remedy`` ends the message for the latter: what to do about it.
    """
    items = (values,) if isinstance(values, str | Variable) else tuple(values)
    if not items:
        msg = "PlotBook needs at least one variable"
        raise ValueError(msg)
    variables = tuple(as_variable(item) for item in items)
    seen: dict[str, Variable] = {}
    for variable in variables:
        key = variable.safe_name  # always a valid file name component, see Variable
        if key in seen:
            msg = (
                f"variables {seen[key].expression!r} and {variable.expression!r} share the "
                f"identifier {key!r}, so their output files would collide; {remedy}"
            )
            raise ValueError(msg)
        seen[key] = variable
    return variables


def _patterns(value: object, *, what: str) -> tuple[str, ...] | None:
    """Coerce ``include=`` or ``exclude=`` to shell patterns; ``None`` means no filter."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or not all(isinstance(item, str) for item in value):
        msg = f"{what} must be a shell pattern or a sequence of them, got {value!r}"
        raise TypeError(msg)
    if not value:
        msg = f"{what} must hold at least one pattern; omit it to keep every variable"
        raise ValueError(msg)
    return tuple(value)


def _check_output_names(
    variables: Sequence[Variable],
    selections: Mapping[str, Any] | None,
    variants: Mapping[str, Any] | None,
    *,
    remedy: str | None = None,
) -> None:
    """Reject two tasks whose output files would be one on a case-insensitive file system."""
    by_file: dict[str, list[str]] = {}
    for variable in variables:
        for selection in selections if selections is not None else [None]:
            for variant in variants if variants is not None else [None]:
                stem = _stem(variable.safe_name, selection, variant)
                by_file.setdefault(_file_key(stem), []).append(stem)
    clashes = [dict.fromkeys(stems) for stems in by_file.values() if len(stems) > 1]
    if clashes:
        shown = "; ".join(" and ".join(repr(stem) for stem in group) for group in clashes)
        msg = (
            f"output names collide: {shown} (compared ignoring case, as case-insensitive "
            "file systems do); rename a variable, selection or variant so that no two tasks "
            f"share <variable>{_SEPARATOR}<selection>{_SEPARATOR}<variant>"
        )
        if remedy is not None:
            msg = f"{msg} ({remedy})"
        raise ValueError(msg)


def _plot_kwargs(values: Mapping[str, Any], *, where: str) -> FrozenMapping[str, Any]:
    """Copy plot keywords, rejecting the ones the book fills in itself and unknown ones."""
    mapping = _as_mapping(values, what=where, holds="plot() keywords to values")
    for key in mapping:
        if not isinstance(key, str):
            msg = f"{where} keys must be plot() keyword names, got {key!r}"
            raise TypeError(msg)
    reserved = sorted(_RESERVED_PLOT_KWARGS.intersection(mapping))
    if reserved:
        msg = (
            f"{where} must not set {', '.join(repr(k) for k in reserved)}: PlotBook passes data, "
            "variable and selection itself, save() owns the output files and every task draws "
            "its own figure (no ax=)"
        )
        raise ValueError(msg)
    unknown = [key for key in mapping if key not in _PLOT_KEYWORDS]
    if unknown:
        described = []
        for key in unknown:
            close = get_close_matches(key, _PLOT_KEYWORDS, n=1)
            described.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
        msg = f"{where} names keywords plot() does not have: {', '.join(described)}"
        raise TypeError(msg)
    return FrozenMapping(mapping)


def _selections(
    values: Mapping[str, CutLike | None] | None,
) -> FrozenMapping[str, Cut | None] | None:
    """Copy the named selections; ``None`` keeps the axis implicit."""
    if values is None:
        return None
    mapping = _as_mapping(values, what="selections=", holds="names to cuts")
    if not mapping:
        msg = "selections= must name at least one selection; omit it to plot without a selection"
        raise ValueError(msg)
    return FrozenMapping(
        {_identifier(name, what="selection"): as_cut(cut) for name, cut in mapping.items()}
    )


def _variants(
    values: Mapping[str, Mapping[str, Any]] | None,
) -> FrozenMapping[str, FrozenMapping[str, Any]] | None:
    """Copy the named variants and their keyword mappings; ``None`` keeps the axis implicit."""
    if values is None:
        return None
    mapping = _as_mapping(values, what="variants=", holds="names to plot() keywords")
    if not mapping:
        msg = "variants= must name at least one variant; omit it to draw every plot once"
        raise ValueError(msg)
    variants: dict[str, FrozenMapping[str, Any]] = {}
    for name, overrides in mapping.items():
        key = _identifier(name, what="variant")
        variants[key] = _plot_kwargs(overrides, where=f"variant {key!r}")
    return FrozenMapping(variants)


def _chosen(
    wanted: str | Sequence[str] | None, available: Sequence[str], *, what: str
) -> list[str] | None:
    """Return the ``available`` names that are ``wanted``, in their order; unknown ones raise."""
    if wanted is None:
        return None
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    if not names:
        msg = f"select({what}s=...) must name at least one {what}; available: {list(available)}"
        raise ValueError(msg)
    unknown = [name for name in names if name not in available]
    if unknown:
        shown = ", ".join(repr(name) for name in unknown)
        msg = f"unknown {what} {shown}; available: {list(available)}"
        raise ValueError(msg)
    return [name for name in available if name in names]


def _prepares(overrides: Mapping[str, Any]) -> bool:
    """Whether a variant's overrides change how the histograms are prepared, not only drawn."""
    return not _PREPARE_KEYWORDS.isdisjoint(overrides)


def _split_options(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a task's keywords into those of ``prepare_plot`` and those of ``draw_plot``."""
    prepare = {key: value for key, value in kwargs.items() if key in _PREPARE_KEYWORDS}
    draw = {key: value for key, value in kwargs.items() if key in _DRAW_KEYWORDS}
    return prepare, draw


def _copied(prepared: PreparedPlot) -> PreparedPlot:
    """Return a copy of ``prepared`` whose histograms are independent of its own.

    Drawing does not modify histograms, but ``hist.Hist`` objects are mutable and
    the plots of several variants must not share them, or changing one figure's
    histogram would change the next variant drawn from the same preparation.
    """
    copies = [histogram.map_hists(lambda h: h.copy()) for histogram in prepared.histograms]
    return replace(prepared, histograms=copies)


@dataclass(frozen=True, init=False, eq=False, repr=False)
class PlotBook:
    """Many :func:`~rootfig.plot` calls described once: variables, selections and variants.

    A book holds what every plot shares (``data`` and ``plot_kwargs``) and the
    three axes that vary. :meth:`tasks` lists the Cartesian product in a fixed
    order (variables, then selections, then variants, each in insertion order),
    :meth:`plots` runs it one plot at a time and :meth:`save` writes one file per
    task under deterministic names. Each plot is what the corresponding
    :func:`~rootfig.plot` call returns, so samples, groups, stored histograms,
    arrays and histogram objects behave exactly as they do there; ``data`` is
    passed on as given, never copied. The book only reads the files less often:
    once per batch of variables, for every selection and variant (see
    :meth:`plots`). The configuration mappings are copied, the values inside them
    (a ``systematics=`` mapping, a ``Style``) are shared. Names of selections and
    variants are file name components and must be usable on every platform
    (:func:`~rootfig.model.check_file_stem`). Two tasks whose file names differ
    only in case or Unicode normalisation are one file on many file systems and
    are rejected as a collision.

    The variables are listed explicitly, in which case nothing is inspected when
    the book is built, or discovered with ``variables=rf.ALL``, which inspects
    the metadata of the inputs (branch types, object classes, array types) and
    never their contents; the result is an ordinary :attr:`variables` tuple.

    Parameters
    ----------
    data
        What to plot, exactly as :func:`~rootfig.plot` takes it.
    variables
        Variables (names, expressions or :class:`~rootfig.model.Variable`
        objects); one may be given bare. :attr:`~rootfig.model.Variable.safe_name`
        identifies each and names its files, so two variables must not share one.
        :data:`ALL` (``rf.ALL``) discovers them instead: every branch of the trees
        whose values are numbers or booleans (lists and fixed-size arrays of them
        included, strings and records left out) and every ``TH1`` stored in files
        read without a tree, kept when every sample, ``observed=`` included,
        provides it the same way under every variant, sorted by name. Stored
        histograms are left out when a selection, a ``weight``,
        ``nonfinite="error"`` or a systematic varying event data rules them out.
        A name that is not an identifier is addressed in backticks. Histogram
        objects as ``data`` carry no names to discover and need an explicit
        variable.
    selections
        Named selections, ``{name: cut}`` with ``cut`` a string, a
        :class:`~rootfig.model.Cut` or ``None``. ``None`` (the default) plots without
        a selection and without a selection component in the file names.
    variants
        Named drawing variants, ``{name: {keyword: value}}``. A variant's keywords
        override ``plot_kwargs`` for its tasks. ``None`` (the default) draws every
        plot once, without a variant component in the file names.
    plot_kwargs
        Keywords passed to every :func:`~rootfig.plot` call (``stack=``,
        ``observed=``, ``style=``, ...). ``data``, ``variable``, ``selection``,
        ``save`` and ``ax`` belong to the book and are rejected here and in variants.
    include, exclude
        With ``variables=rf.ALL`` only: shell patterns (one, or a sequence;
        ``"Muon_*"``, ``["*_cov", "*Index"]``), matched case-sensitively against the
        source name (``jet1_b-tag``, ``sel/mz``, not the file name component). A
        variable is kept when it matches one ``include`` pattern (all do when
        ``include`` is not given) and no ``exclude`` pattern.

    Raises
    ------
    ValueError
        No variable, an empty ``selections=`` or ``variants=``, a reserved plot
        keyword, a name that cannot be a file name component, two tasks whose
        output files would collide, ``include=``/``exclude=`` with an explicit
        variable list, or patterns that leave no variable.
    TypeError
        ``selections=``, ``variants=``, ``plot_kwargs=`` or one variant's overrides
        is not a mapping, a name is not a string, a keyword is not one
        :func:`~rootfig.plot` takes (the message names the closest one), a
        pattern is not a string, or ``rf.ALL`` meets histogram objects.
    SourceError
        With ``rf.ALL``, when the inputs offer nothing every task can plot, or
        a file cannot be surveyed (several trees and no ``tree=``, a missing tree).

    Examples
    --------
    >>> book = rf.PlotBook(  # doctest: +SKIP
    ...     [background, signal],
    ...     [mass, pt],
    ...     selections={"baseline": "nMuon >= 2", "sr": "(nMuon >= 2) & (mass > 120)"},
    ...     variants={"lin": {}, "log": {"logy": True}},
    ...     plot_kwargs={"stack": True, "style": style},
    ... )
    >>> book.save("plots", formats=["pdf", "png"])  # doctest: +SKIP
    >>> rf.PlotBook(  # doctest: +SKIP
    ...     [background, signal], rf.ALL, include=["Muon_*", "MET*"], exclude="*Index"
    ... ).variables
    (Variable('MET'), Variable('Muon_eta'), Variable('Muon_pt'), ...)
    """

    data: Any
    variables: tuple[Variable, ...]
    selections: FrozenMapping[str, Cut | None] | None
    variants: FrozenMapping[str, FrozenMapping[str, Any]] | None
    plot_kwargs: FrozenMapping[str, Any]

    def __init__(
        self,
        data: Any,
        variables: str | Variable | Sequence[str | Variable] | _AllVariables,
        *,
        selections: Mapping[str, CutLike | None] | None = None,
        variants: Mapping[str, Mapping[str, Any]] | None = None,
        plot_kwargs: Mapping[str, Any] | None = None,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
    ) -> None:
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "selections", _selections(selections))
        object.__setattr__(self, "variants", _variants(variants))
        object.__setattr__(
            self,
            "plot_kwargs",
            _plot_kwargs({} if plot_kwargs is None else plot_kwargs, where="plot_kwargs"),
        )
        if isinstance(variables, _AllVariables):
            found = discover_variables(
                data,
                selections=self.selections,
                variants=self.variants,
                plot_kwargs=self.plot_kwargs,
                include=include,
                exclude=exclude,
            )
            object.__setattr__(self, "variables", _variables(found, remedy=_DISCOVERED_REMEDY))
            remedy: str | None = _DISCOVERED_REMEDY
        else:
            if include is not None or exclude is not None:
                msg = (
                    "include= and exclude= are only valid with variables=rf.ALL; an explicit "
                    "list of variables is used as it is"
                )
                raise ValueError(msg)
            object.__setattr__(self, "variables", _variables(variables))
            remedy = None
        _check_output_names(self.variables, self.selections, self.variants, remedy=remedy)

    def _iter_tasks(self) -> Iterator[PlotTask]:
        """Yield the Cartesian product one task at a time, in the documented order."""
        selections: Sequence[tuple[str | None, Cut | None]] = (
            list(self.selections.items()) if self.selections is not None else [(None, None)]
        )
        variants: Sequence[tuple[str | None, Mapping[str, Any]]] = (
            list(self.variants.items()) if self.variants is not None else [(None, {})]
        )
        for variable in self.variables:
            for selection_name, cut in selections:
                for variant_name, overrides in variants:
                    yield PlotTask(
                        variable,
                        selection_name,
                        cut,
                        variant_name,
                        FrozenMapping({**self.plot_kwargs, **overrides}),
                    )

    def tasks(self) -> tuple[PlotTask, ...]:
        """All plots of the book, variables outermost and variants innermost.

        Pure: nothing is read or drawn. An implicit axis contributes one task with
        the name ``None``.
        """
        return tuple(self._iter_tasks())

    def plots(self) -> Iterator[tuple[PlotTask, Plot]]:
        """Run the tasks lazily, yielding ``(task, plot)`` one figure at a time.

        Each result is what ``plot(data, task.variable, selection=task.selection,
        **task.kwargs)`` returns, produced in batches: the variables are taken a
        few dozen at a time, the branches they and every selection need are read
        once per sample for the batch, and variants that only change the drawing
        (``logy``, ``normalize``, ``ratio``, ``style``, ...) are drawn from one
        set of prepared histograms, each from its own copy. A variant that
        changes how the histograms are prepared (``bins``, ``weight``,
        ``observed``, ``systematics``, ...) is prepared on its own. The tasks
        themselves are built one at a time and nothing is read before the first
        one runs. The caller owns the figures that are yielded and closes them
        (:meth:`Plot.close`); the figure a failing call had already created is
        closed here. An error keeps its type and gains a note naming the task
        (``variable=..., selection=..., variant=...``); one raised while reading
        a batch names the batch's first task.
        """
        selections: Sequence[tuple[str | None, Cut | None]] = (
            list(self.selections.items()) if self.selections is not None else [(None, None)]
        )
        variants: Sequence[tuple[str | None, Mapping[str, Any]]] = (
            list(self.variants.items()) if self.variants is not None else [(None, {})]
        )
        cuts = [cut for _, cut in selections]
        # variants drawn from the histograms prepared for the book's own keywords
        drawing_only = [name for name, overrides in variants if not _prepares(overrides)]
        for chunk in itertools.batched(self.variables, _VARIABLE_BATCH_SIZE):
            cache: ReadCache | None = None
            for variable in chunk:
                for selection_name, cut in selections:
                    shared: PreparedPlot | None = None
                    for variant_name, overrides in variants:
                        task = PlotTask(
                            variable,
                            selection_name,
                            cut,
                            variant_name,
                            FrozenMapping({**self.plot_kwargs, **overrides}),
                        )
                        before = open_figure_ids()
                        drawn = False
                        try:
                            if cache is None:
                                cache = ReadCache()
                                # one read per source for every preparation of the batch
                                prefetch_plots(
                                    cache, self.data, chunk, cuts, self._preparation_options()
                                )
                            prepare_options, draw_options = _split_options(task.kwargs)
                            if variant_name not in drawing_only:
                                prepared = prepare_plot(
                                    self.data,
                                    variable,
                                    selection=cut,
                                    cache=cache,
                                    **prepare_options,
                                )
                            else:
                                if shared is None:
                                    shared = prepare_plot(
                                        self.data,
                                        variable,
                                        selection=cut,
                                        cache=cache,
                                        **prepare_options,
                                    )
                                prepared = _copied(shared) if len(drawing_only) > 1 else shared
                            result = draw_plot(prepared, **draw_options)
                            drawn = True
                        except Exception as exc:
                            exc.add_note(f"while running PlotBook task {task.describe()}")
                            raise
                        finally:
                            if not drawn:
                                close_figures_since(before)
                        yield task, result

    def _preparation_options(self) -> list[dict[str, Any]]:
        """Return the distinct preparation keyword sets the tasks use.

        The book's own (when a variant, or the implicit one, draws from it) and
        one per variant that overrides a preparation keyword.
        """
        base = {key: value for key, value in self.plot_kwargs.items() if key in _PREPARE_KEYWORDS}
        overriding = [
            {key: value for key, value in overrides.items() if key in _PREPARE_KEYWORDS}
            for overrides in (self.variants.values() if self.variants is not None else [{}])
        ]
        result = [base] if any(not own for own in overriding) else []
        result.extend({**base, **own} for own in overriding if own)
        return result

    def save(
        self,
        directory: str | os.PathLike[str],
        *,
        formats: str | Sequence[str] = DEFAULT_FORMATS,
        **savefig_kwargs: Any,
    ) -> list[Path]:
        """Draw every task and write it to ``directory`` as ``<stem>.<format>``.

        The directory is created if needed. Every figure is closed once it has been
        written, and also when drawing or writing it fails, so memory stays bounded;
        use :meth:`plots` to keep or customise figures. Writing goes through
        :meth:`Plot.save`.

        Parameters
        ----------
        directory
            Where to write. It is created if it does not exist; an existing path
            that is not a directory raises :class:`NotADirectoryError`.
        formats
            One format or several (``"png"``, ``["pdf", "png"]``), each written for
            every task. A leading dot is accepted and duplicates are dropped.
            Anything matplotlib cannot write raises :class:`ValueError` before the
            first figure is drawn.
        **savefig_kwargs
            Forwarded to :meth:`Plot.save`. ``format`` and ``fname`` are rejected,
            since the file names come from ``formats`` and from the task.

        Returns
        -------
        list[pathlib.Path]
            The written files, in task order and then format order.
        """
        reserved = sorted(_RESERVED_SAVEFIG_KWARGS.intersection(savefig_kwargs))
        if reserved:
            msg = (
                f"PlotBook.save() builds each file name from the task and formats=, so it does "
                f"not take {', '.join(repr(k) for k in reserved)}: write formats=['png'] instead"
            )
            raise ValueError(msg)
        suffixes = normalize_formats(formats)
        target = Path(directory)
        if target.exists() and not target.is_dir():
            msg = (
                f"{target} exists and is not a directory; PlotBook.save() writes one file per task"
            )
            raise NotADirectoryError(msg)
        target.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for task, result in self.plots():
            try:
                written.extend(
                    result.save(
                        target / f"{task.stem}.{suffixes[0]}", formats=suffixes, **savefig_kwargs
                    )
                )
            except Exception as exc:
                exc.add_note(f"while saving PlotBook task {task.describe()} to {str(target)!r}")
                raise
            finally:
                result.close()
        return written

    def select(
        self,
        *,
        variables: str | Sequence[str] | None = None,
        selections: str | Sequence[str] | None = None,
        variants: str | Sequence[str] | None = None,
    ) -> PlotBook:
        """Return a new book restricted to the named variables, selections and variants.

        Names are :attr:`~rootfig.model.Variable.safe_name` values and the keys of
        the ``selections`` and ``variants`` mappings; ``None`` leaves an axis as it
        is and the book's order is kept. An unknown name, or a filter that names
        nothing, raises and lists the choices. An implicit axis (``selections=None``
        or ``variants=None``) has the single choice ``"all"`` or ``"default"`` and
        stays implicit.
        """
        variable_names = _chosen(
            variables, [variable.safe_name for variable in self.variables], what="variable"
        )
        kept_variables = (
            self.variables
            if variable_names is None
            else tuple(v for v in self.variables if v.safe_name in set(variable_names))
        )
        selection_names = _chosen(
            selections,
            list(self.selections) if self.selections is not None else [_IMPLICIT_SELECTION],
            what="selection",
        )
        kept_selections = (
            self.selections
            if selection_names is None or self.selections is None
            else {name: self.selections[name] for name in selection_names}
        )
        variant_names = _chosen(
            variants,
            list(self.variants) if self.variants is not None else [_IMPLICIT_VARIANT],
            what="variant",
        )
        kept_variants = (
            self.variants
            if variant_names is None or self.variants is None
            else {name: self.variants[name] for name in variant_names}
        )
        return PlotBook(
            self.data,
            kept_variables,
            selections=kept_selections,
            variants=kept_variants,
            plot_kwargs=self.plot_kwargs,
        )

    def __repr__(self) -> str:
        selections = list(self.selections) if self.selections is not None else None
        variants = list(self.variants) if self.variants is not None else None
        tasks = len(self.variables) * (len(selections) if selections else 1)
        return (
            f"PlotBook(variables={[v.safe_name for v in self.variables]}, "
            f"selections={selections}, variants={variants}, "
            f"tasks={tasks * (len(variants) if variants else 1)})"
        )
