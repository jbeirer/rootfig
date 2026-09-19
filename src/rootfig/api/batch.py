"""Batch plotting: :class:`PlotBook` repeats :func:`~rootfig.plot` over variables and selections."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rootfig._mapping import FrozenMapping
from rootfig.api.plots1d import plot
from rootfig.model import Cut, CutLike, Variable, as_cut, as_variable
from rootfig.plotting import Plot, close_figures_since, normalize_formats, open_figure_ids

__all__ = ["PlotBook", "PlotTask"]

_RESERVED_PLOT_KWARGS = frozenset({"data", "variable", "selection", "save", "ax"})
"""Keywords of :func:`~rootfig.plot` that :class:`PlotBook` fills in itself."""

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


@dataclass(frozen=True)
class PlotTask:
    """One plot of a :class:`PlotBook`: a variable, a selection and the keywords to draw it with.

    Tasks hash by their :attr:`stem`, which is unique within a book, so they can be
    collected in a set or used as dictionary keys.

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

    def __hash__(self) -> int:
        return hash(self.stem)


def _identifier(value: object, *, what: str) -> str:
    """Return ``value`` if it can name an output file component, else raise."""
    if not isinstance(value, str):
        msg = f"{what} names must be non-empty strings, got {value!r}"
        raise TypeError(msg)
    unusable = (
        not value.strip(" .")
        or any(character < " " or character == "\x7f" for character in value)
        or "/" in value
        or "\\" in value
        or os.sep in value
    )
    if unusable:
        msg = (
            f"{what} name {value!r} cannot be a file name component: it must hold more than "
            "dots and spaces, and no control character, slash or backslash"
        )
        raise ValueError(msg)
    return value


def _as_mapping(values: object, *, what: str, holds: str) -> Mapping[str, Any]:
    """Return ``values`` if it is a mapping, else raise naming what was expected."""
    if not isinstance(values, Mapping):
        msg = f"{what} must map {holds}, got {type(values).__name__}"
        raise TypeError(msg)
    return values


def _variables(values: str | Variable | Sequence[str | Variable]) -> tuple[Variable, ...]:
    """Coerce the variables and reject an empty list or two variables with one identifier."""
    items = (values,) if isinstance(values, str | Variable) else tuple(values)
    if not items:
        msg = "PlotBook needs at least one variable"
        raise ValueError(msg)
    variables = tuple(as_variable(item) for item in items)
    seen: dict[str, Variable] = {}
    for variable in variables:
        key = _identifier(variable.safe_name, what="variable")
        if key in seen:
            msg = (
                f"variables {seen[key].expression!r} and {variable.expression!r} share the "
                f"identifier {key!r}, so their output files would collide; "
                "give one a distinct name="
            )
            raise ValueError(msg)
        seen[key] = variable
    return variables


def _plot_kwargs(values: Mapping[str, Any] | None, *, where: str) -> FrozenMapping[str, Any]:
    """Copy plot keywords, rejecting the ones the book fills in itself."""
    if values is None:
        return FrozenMapping({})
    mapping = _as_mapping(values, what=where, holds="plot() keywords to values")
    reserved = sorted(_RESERVED_PLOT_KWARGS.intersection(mapping))
    if reserved:
        msg = (
            f"{where} must not set {', '.join(repr(k) for k in reserved)}: PlotBook passes data, "
            "variable and selection itself, save() owns the output files and every task draws "
            "its own figure (no ax=)"
        )
        raise ValueError(msg)
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


@dataclass(frozen=True, init=False, eq=False, repr=False)
class PlotBook:
    """Many :func:`~rootfig.plot` calls described once: variables, selections and variants.

    A book holds what every plot shares (``data`` and ``plot_kwargs``) and the
    three axes that vary. :meth:`tasks` lists the Cartesian product in a fixed
    order (variables, then selections, then variants, each in insertion order),
    :meth:`plots` runs it one plot at a time and :meth:`save` writes one file per
    task under deterministic names. Each plot is an ordinary
    :func:`~rootfig.plot` call, so samples, groups, stored histograms, arrays and
    histogram objects behave exactly as they do there; ``data`` is passed on as
    given, never copied or inspected. Names of selections and variants are file
    name components: they must hold more than dots and spaces, and no control
    character, slash or backslash.

    Parameters
    ----------
    data
        What to plot, exactly as :func:`~rootfig.plot` takes it.
    variables
        Variables (names, expressions or :class:`~rootfig.model.Variable`
        objects); one may be given bare. :attr:`~rootfig.model.Variable.safe_name`
        identifies each and names its files, so two variables must not share one.
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

    Raises
    ------
    ValueError
        No variable, an empty ``selections=`` or ``variants=``, a reserved plot
        keyword, a name that cannot be a file name component, or two tasks whose
        output files would collide.
    TypeError
        ``selections=``, ``variants=``, ``plot_kwargs=`` or one variant's overrides
        is not a mapping, or a name is not a string.

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
    """

    data: Any
    variables: tuple[Variable, ...]
    selections: FrozenMapping[str, Cut | None] | None
    variants: FrozenMapping[str, FrozenMapping[str, Any]] | None
    plot_kwargs: FrozenMapping[str, Any]

    def __init__(
        self,
        data: Any,
        variables: str | Variable | Sequence[str | Variable],
        *,
        selections: Mapping[str, CutLike | None] | None = None,
        variants: Mapping[str, Mapping[str, Any]] | None = None,
        plot_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "variables", _variables(variables))
        object.__setattr__(self, "selections", _selections(selections))
        object.__setattr__(self, "variants", _variants(variants))
        object.__setattr__(self, "plot_kwargs", _plot_kwargs(plot_kwargs, where="plot_kwargs"))
        counts = Counter(
            _stem(variable.safe_name, selection, variant)
            for variable in self.variables
            for selection in (self.selections if self.selections is not None else [None])
            for variant in (self.variants if self.variants is not None else [None])
        )
        clashes = sorted(stem for stem, count in counts.items() if count > 1)
        if clashes:
            msg = (
                f"output names collide: {clashes}; rename a variable, selection or variant "
                f"so that no two tasks share <variable>{_SEPARATOR}<selection>{_SEPARATOR}<variant>"
            )
            raise ValueError(msg)

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

        Each step is ``plot(data, task.variable, selection=task.selection,
        **task.kwargs)``, and the tasks themselves are built one at a time. The
        caller owns the figures that are yielded and closes them
        (:meth:`Plot.close`); the figure a failing call had already created is
        closed here. An error keeps its type and gains a note naming the task
        (``variable=..., selection=..., variant=...``).
        """
        for task in self._iter_tasks():
            before = open_figure_ids()
            drawn = False
            try:
                result = plot(self.data, task.variable, selection=task.selection, **task.kwargs)
                drawn = True
            except Exception as exc:
                exc.add_note(f"while running PlotBook task {task.describe()}")
                raise
            finally:
                if not drawn:
                    close_figures_since(before)
            yield task, result

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
