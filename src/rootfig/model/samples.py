"""The :class:`Sample` description of a dataset."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal, TypeAlias

from rootfig._mapping import FrozenMapping
from rootfig.errors import LuminosityError, SourceError, SystematicError
from rootfig.io import FileSource, Source, as_source
from rootfig.model.cuts import Cut, CutLike, as_cut
from rootfig.model.systematics import Systematic, SystematicLike, as_systematics
from rootfig.model.units import cross_section_pb, luminosity_fb

__all__ = ["HistType", "Sample"]

HistType: TypeAlias = Literal["step", "fill", "errorbar", "band"]
"""How a histogram is drawn: outline, filled area, points with error bars, or a band."""


@dataclass(frozen=True, init=False)
class Sample:
    """A dataset: where it comes from, how it is labelled, and how it is drawn.

    Parameters
    ----------
    data
        File path(s), glob pattern(s), ``"path:tree"`` strings, a mapping of
        arrays, an Awkward record array, or an existing
        :class:`~rootfig.io.Source`.
    tree
        Tree name for file inputs. Auto-detected when the file holds exactly one
        ``TTree``/``RNTuple``.
    label
        Legend label. Defaults to the file stem.
    selection
        A selection applied to this sample only, combined with the selection
        given to :func:`rootfig.plot` with ``&``.
    weight
        Expression for per-event (or per-object) weights, e.g. ``"mc_weight"``
        or ``"mc_weight * pileup_sf"``. Combined multiplicatively with the
        weight given to :func:`rootfig.plot`.
    is_data
        Mark as observed data: drawn as points with error bars, excluded
        from stacks, and by default compared with the prediction in a ratio,
        relative difference, difference or pull panel (never a signal of a
        significance).
    color
        Matplotlib colour. Defaults to the style's colour cycle.
    histtype
        How to draw this sample; see :data:`HistType`. Defaults depend on the
        plot type.
    scale
        Constant multiplied into every weight.
    xsec
        Cross section of the process, in pb or as a string with a unit
        (``"1.2 fb"``). With a luminosity given to :func:`rootfig.plot`
        (``lumi=``), every weight is multiplied by ``xsec * lumi / ngen`` so
        that the histogram is the expected yield.
    ngen
        Number of generated events the cross section refers to: a number, the
        name of an object in the file(s) holding it (a sum-of-weights histogram
        or a ``TParameter`` such as ``"eventsProcessed"``), or
        ``None`` to use the number of entries in the source.
    entry_start, entry_stop
        Read only this range of entries (counted across files).
    systematics
        Sources of systematic uncertainty, ``{name: variation}``: a weight
        expression or ``(up, down)`` pair of them replacing ``weight``, a relative
        normalisation uncertainty (``0.05``), ``(up, down)`` normalisation
        factors, a mapping of branches to their ``(up, down)`` replacements
        (``{"Jet_pt": ("Jet_pt_up", "Jet_pt_down")}``), or
        ``Systematic.samples(...)`` for varied files. Sources
        with the same name in several samples are fully correlated; different
        names are independent. Not allowed together with ``is_data=True``. The mapping is copied and
        made read-only; use ``sample.replace(systematics=...)`` to change it.

    Examples
    --------
    >>> import rootfig as rf
    >>> sig = rf.Sample("sig.root", tree="events", label="Signal", weight="mc_w")
    >>> bkg = rf.Sample(["bkg_*.root"], tree="events", label="Background", color="gray")
    >>> data = rf.Sample("data.root", tree="events", label="Data", is_data=True)
    >>> mc = rf.Sample(
    ...     "mc.root", weight="w", systematics={"pileup": ("w_pu_up", "w_pu_down"), "xsec": 0.05}
    ... )  # doctest: +SKIP
    """

    source: Source
    label: str
    selection: Cut | None = None
    weight: str | None = None
    is_data: bool = False
    color: str | None = None
    histtype: HistType | None = None
    scale: float = 1.0
    xsec: float | str | None = None
    ngen: float | str | None = None
    systematics: Mapping[str, Systematic] = field(default_factory=dict)

    def __init__(
        self,
        data: Any,
        *,
        tree: str | None = None,
        label: str | None = None,
        selection: CutLike | None = None,
        weight: str | None = None,
        is_data: bool = False,
        color: str | None = None,
        histtype: HistType | None = None,
        scale: float = 1.0,
        xsec: float | str | None = None,
        ngen: float | str | None = None,
        entry_start: int | None = None,
        entry_stop: int | None = None,
        systematics: Mapping[str, SystematicLike] | None = None,
    ) -> None:
        source = as_source(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
        if label is None:
            label = getattr(source, "default_label", None) or source.describe()
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "selection", as_cut(selection))
        object.__setattr__(self, "weight", _normalise_weight(weight, label))
        object.__setattr__(self, "is_data", is_data)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "histtype", histtype)
        object.__setattr__(self, "scale", _check_scale(scale, label))
        object.__setattr__(self, "xsec", _check_xsec(xsec, label))
        object.__setattr__(self, "ngen", _check_ngen(ngen, label))
        object.__setattr__(self, "systematics", _check_systematics(systematics, label))
        _check_data_systematics(self)

    def __repr__(self) -> str:
        parts = [repr(self.source.describe()), f"label={self.label!r}"]
        if self.selection is not None:
            parts.append(f"selection={self.selection.expression!r}")
        if self.weight is not None:
            parts.append(f"weight={self.weight!r}")
        if self.is_data:
            parts.append("is_data=True")
        if self.xsec is not None:
            parts.append(f"xsec={self.xsec!r}")
        if self.systematics:
            parts.append(f"systematics={sorted(self.systematics)!r}")
        return f"Sample({', '.join(parts)})"

    def generated_events(self) -> float:
        """Return the number of generated events ``ngen`` refers to (see the class docstring)."""
        if isinstance(self.ngen, int | float):
            return float(self.ngen)
        if isinstance(self.ngen, str):
            reader = getattr(self.source, "read_scalar", None)
            if not callable(reader):
                msg = f"sample {self.label!r}: ngen={self.ngen!r} needs a file source"
                raise LuminosityError(msg)
            return float(reader(self.ngen))
        counter = getattr(self.source, "num_entries", None)
        try:
            if callable(counter):
                return float(counter())
            first = self.source.branches()[:1]
            return float(len(next(iter(self.source.arrays(first).values()))))
        except SourceError as exc:
            # a file of stored histograms has no tree whose entries could stand in for ngen
            msg = (
                f"sample {self.label!r} has a cross section but no ngen=, so its generated "
                f"events would be counted from the tree, which cannot be read ({exc}). Pass "
                "ngen=<number>, or ngen='<object>' naming a sum-of-weights histogram or "
                "TParameter in the file (such as 'eventsProcessed')"
            )
            raise LuminosityError(msg) from exc

    def lumi_scale(self, lumi: float | str | None) -> float:
        """Factor turning weights into expected yields at ``lumi``; 1 without a cross section.

        ``xsec`` (pb) times ``lumi`` (fb^-1, or a string with a unit) divided by
        :meth:`generated_events`.
        """
        if self.xsec is None:
            return 1.0
        if lumi is None:
            msg = (
                f"sample {self.label!r} has a cross section ({self.xsec!r}) but no luminosity "
                "was given; pass lumi=... (in fb^-1, or a string such as '10.8 ab^-1')"
            )
            raise LuminosityError(msg)
        ngen = self.generated_events()
        if not (math.isfinite(ngen) and ngen > 0):
            msg = f"sample {self.label!r}: the number of generated events is {ngen}"
            raise LuminosityError(msg)
        factor = cross_section_pb(self.xsec) * 1e3 * luminosity_fb(lumi) / ngen
        if not math.isfinite(factor):
            msg = (
                f"sample {self.label!r}: xsec * lumi / ngen = {factor} is not finite "
                f"(xsec={self.xsec!r}, lumi={lumi!r}, ngen={ngen})"
            )
            raise LuminosityError(msg)
        return factor

    @property
    def files(self) -> tuple[str, ...]:
        """The resolved input files, or an empty tuple for in-memory sources."""
        return self.source.files if isinstance(self.source, FileSource) else ()

    def replace(self, **changes: Any) -> Sample:
        """Return a copy with the given fields changed, e.g. ``sample.replace(label="B")``.

        New values are validated and normalised exactly as by the
        constructor (``selection`` accepts a string, ``source`` anything
        :func:`~rootfig.io.as_source` accepts, ``scale`` must be finite, ...).
        """
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            msg = f"unknown Sample field(s) {sorted(unknown)}; valid fields: {sorted(known)}"
            raise TypeError(msg)
        label = changes.get("label", self.label)
        if not isinstance(label, str):
            msg = f"sample label must be a string, got {type(label).__name__}"
            raise TypeError(msg)
        clone = copy.copy(self)
        for name, value in changes.items():
            check = _FIELD_CHECKS.get(name)
            object.__setattr__(clone, name, value if check is None else check(value, label))
        _check_data_systematics(clone)
        return clone


# -- field validation shared by the constructor and replace() ---------------------------------


def _normalise_weight(weight: str | None, label: str) -> str | None:
    if weight is None:
        return None
    if not isinstance(weight, str):  # runtime guard for untyped callers
        msg = (  # type: ignore[unreachable]
            f"sample {label!r}: weight must be an expression string or None, "
            f"got {type(weight).__name__}"
        )
        raise TypeError(msg)
    return weight if weight.strip() else None


def _check_systematics(
    systematics: Mapping[str, SystematicLike] | None, label: str
) -> Mapping[str, Systematic]:
    return FrozenMapping(as_systematics(systematics, f"sample {label!r}"))


def _check_data_systematics(sample: Sample) -> None:
    """Refuse a data sample with systematics: uncertainties belong to the simulation."""
    if sample.is_data and sample.systematics:
        msg = (
            f"sample {sample.label!r} is observed data and cannot carry systematics "
            f"({sorted(sample.systematics)}); attach them to the simulated samples"
        )
        raise SystematicError(msg)


def _check_scale(scale: float, label: str) -> float:
    try:
        finite = math.isfinite(scale)
    except TypeError:
        finite = False
    if not finite:
        msg = f"sample {label!r}: scale must be a finite number, got {scale!r}"
        raise ValueError(msg)
    return float(scale)


def _check_xsec(xsec: float | str | None, _label: str) -> float | str | None:
    if xsec is not None:
        cross_section_pb(xsec)  # validate early; keep the user's spelling for labels
    return xsec


def _check_ngen(ngen: float | str | None, label: str) -> float | str | None:
    if isinstance(ngen, int | float) and not math.isfinite(ngen):
        msg = f"sample {label!r}: ngen must be a finite number, got {ngen!r}"
        raise LuminosityError(msg)
    return ngen


_FIELD_CHECKS: dict[str, Callable[[Any, str], Any]] = {
    "source": lambda value, _label: as_source(value),
    "selection": lambda value, _label: as_cut(value),
    "weight": _normalise_weight,
    "scale": _check_scale,
    "xsec": _check_xsec,
    "ngen": _check_ngen,
    "systematics": _check_systematics,
}
