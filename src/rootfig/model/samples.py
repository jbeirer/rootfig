"""The :class:`Sample` description of a dataset."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike
from typing import Any, Literal, TypeAlias

import awkward as ak

from rootfig.errors import LuminosityError, SourceError
from rootfig.io import FileSource, Source, as_source
from rootfig.model.cuts import Cut, CutLike, as_cut
from rootfig.model.units import cross_section_pb, luminosity_fb

__all__ = ["HistType", "Sample", "SampleLike", "as_samples"]

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
        Mark as observed data: drawn as black points with error bars, excluded
        from stacks, and used as the numerator of data/MC ratios.
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
        or a ``TParameter`` such as FCCAnalyses' ``"eventsProcessed"``), or
        ``None`` to use the number of entries in the source.
    entry_start, entry_stop
        Read only this range of entries (counted across files).

    Examples
    --------
    >>> import rootfig as rf
    >>> sig = rf.Sample("sig.root", tree="events", label="Signal", weight="mc_w")
    >>> bkg = rf.Sample(["bkg_*.root"], tree="events", label="Background", color="gray")
    >>> data = rf.Sample("data.root", tree="events", label="Data", is_data=True)
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
    ) -> None:
        source = as_source(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
        if label is None:
            label = getattr(source, "default_label", None) or source.describe()
        if weight is not None and not weight.strip():
            weight = None
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "selection", as_cut(selection))
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "is_data", is_data)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "histtype", histtype)
        if not math.isfinite(scale):
            msg = f"sample {label!r}: scale must be a finite number, got {scale!r}"
            raise ValueError(msg)
        object.__setattr__(self, "scale", float(scale))
        if xsec is not None:
            cross_section_pb(xsec)  # validate early
        if isinstance(ngen, int | float) and not math.isfinite(ngen):
            msg = f"sample {label!r}: ngen must be a finite number, got {ngen!r}"
            raise LuminosityError(msg)
        object.__setattr__(self, "xsec", xsec)
        object.__setattr__(self, "ngen", ngen)

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
        if callable(counter):
            return float(counter())
        first = self.source.branches()[:1]
        return float(len(next(iter(self.source.arrays(first).values()))))

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

    def with_(self, **changes: Any) -> Sample:
        """Return a copy with the given fields replaced, e.g. ``sample.with_(label="B")``."""
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            msg = f"unknown Sample field(s) {sorted(unknown)}; valid fields: {sorted(known)}"
            raise TypeError(msg)
        if "selection" in changes:
            changes["selection"] = as_cut(changes["selection"])
        clone = copy.copy(self)
        for name, value in changes.items():
            object.__setattr__(clone, name, value)
        return clone


SampleLike: TypeAlias = Sample | str | Sequence[Any] | Any
"""A :class:`Sample`, file specification(s), or in-memory arrays."""


def as_samples(
    data: Any,
    *,
    tree: str | None = None,
    labels: str | Sequence[str] | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> list[Sample]:
    """Normalise the ``data`` argument of :func:`rootfig.plot` into a list of samples.

    * A :class:`Sample` gives one sample.
    * A single path/glob/mapping/array gives one sample.
    * A list gives one sample per element. Elements may mix ``Sample`` objects
      with raw file specifications; each raw element becomes its own sample,
      so ``["sig.root", "bkg.root"]`` produces two samples. To combine several
      files into *one* sample wrap them in a ``Sample`` or pass a glob.
    * A mapping from label to file specification, in-memory arrays or
      ``Sample`` gives one labelled sample per entry (a ``Sample`` keeps its
      selection, weight and cross section and only takes the label). A mapping
      whose values are columns (arrays or lists of numbers) is one in-memory
      sample instead.

    Raises
    ------
    SourceError
        If ``labels`` does not match the number of samples.
    """
    if isinstance(data, Sample):
        samples = [data]
    elif isinstance(data, Mapping) and _looks_like_label_map(data):
        samples = [
            value.with_(label=key)
            if isinstance(value, Sample)
            else Sample(value, tree=tree, label=key, entry_start=entry_start, entry_stop=entry_stop)
            for key, value in data.items()
        ]
    elif isinstance(data, list | tuple):
        if not data:
            msg = "no samples given"
            raise SourceError(msg)
        samples = [
            item
            if isinstance(item, Sample)
            else Sample(item, tree=tree, entry_start=entry_start, entry_stop=entry_stop)
            for item in data
        ]
    else:
        samples = [Sample(data, tree=tree, entry_start=entry_start, entry_stop=entry_stop)]

    if labels is not None:
        label_list = [labels] if isinstance(labels, str) else list(labels)
        if len(label_list) != len(samples):
            msg = f"got {len(label_list)} labels for {len(samples)} samples"
            raise SourceError(msg)
        samples = [s.with_(label=lab) for s, lab in zip(samples, label_list, strict=True)]
    return samples


def _looks_like_label_map(data: Mapping[Any, Any]) -> bool:
    """Distinguish ``{"Signal": "sig.root"}`` from a mapping of column arrays.

    Values that describe a dataset (paths, lists of paths, ``Sample`` objects,
    mappings of arrays, Awkward record arrays) make a label map; anything else
    (NumPy arrays, lists of numbers, flat Awkward arrays) is a column.
    """
    return bool(data) and all(isinstance(k, str) and _is_dataset_spec(v) for k, v in data.items())


def _is_dataset_spec(value: Any) -> bool:
    if isinstance(value, str | PathLike | Sample | Mapping):
        return True
    if isinstance(value, ak.Array):
        return bool(value.fields)
    if isinstance(value, list | tuple):
        return bool(value) and all(isinstance(item, str | PathLike) for item in value)
    return False
