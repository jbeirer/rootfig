"""The :class:`Group` composition of samples drawn as one histogram."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from rootfig.model.samples import HistType, Sample

__all__ = ["Group"]


@dataclass(frozen=True, init=False)
class Group:
    """Several samples drawn as one histogram.

    The components are filled independently, each with its own source,
    selection, weight, scale, cross section and systematics, and their
    histograms are summed afterwards; same-named systematic sources stay
    correlated (see :func:`~rootfig.histograms.sum_histograms`). Normalisation
    applies to the sum. The summed histogram carries the group's label and
    drawing hints, no ``sample`` and no unbinned statistics.

    Parameters
    ----------
    components
        :class:`Sample` objects and nested groups, in drawing order. All
        observed data or all simulation. Raw file paths are not accepted; wrap
        them in a ``Sample``.
    label
        Legend label of the summed histogram.
    color, histtype
        Drawing hints for the summed histogram, ``None`` for the style defaults.
        The components' own are not used.

    Examples
    --------
    >>> import rootfig as rf
    >>> ww = rf.Sample("ww.root", label="WW", xsec="16.4 pb", ngen="eventsProcessed")
    >>> zz = rf.Sample("zz.root", label="ZZ", xsec="1.4 pb", ngen="eventsProcessed")
    >>> vv = rf.Group([ww, zz], label="VV", color="C0")  # doctest: +SKIP
    """

    components: tuple[Sample | Group, ...]
    label: str
    color: str | None = None
    histtype: HistType | None = None

    def __init__(
        self,
        components: Sequence[Sample | Group],
        *,
        label: str,
        color: str | None = None,
        histtype: HistType | None = None,
    ) -> None:
        object.__setattr__(self, "label", _check_label(label))
        object.__setattr__(self, "components", _check_components(components, label))
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "histtype", histtype)
        _check_data_flags(self)

    def __repr__(self) -> str:
        parts = [repr([c.label for c in self.components]), f"label={self.label!r}"]
        if self.color is not None:
            parts.append(f"color={self.color!r}")
        if self.histtype is not None:
            parts.append(f"histtype={self.histtype!r}")
        return f"Group({', '.join(parts)})"

    @property
    def samples(self) -> tuple[Sample, ...]:
        """The leaf samples, nested groups expanded, left to right."""
        return tuple(
            leaf
            for component in self.components
            for leaf in (component.samples if isinstance(component, Group) else (component,))
        )

    @property
    def is_data(self) -> bool:
        """Whether the group is observed data (all of its samples are)."""
        return self.samples[0].is_data

    def replace(self, **changes: Any) -> Group:
        """Return a copy with the given fields changed, validated like the constructor."""
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            msg = f"unknown Group field(s) {sorted(unknown)}; valid fields: {sorted(known)}"
            raise TypeError(msg)
        current = {name: getattr(self, name) for name in known}
        return Group(**{**current, **changes})


def _check_label(label: str) -> str:
    if not isinstance(label, str):  # runtime guard for untyped callers
        msg = f"group label must be a string, got {type(label).__name__}"  # type: ignore[unreachable]
        raise TypeError(msg)
    if not label.strip():
        msg = "group label must not be blank"
        raise ValueError(msg)
    return label


def _check_components(
    components: Sequence[Sample | Group], label: str
) -> tuple[Sample | Group, ...]:
    if isinstance(components, str | bytes | Mapping) or not isinstance(components, Sequence):
        msg = (
            f"group {label!r}: components must be a sequence of Sample or Group objects, "
            f"got {type(components).__name__}"
        )
        raise TypeError(msg)
    if not components:
        msg = f"group {label!r} has no components"
        raise ValueError(msg)
    for index, component in enumerate(components):
        if not isinstance(component, Sample | Group):
            msg = (  # type: ignore[unreachable]
                f"component {index} of group {label!r} is a {type(component).__name__} "
                f"({component!r}), not a Sample or Group; wrap it in rf.Sample(...)"
            )
            raise TypeError(msg)
    return tuple(components)


def _check_data_flags(group: Group) -> None:
    """Refuse a group mixing data and simulation: it is drawn as one histogram, which is either."""
    samples = group.samples
    data = [s.label for s in samples if s.is_data]
    if data and len(data) != len(samples):
        simulated = [s.label for s in samples if not s.is_data]
        msg = (
            f"group {group.label!r} mixes observed data ({data}) and simulated samples "
            f"({simulated}); a group is drawn as one histogram, which is either data or simulation"
        )
        raise ValueError(msg)
