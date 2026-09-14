"""Internal immutable mappings for validated model metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType


class FrozenMapping[K, V](Mapping[K, V]):
    """A copied, read-only mapping that supports copying and pickling.

    Values retain their existing mutability; only the mapping is frozen.
    """

    __slots__ = ("_data",)

    _data: Mapping[K, V]

    def __init__(self, data: Mapping[K, V]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(data)))

    def __setattr__(self, name: str, value: object) -> None:
        msg = "FrozenMapping is read-only"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        msg = "FrozenMapping is read-only"
        raise AttributeError(msg)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(dict(self._data))

    def __reduce__(self) -> tuple[type[FrozenMapping[K, V]], tuple[dict[K, V]]]:
        return type(self), (dict(self._data),)
