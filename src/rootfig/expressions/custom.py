"""Functions the caller adds to expressions, for the duration of one rootfig call.

Expressions are parsed where they are used, deep in the pipeline and in its worker
threads, often from strings combined there (a sample weight times a plot weight).
The functions given to an ``rf`` call are therefore held in a context variable for
the call's duration: :func:`~rootfig.expressions.parse` reads it, and the threads
preparing chunks of events run in a copy of the caller's context.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, Final

from rootfig.errors import ExpressionError
from rootfig.expressions.functions import CONSTANTS, FUNCTIONS

__all__ = ["RESERVED_PREFIX", "active_functions", "check_functions", "function_scope"]

RESERVED_PREFIX: Final = "__rootfig_"
"""Names rootfig gives the functions and quoted branches of a parsed expression begin so."""

_NONE: Final[Mapping[str, Callable[..., Any]]] = MappingProxyType({})

_ACTIVE: ContextVar[Mapping[str, Callable[..., Any]]] = ContextVar(
    "rootfig_functions", default=_NONE
)


def check_functions(functions: object) -> Mapping[str, Callable[..., Any]]:
    """Return a read-only copy of ``functions`` once every name and value is usable.

    ``functions`` is what the caller gave as ``functions=``: ``{name: callable}``
    or ``None`` for none.

    Raises
    ------
    TypeError
        ``functions`` is not a mapping, a name is not a string or a value is not callable.
    ExpressionError
        A name cannot be called from an expression (not an identifier, a keyword, or
        reserved) or is a built-in function or constant.
    """
    if functions is None:
        return _NONE
    if not isinstance(functions, Mapping):
        msg = f"functions= must map names to callables, got {type(functions).__name__}"
        raise TypeError(msg)
    for name, function in functions.items():
        if not isinstance(name, str):
            msg = f"functions= names must be strings, got {name!r}"
            raise TypeError(msg)
        if not name.isidentifier() or keyword.iskeyword(name):
            msg = f"function name {name!r} cannot be called from an expression; use an identifier"
            raise ExpressionError(msg)
        if name.startswith(RESERVED_PREFIX):
            msg = f"function name {name!r} is reserved for rootfig; give yours another name"
            raise ExpressionError(msg)
        if name in FUNCTIONS or name in CONSTANTS:
            what = "function" if name in FUNCTIONS else "constant"
            msg = f"{name!r} is a built-in {what} of rootfig expressions; give yours another name"
            raise ExpressionError(msg)
        if not callable(function):
            msg = f"functions= value for {name!r} must be callable, got {type(function).__name__}"
            raise TypeError(msg)
    return MappingProxyType(dict(functions))


@contextmanager
def function_scope(functions: Mapping[str, Callable[..., Any]] | None) -> Iterator[None]:
    """Make ``functions`` the ones expressions parsed inside the block may call.

    ``None`` means none: an ``rf`` call never sees the functions of a call it runs
    in. Only a plain function may enter it, not a generator, which would hand the
    functions to its caller between its yields.
    """
    token = _ACTIVE.set(check_functions(functions))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_functions() -> Mapping[str, Callable[..., Any]]:
    """Return the functions of the ``rf`` call running; empty outside one."""
    return _ACTIVE.get()
