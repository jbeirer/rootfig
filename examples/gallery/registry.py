"""The example registry and the source extraction used by the documentation.

Each example is a plain function decorated with :func:`example`. Its parameters
are names of ``Dataset`` attributes (samples, variables, style), so the
body reads exactly like user code; :func:`body_source` returns that body for
``docs/gallery.md``.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import rootfig as rf


@dataclass(frozen=True)
class Example:
    """One gallery entry: a name (file stem), a title and the function producing the plot."""

    name: str
    title: str
    func: Callable[..., rf.Plot]
    image_width: str = "75%"

    @property
    def description(self) -> str:
        """The function's docstring, dedented."""
        return inspect.cleandoc(self.func.__doc__ or "")

    def run(self, dataset: Any, cwd: Path) -> rf.Plot:
        """Call the example inside ``cwd`` with the attributes of ``dataset`` it asks for.

        ``cwd`` is the directory holding the toy files, so the examples name them
        with bare relative paths (``"signal.root"``) exactly as a user would.
        """
        names = inspect.signature(self.func).parameters
        with contextlib.chdir(cwd):
            return self.func(**{name: getattr(dataset, name) for name in names})


EXAMPLES: list[Example] = []


def example(
    name: str, title: str, *, image_width: str = "75%"
) -> Callable[[Callable[..., rf.Plot]], Callable[..., rf.Plot]]:
    """Register a gallery example."""

    def register(func: Callable[..., rf.Plot]) -> Callable[..., rf.Plot]:
        EXAMPLES.append(Example(name, title, func, image_width=image_width))
        return func

    return register


def body_source(func: Callable[..., Any], *, returns: Literal["strip", "omit"] = "strip") -> str:
    """Return the body of ``func`` as user-facing code: no signature, docstring or ``return``.

    Blank lines and comments are kept, so the code blocks in the documentation
    are grouped exactly like the source. ``returns="strip"`` keeps the returned
    expression as a bare statement (``return rf.plot(...)`` becomes
    ``rf.plot(...)``); ``"omit"`` drops the return statement entirely. A
    ``return name`` is always dropped.
    """
    source = textwrap.dedent(inspect.getsource(func))
    node = ast.parse(source).body[0]
    if not isinstance(node, ast.FunctionDef):
        msg = f"{func!r} is not a plain function"
        raise TypeError(msg)
    statements = list(node.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
    ):
        statements = statements[1:]  # docstring
    if not statements:
        return "\n"
    first, last = statements[0].lineno - 1, statements[-1].end_lineno
    assert last is not None
    source_lines = source.splitlines()
    while first > 0 and source_lines[first - 1].lstrip().startswith("#"):
        first -= 1  # a comment introducing the first statement (never the docstring or def)
    lines: list[str | None] = list(source_lines[first:last])
    for statement in statements:
        if not isinstance(statement, ast.Return):
            continue
        assert statement.end_lineno is not None
        at = statement.lineno - 1 - first
        if returns == "omit" or statement.value is None or isinstance(statement.value, ast.Name):
            lines[at : statement.end_lineno - first] = [None] * (
                statement.end_lineno - statement.lineno + 1
            )
        else:
            line = lines[at]
            assert line is not None
            lines[at] = line.replace("return ", "", 1)
    body = "\n".join(line for line in lines if line is not None)
    return textwrap.dedent(body).strip("\n") + "\n"
