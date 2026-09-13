"""The example registry and the source extraction used by the documentation.

Each example is a plain function decorated with :func:`example`. Its parameters
are names of ``Dataset`` attributes (samples, variables), plus ``style`` for an
example drawn in every gallery style, so the body reads exactly like user code;
:func:`body_source` returns that body and :func:`style_source` the line that
makes the style, for the gallery pages of the documentation.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import inspect
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import rootfig as rf

DEFAULT_STYLE = "Default"
"""Name of the neutral style; its images carry the bare example name."""


@dataclass(frozen=True)
class Example:
    """One gallery entry: a name (file stem), a title, a section and the plotting function.

    ``toy_data`` says whether the example reads the toy files, which its page then
    explains how to make.
    """

    name: str
    title: str
    section: str
    func: Callable[..., rf.Plot]
    image_width: str = "75%"
    toy_data: bool = True

    @property
    def description(self) -> str:
        """The function's docstring, dedented."""
        return inspect.cleandoc(self.func.__doc__ or "")

    @property
    def parameters(self) -> list[str]:
        """The names the function asks for."""
        return list(inspect.signature(self.func).parameters)

    @property
    def styled(self) -> bool:
        """Whether the example takes a ``style``, and so is drawn in every gallery style."""
        return "style" in self.parameters

    def run(self, dataset: Any, cwd: Path, style: rf.Style | None = None) -> rf.Plot:
        """Call the example inside ``cwd`` with the attributes of ``dataset`` it asks for.

        ``cwd`` is the directory holding the toy files, so the examples name them
        with bare relative paths (``"signal.root"``) exactly as a user would. An
        example that takes a ``style`` gets ``style`` (the neutral default if omitted).
        """
        values = {**vars(dataset), "style": style if style is not None else rf.Style()}
        with contextlib.chdir(cwd):
            return self.func(**{name: values[name] for name in self.parameters})

    def image(self, style: str = DEFAULT_STYLE, *, dark: bool = False) -> str:
        """File name of one rendering: ``<name>[-<style>][-dark].png``, lower-case style."""
        parts = [self.name, *([style.lower()] if style != DEFAULT_STYLE else [])]
        return "-".join([*parts, *(["dark"] if dark else [])]) + ".png"


EXAMPLES: list[Example] = []


def example(
    name: str, title: str, *, section: str, image_width: str = "75%", toy_data: bool = True
) -> Callable[[Callable[..., rf.Plot]], Callable[..., rf.Plot]]:
    """Register a gallery example under ``section`` (the gallery groups examples by it)."""

    def register(func: Callable[..., rf.Plot]) -> Callable[..., rf.Plot]:
        EXAMPLES.append(
            Example(name, title, section, func, image_width=image_width, toy_data=toy_data)
        )
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


def style_source(style: rf.Style) -> str:
    """Return ``style`` as the ``rf.Style(...)`` call that makes it.

    Only the fields that differ from the default are written, in field order and
    with double-quoted strings, so the call reads like the rest of the gallery.
    """
    default = rf.Style()
    arguments = ", ".join(
        f"{field.name}={json.dumps(value) if isinstance(value, str) else repr(value)}"
        for field in dataclasses.fields(style)
        if (value := getattr(style, field.name)) != getattr(default, field.name)
    )
    return f"rf.Style({arguments})"
