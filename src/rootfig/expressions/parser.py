"""Parsing, validation and evaluation of expression strings."""

from __future__ import annotations

import ast
import difflib
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, Final

import awkward as ak
import numpy as np

from rootfig.errors import ExpressionError, MissingBranchError
from rootfig.expressions.functions import CONSTANTS, FUNCTIONS

__all__ = ["Expression", "ExpressionLike", "evaluate", "parse"]


# --------------------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------------------

_FUNCTION_PREFIX: Final = "__rootfig_fn_"
_BACKTICK_PREFIX: Final = "__rootfig_bt_"
_NOT_NAME: Final = "__rootfig_not"
_BACKTICK_RE: Final = re.compile(r"`([^`]*)`")

_ALLOWED_BINOPS: Final[dict[type[ast.operator], str]] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
}
_ALLOWED_UNARYOPS: Final[tuple[type[ast.unaryop], ...]] = (ast.USub, ast.UAdd, ast.Invert)
_ALLOWED_CMPOPS: Final[tuple[type[ast.cmpop], ...]] = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _mangle_backticks(text: str) -> tuple[str, dict[str, str]]:
    """Replace backtick-quoted names with valid identifiers; return the reverse mapping."""
    mapping: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        original = match.group(1)
        if not original:
            msg = "empty backticks `` in expression"
            raise ExpressionError(msg)
        for mangled, name in mapping.items():
            if name == original:
                return mangled
        mangled = f"{_BACKTICK_PREFIX}{len(mapping)}"
        mapping[mangled] = original
        return mangled

    return _BACKTICK_RE.sub(repl, text), mapping


class _Rewriter(ast.NodeTransformer):
    """Validate the AST and rewrite Python-only constructs to element-wise ones.

    Collects the referenced names into ``self.names`` (in order of first
    appearance) and the called functions into ``self.functions``.
    """

    def __init__(self, text: str, backticks: dict[str, str]) -> None:
        self.text = text
        self.backticks = backticks
        self.names: list[str] = []
        self.functions: list[str] = []

    # -- helpers -------------------------------------------------------------------

    def _fail(self, node: ast.AST, what: str) -> ExpressionError:
        segment = ast.get_source_segment(self.text, node) or type(node).__name__
        return ExpressionError(f"{what} is not allowed in expressions: {segment!r}")

    def _display_name(self, ident: str) -> str:
        return self.backticks.get(ident, ident)

    # -- allowed nodes ---------------------------------------------------------------

    def visit_Expression(self, node: ast.Expression) -> ast.AST:
        node.body = self.visit(node.body)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            raise self._fail(node, "assignment")
        display = self._display_name(node.id)
        if display not in self.names:
            self.names.append(display)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """``Collection.field.sub`` is one dotted branch name, as in EDM4hep/podio files."""
        parts: list[str] = []
        base: ast.expr = node
        while isinstance(base, ast.Attribute):
            if not isinstance(base.ctx, ast.Load):
                raise self._fail(node, "assignment")
            parts.append(base.attr)
            base = base.value
        if not isinstance(base, ast.Name):
            raise self._fail(node, "attribute access on this construct")
        dotted = ".".join([self._display_name(base.id), *reversed(parts)])
        mangled = self._mangle(dotted)
        if dotted not in self.names:
            self.names.append(dotted)
        return ast.copy_location(ast.Name(id=mangled, ctx=ast.Load()), node)

    def _mangle(self, name: str) -> str:
        """Register ``name`` (not a valid identifier) and return its stand-in identifier."""
        for mangled, original in self.backticks.items():
            if original == name:
                return mangled
        mangled = f"{_BACKTICK_PREFIX}{len(self.backticks)}"
        self.backticks[mangled] = name
        return mangled

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool | int | float):
            return node
        raise self._fail(node, f"a {type(node.value).__name__} literal")

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if type(node.op) not in _ALLOWED_BINOPS:
            raise self._fail(node, f"operator {type(node.op).__name__}")
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            # ``not x`` as a logical negation: ``~`` would turn a Python ``True`` into ``-2``
            # and an integer array into bit patterns instead of booleans.
            func = ast.copy_location(ast.Name(id=_NOT_NAME, ctx=ast.Load()), node)
            return ast.copy_location(ast.Call(func=func, args=[operand], keywords=[]), node)
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise self._fail(node, f"operator {type(node.op).__name__}")
        node.operand = operand
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        op: ast.operator = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        values = [self.visit(v) for v in node.values]
        result: ast.expr = values[0]
        for value in values[1:]:
            result = ast.copy_location(ast.BinOp(left=result, op=op, right=value), node)
        return result

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        for op in node.ops:
            if not isinstance(op, _ALLOWED_CMPOPS):
                raise self._fail(node, f"comparison {type(op).__name__}")
        operands = [self.visit(node.left), *(self.visit(c) for c in node.comparators)]
        comparisons = [
            ast.copy_location(ast.Compare(left=lhs, ops=[op], comparators=[rhs]), node)
            for lhs, op, rhs in zip(operands[:-1], node.ops, operands[1:], strict=True)
        ]
        result: ast.expr = comparisons[0]
        for comparison in comparisons[1:]:
            result = ast.copy_location(
                ast.BinOp(left=result, op=ast.BitAnd(), right=comparison), node
            )
        return result

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func, ast.Name):
            raise self._fail(node.func, "calling anything but a known function by name")
        name = node.func.id
        if name not in FUNCTIONS:
            suggestions = difflib.get_close_matches(name, FUNCTIONS, n=3)
            hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
            msg = f"unknown function {name!r}.{hint} Available functions: " + ", ".join(
                sorted(FUNCTIONS)
            )
            raise ExpressionError(msg)
        if name not in self.functions:
            self.functions.append(name)
        node.func = ast.copy_location(
            ast.Name(id=f"{_FUNCTION_PREFIX}{name}", ctx=ast.Load()), node
        )
        node.args = [self.visit(a) for a in node.args]
        for keyword in node.keywords:
            if keyword.arg is None:
                raise self._fail(node, "** argument unpacking")
            keyword.value = self.visit(keyword.value)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return node

    def visit_Slice(self, node: ast.Slice) -> ast.AST:
        for attr in ("lower", "upper", "step"):
            value = getattr(node, attr)
            if value is not None:
                setattr(node, attr, self.visit(value))
        return node

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        node.elts = [self.visit(e) for e in node.elts]
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        # ``a if cond else b`` is ambiguous for arrays; steer users to where().
        raise self._fail(node, "the conditional expression (use where(cond, a, b))")

    def generic_visit(self, node: ast.AST) -> ast.AST:
        raise self._fail(node, f"the construct {type(node).__name__}")


# --------------------------------------------------------------------------------------
# Public objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Expression:
    """A parsed, validated expression ready to be evaluated on arrays.

    Create instances with :func:`parse` (or pass strings anywhere rootfig
    accepts an expression; they are parsed on the fly).

    Attributes
    ----------
    text
        The original expression string.
    names
        Names referenced by the expression in order of first appearance. These
        are candidate branch names; a name that is not a branch may still
        resolve to a constant (``pi``, ``e``, ``inf``, ``nan``).
    functions
        Names of the functions called by the expression.
    """

    text: str
    names: tuple[str, ...]
    functions: tuple[str, ...]
    _code: CodeType = field(repr=False, compare=False)
    _backticks: Mapping[str, str] = field(repr=False, compare=False, default_factory=dict)

    def __str__(self) -> str:
        return self.text

    @property
    def is_trivial(self) -> bool:
        """True if the expression is a bare name (a single branch, no computation)."""
        return (
            len(self.names) == 1
            and not self.functions
            and self.text.strip()
            in (
                self.names[0],
                f"`{self.names[0]}`",
            )
        )

    def required_branches(self, available: Collection[str]) -> list[str]:
        """Return the referenced names that must be read from ``available`` branches.

        Names that are not available but match a constant are skipped. Any
        other unknown name raises :class:`~rootfig.errors.MissingBranchError`
        with close-match suggestions.
        """
        required: list[str] = []
        for name in self.names:
            if name in available:
                required.append(name)
            elif name not in CONSTANTS:
                suggestions = difflib.get_close_matches(name, list(available), n=3)
                raise MissingBranchError(
                    name,
                    available=sorted(available),
                    suggestions=suggestions,
                    context=f"expression {self.text!r}",
                )
        return required

    def evaluate(
        self, arrays: Mapping[str, Any] | ak.Array, *, length: int | None = None
    ) -> ak.Array:
        """Evaluate the expression using ``arrays`` to resolve branch names.

        Parameters
        ----------
        arrays
            Either a mapping from branch name to array, or an Awkward record
            array whose fields are the branches.
        length
            Number of events a constant expression (``"1"``, ``"True"``) is
            broadcast to when ``arrays`` holds nothing to take the length from.

        Returns
        -------
        awkward.Array
            The result, converted to an Awkward array if the expression produced
            a NumPy array or a scalar.
        """
        lookup = _as_mapping(arrays)
        self.required_branches(list(lookup.keys()))
        namespace: dict[str, Any] = {
            f"{_FUNCTION_PREFIX}{name}": fn for name, fn in FUNCTIONS.items()
        }
        namespace[_NOT_NAME] = np.logical_not
        for mangled, original in self._backticks.items():
            namespace[mangled] = (
                _bind(lookup[original]) if original in lookup else CONSTANTS[original]
            )
        # A name may occur both quoted and unquoted (``x + `x```); bind both spellings.
        for name in self.names:
            if name.isidentifier():
                namespace[name] = _bind(lookup[name]) if name in lookup else CONSTANTS[name]
        try:
            result = eval(self._code, {"__builtins__": {}}, namespace)  # validated AST
        except ExpressionError:
            raise
        except Exception as exc:
            msg = f"failed to evaluate expression {self.text!r}: {type(exc).__name__}: {exc}"
            raise ExpressionError(msg) from exc
        return _as_awkward(result, self.text, lookup, length)


ExpressionLike = str | Expression
"""Anything accepted where an expression is expected."""


def parse(expression: ExpressionLike) -> Expression:
    """Parse and validate an expression string.

    Raises
    ------
    ExpressionError
        If the text is not a single Python expression or uses a disallowed
        construct (attribute access, unknown functions, string literals, ...).
    """
    if isinstance(expression, Expression):
        return expression
    if not isinstance(expression, str):  # runtime guard for untyped callers
        msg = f"expression must be a string, got {type(expression).__name__}"  # type: ignore[unreachable]
        raise ExpressionError(msg)
    text = expression.strip()
    if not text:
        msg = "expression is empty"
        raise ExpressionError(msg)
    mangled, backticks = _mangle_backticks(text)
    try:
        tree = ast.parse(mangled, mode="eval")
    except SyntaxError as exc:
        msg = f"invalid syntax in expression {text!r}: {exc.msg}"
        raise ExpressionError(msg) from exc
    rewriter = _Rewriter(mangled, backticks)
    tree = rewriter.visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<rootfig expression>", "eval")
    return Expression(
        text=text,
        names=tuple(rewriter.names),
        functions=tuple(rewriter.functions),
        _code=code,
        _backticks=backticks,
    )


def evaluate(
    expression: ExpressionLike, arrays: Mapping[str, Any] | ak.Array, *, length: int | None = None
) -> ak.Array:
    """Parse (if needed) and evaluate ``expression`` on ``arrays``.

    This is a convenience wrapper around :func:`parse` and
    :meth:`Expression.evaluate`; ``length`` sizes constant expressions when
    ``arrays`` is empty.

    Examples
    --------
    >>> import awkward as ak
    >>> arrays = {"pt": ak.Array([[10.0, 30.0], [], [50.0]])}
    >>> evaluate("pt > 20", arrays).tolist()
    [[False, True], [], [True]]
    >>> evaluate("count(pt)", arrays).tolist()
    [2, 0, 1]
    """
    return parse(expression).evaluate(arrays, length=length)


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _as_mapping(arrays: object) -> Mapping[str, Any]:
    if isinstance(arrays, Mapping):
        return arrays
    if isinstance(arrays, ak.Array):
        if not arrays.fields:
            msg = "expected a record array with named fields, got an array without fields"
            raise ExpressionError(msg)
        return {name: arrays[name] for name in arrays.fields}
    msg = f"cannot interpret {type(arrays).__name__} as a mapping of branch arrays"
    raise ExpressionError(msg)


def _bind(value: Any) -> Any:
    """Present fixed-size (regular) dimensions as variable-length lists before evaluation.

    Regular arrays broadcast like NumPy matrices, so ``x >= 50 and met > 10`` with a
    ``float x[3]`` branch would fail (or silently align the wrong axis); as lists of
    objects they follow rootfig's per-event/per-object rules.
    """
    if isinstance(value, np.ndarray) and value.ndim > 1:
        value = ak.Array(value)
    if isinstance(value, ak.Array) and ak.to_layout(value).purelist_depth > 1:
        return ak.from_regular(value, axis=None)
    return value


def _as_awkward(
    result: Any, text: str, lookup: Mapping[str, Any], length: int | None = None
) -> ak.Array:
    if isinstance(result, ak.Array):
        return result
    if isinstance(result, np.ndarray):
        return ak.Array(result)
    if isinstance(result, bool | int | float | np.generic):
        # Broadcast a scalar over events, e.g. weight="1.5" or selection="True".
        if length is None:
            length = _common_length(lookup.values())
        if length is None:
            msg = (
                f"expression {text!r} is a constant and no branch arrays were available to "
                "determine the number of events"
            )
            raise ExpressionError(msg)
        return ak.Array(np.full(length, result))
    msg = f"expression {text!r} produced an unsupported result of type {type(result).__name__}"
    raise ExpressionError(msg)


def _common_length(arrays: Iterable[Any]) -> int | None:
    for array in arrays:
        try:
            return len(array)
        except TypeError:
            continue
    return None
