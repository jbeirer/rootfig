"""Tests for the expression layer: parsing, validation, rewriting, evaluation."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from rootfig.errors import ExpressionError, MissingBranchError
from rootfig.expressions import CONSTANTS, FUNCTIONS, Expression, evaluate, parse
from rootfig.expressions.custom import active_functions, function_scope
from rootfig.expressions.parser import validate


@pytest.fixture
def arrays() -> dict[str, ak.Array]:
    return {
        "x": ak.Array([1.0, 2.0, 3.0, 4.0]),
        "y": ak.Array([10.0, 20.0, 30.0, 40.0]),
        "n": ak.Array([2, 0, 1, 3]),
        "pt": ak.Array([[10.0, 30.0], [], [50.0], [5.0, 60.0, 70.0]]),
        "eta": ak.Array([[0.5, -1.5], [], [2.0], [0.1, -0.2, 3.0]]),
        "jet1_b-tag": ak.Array([0.9, 0.1, 0.5, 0.7]),
    }


class TestParse:
    def test_single_name(self) -> None:
        expr = parse("x")
        assert isinstance(expr, Expression)
        assert expr.names == ("x",)
        assert expr.functions == ()
        assert expr.is_trivial
        assert str(expr) == "x"

    def test_names_in_order_of_appearance(self) -> None:
        assert parse("y + x * y - n").names == ("y", "x", "n")

    def test_functions_recorded(self) -> None:
        expr = parse("sqrt(abs(x)) + count(pt)")
        assert expr.functions == ("sqrt", "abs", "count")
        assert not expr.is_trivial

    def test_parse_is_idempotent_on_expression(self) -> None:
        expr = parse("x + 1")
        assert parse(expr) is expr

    def test_backticks(self) -> None:
        expr = parse("`jet1_b-tag` > 0.5")
        assert expr.names == ("jet1_b-tag",)

    def test_backtick_trivial(self) -> None:
        assert parse("`jet1_b-tag`").is_trivial

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty(self, text: str) -> None:
        with pytest.raises(ExpressionError, match="empty"):
            parse(text)

    def test_not_a_string(self) -> None:
        with pytest.raises(ExpressionError, match="must be a string"):
            parse(3)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "text",
        ["x >", "x +* y", "(x", "x = 1", "x; y", "import os", "x if y else n"],
    )
    def test_syntax_errors(self, text: str) -> None:
        with pytest.raises(ExpressionError):
            parse(text)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("(lambda: 1)()", "calling anything but"),
            ("[i for i in x]", "ListComp"),
            ("'abc'", "str literal"),
            ("x in y", "comparison In"),
            ("x is y", "comparison Is"),
            ("__import__('os')", "unknown function"),
            ("open('f')", "unknown function"),
            ("x @ y", "operator MatMult"),
            ("x << 1", "operator LShift"),
            ("f(**x)", "unknown function"),
            ("sqrt(**x)", "unpacking"),
            ("{1: 2}", "Dict"),
        ],
    )
    def test_disallowed_constructs(self, text: str, match: str) -> None:
        with pytest.raises(ExpressionError, match=match):
            parse(text)

    def test_unknown_function_suggests(self) -> None:
        with pytest.raises(ExpressionError, match="Did you mean sqrt"):
            parse("sqr(x)")

    def test_empty_backticks(self) -> None:
        with pytest.raises(ExpressionError, match="empty backticks"):
            parse("`` > 1")


class TestRequiredBranches:
    def test_constants_are_not_branches(self) -> None:
        expr = parse("x * pi + e")
        assert expr.required_branches(["x", "y"]) == ["x"]

    def test_branch_shadows_constant(self) -> None:
        expr = parse("e + 1")
        assert expr.required_branches(["e"]) == ["e"]

    def test_missing_branch_with_suggestion(self) -> None:
        expr = parse("Muon_pt > 20")
        with pytest.raises(MissingBranchError, match="Did you mean 'Muon_Pt'") as info:
            expr.required_branches(["Muon_Pt", "Muon_eta", "nMuon"])
        assert info.value.name == "Muon_pt"
        assert info.value.suggestions[0] == "Muon_Pt"

    def test_missing_branch_lists_available(self) -> None:
        with pytest.raises(MissingBranchError, match="Available names: 'a', 'b'"):
            parse("zzz").required_branches(["b", "a"])


class TestEvaluate:
    def test_scalar_arithmetic(self, arrays: dict[str, ak.Array]) -> None:
        result = evaluate("(x + y) / 2 - x**2 % 3", arrays)
        expected = (arrays["x"] + arrays["y"]) / 2 - arrays["x"] ** 2 % 3
        assert result.tolist() == expected.tolist()

    def test_jagged_comparison(self, arrays: dict[str, ak.Array]) -> None:
        assert evaluate("pt > 20", arrays).tolist() == [
            [False, True],
            [],
            [True],
            [False, True, True],
        ]

    def test_and_or_not_rewritten(self, arrays: dict[str, ak.Array]) -> None:
        keyword = evaluate("pt > 20 and not eta < 0 or pt > 65", arrays)
        bitwise = evaluate("((pt > 20) & ~(eta < 0)) | (pt > 65)", arrays)
        assert keyword.tolist() == bitwise.tolist()
        assert keyword.tolist() == [[False, False], [], [True], [False, False, True]]

    def test_chained_comparison(self, arrays: dict[str, ak.Array]) -> None:
        result = evaluate("20 < pt <= 60", arrays)
        assert result.tolist() == [[False, True], [], [True], [False, True, False]]

    def test_chained_comparison_with_and(self, arrays: dict[str, ak.Array]) -> None:
        result = evaluate("1 < x < 4 and y > 15", arrays)
        assert result.tolist() == [False, True, True, False]

    def test_functions_elementwise(self, arrays: dict[str, ak.Array]) -> None:
        result = evaluate("sqrt(abs(eta)) + log(pt)", arrays)
        expected = np.sqrt(np.abs(arrays["eta"])) + np.log(arrays["pt"])
        assert result.tolist() == expected.tolist()

    def test_where_and_clip(self, arrays: dict[str, ak.Array]) -> None:
        assert evaluate("where(x > 2, x, -1)", arrays).tolist() == [-1.0, -1.0, 3.0, 4.0]
        assert evaluate("clip(x, 2, 3)", arrays).tolist() == [2.0, 2.0, 3.0, 3.0]

    def test_reductions(self, arrays: dict[str, ak.Array]) -> None:
        assert evaluate("count(pt)", arrays).tolist() == [2, 0, 1, 3]
        assert evaluate("len(pt)", arrays).tolist() == [2, 0, 1, 3]
        assert evaluate("sum(pt)", arrays).tolist() == [40.0, 0.0, 50.0, 135.0]
        assert evaluate("any(pt > 40)", arrays).tolist() == [False, False, True, True]
        assert evaluate("all(pt > 40)", arrays).tolist() == [False, True, True, False]
        assert evaluate("max(pt)", arrays).tolist() == [30.0, None, 50.0, 70.0]
        assert evaluate("first(pt)", arrays).tolist() == [10.0, None, 50.0, 5.0]
        assert evaluate("count(pt[pt > 20])", arrays).tolist() == [1, 0, 1, 2]

    def test_reduction_requires_jagged(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(ExpressionError, match="count\\(\\) reduces over the objects"):
            evaluate("count(x)", arrays)

    def test_indexing_and_slicing(self, arrays: dict[str, ak.Array]) -> None:
        assert evaluate("pt[:, :1]", arrays).tolist() == [[10.0], [], [50.0], [5.0]]
        assert evaluate("x[1:3]", arrays).tolist() == [2.0, 3.0]

    def test_backtick_names(self, arrays: dict[str, ak.Array]) -> None:
        assert evaluate("`jet1_b-tag` > 0.5 and x > 1", arrays).tolist() == [
            False,
            False,
            False,
            True,
        ]

    def test_constants(self, arrays: dict[str, ak.Array]) -> None:
        assert evaluate("x * pi", arrays).tolist() == (arrays["x"] * np.pi).tolist()
        assert evaluate("x > -inf", arrays).tolist() == [True] * 4
        assert set(CONSTANTS) == {"pi", "e", "inf", "nan"}

    def test_branch_shadows_constant_and_function_name(self, arrays: dict[str, ak.Array]) -> None:
        arrays = {**arrays, "e": ak.Array([1.0, 1.0, 1.0, 1.0]), "sum": ak.Array([7.0] * 4)}
        assert evaluate("e * 2", arrays).tolist() == [2.0] * 4
        # ``sum`` in call position is the function; as a bare name it is the branch.
        assert evaluate("sum(pt) + sum", arrays).tolist() == [47.0, 7.0, 57.0, 142.0]

    def test_scalar_result_broadcast_to_events(self, arrays: dict[str, ak.Array]) -> None:
        assert evaluate("1.5", arrays).tolist() == [1.5] * 4
        assert evaluate("2 * 3", arrays).tolist() == [6] * 4

    def test_scalar_result_without_arrays(self) -> None:
        with pytest.raises(ExpressionError, match="constant"):
            evaluate("1.5", {})

    def test_record_array_input(self, arrays: dict[str, ak.Array]) -> None:
        record = ak.Array({"x": arrays["x"], "pt": arrays["pt"]})
        assert evaluate("x + count(pt)", record).tolist() == [3.0, 2.0, 4.0, 7.0]

    def test_record_array_without_fields(self) -> None:
        with pytest.raises(ExpressionError, match="named fields"):
            evaluate("x", ak.Array([1, 2, 3]))

    def test_numpy_input(self) -> None:
        result = evaluate("a * 2", {"a": np.array([1, 2])})
        assert isinstance(result, ak.Array)
        assert result.tolist() == [2, 4]

    def test_missing_branch_at_evaluation(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(MissingBranchError, match="'zz'"):
            evaluate("zz + x", arrays)

    def test_runtime_error_is_wrapped(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(ExpressionError, match="failed to evaluate"):
            evaluate("x[:, 0]", arrays)  # too many indices on a flat array

    def test_builtins_are_not_reachable(self, arrays: dict[str, ak.Array]) -> None:
        with pytest.raises(ExpressionError):
            evaluate("__builtins__", arrays)

    def test_function_table_is_readonly(self) -> None:
        with pytest.raises(TypeError):
            FUNCTIONS["evil"] = print  # type: ignore[index]


class TestDottedNames:
    def test_parse_and_evaluate(self) -> None:
        expr = parse("pt(RP.momentum.x, RP.momentum.y) > 4 and RP.charge > 0")
        assert expr.names == ("RP.momentum.x", "RP.momentum.y", "RP.charge")
        arrays = {
            "RP.momentum.x": ak.Array([[3.0, 0.0]]),
            "RP.momentum.y": ak.Array([[4.0, 1.0]]),
            "RP.charge": ak.Array([[1, 1]]),
        }
        assert expr.evaluate(arrays).tolist() == [[True, False]]
        assert expr.required_branches(list(arrays)) == list(arrays)

    def test_missing_dotted_name(self) -> None:
        expr = parse("RP.energy * 2")
        with pytest.raises(MissingBranchError, match=r"RP\.energy"):
            expr.required_branches(["RP.momentum.x"])

    def test_disallowed_attribute_forms(self) -> None:
        with pytest.raises(ExpressionError):
            parse("a.b(1)")
        with pytest.raises(ExpressionError):
            parse("(a + b).c")
        with pytest.raises(ExpressionError):
            parse("f(1).x")

    def test_backticks_and_dots_are_the_same_name(self) -> None:
        arrays = {"RP.energy": ak.Array([[1.0, 2.0]])}
        assert parse("`RP.energy` + RP.energy").evaluate(arrays).tolist() == [[2.0, 4.0]]


class TestKinematics:
    def test_functions(self) -> None:
        arrays = {
            "px": ak.Array([[3.0, 0.0]]),
            "py": ak.Array([[4.0, 0.0]]),
            "pz": ak.Array([[0.0, 2.0]]),
            "e": ak.Array([[13.0, 2.0]]),
        }

        def ev(text: str) -> list[float]:
            return parse(text).evaluate(arrays).tolist()[0]

        assert ev("pt(px, py)") == [5.0, 0.0]
        assert ev("p(px, py, pz)") == [5.0, 2.0]
        assert ev("costheta(px, py, pz)") == [0.0, 1.0]
        assert ev("theta(px, py, pz)") == pytest.approx([np.pi / 2, 0.0])
        assert ev("phi(px, py)") == pytest.approx([np.arctan2(4, 3), 0.0])
        assert ev("eta(px, py, pz)")[0] == 0.0
        assert np.isinf(ev("eta(px, py, pz)")[1])
        assert ev("mass(e, px, py, pz)") == pytest.approx([12.0, 0.0])
        # never negative under the square root
        assert parse("mass(e, px, py, pz)").evaluate(
            {**arrays, "e": ak.Array([[1.0, 1.0]])}
        ).tolist() == [[0.0, 0.0]]


class TestReviewRegressions:
    """Scalar booleans, mixed quoting and constant lengths (review findings)."""

    def test_not_keeps_booleans_boolean(self) -> None:
        arrays = {"x": ak.Array([0.0, 1.0, 2.0])}
        assert evaluate("not True", arrays).tolist() == [False, False, False]
        assert evaluate("not (x > 0)", arrays).tolist() == [True, False, False]
        assert evaluate("not x", {"x": ak.Array([0, 3])}).tolist() == [True, False]
        assert evaluate("not (x > 0)", {"x": ak.Array([[1.0, -1.0], []])}).tolist() == [
            [False, True],
            [],
        ]

    def test_quoted_and_unquoted_same_name(self) -> None:
        assert evaluate("x + `x`", {"x": ak.Array([1.0, 2.0])}).tolist() == [2.0, 4.0]

    def test_constant_with_explicit_length(self) -> None:
        assert evaluate("1", {}, length=3).tolist() == [1, 1, 1]
        assert evaluate("True", {}, length=2).tolist() == [True, True]
        assert parse("2").evaluate({}, length=4).tolist() == [2, 2, 2, 2]
        with pytest.raises(ExpressionError, match="constant"):
            evaluate("1", {})


class TestQuoteName:
    @pytest.mark.parametrize(
        ("name", "expression"),
        [
            ("Muon_pt", "Muon_pt"),
            ("ReconstructedParticles.momentum.x", "ReconstructedParticles.momentum.x"),
            ("pi", "pi"),  # a branch of a constant's name is read as the branch
            ("count", "count"),  # functions live in their own namespace
            ("jet1_b-tag", "`jet1_b-tag`"),
            ("sel/mz", "`sel/mz`"),
            ("1abc", "`1abc`"),
            ("if", "`if`"),
            ("True", "`True`"),
            ("with space", "`with space`"),
            ("x ", "`x `"),
            ("__rootfig_x", "`__rootfig_x`"),  # reserved names are only written quoted
        ],
    )
    def test_addresses_exactly_the_name(self, name: str, expression: str) -> None:
        from rootfig.expressions import quote_name

        assert quote_name(name) == expression
        parsed = parse(expression)
        assert parsed.is_trivial
        assert parsed.names == (name,)

    @pytest.mark.parametrize("name", ["", "a`b", "`x`"])
    def test_unaddressable_names(self, name: str) -> None:
        from rootfig.expressions import quote_name

        assert quote_name(name) is None

    def test_quoted_name_evaluates_the_branch(self) -> None:
        from rootfig.expressions import quote_name

        arrays = {"jet1_b-tag": ak.Array([0.1, 0.9]), "pi": ak.Array([1.0, 2.0])}
        assert evaluate(quote_name("jet1_b-tag") or "", arrays).tolist() == [0.1, 0.9]
        assert evaluate(quote_name("pi") or "", arrays).tolist() == [1.0, 2.0]


def _delta_phi(phi1: Any, phi2: Any) -> Any:
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def _twice(values: Any) -> Any:
    return 2 * values


class TestCustomFunctions:
    """Functions given with functions=: bound when parsed, never looked up globally."""

    def test_flat_jagged_and_keywords(self, arrays: dict[str, ak.Array]) -> None:
        def scaled(values: Any, *, factor: float = 1.0) -> Any:
            return values * factor

        functions = {"twice": _twice, "scaled": scaled}
        assert evaluate("twice(x) + 1", arrays, functions=functions).tolist() == [3, 5, 7, 9]
        assert evaluate("twice(pt[pt > 20])", arrays, functions=functions).tolist() == [
            [60.0],
            [],
            [100.0],
            [120.0, 140.0],
        ]
        assert evaluate("scaled(x, factor=3)", arrays, functions=functions).tolist() == [
            3,
            6,
            9,
            12,
        ]

    def test_delta_phi_wraps(self) -> None:
        arrays = {"phi": ak.Array([[3.0, -3.0], [0.5, 0.25]])}
        result = evaluate("dphi(phi[:, 0], phi[:, 1])", arrays, functions={"dphi": _delta_phi})
        np.testing.assert_allclose(ak.to_numpy(result), [6.0 - 2 * np.pi, 0.25])

    def test_function_names_are_not_branches(self) -> None:
        expr = parse("dphi(phi1, phi2) > x", functions={"dphi": _delta_phi})
        assert expr.names == ("phi1", "phi2", "x")
        assert expr.functions == ("dphi",)
        assert expr.required_branches(["x", "phi2", "phi1", "y"]) == ["phi1", "phi2", "x"]

    def test_unknown_function(self) -> None:
        with pytest.raises(ExpressionError, match="Did you mean delta_phi") as info:
            parse("delta_ph(a, b)", functions={"delta_phi": _delta_phi})
        assert "given with functions=: delta_phi" in str(info.value)
        with pytest.raises(ExpressionError, match=r"functions=\{'name': callable\}"):
            parse("delta_phi(a, b)")

    @pytest.mark.parametrize(
        ("functions", "error", "match"),
        [
            ({"sqrt": np.sqrt}, ExpressionError, "built-in function"),
            ({"pi": np.sqrt}, ExpressionError, "built-in constant"),
            ({"delta-phi": _delta_phi}, ExpressionError, "use an identifier"),
            ({"lambda": _delta_phi}, ExpressionError, "use an identifier"),
            ({"__rootfig_fn_f": _delta_phi}, ExpressionError, "reserved"),
            ({"f": 3}, TypeError, "must be callable"),
            ({1: _delta_phi}, TypeError, "names must be strings"),
            ([("f", _delta_phi)], TypeError, "must map names"),
        ],
    )
    def test_refused(self, functions: Any, error: type[Exception], match: str) -> None:
        with pytest.raises(error, match=match):
            parse("x", functions=functions)
        with pytest.raises(error, match=match), function_scope(functions):
            pass

    def test_parsed_expression_keeps_its_functions(self, arrays: dict[str, ak.Array]) -> None:
        expr = parse("twice(x)", functions={"twice": _twice})
        assert active_functions() == {}
        assert expr.evaluate(arrays).tolist() == [2, 4, 6, 8]
        with function_scope({"twice": lambda values: 3 * values}):
            assert expr.evaluate(arrays).tolist() == [2, 4, 6, 8]
            assert evaluate(expr, arrays).tolist() == [2, 4, 6, 8]

    def test_functions_decide_equality(self, arrays: dict[str, ak.Array]) -> None:
        absolute = parse("f(x - 2)", functions={"f": np.abs})
        squared = parse("f(x - 2)", functions={"f": np.square})
        assert absolute.evaluate(arrays).tolist() == [1, 0, 1, 2]
        assert squared.evaluate(arrays).tolist() == [1, 0, 1, 4]
        assert absolute != squared
        again = parse("f(x - 2)", functions={"f": np.abs})
        assert absolute == again
        assert hash(absolute) == hash(again)
        assert parse("abs(x)") == parse("abs(x)")

    def test_parsed_expression_refuses_functions(self, arrays: dict[str, ak.Array]) -> None:
        expr = parse("f(x)", functions={"f": np.abs})
        assert parse(expr) is expr
        with pytest.raises(ExpressionError, match="already parsed"):
            parse(expr, functions={"f": np.square})
        with pytest.raises(ExpressionError, match="already parsed"):
            evaluate(expr, arrays, functions={"f": np.abs})

    def test_scope(self, arrays: dict[str, ak.Array]) -> None:
        with function_scope({"twice": _twice}):
            assert parse("twice(x)").evaluate(arrays).tolist() == [2, 4, 6, 8]
            with function_scope(None), pytest.raises(ExpressionError, match="unknown function"):
                parse("twice(x)")
            assert set(active_functions()) == {"twice"}
        assert active_functions() == {}
        with pytest.raises(RuntimeError), function_scope({"twice": _twice}):
            raise RuntimeError
        assert active_functions() == {}

    def test_evaluate_sees_only_its_own_functions(self, arrays: dict[str, ak.Array]) -> None:
        def inner(values: Any) -> Any:
            return values + 1

        def outer(values: Any) -> Any:
            return evaluate("inner(v)", {"v": values})

        def outer_given(values: Any) -> Any:
            return evaluate("inner(v)", {"v": values}, functions={"inner": inner})

        with pytest.raises(ExpressionError, match="unknown function 'inner'"):
            evaluate("outer(x)", arrays, functions={"outer": outer, "inner": inner})
        result = evaluate("outer(x)", arrays, functions={"outer": outer_given})
        assert result.tolist() == [2, 3, 4, 5]

    def test_validate_checks_syntax_only(self) -> None:
        validate("delta_phi(a, b) > 0.4")
        for text, match in [
            ("a +", "invalid syntax"),
            ("f(**x)", "unpacking"),
            ("f('a')", "str literal"),
            ("__rootfig_x > 0", "reserved"),
        ]:
            with pytest.raises(ExpressionError, match=match):
                validate(text)

    def test_errors_of_a_function(self, arrays: dict[str, ak.Array]) -> None:
        def broken(values: Any) -> Any:
            msg = "boom"
            raise ValueError(msg)

        with pytest.raises(ExpressionError, match=r"failed to evaluate.*ValueError: boom") as info:
            evaluate("broken(x)", arrays, functions={"broken": broken})
        assert isinstance(info.value.__cause__, ValueError)
        with pytest.raises(ExpressionError, match="unsupported result of type list"):
            evaluate("listed(x)", arrays, functions={"listed": lambda values: values.tolist()})


class TestReservedNames:
    """Names beginning with __rootfig_ are rootfig's own: quoted, they are branches."""

    @pytest.mark.parametrize(
        "text", ["__rootfig_fn_f + f(x)", "__rootfig_not & ~x", "a.__rootfig_x", "__rootfig_bt_0"]
    )
    def test_bare_reserved_name(self, text: str) -> None:
        with pytest.raises(ExpressionError, match="reserved for rootfig"):
            parse(text, functions={"f": np.abs})

    def test_quoted_reserved_names_are_branches(self) -> None:
        arrays = {
            "__rootfig_fn_f": ak.Array([10.0, 20.0]),
            "__rootfig_not": ak.Array([True, True]),
            "__rootfig_bt_0": ak.Array([100.0, 200.0]),
            "a-b": ak.Array([1.0, 2.0]),
            "x": ak.Array([-1.0, 2.0]),
        }
        expr = parse("`__rootfig_fn_f` + f(x)", functions={"f": np.abs})
        assert expr.names == ("__rootfig_fn_f", "x")
        assert expr.evaluate(arrays).tolist() == [11.0, 22.0]
        assert evaluate("`__rootfig_not` & (not (x > 0))", arrays).tolist() == [True, False]
        assert evaluate("`a-b` + `__rootfig_bt_0`", arrays).tolist() == [101.0, 202.0]

    def test_names_containing_the_prefix_are_not_reserved(self) -> None:
        arrays = {"Jet__rootfig_pt": ak.Array([1.0, -2.0]), "x___rootfig_y": ak.Array([3.0, 4.0])}
        functions = {"my__rootfig_abs": np.abs}
        expr = parse("my__rootfig_abs(Jet__rootfig_pt) + x___rootfig_y", functions=functions)
        assert expr.names == ("Jet__rootfig_pt", "x___rootfig_y")
        assert expr.functions == ("my__rootfig_abs",)
        assert expr.evaluate(arrays).tolist() == [4.0, 6.0]
