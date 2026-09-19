"""PlotBook: configuration, task expansion, lazy delegation to plot(), saving and filtering."""

from __future__ import annotations

import dataclasses
import itertools
import re
from pathlib import Path
from typing import Any

import hist
import matplotlib.pyplot as plt
import numpy as np
import pytest

import rootfig as rf
from rootfig._mapping import FrozenMapping
from rootfig.api import batch

X = rf.Variable("x", bins=(5, 0, 1))
Y = rf.Variable("y", bins=(4, 0, 4))


@pytest.fixture
def samples() -> list[rf.Sample]:
    a = rf.Sample({"x": [0.2, 0.3, 0.5, 0.7], "y": [1.0, 2.0, 3.0, 4.0]}, label="A")
    b = rf.Sample({"x": [0.4, 0.6], "y": [1.5, 2.5]}, label="B")
    return [a, b]


class TestConstruction:
    def test_one_bare_variable(self, samples: list[rf.Sample]) -> None:
        assert rf.PlotBook(samples, "x").variables == (rf.Variable("x"),)
        assert rf.PlotBook(samples, X).variables == (X,)

    def test_variables_are_coerced_in_order(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(samples, [X, "y"])
        assert [v.safe_name for v in book.variables] == ["x", "y"]
        assert book.variables[0] is X
        assert isinstance(book.variables[1], rf.Variable)

    def test_implicit_axes(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(samples, [X, Y])
        assert book.selections is None
        assert book.variants is None
        assert book.plot_kwargs == {}
        tasks = book.tasks()
        assert len(tasks) == 2
        (task, _) = tasks
        assert (task.selection_name, task.selection, task.variant_name) == (None, None, None)
        assert (task.selection_id, task.variant_id) == ("all", "default")
        assert task.kwargs == {}

    def test_named_axes(self, samples: list[rf.Sample]) -> None:
        cut = rf.Cut("y > 1", label="high")
        book = rf.PlotBook(
            samples,
            [X],
            selections={"none": None, "text": "y > 2", "cut": cut},
            variants={"lin": {}, "log": {"logy": True}},
            plot_kwargs={"stack": True},
        )
        assert isinstance(book.selections, FrozenMapping)
        assert list(book.selections) == ["none", "text", "cut"]
        assert book.selections["none"] is None
        assert book.selections["text"] == rf.Cut("y > 2")
        assert book.selections["cut"] is cut
        assert isinstance(book.variants, FrozenMapping)
        assert isinstance(book.variants["log"], FrozenMapping)
        assert book.variants["log"] == {"logy": True}
        assert isinstance(book.plot_kwargs, FrozenMapping)

    def test_empty_variables_rejected(self, samples: list[rf.Sample]) -> None:
        with pytest.raises(ValueError, match="at least one variable"):
            rf.PlotBook(samples, [])

    def test_empty_explicit_axes_rejected(self, samples: list[rf.Sample]) -> None:
        with pytest.raises(ValueError, match="selections= must name at least one"):
            rf.PlotBook(samples, [X], selections={})
        with pytest.raises(ValueError, match="variants= must name at least one"):
            rf.PlotBook(samples, [X], variants={})

    @pytest.mark.parametrize(
        "name", ["", ".", "..", "a/b", "a\\b", "a\0b", "a:b", "lin.", "NUL", "com1.x"]
    )
    def test_unsafe_names_rejected(self, samples: list[rf.Sample], name: str) -> None:
        # The rules are check_file_stem's, tested in test_model; here that both axes use them.
        with pytest.raises(ValueError, match=r"selection name .* cannot be a file name component"):
            rf.PlotBook(samples, [X], selections={name: None})
        with pytest.raises(ValueError, match=r"variant name .* cannot be a file name component"):
            rf.PlotBook(samples, [X], variants={name: {}})

    def test_non_string_names_rejected(self, samples: list[rf.Sample]) -> None:
        with pytest.raises(TypeError, match="selection names must be non-empty strings"):
            rf.PlotBook(samples, [X], selections={1: None})  # type: ignore[dict-item]
        with pytest.raises(TypeError, match="variant names must be non-empty strings"):
            rf.PlotBook(samples, [X], variants={None: {}})  # type: ignore[dict-item]

    def test_duplicate_variable_identifier_rejected(self, samples: list[rf.Sample]) -> None:
        with pytest.raises(ValueError, match="share the identifier 'pt'"):
            rf.PlotBook(
                samples,
                [rf.Variable("Muon_pt", name="pt"), rf.Variable("Electron_pt", name="pt")],
            )
        with pytest.raises(ValueError, match="share the identifier 'x_1'"):
            rf.PlotBook(samples, ["x+1", "x-1"])

    def test_variable_identifier_must_be_a_file_component(self) -> None:
        # An explicit name= is checked by Variable itself; a sanitised expression only here.
        with pytest.raises(ValueError, match="variable name 'nul' cannot be a file name component"):
            rf.PlotBook(None, "nul")

    def test_stem_collision_rejected(self, samples: list[rf.Sample]) -> None:
        # variable a__b with selection c and variable a with selection b__c both spell a__b__c
        with pytest.raises(ValueError, match=re.escape("output names collide: 'a__b__c' (")):
            rf.PlotBook(
                samples,
                [rf.Variable("y", name="a__b"), rf.Variable("x", name="a")],
                selections={"c": None, "b__c": None},
            )

    def test_stems_that_are_one_file_on_case_insensitive_systems_collide(
        self, samples: list[rf.Sample]
    ) -> None:
        # lin and LIN are one file on macOS and Windows; the message lists both spellings.
        with pytest.raises(ValueError, match=re.escape("collide: 'x__lin' and 'x__LIN' (")):
            rf.PlotBook(samples, [X], variants={"lin": {}, "LIN": {"logy": True}})
        with pytest.raises(ValueError, match="'Mass' and 'mass'"):
            rf.PlotBook(samples, [rf.Variable("y", name="Mass"), rf.Variable("x", name="mass")])
        # Composed and decomposed spellings of an accented letter, and a ligature.
        with pytest.raises(ValueError, match="collide"):
            rf.PlotBook(samples, [X], selections={"\u00e9": None, "e\u0301": None})
        with pytest.raises(ValueError, match="collide"):
            rf.PlotBook(samples, [X], selections={"\ufb01t": None, "fit": None})
        # Case only matters within one book's names, not for what plot() receives.
        assert [t.stem for t in rf.PlotBook(samples, [X], variants={"LIN": {}}).tasks()] == [
            "x__LIN"
        ]

    @pytest.mark.parametrize("keyword", sorted(batch._RESERVED_PLOT_KWARGS))
    def test_reserved_keywords_rejected(self, samples: list[rf.Sample], keyword: str) -> None:
        with pytest.raises(ValueError, match=f"plot_kwargs must not set '{keyword}'"):
            rf.PlotBook(samples, [X], plot_kwargs={keyword: None})
        with pytest.raises(ValueError, match=f"variant 'v' must not set '{keyword}'"):
            rf.PlotBook(samples, [X], variants={"v": {keyword: None}})

    def test_unknown_keywords_rejected_when_built(self, samples: list[rf.Sample]) -> None:
        # plot() would raise the same TypeError, but only once the batch is running.
        with pytest.raises(TypeError, match=r"plot_kwargs names keywords plot\(\) does not have"):
            rf.PlotBook(samples, [X], plot_kwargs={"log_y": True})
        with pytest.raises(TypeError, match=r"'log_y' \(did you mean 'logy'\?\)"):
            rf.PlotBook(samples, [X], plot_kwargs={"log_y": True})
        with pytest.raises(TypeError, match=r"variant 'v' names keywords .*: 'nothing_like_it'$"):
            rf.PlotBook(samples, [X], variants={"v": {"nothing_like_it": 1}})
        with pytest.raises(TypeError, match=r"'log_y' \(did you mean 'logy'\?\), 'zz'$"):
            rf.PlotBook(samples, [X], plot_kwargs={"log_y": True, "zz": 1})
        with pytest.raises(TypeError, match=r"plot_kwargs keys must be .* names, got 1"):
            rf.PlotBook(samples, [X], plot_kwargs={1: True})  # type: ignore[dict-item]
        with pytest.raises(TypeError, match=r"variant 'v' keys must be .*, got None"):
            rf.PlotBook(samples, [X], variants={"v": {None: True}})  # type: ignore[dict-item]

    def test_every_plot_keyword_is_accepted(self, samples: list[rf.Sample]) -> None:
        accepted = set(batch._PLOT_KEYWORDS)
        assert {"logy", "stack", "observed", "systematics", "style", "figsize"} <= accepted
        assert accepted.isdisjoint(batch._RESERVED_PLOT_KWARGS)
        book = rf.PlotBook(samples, [X], plot_kwargs=dict.fromkeys(accepted))
        assert set(book.plot_kwargs) == accepted

    def test_ax_rejected_with_explanation(self, samples: list[rf.Sample]) -> None:
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="every task draws its own figure"):
            rf.PlotBook(samples, [X], plot_kwargs={"ax": ax})

    def test_variant_must_be_a_mapping(self, samples: list[rf.Sample]) -> None:
        with pytest.raises(TypeError, match="variant 'log' must map plot\\(\\) keywords"):
            rf.PlotBook(samples, [X], variants={"log": True})  # type: ignore[dict-item]
        # None means "no keywords" for plot_kwargs=, but a variant without overrides is {}.
        with pytest.raises(TypeError, match=r"variant 'lin' must map .*, got NoneType"):
            rf.PlotBook(samples, [X], variants={"lin": None})  # type: ignore[dict-item]
        assert rf.PlotBook(samples, [X], plot_kwargs=None).plot_kwargs == {}

    @pytest.mark.parametrize("axis", ["selections", "variants", "plot_kwargs"])
    def test_configuration_must_be_a_mapping(self, samples: list[rf.Sample], axis: str) -> None:
        # A sequence of pairs would otherwise slip a reserved keyword past the check.
        with pytest.raises(TypeError, match="must map"):
            rf.PlotBook(samples, [X], **{axis: [("save", "out.pdf")]})

    def test_data_is_kept_as_given(self, samples: list[rf.Sample]) -> None:
        assert rf.PlotBook(samples, [X]).data is samples
        opaque = object()  # tasks never look at the data
        assert len(rf.PlotBook(opaque, [X, Y], selections={"a": None}).tasks()) == 2

    def test_repr(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(samples, [X, Y], variants={"lin": {}, "log": {"logy": True}})
        assert repr(book) == (
            "PlotBook(variables=['x', 'y'], selections=None, variants=['lin', 'log'], tasks=4)"
        )


class TestFreezing:
    def test_caller_mutations_do_not_reach_the_book(self, samples: list[rf.Sample]) -> None:
        variables = [X]
        selections: dict[str, str | None] = {"sr": "y > 1"}
        log: dict[str, Any] = {"logy": True}
        variants = {"log": log}
        plot_kwargs: dict[str, Any] = {"stack": True}
        book = rf.PlotBook(
            samples, variables, selections=selections, variants=variants, plot_kwargs=plot_kwargs
        )
        variables.append(Y)
        selections["extra"] = None
        variants["lin"] = {}
        log["logy"] = False
        plot_kwargs["stack"] = False
        assert book.variables == (X,)
        assert list(book.selections or {}) == ["sr"]
        assert list(book.variants or {}) == ["log"]
        assert (book.variants or {})["log"] == {"logy": True}
        assert book.plot_kwargs == {"stack": True}
        assert book.tasks()[0].kwargs == {"stack": True, "logy": True}

    def test_book_and_tasks_are_read_only(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(
            samples, [X], variants={"log": {"logy": True}}, plot_kwargs={"stack": True}
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            book.variables = ()  # type: ignore[misc]
        with pytest.raises(TypeError):
            book.plot_kwargs["stack"] = False  # type: ignore[index]
        with pytest.raises(TypeError):
            (book.variants or {})["log"]["logy"] = False  # type: ignore[index]
        task = book.tasks()[0]
        assert isinstance(task.kwargs, FrozenMapping)
        with pytest.raises(dataclasses.FrozenInstanceError):
            task.variant_name = "x"  # type: ignore[misc]


class TestTasks:
    def test_cartesian_product_in_documented_order(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(
            samples,
            [X, Y],
            selections={"a": None, "b": "y > 1", "c": "y > 2"},
            variants={"lin": {}, "log": {"logy": True}},
        )
        tasks = book.tasks()
        assert len(tasks) == 12
        assert [(t.variable.safe_name, t.selection_name, t.variant_name) for t in tasks] == list(
            itertools.product(["x", "y"], ["a", "b", "c"], ["lin", "log"])
        )
        assert tasks[3].selection == rf.Cut("y > 1")
        assert book.tasks() == tasks

    def test_variant_overrides_common_kwargs(self, samples: list[rf.Sample]) -> None:
        plot_kwargs = {"stack": True, "logy": False}
        variants = {"lin": {}, "log": {"logy": True}}
        book = rf.PlotBook(samples, [X], variants=variants, plot_kwargs=plot_kwargs)
        lin, log = book.tasks()
        assert dict(lin.kwargs) == {"stack": True, "logy": False}
        assert dict(log.kwargs) == {"stack": True, "logy": True}
        assert plot_kwargs == {"stack": True, "logy": False}
        assert variants == {"lin": {}, "log": {"logy": True}}

    @pytest.mark.parametrize(
        ("options", "stem"),
        [
            ({}, "x"),
            ({"selections": {"all": None}}, "x__all"),
            ({"variants": {"default": {"logy": True}}}, "x__default"),
            ({"selections": {"sr": "y > 1"}, "variants": {"log": {"logy": True}}}, "x__sr__log"),
        ],
    )
    def test_stem_lists_explicit_axes_only(
        self, samples: list[rf.Sample], options: dict[str, Any], stem: str
    ) -> None:
        (task,) = rf.PlotBook(samples, [X], **options).tasks()
        assert task.stem == stem

    def test_describe_uses_display_aliases(self, samples: list[rf.Sample]) -> None:
        (implicit,) = rf.PlotBook(samples, [X]).tasks()
        assert implicit.describe() == "variable='x', selection='all', variant='default'"
        (explicit,) = rf.PlotBook(
            samples, [X], selections={"sr": None}, variants={"log": {}}
        ).tasks()
        assert explicit.describe() == "variable='x', selection='sr', variant='log'"

    def test_tasks_hash_by_their_stem(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(samples, [X, Y], selections={"a": None, "b": "y > 1"})
        tasks = book.tasks()
        assert len(set(tasks)) == len(tasks)
        assert {task: task.stem for task in tasks}[tasks[0]] == "x__a"

    def test_tasks_compare_by_stem_whatever_the_keywords_hold(
        self, samples: list[rf.Sample]
    ) -> None:
        # Comparing kwargs would evaluate np.array == np.array and raise on its truth value.
        edges = np.array([0.0, 0.5, 1.0])
        book = rf.PlotBook(samples, ["x", "y"], plot_kwargs={"bins": edges})
        first, other = book.tasks()
        again = book.tasks()[0]
        assert first == again
        assert first != other
        assert first != "x"
        assert {first, again, other} == {first, other}
        assert first.kwargs["bins"] is edges


class FakePlot:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestPlots:
    def test_tasks_are_created_one_at_a_time(
        self, samples: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        book = rf.PlotBook(samples, [X, Y], variants={"lin": {}, "log": {"logy": True}})
        created: list[rf.PlotTask] = []

        def task(*args: Any, **kwargs: Any) -> rf.PlotTask:
            result = rf.PlotTask(*args, **kwargs)
            created.append(result)
            return result

        monkeypatch.setattr(batch, "PlotTask", task)
        monkeypatch.setattr(batch, "plot", lambda *a, **k: FakePlot())
        iterator = book.plots()
        assert created == []
        next(iterator)
        assert len(created) == 1
        next(iterator)
        assert len(created) == 2

    def test_lazy_one_task_per_step(
        self, samples: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[Any, ...]] = []

        def fake(*args: Any, **kwargs: Any) -> FakePlot:
            calls.append((args, kwargs))
            return FakePlot()

        monkeypatch.setattr(batch, "plot", fake)
        book = rf.PlotBook(samples, [X, Y], variants={"lin": {}, "log": {"logy": True}})
        iterator = book.plots()
        assert calls == []
        task, result = next(iterator)
        assert len(calls) == 1
        assert isinstance(result, FakePlot)
        assert task == book.tasks()[0]
        next(iterator)
        assert len(calls) == 2
        assert len(list(iterator)) == 2
        assert len(calls) == 4

    def test_delegates_data_variable_selection_and_kwargs(
        self, samples: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(batch, "plot", lambda *a, **k: calls.append((a, k)) or FakePlot())
        cut = rf.Cut("y > 1")
        book = rf.PlotBook(
            samples,
            [X],
            selections={"sr": cut},
            variants={"log": {"logy": True}},
            plot_kwargs={"stack": True, "logy": False},
        )
        list(book.plots())
        ((args, kwargs),) = calls
        assert args[0] is samples
        assert args[1] is X
        assert kwargs == {"selection": cut, "stack": True, "logy": True}

    def test_results_reflect_the_task(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(
            samples,
            [X],
            selections={"all": None, "hi": "y > 1"},
            variants={"lin": {}, "log": {"logy": True}},
            plot_kwargs={"stack": True},
        )
        entries = {"all": 6.0, "hi": 5.0}
        for task, result in book.plots():
            assert result.variable == X
            assert result.ax.get_yscale() == ("log" if task.variant_name == "log" else "linear")
            assert [h.label for h in result.histograms] == ["A", "B"]
            total = sum(h.sum().value for h in result.hists)
            assert total == pytest.approx(entries[task.selection_id])
            result.close()

    def test_error_keeps_type_and_names_the_task(self, samples: list[rf.Sample]) -> None:
        book = rf.PlotBook(samples, ["x", "nosuch"], selections={"sr": "y > 1"})
        with pytest.raises(rf.MissingBranchError) as info:
            list(book.plots())
        assert info.value.__notes__ == [
            "while running PlotBook task variable='nosuch', selection='sr', variant='default'"
        ]

    def test_non_rootfig_errors_are_annotated_too(
        self, samples: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(*_: Any, **__: Any) -> FakePlot:
            msg = "boom"
            raise RuntimeError(msg)

        monkeypatch.setattr(batch, "plot", failing)
        with pytest.raises(RuntimeError, match="boom") as info:
            next(rf.PlotBook(samples, [X]).plots())
        assert info.value.__notes__ == [
            "while running PlotBook task variable='x', selection='all', variant='default'"
        ]

    def test_drawing_failure_closes_only_the_failed_task(self, samples: list[rf.Sample]) -> None:
        existing = plt.figure()
        book = rf.PlotBook(samples, X, variants={"good": {}, "bad": {"ratio": "missing"}})
        iterator = book.plots()
        _, completed = next(iterator)
        with pytest.raises(ValueError, match="ratio reference 'missing'") as info:
            next(iterator)
        assert plt.get_fignums() == [existing.number, completed.fig.number]
        assert info.value.__notes__ == [
            "while running PlotBook task variable='x', selection='all', variant='bad'"
        ]

    def test_interrupted_drawing_closes_the_new_figure(
        self, samples: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = plt.figure()

        def interrupted(*args: Any, **kwargs: Any) -> rf.Plot:
            plt.figure()
            raise KeyboardInterrupt

        monkeypatch.setattr(batch, "plot", interrupted)
        with pytest.raises(KeyboardInterrupt):
            next(rf.PlotBook(samples, X).plots())
        assert plt.get_fignums() == [existing.number]


class TestPassthrough:
    def test_group_is_one_histogram(self, samples: list[rf.Sample]) -> None:
        signal = rf.Sample({"x": [0.8, 0.9], "y": [0.5, 0.7]}, label="Signal")
        book = rf.PlotBook(
            [rf.Group(samples, label="AB"), signal],
            [X],
            variants={"lin": {}, "log": {"logy": True}},
            plot_kwargs={"stack": True},
        )
        results = list(book.plots())
        assert len(results) == 2
        for _, result in results:
            assert [h.label for h in result.histograms] == ["AB", "Signal"]
            result.close()

    def test_nested_group(self, samples: list[rf.Sample]) -> None:
        inner = rf.Group(samples, label="AB")
        outer = rf.Group([inner, rf.Sample({"x": [0.1], "y": [0.1]}, label="C")], label="ABC")
        ((_, result),) = list(rf.PlotBook(outer, [X]).plots())
        assert [h.label for h in result.histograms] == ["ABC"]
        assert sum(h.sum().value for h in result.hists) == 7.0

    def test_observed_group_matches_direct_plot(self, samples: list[rf.Sample]) -> None:
        mc = rf.Group(samples, label="MC")
        observed = rf.Group(
            [
                rf.Sample({"x": [0.25, 0.35, 0.55]}, label="D1", is_data=True),
                rf.Sample({"x": [0.45, 0.65, 0.75]}, label="D2", is_data=True),
            ],
            label="Data",
        )
        kwargs = {"observed": observed, "stack": True, "ratio": True}
        ((_, from_book),) = list(rf.PlotBook(mc, [X], plot_kwargs=kwargs).plots())
        direct = rf.plot(mc, X, **kwargs)
        assert [h.label for h in from_book.histograms] == ["MC", "Data"]
        assert from_book.histograms[1].is_data
        assert from_book.ratio_ax is not None
        np.testing.assert_allclose(from_book.ratios[0].values, direct.ratios[0].values)

    def test_histogram_objects(self) -> None:
        h = hist.Hist(hist.axis.Regular(4, 0, 1, name="x"), storage=hist.storage.Weight())
        h.fill([0.1, 0.5, 0.6])
        book = rf.PlotBook(h, [rf.Variable("x")], plot_kwargs={"label": "MC"})
        task, result = next(iter(book.plots()))
        assert task.stem == "x"
        assert [hh.label for hh in result.histograms] == ["MC"]
        assert result.hists[0].sum().value == 3.0
        wrapped = rf.Histogram(h, label="Ready")
        ((_, again),) = list(rf.PlotBook([wrapped], "x").plots())
        assert [hh.label for hh in again.histograms] == ["Ready"]

    def test_stored_histogram_group(self, stored_dir: Path) -> None:
        vv = rf.Group(
            [
                rf.Sample(stored_dir / "WW_sel0_histo.root", label="WW"),
                rf.Sample(stored_dir / "ZZ_sel0_histo.root", label="ZZ"),
            ],
            label="VV",
        )
        ((task, result),) = list(rf.PlotBook(vv, ["mz"]).plots())
        assert task.stem == "mz"
        assert [h.label for h in result.histograms] == ["VV"]
        assert result.hists[0].sum().value == pytest.approx(rf.plot(vv, "mz").hists[0].sum().value)


class TestSave:
    @pytest.mark.parametrize("formats", ["png", [".PNG", "png"]])
    def test_single_format_and_normalized_duplicates(
        self, samples: list[rf.Sample], tmp_path: Path, formats: str | list[str]
    ) -> None:
        written = rf.PlotBook(samples, X).save(tmp_path, formats=formats)
        assert written == [tmp_path / "x.png"]
        assert written[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    @pytest.mark.parametrize("formats", [[], "", ["png", "invalid"], ["../png"], ["\0"]])
    def test_invalid_formats_fail_before_drawing_or_creating_directory(
        self, tmp_path: Path, formats: Any
    ) -> None:
        directory = tmp_path / "plots"
        with pytest.raises(ValueError, match="format"):
            rf.PlotBook(object(), X).save(directory, formats=formats)
        assert not directory.exists()
        assert plt.get_fignums() == []

    def test_singular_format_cannot_override_file_extensions(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="formats="):
            rf.PlotBook(object(), X).save(tmp_path / "plots", format="png")
        assert not (tmp_path / "plots").exists()

    def test_dotted_task_names_are_preserved(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        book = rf.PlotBook(
            samples,
            X.replace(name="x.v1"),
            selections={"sr.1": None},
            variants={"lin.2": {}},
        )
        assert book.save(tmp_path) == [tmp_path / "x.v1__sr.1__lin.2.pdf"]

    def test_drawing_failure_closes_figure(self, samples: list[rf.Sample], tmp_path: Path) -> None:
        existing = plt.figure()
        book = rf.PlotBook(samples, X, plot_kwargs={"ratio": "missing"})
        with pytest.raises(ValueError, match="ratio reference 'missing'"):
            book.save(tmp_path)
        assert plt.get_fignums() == [existing.number]

    def test_writes_deterministic_files_and_closes_figures(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        book = rf.PlotBook(
            samples,
            [X, Y],
            selections={"all": None, "hi": "y > 1"},
            variants={"lin": {}, "log": {"logy": True}},
        )
        directory = tmp_path / "plots" / "nested"
        written = book.save(directory, formats=["pdf", "png"], dpi=50)
        assert directory.is_dir()
        assert [p.name for p in written] == [
            f"{t.stem}.{fmt}" for t in book.tasks() for fmt in ("pdf", "png")
        ]
        assert [p.name for p in written][:4] == [
            "x__all__lin.pdf",
            "x__all__lin.png",
            "x__all__log.pdf",
            "x__all__log.png",
        ]
        assert all(p.exists() and p.parent == directory for p in written)
        assert plt.get_fignums() == []

    def test_default_format_is_pdf(self, samples: list[rf.Sample], tmp_path: Path) -> None:
        written = rf.PlotBook(samples, "x").save(tmp_path)
        assert written == [tmp_path / "x.pdf"]
        assert written[0].exists()

    def test_existing_file_is_not_a_directory(
        self, samples: list[rf.Sample], tmp_path: Path
    ) -> None:
        target = tmp_path / "file.pdf"
        target.write_text("")
        with pytest.raises(NotADirectoryError, match="is not a directory"):
            rf.PlotBook(samples, "x").save(target)

    def test_failing_save_still_closes_the_figure(
        self, samples: list[rf.Sample], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        numbers: list[int] = []

        def failing_save(self: rf.Plot, *_: Any, **__: Any) -> list[Path]:
            numbers.append(self.fig.number)
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(rf.Plot, "save", failing_save)
        with pytest.raises(OSError, match="disk full") as info:
            rf.PlotBook(samples, [X, Y]).save(tmp_path)
        assert len(numbers) == 1  # fail-fast: the second task never ran
        assert not plt.fignum_exists(numbers[0])
        assert info.value.__notes__ == [
            f"while saving PlotBook task variable='x', selection='all', variant='default' "
            f"to {str(tmp_path)!r}"
        ]


class TestSelect:
    @pytest.mark.parametrize("axis", ["variables", "selections", "variants"])
    @pytest.mark.parametrize("explicit", [False, True])
    def test_empty_filter_is_rejected_on_every_axis(
        self, samples: list[rf.Sample], axis: str, explicit: bool
    ) -> None:
        book = rf.PlotBook(
            samples,
            [X, Y],
            selections={"all": None} if explicit else None,
            variants={"default": {}} if explicit else None,
        )
        with pytest.raises(ValueError, match="at least one"):
            book.select(**{axis: []})

    @pytest.fixture
    def book(self, samples: list[rf.Sample]) -> rf.PlotBook:
        return rf.PlotBook(
            samples,
            [X, Y],
            selections={"all": None, "hi": "y > 1"},
            variants={"lin": {}, "log": {"logy": True}},
            plot_kwargs={"stack": True},
        )

    def test_each_axis_and_combined(self, book: rf.PlotBook) -> None:
        assert [t.stem for t in book.select(variables=["y"]).tasks()] == [
            "y__all__lin",
            "y__all__log",
            "y__hi__lin",
            "y__hi__log",
        ]
        assert [t.stem for t in book.select(selections="hi").tasks()] == [
            "x__hi__lin",
            "x__hi__log",
            "y__hi__lin",
            "y__hi__log",
        ]
        assert [t.stem for t in book.select(variants=["log"]).tasks()] == [
            "x__all__log",
            "x__hi__log",
            "y__all__log",
            "y__hi__log",
        ]
        (only,) = book.select(variables="x", selections=["hi"], variants=["log"]).tasks()
        assert only.stem == "x__hi__log"
        assert only.kwargs == {"stack": True, "logy": True}

    def test_keeps_the_book_order_not_the_filter_order(self, book: rf.PlotBook) -> None:
        subset = book.select(variables=["y", "x"], selections=["hi", "all"])
        assert [v.safe_name for v in subset.variables] == ["x", "y"]
        assert list(subset.selections or {}) == ["all", "hi"]

    def test_returns_a_new_book_sharing_data_and_kwargs(self, book: rf.PlotBook) -> None:
        subset = book.select(variables=["x"])
        assert subset is not book
        assert subset.data is book.data
        assert subset.plot_kwargs == book.plot_kwargs
        assert len(book.tasks()) == 8
        assert len(subset.tasks()) == 4

    def test_unknown_names_list_the_choices(self, book: rf.PlotBook) -> None:
        with pytest.raises(
            ValueError, match=re.escape("unknown variable 'z'; available: ['x', 'y']")
        ):
            book.select(variables=["x", "z"])
        with pytest.raises(
            ValueError, match=re.escape("unknown selection 'sr'; available: ['all', 'hi']")
        ):
            book.select(selections="sr")
        with pytest.raises(
            ValueError, match=re.escape("unknown variant 'nostack'; available: ['lin', 'log']")
        ):
            book.select(variants=["nostack"])

    def test_implicit_axes_accept_their_alias_and_stay_implicit(
        self, samples: list[rf.Sample]
    ) -> None:
        book = rf.PlotBook(samples, [X, Y])
        subset = book.select(selections=["all"], variants="default")
        assert subset.selections is None
        assert subset.variants is None
        assert subset.tasks() == book.tasks()
        with pytest.raises(
            ValueError, match=re.escape("unknown selection 'sr'; available: ['all']")
        ):
            book.select(selections=["sr"])
        with pytest.raises(
            ValueError, match=re.escape("unknown variant 'log'; available: ['default']")
        ):
            book.select(variants=["log"])
