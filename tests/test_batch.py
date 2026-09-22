"""PlotBook: configuration, task expansion, lazy delegation to plot(), saving and filtering."""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import os
import re
from pathlib import Path
from typing import Any

import hist
import matplotlib.pyplot as plt
import numpy as np
import pytest
import uproot

import rootfig as rf
from rootfig._mapping import FrozenMapping
from rootfig.api import batch
from rootfig.io import FileSource, ReadCache

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

    def test_variable_names_are_file_safe_without_a_check_here(self) -> None:
        # Variable validates an explicit name= and generates a safe one otherwise.
        with pytest.raises(ValueError, match="Variable name"):
            rf.PlotBook(None, rf.Variable("x", name="CON"))
        assert [t.stem for t in rf.PlotBook(None, ["x", "nul"]).tasks()] == ["x", "nul_"]

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


FAKE_PREPARED = batch.PreparedPlot([], None)


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
        monkeypatch.setattr(batch, "prepare_plot", lambda *a, **k: FAKE_PREPARED)
        monkeypatch.setattr(batch, "draw_plot", lambda *a, **k: FakePlot())
        iterator = book.plots()
        assert created == []
        next(iterator)
        assert len(created) == 1
        next(iterator)
        assert len(created) == 2

    def test_lazy_one_task_per_step(
        self, samples: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prepared: list[tuple[Any, ...]] = []
        drawn: list[tuple[Any, ...]] = []

        def fake_prepare(*args: Any, **kwargs: Any) -> batch.PreparedPlot:
            prepared.append((args, kwargs))
            return FAKE_PREPARED

        def fake_draw(*args: Any, **kwargs: Any) -> FakePlot:
            drawn.append((args, kwargs))
            return FakePlot()

        monkeypatch.setattr(batch, "prepare_plot", fake_prepare)
        monkeypatch.setattr(batch, "draw_plot", fake_draw)
        book = rf.PlotBook(samples, [X, Y], variants={"lin": {}, "log": {"logy": True}})
        iterator = book.plots()
        assert (prepared, drawn) == ([], [])
        task, result = next(iterator)
        assert (len(prepared), len(drawn)) == (1, 1)
        assert isinstance(result, FakePlot)
        assert task == book.tasks()[0]
        next(iterator)  # the log variant draws from the same preparation
        assert (len(prepared), len(drawn)) == (1, 2)
        assert len(list(iterator)) == 2
        assert (len(prepared), len(drawn)) == (2, 4)

    def test_delegates_data_variable_selection_and_kwargs(
        self, samples: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prepared: list[tuple[Any, ...]] = []
        drawn: list[tuple[Any, ...]] = []

        def fake_prepare(*args: Any, **kwargs: Any) -> batch.PreparedPlot:
            prepared.append((args, kwargs))
            return FAKE_PREPARED

        def fake_draw(*args: Any, **kwargs: Any) -> FakePlot:
            drawn.append((args, kwargs))
            return FakePlot()

        monkeypatch.setattr(batch, "prepare_plot", fake_prepare)
        monkeypatch.setattr(batch, "draw_plot", fake_draw)
        cut = rf.Cut("y > 1")
        book = rf.PlotBook(
            samples,
            [X],
            selections={"sr": cut},
            variants={"log": {"logy": True}},
            plot_kwargs={"stack": True, "logy": False, "weight": "y", "lumi": 2.0},
        )
        list(book.plots())
        ((args, kwargs),) = prepared
        assert args[0] is samples
        assert args[1] is X
        assert kwargs["selection"] is cut
        assert isinstance(kwargs["cache"], ReadCache)
        assert {k: v for k, v in kwargs.items() if k not in ("selection", "cache")} == {
            "weight": "y",
            "lumi": 2.0,
        }
        ((args, kwargs),) = drawn
        assert args == (FAKE_PREPARED,)
        assert kwargs == {"stack": True, "logy": True}

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

        monkeypatch.setattr(batch, "draw_plot", failing)
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

        monkeypatch.setattr(batch, "draw_plot", interrupted)
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


def _reads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the branches of every ``FileSource.arrays`` call."""
    calls: list[list[str]] = []
    original = FileSource.arrays

    def counting(self: FileSource, branches: Any) -> Any:
        calls.append(list(branches))
        return original(self, branches)

    monkeypatch.setattr(FileSource, "arrays", counting)
    return calls


def _opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the file name of every ``uproot.open`` call made with a path."""
    calls: list[str] = []
    original = uproot.open

    def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, str | os.PathLike):
            calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(uproot, "open", counting)
    return calls


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple[Any, ...]]:
    """Record the calls of ``batch.<name>`` while forwarding them."""
    calls: list[tuple[Any, ...]] = []
    original = getattr(batch, name)

    def forwarding(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(batch, name, forwarding)
    return calls


def assert_same_plot(result: rf.Plot, direct: rf.Plot) -> None:
    """Assert that two plots hold the same histograms, ratios and axes, bit for bit."""
    assert [h.label for h in result.histograms] == [h.label for h in direct.histograms]
    for got, want in zip(result.histograms, direct.histograms, strict=True):
        np.testing.assert_array_equal(got.values(flow=True), want.values(flow=True))
        np.testing.assert_array_equal(got.variances(flow=True), want.variances(flow=True))
        np.testing.assert_array_equal(got.edges, want.edges)
        assert got.axis.label == want.axis.label
        assert (got.is_data, got.normalization, got.per_object, got.color) == (
            want.is_data,
            want.normalization,
            want.per_object,
            want.color,
        )
        assert got.stats == want.stats
        assert sorted(got.variations) == sorted(want.variations)
        for name, (up, down) in got.variations.items():
            want_up, want_down = want.variations[name]
            np.testing.assert_array_equal(up.values(flow=True), want_up.values(flow=True))
            np.testing.assert_array_equal(up.variances(flow=True), want_up.variances(flow=True))
            np.testing.assert_array_equal(down.values(flow=True), want_down.values(flow=True))
    assert len(result.ratios) == len(direct.ratios)
    for got_ratio, want_ratio in zip(result.ratios, direct.ratios, strict=True):
        np.testing.assert_array_equal(got_ratio.values, want_ratio.values)
    assert (result.ratio_ax is None) == (direct.ratio_ax is None)
    assert result.ax.get_yscale() == direct.ax.get_yscale()
    assert result.ax.get_xlabel() == direct.ax.get_xlabel()
    assert result.ax.get_ylabel() == direct.ax.get_ylabel()
    assert result.variable == direct.variable


class TestBatching:
    """Reads are shared across a batch's variables, selections and variants; plots equal plot()."""

    @pytest.fixture
    def files(self, signal_file: Path, background_file: Path) -> list[rf.Sample]:
        return [
            rf.Sample(signal_file, tree="events", label="Signal", weight="weight"),
            rf.Sample(background_file, tree="events", label="Background", weight="weight"),
        ]

    def test_keyword_sets_cover_plot(self) -> None:
        every = set(inspect.signature(rf.plot).parameters) - {"data", "variable"}
        assert every == batch._PREPARE_KEYWORDS | batch._DRAW_KEYWORDS
        assert batch._PREPARE_KEYWORDS.isdisjoint(batch._DRAW_KEYWORDS)
        prepare = {"selection", "tree", "bins", "range", "weight", "lumi", "observed"}
        prepare |= {"systematics", "assume_poisson", "nonfinite", "label", "xlabel", "unit"}
        assert prepare <= batch._PREPARE_KEYWORDS
        draw = {"logy", "logx", "normalize", "ratio", "stack", "style", "text", "stats"}
        assert draw | {"save", "ax"} <= batch._DRAW_KEYWORDS

    def test_variables_of_a_batch_share_one_read_per_sample(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        variables = ["MET", "Muon_pt", "nMuon"]
        results = list(rf.PlotBook(files, variables).plots())
        assert len(reads) == 2
        assert all(set(call) == {"MET", "Muon_pt", "nMuon", "weight"} for call in reads)
        for (_, result), variable in zip(results, variables, strict=True):
            assert_same_plot(result, rf.plot(files, variable))

    def test_batch_size_bounds_the_reads(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch, "_VARIABLE_BATCH_SIZE", 1)
        reads = _reads(monkeypatch)
        list(rf.PlotBook(files, ["MET", "Muon_pt", "nMuon"]).plots())
        assert len(reads) == 6
        assert [set(call) for call in reads[:2]] == [{"MET", "weight"}] * 2

    def test_drawing_only_variants_reuse_the_preparation(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        prepared = _spy(monkeypatch, "prepare_plot")
        book = rf.PlotBook(
            files,
            ["MET", "Muon_pt"],
            variants={"lin": {}, "log": {"logy": True}, "overlay": {"stack": ["Background"]}},
            plot_kwargs={"stack": True},
        )
        results = {task.stem: result for task, result in book.plots()}
        assert len(reads) == 2
        assert len(prepared) == 2
        for variable in ("MET", "Muon_pt"):
            lin, log = results[f"{variable}__lin"], results[f"{variable}__log"]
            assert (lin.ax.get_yscale(), log.ax.get_yscale()) == ("linear", "log")
            for a, b in zip(lin.histograms, log.histograms, strict=True):
                np.testing.assert_array_equal(a.values(flow=True), b.values(flow=True))
                assert a.hist is not b.hist  # each figure holds its own copy
            assert_same_plot(log, rf.plot(files, variable, stack=True, logy=True))

        assert_same_plot(results["MET__overlay"], rf.plot(files, "MET", stack=["Background"]))

    def test_variants_changing_the_preparation_are_prepared_apart(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        prepared = _spy(monkeypatch, "prepare_plot")
        book = rf.PlotBook(
            files, ["MET", "Muon_pt"], variants={"fine": {"bins": 20}, "coarse": {"bins": 10}}
        )
        results = {task.stem: result for task, result in book.plots()}
        assert len(reads) == 2  # the cache serves both preparations
        assert len(prepared) == 4
        assert results["MET__fine"].histograms[0].axis.size == 20
        assert results["MET__coarse"].histograms[0].axis.size == 10
        assert_same_plot(results["Muon_pt__coarse"], rf.plot(files, "Muon_pt", bins=10))
        assert_same_plot(results["MET__fine"], rf.plot(files, "MET", bins=20))

    def test_preparations_needing_different_branches_share_one_read(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Variants whose preparation reads other branches are planned together, so a batch
        # still costs one pass over each file rather than one per preparation.
        reads = _reads(monkeypatch)
        book = rf.PlotBook(
            files,
            ["MET"],
            variants={
                "plain": {},
                "weighted": {"weight": "nMuon"},
                "varied": {"systematics": {"scale": {"MET": "sentinel"}}},
            },
        )
        results = list(book.plots())
        assert len(reads) == 2
        assert all(set(call) == {"MET", "weight", "nMuon", "sentinel"} for call in reads)
        by_stem = {task.stem: result for task, result in results}
        assert by_stem["MET__plain"].histograms[0].variations == {}
        assert sorted(by_stem["MET__varied"].histograms[0].variations) == ["scale"]
        for task, result in results:
            assert_same_plot(result, rf.plot(files, task.variable, **task.kwargs))

    def test_normalisation_is_a_drawing_variant(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        book = rf.PlotBook(files, ["MET"], variants={"raw": {}, "norm": {"normalize": True}})
        (_, raw), (_, norm) = list(book.plots())
        assert len(reads) == 2
        assert [h.normalization for h in raw.histograms] == [None, None]
        assert [h.normalization for h in norm.histograms] == ["Normalised to unity"] * 2
        assert norm.histograms[0].integral == pytest.approx(1.0)
        assert_same_plot(raw, rf.plot(files, "MET"))
        assert_same_plot(norm, rf.plot(files, "MET", normalize=True))

    def test_figures_of_variants_do_not_share_histograms(self, files: list[rf.Sample]) -> None:
        book = rf.PlotBook(files, ["MET"], variants={"lin": {}, "log": {"logy": True}})
        plots = book.plots()
        _, lin = next(plots)
        lin.hists[0].view(flow=True).value[:] = 0.0  # a user editing one figure's histogram
        _, log = next(plots)
        assert_same_plot(log, rf.plot(files, "MET", logy=True))

    def test_selections_share_the_read(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        book = rf.PlotBook(files, ["MET", "Muon_pt"], selections={"all": None, "hi": "nMuon > 1"})
        results = list(book.plots())
        assert len(reads) == 2
        assert all(set(call) == {"MET", "Muon_pt", "nMuon", "weight"} for call in reads)
        for task, result in results:
            assert_same_plot(result, rf.plot(files, task.variable, selection=task.selection))

    def test_groups_read_their_leaves_once(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        group = rf.Group(files, label="All")
        book = rf.PlotBook(group, ["MET", "Muon_pt"], variants={"lin": {}, "log": {"logy": True}})
        results = list(book.plots())
        assert len(reads) == 2
        assert all([h.label for h in result.histograms] == ["All"] for _, result in results)
        for task, result in results:
            assert_same_plot(result, rf.plot(group, task.variable, **task.kwargs))

    def test_observed_data_joins_the_batch(
        self, signal_file: Path, background_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reads(monkeypatch)
        mc = rf.Sample(signal_file, tree="events", label="MC", weight="weight")
        data = rf.Sample(background_file, tree="events", label="Data", is_data=True)
        kwargs: dict[str, Any] = {"observed": data, "stack": True, "ratio": True}
        results = list(rf.PlotBook(mc, ["MET", "Muon_pt"], plot_kwargs=kwargs).plots())
        assert len(reads) == 2
        for task, result in results:
            assert result.histograms[-1].is_data
            assert result.ratio_ax is not None
            assert_same_plot(result, rf.plot(mc, task.variable, **kwargs))

    def test_stored_histograms_open_each_file_once_per_batch(
        self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opens = _opens(monkeypatch)
        samples = [
            rf.Sample(stored_dir / "WW_sel0_histo.root", label="WW"),
            rf.Sample(stored_dir / "ZZ_sel0_histo.root", label="ZZ"),
        ]
        book = rf.PlotBook(
            samples, ["mz", "mz_raw", "cutflow"], variants={"lin": {}, "log": {"logy": True}}
        )
        results = list(book.plots())
        # one open lists the objects of each file, one reads all three histograms from it
        assert (opens.count("WW_sel0_histo.root"), opens.count("ZZ_sel0_histo.root")) == (2, 2)
        for task, result in results:
            assert_same_plot(result, rf.plot(samples, task.variable, **task.kwargs))

    def test_stored_histograms_of_several_files_are_summed(
        self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opens = _opens(monkeypatch)
        vv = rf.Sample(
            [stored_dir / "WW_sel0_histo.root", stored_dir / "ZZ_sel0_histo.root"], label="VV"
        )
        results = list(rf.PlotBook(vv, ["mz", "mz_raw"]).plots())
        # the first file is also opened to list its objects; the second only to read both names
        assert (opens.count("WW_sel0_histo.root"), opens.count("ZZ_sel0_histo.root")) == (2, 1)
        assert results[0][1].histograms[0].sum_weights == pytest.approx(0.5 * 3000)
        for task, result in results:
            assert_same_plot(result, rf.plot(vv, task.variable))

    def test_assume_poisson_warns_once_per_batch(self, stored_dir: Path) -> None:
        sample = rf.Sample(stored_dir / "negative.root", label="N")
        book = rf.PlotBook(
            sample,
            ["mz"],
            variants={"plain": {}, "titled": {"title": "signed"}},
            plot_kwargs={"assume_poisson": True},
        )
        with pytest.warns(rf.RootfigWarning, match="Poisson guess"):
            results = list(book.plots())
        for task, result in results:
            with pytest.warns(rf.RootfigWarning, match="Poisson guess"):
                direct = rf.plot(sample, "mz", **task.kwargs)
            assert_same_plot(result, direct)

    def test_batches_are_read_lazily_in_task_order(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch, "_VARIABLE_BATCH_SIZE", 2)
        reads = _reads(monkeypatch)
        variables = ["MET", "Muon_pt", "nMuon", "Muon_eta", "Muon_phi"]
        book = rf.PlotBook(files, variables, variants={"lin": {}, "log": {"logy": True}})
        tasks = book.tasks()
        plots = book.plots()
        assert reads == []  # neither the book nor its tasks read anything
        stems = [next(plots)[0].stem]
        assert len(reads) == 2
        assert all(set(call) == {"MET", "Muon_pt", "weight"} for call in reads)
        stems.extend(next(plots)[0].stem for _ in range(3))
        assert len(reads) == 2  # the rest of the first batch is served from the cache
        stems.append(next(plots)[0].stem)  # the first task of the second batch
        assert len(reads) == 4
        assert set(reads[2]) == {"nMuon", "Muon_eta", "weight"}
        stems.extend(task.stem for task, _ in plots)
        assert len(reads) == 6
        assert stems == [task.stem for task in tasks]

    def test_a_bad_variable_fails_at_its_own_task(self, files: list[rf.Sample]) -> None:
        existing = plt.figure()
        book = rf.PlotBook(files, ["MET", "nosuch", "Muon_pt"], selections={"sr": "nMuon > 0"})
        plots = book.plots()
        _, first = next(plots)
        with pytest.raises(rf.MissingBranchError, match="'nosuch'") as info:
            next(plots)
        assert info.value.__notes__ == [
            "while running PlotBook task variable='nosuch', selection='sr', variant='default'"
        ]
        assert plt.get_fignums() == [existing.number, first.fig.number]

    def test_a_typo_deep_in_a_batch_fails_there(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch, "_VARIABLE_BATCH_SIZE", 8)
        variables = [f"MET + {i}" for i in range(5)] + ["nosuch", "MET + 7"]
        book = rf.PlotBook(files, variables)
        plots = book.plots()
        yielded = [next(plots)[0].stem for _ in range(5)]
        assert yielded == [task.stem for task in book.tasks()[:5]]
        with pytest.raises(rf.MissingBranchError, match="'nosuch'") as info:
            next(plots)
        assert info.value.__notes__ == [
            "while running PlotBook task variable='nosuch', selection='all', variant='default'"
        ]

    def test_a_bad_variant_fails_at_its_own_task(self, files: list[rf.Sample]) -> None:
        # label=[123] is a TypeError, not a rootfig error: reading ahead for the good variant
        # must not surface it, and it must carry the bad variant's note when its task runs.
        book = rf.PlotBook(files[0], ["MET"], variants={"good": {}, "bad": {"label": [123]}})
        plots = book.plots()
        _, good = next(plots)
        assert_same_plot(good, rf.plot(files[0], "MET"))
        with pytest.raises(TypeError, match="label must be a string") as info:
            next(plots)
        assert info.value.__notes__ == [
            "while running PlotBook task variable='MET', selection='all', variant='bad'"
        ]

    def test_raw_paths_learn_their_files_once(
        self, signal_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        variables = ["MET", "Muon_pt", "nMuon"]
        opens = _opens(monkeypatch)
        list(rf.PlotBook(str(signal_file), variables, plot_kwargs={"tree": "events"}).plots())
        from_paths = opens.count("signal.root")
        del opens[:]
        list(rf.PlotBook(rf.Sample(signal_file, tree="events"), variables).plots())
        # a path rebuilds its FileSource for every task; the cache hands out the first one
        assert from_paths == opens.count("signal.root")

    def test_variation_files_learn_their_files_once(
        self, signal_file: Path, background_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        variables = ["MET", "Muon_pt", "nMuon"]
        opens = _opens(monkeypatch)
        by_path = rf.Sample(
            signal_file,
            tree="events",
            systematics={"alt": rf.Systematic.samples(background_file)},
        )
        results = list(rf.PlotBook(by_path, variables).plots())
        from_paths = opens.count("background.root")
        del opens[:]
        by_sample = by_path.replace(
            systematics={"alt": rf.Systematic.samples(rf.Sample(background_file, tree="events"))}
        )
        list(rf.PlotBook(by_sample, variables).plots())
        assert from_paths == opens.count("background.root")
        for task, result in results:
            assert_same_plot(result, rf.plot(by_path, task.variable))

    def test_variation_files_are_read_once_for_the_batch(
        self, signal_file: Path, background_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An alternative generator is one file for many variables, so it must not be read
        # once per variable; its branches join the batch's plan like the nominal ones.
        sample = rf.Sample(
            signal_file,
            tree="events",
            label="S",
            weight="weight",
            systematics={"generator": rf.Systematic.samples(background_file)},
        )
        variables = ["MET", "Muon_pt", "nMuon"]
        reads = _reads(monkeypatch)
        results = list(rf.PlotBook(sample, variables).plots())
        assert len(reads) == 2  # the nominal file and the variation's, once each
        assert all(set(call) == {"MET", "Muon_pt", "nMuon", "weight"} for call in reads)
        for task, result in results:
            assert sorted(result.histograms[0].variations) == ["generator"]
            assert_same_plot(result, rf.plot(sample, task.variable))

    def test_stored_variation_files_are_read_once_for_the_batch(
        self, stored_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample = rf.Sample(
            stored_dir / "WW_sel0_histo.root",
            label="WW",
            systematics={"alt": rf.Systematic.samples(stored_dir / "ZZ_sel0_histo.root")},
        )
        opens = _opens(monkeypatch)
        results = list(rf.PlotBook(sample, ["mz", "mz_raw", "cutflow"]).plots())
        # the nominal file is listed (stored_mode) and read; the variation's is only read
        assert opens.count("ZZ_sel0_histo.root") == 1
        assert opens.count("WW_sel0_histo.root") == 2
        for task, result in results:
            assert sorted(result.histograms[0].variations) == ["alt"]
            assert_same_plot(result, rf.plot(sample, task.variable))

    def test_replace_systematic_missing_branch_fails_only_where_used(
        self, signal_file: Path
    ) -> None:
        sample = rf.Sample(
            signal_file,
            tree="events",
            label="S",
            systematics={"scale": {"Muon_pt": ("Muon_pt_up", "Muon_pt_dn")}},
        )
        plots = rf.PlotBook(sample, ["MET", "Muon_pt"]).plots()
        _, met = next(plots)  # does not use Muon_pt, so the replacement is not looked up
        assert_same_plot(met, rf.plot(sample, "MET"))
        with pytest.raises(rf.MissingBranchError, match="'Muon_pt_up'") as info:
            next(plots)
        assert info.value.__notes__ == [
            "while running PlotBook task variable='Muon_pt', selection='all', variant='default'"
        ]

    def test_failing_batch_read_is_left_to_the_tasks(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = FileSource.arrays
        failed: list[list[str]] = []

        def flaky(self: FileSource, branches: Any) -> Any:
            if not failed:
                failed.append(list(branches))
                msg = "transient"
                raise OSError(msg)
            return original(self, branches)

        monkeypatch.setattr(FileSource, "arrays", flaky)
        results = list(rf.PlotBook(files, ["MET", "Muon_pt"]).plots())
        assert [set(call) for call in failed] == [{"MET", "Muon_pt", "weight"}]
        for task, result in results:
            assert_same_plot(result, rf.plot(files, task.variable))

    def test_prefetch_only_saves_reads(
        self, files: list[rf.Sample], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        book = rf.PlotBook(files, ["MET", "nosuch"], variants={"lin": {}, "log": {"logy": True}})

        def run() -> tuple[rf.Plot, rf.Plot, str, list[str]]:
            plots = book.plots()
            (_, lin), (_, log) = next(plots), next(plots)
            with pytest.raises(rf.MissingBranchError) as info:
                next(plots)
            return lin, log, str(info.value), list(info.value.__notes__)

        lin, log, message, notes = run()
        monkeypatch.setattr(batch, "prefetch_plots", lambda *args, **kwargs: None)
        lin_again, log_again, message_again, notes_again = run()
        assert_same_plot(lin_again, lin)
        assert_same_plot(log_again, log)
        assert (message_again, notes_again) == (message, notes)

    @pytest.mark.parametrize(
        "case", ["systematics", "lumi", "group_observed_ratio", "stored", "rntuple", "binning"]
    )
    def test_every_plot_equals_plot(self, case: str, data_dir: Path, stored_dir: Path) -> None:
        book = _books(data_dir, stored_dir)[case]
        for task, result in book.plots():
            direct = rf.plot(book.data, task.variable, selection=task.selection, **task.kwargs)
            assert_same_plot(result, direct)


def _books(data_dir: Path, stored_dir: Path) -> dict[str, rf.PlotBook]:
    """Books covering the inputs and options whose results must equal plot()'s, task by task."""
    signal = rf.Sample(data_dir / "signal.root", tree="events", label="Signal", weight="weight")
    background = rf.Sample(
        data_dir / "background.root", tree="events", label="Background", weight="weight"
    )
    varied = signal.replace(
        systematics={
            "w": ("weight * 1.1", "weight * 0.9"),
            "norm": 0.05,
            "alt": rf.Systematic.samples(data_dir / "background.root"),
        }
    )
    data = rf.Sample(data_dir / "background.root", tree="events", label="Data", is_data=True)
    return {
        "systematics": rf.PlotBook(
            [varied, background],
            ["MET", "Muon_pt"],
            selections={"all": None, "hi": "nMuon > 1"},
            plot_kwargs={"stack": True},
        ),
        "lumi": rf.PlotBook(
            [
                signal.replace(xsec=1.5, ngen=1000),
                background.replace(xsec="2 pb", ngen=2000),
            ],
            ["MET", rf.Variable("nMuon", bins=(6, 0, 6))],
            plot_kwargs={"lumi": 10.0, "stack": True},
        ),
        "group_observed_ratio": rf.PlotBook(
            rf.Group([signal, background], label="MC"),
            ["MET", "Muon_pt"],
            variants={"lin": {}, "norm": {"normalize": True, "stack": False}},
            plot_kwargs={"observed": data, "stack": True, "ratio": True},
        ),
        "stored": rf.PlotBook(
            [
                rf.Sample(stored_dir / "WW_sel0_histo.root", label="WW", scale=2.0),
                rf.Sample(stored_dir / "ZZ_sel0_histo.root", label="ZZ"),
            ],
            ["mz", "mz_raw", rf.Variable("mz", bins=50, name="mz50")],
            variants={"lin": {}, "log": {"logy": True}},
            plot_kwargs={"stack": True},
        ),
        "rntuple": rf.PlotBook(
            rf.Sample(data_dir / "signal_rntuple.root", tree="events", weight="weight"),
            ["MET", "Muon_pt"],
        ),
        "binning": rf.PlotBook(
            [signal, background],
            [
                rf.Variable("MET", bins=(20, 0, 100), unit="GeV"),
                rf.Variable("Muon_pt", bins=15, range="auto"),
                "Muon_eta",
            ],
            plot_kwargs={"flow": "show"},
        ),
    }
