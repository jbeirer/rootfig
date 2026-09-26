"""Tests for the plotting layer: styles, layout, drawing, lower panel, annotations."""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import partial
from typing import Any

import hist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.gridspec import GridSpec
from matplotlib.patches import StepPatch
from matplotlib.text import Text

from rootfig.errors import BinningError, RootfigWarning
from rootfig.histograms import (
    Comparison,
    ComparisonKind,
    Efficiency,
    Histogram,
    Profile,
    compare,
    fill,
    summarize,
    uncertainty,
)
from rootfig.model import Style
from rootfig.model.style import EXPERIMENT_STYLES
from rootfig.plotting import (
    DEFAULT_COLORS,
    ROOTFIG_STYLE,
    Finish,
    Layout,
    Plot,
    StackSpec,
    add_experiment_label,
    add_legend,
    add_stats_box,
    add_text,
    apply_xbreak,
    break_segments,
    color_cycle,
    correlation_figsize,
    draw_correlation,
    draw_hist2d,
    draw_histograms,
    draw_panel,
    envelope,
    finish_axes,
    finish_figure,
    finishing_together,
    fold_flow_bins,
    label_flow_bins,
    make_figure,
    overlay_artists,
    panel_ylim,
    raise_ylim_above,
    show_flow_bins,
    split_stack,
    style_context,
    use_style,
    ylabel_for,
)
from rootfig.plotting.figure import (
    PANEL_HEIGHT_FRACTION,
    PANEL_LABEL_MIN_SCALE,
    _balanced_wrap,
    _renderer,
    figure_size,
    fit_ylabel,
    without_redraw,
)
from rootfig.plotting.panel import comparison_label
from rootfig.plotting.style import align_experiment_label, foreground, pin_fonts, resolve_rc
from rootfig.selection import Columns


def make_hist(
    values: list[float], weights: list[float] | None = None, **kwargs: object
) -> Histogram:
    cols = Columns(
        arrays=(np.asarray(values, dtype=float),),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        n_events=len(values),
        n_selected_events=len(values),
    )
    return Histogram(fill([hist.axis.Regular(4, 0, 4)], cols), stats=summarize(cols), **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def mc_hists() -> list[Histogram]:
    return [
        make_hist([0.5, 1.5, 1.5, 2.5], label="A"),
        make_hist([0.5, 0.5, 2.5, 3.5], [2.0, 2.0, 1.0, 1.0], label="B", color="green"),
    ]


@pytest.fixture
def data_hist() -> Histogram:
    return make_hist([0.5, 1.5, 2.5, 2.5, 3.5], label="Data", is_data=True)


class TestStyle:
    def test_resolve_default(self) -> None:
        sheets = resolve_rc(Style())
        assert sheets == [ROOTFIG_STYLE]

    def test_resolve_experiment_and_overrides(self) -> None:
        sheets = resolve_rc(Style(experiment="ATLAS", colors=["red"], rc={"font.size": 9}))
        assert len(sheets) == 3
        assert sheets[0]["figure.figsize"] == (8.0, 6.0)  # mplhep ATLAS
        assert sheets[2] == {"font.size": 9}

    def test_resolve_named_and_mapping(self) -> None:
        assert resolve_rc(Style(base="CMS"))[0]["figure.figsize"] == (10.0, 10.0)
        assert resolve_rc(Style(base="ggplot")) == ["ggplot"]
        assert resolve_rc(Style(base={"font.size": 7}))[1] == {"font.size": 7}
        with pytest.raises(ValueError, match="unknown style"):
            resolve_rc(Style(base="not-a-style"))

    def test_unknown_experiment_uses_neutral_base(self) -> None:
        assert resolve_rc(Style(experiment="MyExp"))[0] is ROOTFIG_STYLE

    def test_context_restores_rcparams(self) -> None:
        before = matplotlib.rcParams["font.size"]
        with style_context(Style(rc={"font.size": 33})) as st:
            assert matplotlib.rcParams["font.size"] == 33
            assert isinstance(st, Style)
        assert matplotlib.rcParams["font.size"] == before

    def test_context_activates_backend_before_applying_rcparams(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # matplotlib activates its backend on the first pyplot call. In Jupyter that
        # sets interactive mode; if it happened inside the rc context it would be
        # rolled back on exit and inline display would stop working (see _activate_backend).
        seen: list[float] = []

        def fake_draw_if_interactive() -> None:
            seen.append(matplotlib.rcParams["font.size"])
            matplotlib.interactive(True)

        monkeypatch.setattr(plt, "_backend_mod", None, raising=False)
        monkeypatch.setattr(plt, "draw_if_interactive", fake_draw_if_interactive)
        monkeypatch.setitem(matplotlib.rcParams, "interactive", False)
        before = matplotlib.rcParams["font.size"]

        with style_context(Style(rc={"font.size": 33})):
            pass

        assert seen == [before]  # ran outside the context, on the unstyled rcParams
        assert matplotlib.is_interactive()  # and its change survived the rollback

    def test_context_skips_activation_once_backend_is_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plt.draw_if_interactive()  # resolve the backend, as any earlier drawing would have
        calls = []
        monkeypatch.setattr(plt, "draw_if_interactive", lambda: calls.append(1))
        with style_context():
            pass
        assert calls == []

    @pytest.mark.parametrize("experiment", [None, "ATLAS", "CMS", "LHCb", "ALICE"])
    def test_pin_fonts_pins_only_installed_families(self, experiment: str | None) -> None:
        # matplotlib walks the whole family list on every draw and logs a warning per
        # miss, so a family that cannot be resolved must never be pinned.
        with style_context(Style(experiment=experiment)):
            layout = make_figure(Style(), panel=False)
            layout.main.set_xlabel("x")
            pin_fonts(layout.fig)
            families = [name for text in layout.fig.findobj(Text) for name in text.get_fontfamily()]
        assert families
        for name in families:
            assert name not in font_manager.font_family_aliases  # never generic
            findfont(FontProperties(family=name), fallback_to_default=False)  # resolves
        plt.close(layout.fig)

    def test_pin_fonts_keeps_case_insensitive_matches(self) -> None:
        # mplhep's LHCb2 sheet spells it "Tex Gyre Termes", which is not among the
        # installed font names but does resolve; filtering must not drop it.
        with style_context(Style(rc={"font.family": ["Tex Gyre Termes"]})):
            layout = make_figure(Style(), panel=False)
            layout.main.set_xlabel("x")
            pin_fonts(layout.fig)
            assert layout.main.xaxis.label.get_fontfamily() == ["Tex Gyre Termes"]
        plt.close(layout.fig)

    @pytest.mark.parametrize("rc_key", ["font.family", "font.sans-serif"])
    def test_pin_fonts_falls_back_when_nothing_is_installed(self, rc_key: str) -> None:
        with style_context(Style(rc={rc_key: ["No Such Font XYZ"]})):
            layout = make_figure(Style(), panel=False)
            layout.main.set_xlabel("x")
            with pytest.warns(RootfigWarning, match="none of the requested fonts") as caught:
                pin_fonts(layout.fig)
            assert len(caught) == 1
            assert layout.main.xaxis.label.get_fontfamily() == [
                font_manager.fontManager.defaultFamily["ttf"]
            ]
        plt.close(layout.fig)

    def test_pin_fonts_warns_once_per_missing_configuration_per_call(self) -> None:
        with style_context(Style(rc={"font.family": ["No Such Font XYZ"]})):
            # Each figure should report its missing configurations independently.
            for _ in range(2):
                layout = make_figure(Style(), panel=False)
                for y in (0.3, 0.6):
                    layout.main.text(0.5, y, "label", fontfamily=["Another Missing Font XYZ"])
                with pytest.warns(RootfigWarning, match="none of the requested fonts") as caught:
                    pin_fonts(layout.fig)
                assert len(caught) == 2
                messages = [str(warning.message) for warning in caught]
                assert sum("(No Such Font XYZ)" in message for message in messages) == 1
                assert sum("(Another Missing Font XYZ)" in message for message in messages) == 1
                plt.close(layout.fig)

    def test_use_style_is_global(self) -> None:
        before = dict(matplotlib.rcParams)
        try:
            use_style(Style(rc={"font.size": 31}))
            assert matplotlib.rcParams["font.size"] == 31
            use_style("ggplot")
        finally:
            matplotlib.rcParams.update(before)

    def test_color_cycle(self) -> None:
        with style_context():
            assert color_cycle(3) == list(DEFAULT_COLORS[:3])
            assert len(color_cycle(15)) == 15
        assert color_cycle(2, Style(colors=["r", "g"])) == ["r", "g"]
        assert color_cycle(3, Style(colors=["r", "g"])) == ["r", "g", "r"]

    def test_experiment_labels(self) -> None:
        for experiment in ["ATLAS", "CMS", "LHCb", "ALICE", "DUNE", "MyExperiment"]:
            fig, ax = plt.subplots()
            before = len(ax.texts)
            add_experiment_label(
                ax,
                Style(experiment=experiment, status="Internal", lumi=10, com=13, text="x"),
                has_data=True,
            )
            assert len(ax.texts) > before, experiment
            plt.close(fig)

    def test_label_words_are_separated(self) -> None:
        with style_context(Style(experiment="ATLAS")):
            fig, ax = plt.subplots()
            add_experiment_label(ax, Style(experiment="ATLAS", status="Internal"), has_data=True)
            align_experiment_label(ax)  # the plotting functions do so on the finished figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            texts = {t.get_text(): t for t in ax.texts}
            exp_box = texts["ATLAS"].get_window_extent(renderer)
            suffix_box = texts["Internal"].get_window_extent(renderer)
            gap_em = (suffix_box.x0 - exp_box.x1) / (
                texts["Internal"].get_fontsize() / 72 * fig.dpi
            )
        assert gap_em > 0.3

    def test_text_only_label(self) -> None:
        fig, ax = plt.subplots()
        add_experiment_label(
            ax, Style(status="Preliminary", com=13.6, lumi=140, text=["a", "b"]), has_data=False
        )
        assert len(ax.artists) == 1
        add_experiment_label(ax, Style(), has_data=False)
        assert len(ax.artists) == 1

    @pytest.mark.parametrize("experiment", [*EXPERIMENT_STYLES, "FCC-ee"])
    def test_text_lines_stay_inside_the_frame(self, experiment: str) -> None:
        # mplhep turns supp= into "Supplementary" in the label and, for a label above the
        # frame, into a note rotated along the frame's right edge
        style = Style(
            experiment=experiment, status="Internal", lumi=140, com=13.6, text=["one", "two"]
        )
        with style_context(style):
            fig, ax = plt.subplots()
            add_experiment_label(ax, style, has_data=False)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            texts = [t for t in fig.findobj(Text) if t.get_text()]
            assert not any("Supplementary" in t.get_text() for t in texts)
            assert "Simulation Internal" in {t.get_text() for t in texts}
            [lines] = [t for t in texts if t.get_text() == "one\ntwo"]
            assert lines.get_rotation() == 0
            box, frame = lines.get_window_extent(renderer), ax.get_window_extent(renderer)
            assert frame.x0 <= box.x0 < box.x1 <= frame.x1
            assert frame.y0 <= box.y0 < box.y1 <= frame.y1
        plt.close(fig)


class TestFigure:
    def test_make_figure_single(self) -> None:
        with style_context():
            layout = make_figure(Style(), panel=False)
        assert isinstance(layout, Layout)
        assert isinstance(layout.fig, Figure)
        assert isinstance(layout.main, Axes)
        assert layout.panel is None
        assert not layout.is_broken
        assert layout.main_axes == (layout.main,)
        assert layout.panel_axes == ()
        assert layout.legend_axes is layout.main
        assert layout.xlabel_axes is layout.main
        assert tuple(layout.fig.get_size_inches()) == pytest.approx(ROOTFIG_STYLE["figure.figsize"])

    def test_make_figure_panel(self) -> None:
        with style_context():
            layout = make_figure(Style(), panel=True, figsize=(5, 6))
        assert layout.panel is not None
        assert layout.panel.get_shared_x_axes().joined(layout.main, layout.panel)
        assert layout.xlabel_axes is layout.panel
        assert tuple(layout.fig.get_size_inches()) == (5, 6)

    def test_make_figure_broken(self) -> None:
        with style_context():
            layout = make_figure(Style(), panel=True, break_widths=(0.7, 0.3))
        assert layout.is_broken
        assert layout.main_right is not None
        assert layout.panel_right is not None
        assert len(layout.fig.axes) == 4
        assert layout.main_right.get_shared_y_axes().joined(layout.main, layout.main_right)
        assert layout.panel_right.get_shared_x_axes().joined(layout.main_right, layout.panel_right)
        assert layout.legend_axes is layout.main_right
        assert layout.xlabel_axes is layout.panel_right
        left_width = layout.main.get_position().width
        right_width = layout.main_right.get_position().width
        assert left_width / right_width == pytest.approx(0.7 / 0.3, rel=0.05)

    def test_make_figure_existing_axes(self) -> None:
        fig, axes = plt.subplots(2)
        assert make_figure(Style(), panel=False, ax=axes[0]).main is axes[0]
        layout = make_figure(Style(), panel=True, ax=(axes[0], axes[1]))
        assert (layout.main, layout.panel) == (axes[0], axes[1])
        assert make_figure(Style(), panel=False, ax=(axes[0], axes[1])).panel is None
        with pytest.raises(ValueError, match="two axes"):
            make_figure(Style(), panel=True, ax=axes[0])
        with pytest.raises(ValueError, match="pair"):
            make_figure(Style(), panel=True, ax=(axes[0],))
        with pytest.raises(ValueError, match="existing axes"):
            make_figure(Style(), panel=False, ax=axes[0], break_widths=(0.5, 0.5))

    def test_figure_size(self) -> None:
        with style_context():
            width, height = ROOTFIG_STYLE["figure.figsize"]
            assert figure_size(Style(), panel=False) == pytest.approx((width, height))
            assert figure_size(Style(), panel=True) == pytest.approx(
                (width, height * (1 + PANEL_HEIGHT_FRACTION * 0.85))
            )
            assert figure_size(Style(figsize=(5, 4)), panel=True) == (5, 4)  # the style's, as is
            assert figure_size(Style(figsize=(5, 4)), panel=False, figsize=(3, 2)) == (3, 2)

    def test_layout_axes_in_reading_order(self) -> None:
        with style_context():
            layout = make_figure(Style(), panel=True, break_widths=(0.7, 0.3))
            single = make_figure(Style(), panel=False)
        assert layout.axes == (layout.main, layout.main_right, layout.panel, layout.panel_right)
        assert single.axes == (single.main,)
        plt.close(layout.fig)
        plt.close(single.fig)

    @pytest.mark.parametrize("panel", [False, True])
    @pytest.mark.parametrize("break_widths", [None, (0.7, 0.3)])
    def test_make_figure_in_a_cell(
        self, panel: bool, break_widths: tuple[float, float] | None
    ) -> None:
        page = plt.figure(figsize=(12, 8), layout="constrained")
        outer = page.add_gridspec(1, 2)
        with style_context():
            left = make_figure(Style(), panel=panel, break_widths=break_widths, cell=outer[0, 0])
            right = make_figure(Style(), panel=panel, break_widths=break_widths, cell=outer[0, 1])
        assert left.fig is page
        assert right.fig is page
        assert page.get_size_inches().tolist() == [12, 8]  # the page keeps its size
        expected = (1 + panel) * (2 if break_widths else 1)
        assert len(left.axes) == len(right.axes) == expected
        assert len(page.axes) == 2 * expected
        for layout, spec in ((left, outer[0, 0]), (right, outer[0, 1])):
            assert layout.is_broken is (break_widths is not None)
            assert (layout.panel is not None) is panel
            for ax in layout.axes:
                subplotspec = ax.get_subplotspec()
                assert subplotspec is not None
                assert subplotspec.get_topmost_subplotspec() == spec
        if panel:
            assert left.panel is not None
            assert left.panel.get_shared_x_axes().joined(left.main, left.panel)
        page.canvas.draw()
        # the same relative geometry as on a figure of its own, inside the cell
        if break_widths:
            assert left.main_right is not None
            ratio_of_widths = left.main.get_position().width / left.main_right.get_position().width
            assert ratio_of_widths == pytest.approx(0.7 / 0.3, rel=0.05)
        if panel:
            assert left.panel is not None
            heights = left.panel.get_position().height / left.main.get_position().height
            assert heights == pytest.approx(PANEL_HEIGHT_FRACTION, rel=0.05)
        assert max(ax.get_position().x1 for ax in left.axes) < min(
            ax.get_position().x0 for ax in right.axes
        )
        plt.close(page)

    def test_make_figure_cell_rejects_figsize_and_axes(self) -> None:
        page = plt.figure()
        cell = page.add_gridspec(1, 1)[0, 0]
        with pytest.raises(ValueError, match="page"):
            make_figure(Style(), panel=False, cell=cell, figsize=(3, 3))
        with pytest.raises(ValueError, match="one of them"):
            make_figure(Style(), panel=False, cell=cell, ax=page.add_subplot(cell))
        plt.close(page)
        with pytest.raises(TypeError, match="grid on a figure"):
            make_figure(Style(), panel=False, cell=GridSpec(1, 1)[0, 0])

    def test_break_segments(self) -> None:
        left, right, widths = break_segments((0.0, 100.0), (20.0, 80.0))
        assert left == (0.0, 20.0)
        assert right == (80.0, 100.0)
        assert widths == pytest.approx((0.5, 0.5))
        _, _, widths = break_segments((0.0, 100.0), (5.0, 95.0))
        assert widths == pytest.approx((0.5, 0.5))
        _, _, widths = break_segments((0.0, 100.0), (90.0, 99.0))
        assert widths[1] == pytest.approx(0.15)  # minimum width for a thin segment
        _, _, widths = break_segments((1.0, 1000.0), (10.0, 100.0), logx=True)
        assert widths == pytest.approx((0.5, 0.5))
        for bad in [(20.0, 10.0), (-1.0, 50.0), (10.0, 100.0), (50.0, 50.0)]:
            with pytest.raises(ValueError, match="xbreak"):
                break_segments((0.0, 100.0), bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="pair"):
            break_segments((0.0, 100.0), (1.0,))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive"):
            break_segments((0.0, 100.0), (10.0, 50.0), logx=True)

    def test_apply_xbreak(self) -> None:
        fig, (left, right) = plt.subplots(1, 2, sharey=True)
        apply_xbreak(left, right, (0.0, 20.0), (80.0, 100.0))
        assert left.get_xlim() == (0.0, 20.0)
        assert right.get_xlim() == (80.0, 100.0)
        assert not left.spines["right"].get_visible()
        assert not right.spines["left"].get_visible()
        assert len(left.lines) == 2
        assert len(right.lines) == 2
        assert 20.0 not in left.get_xticks()
        assert 80.0 not in right.get_xticks()
        assert (
            not right.yaxis.get_ticklabels()[0].get_visible()
            if right.yaxis.get_ticklabels()
            else True
        )

    def test_ylabel_for(self) -> None:
        assert ylabel_for(normalization=None, unit=None, widths=None, per_object=False) == "Events"
        assert (
            ylabel_for(normalization=None, unit=None, widths=np.array([1.0, 1.0]), per_object=True)
            == "Entries"
        )
        assert (
            ylabel_for(
                normalization=None, unit="GeV", widths=np.array([2.0, 2.0]), per_object=False
            )
            == "Events / 2 GeV"
        )
        assert (
            ylabel_for(normalization=None, unit=None, widths=np.array([0.5, 0.5]), per_object=False)
            == "Events / 0.5"
        )
        # awkward widths are not shown
        assert (
            ylabel_for(
                normalization=None,
                unit="GeV",
                widths=np.array([7.64866, 7.64866]),
                per_object=False,
            )
            == "Events"
        )
        assert (
            ylabel_for(normalization=None, unit=None, widths=np.array([1.0, 2.0]), per_object=False)
            == "Events"
        )
        assert (
            ylabel_for(
                normalization="Normalised to unity", unit="GeV", widths=None, per_object=False
            )
            == "Normalised to unity"
        )
        assert (
            ylabel_for(normalization="Events / unit", unit="GeV", widths=None, per_object=False)
            == "Entries / GeV"
        )
        assert (
            ylabel_for(normalization="Events / unit", unit=None, widths=None, per_object=False)
            == "Entries / unit"
        )

    def test_finish_axes_linear(self) -> None:
        fig, ax = plt.subplots()
        finish_axes(
            ax,
            data_range=(0.0, 10.0),
            xlabel="x",
            ylabel="y",
            xlim=(0, 1),
            ylim=None,
            logx=False,
            logy=False,
        )
        assert ax.get_xlabel() == "x"
        assert ax.get_xlim() == (0, 1)
        bottom, top = ax.get_ylim()
        assert bottom == 0.0
        assert top == pytest.approx(12.0)

    def test_finish_axes_log_and_user_limits(self) -> None:
        fig, ax = plt.subplots()
        finish_axes(
            ax,
            data_range=(2.0, 100.0),
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=(None, 5000.0),
            logx=True,
            logy=True,
        )
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        assert ax.get_ylim() == (1.0, 5000.0)
        finish_axes(
            ax,
            data_range=(2.0, 100.0),
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=(1.0, 2.0),
            logx=False,
            logy=False,
        )
        assert ax.get_ylim() == (1.0, 2.0)

    def test_finish_axes_degenerate(self) -> None:
        fig, ax = plt.subplots()
        finish_axes(
            ax,
            data_range=(0.0, 0.0),
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=None,
            logx=False,
            logy=False,
        )
        assert ax.get_ylim() == (0.0, 1.0)
        finish_axes(
            ax,
            data_range=(float("nan"), float("nan")),
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=None,
            logx=False,
            logy=True,
        )
        assert ax.get_ylim()[0] > 0
        _, ax = plt.subplots()
        finish_axes(
            ax,
            data_range=(-5.0, 5.0),
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=None,
            logx=False,
            logy=False,
        )
        assert ax.get_ylim()[0] < -5.0


class TestDrawHistograms:
    def test_overlay_step(self, mc_hists: list[Histogram]) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms(mc_hists, ax, style=st)
        assert drawn.labels == ["A", "B"]
        assert drawn.colors == [DEFAULT_COLORS[0], "green"]
        assert drawn.ymax == pytest.approx(4.0)
        assert drawn.ymin_positive == pytest.approx(1.0)
        assert len(ax.get_legend_handles_labels()[1]) == 2

    @pytest.mark.parametrize("histtype", ["fill", "errorbar", "step", "band"])
    def test_histtypes(self, mc_hists: list[Histogram], histtype: str) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms(mc_hists, ax, style=st, histtype=histtype, errorbars=True)  # type: ignore[arg-type]
        assert drawn.artists
        # with error bars the range includes them
        assert drawn.ymax > 4.0

    @pytest.mark.parametrize("stack", [True, ["A", "B"]])
    def test_stack_with_data(
        self, mc_hists: list[Histogram], data_hist: Histogram, stack: StackSpec
    ) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms([*mc_hists, data_hist], ax, style=st, stack=stack)
        assert drawn.labels == ["A", "B", "Stat. unc.", "Data"]
        total = mc_hists[0].values() + mc_hists[1].values()
        assert drawn.stack is not None
        np.testing.assert_allclose(drawn.stack.values(), total)
        assert drawn.ymax >= total.max()
        assert drawn.colors[-1] == "black"

    def test_partial_stack(self, mc_hists: list[Histogram], data_hist: Histogram) -> None:
        histograms = [*mc_hists, data_hist]
        with style_context() as st:
            _, ax = plt.subplots()
            drawn = draw_histograms(histograms, ax, style=st, stack=["A"])
            colors = []
            for stack in (True, False):
                _, other = plt.subplots()
                colors.append(draw_histograms(histograms, other, style=st, stack=stack).colors)
        assert drawn.labels == ["A", "Stat. unc.", "B", "Data"]
        assert drawn.stack is not None
        np.testing.assert_allclose(drawn.stack.values(), mc_hists[0].values())
        assert drawn.ymax == pytest.approx(4.0)
        assert drawn.colors == colors[0] == colors[1]

    def test_stack_labels_are_checked(
        self, mc_hists: list[Histogram], data_hist: Histogram
    ) -> None:
        histograms = [*mc_hists, data_hist]
        assert split_stack(histograms, "A") == split_stack(histograms, ["A"])
        assert split_stack(histograms, []) == split_stack(histograms, False)
        assert split_stack(histograms, ["B", "A", "B"])[0] == mc_hists
        with pytest.raises(ValueError, match="observed data"):
            split_stack(histograms, "Data")
        with pytest.raises(ValueError, match="stack= names") as exc:
            split_stack(histograms, ["WW", "ZZ"])
        assert str(exc.value) == (
            "stack= names 'WW', 'ZZ', which are not labels of drawn histograms "
            "(labels: ['A', 'B', 'Data']); a Group is stacked by its own label, "
            "not by those of its components"
        )
        with pytest.raises(ValueError, match="stack= names") as exc:
            split_stack(histograms, "WW")
        assert str(exc.value) == (
            "stack= names 'WW', which is not the label of a drawn histogram "
            "(labels: ['A', 'B', 'Data']); a Group is stacked by its own label, "
            "not by those of its components"
        )
        for bad in (None, 1, [1], {"A"}):
            with pytest.raises(TypeError) as exc:
                split_stack(histograms, bad)  # type: ignore[arg-type]
            assert str(exc.value) == (
                f"stack= must be True, False, a label or a list of labels, got {bad!r}"
            )

    def test_duplicate_labels_select_all_non_data(self, mc_hists: list[Histogram]) -> None:
        a, b = mc_hists
        histograms = [a, b.replace(label="A"), a.replace(is_data=True)]
        stacked, overlaid, data = split_stack(histograms, "A")
        assert stacked == histograms[:2]
        assert overlaid == []
        assert data == histograms[2:]
        _, ax = plt.subplots()
        drawn = draw_histograms(histograms, ax, style=Style(), stack="A")
        assert drawn.colors == [color_cycle(1, Style())[0], "green", "black"]
        assert drawn.stack is not None
        np.testing.assert_allclose(drawn.stack.values(), a.values() + b.values())

    def test_stack_without_uncertainty_band(self, mc_hists: list[Histogram]) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms(
                mc_hists, ax, style=st, stack=True, stack_uncertainty=False, alpha=0.5
            )
        assert "Stat. unc." not in drawn.labels

    def test_empty_histograms(self) -> None:
        empty = make_hist([], label="empty")
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms([empty], ax, style=st)
        assert drawn.ymax == 0.0
        assert np.isnan(drawn.ymin_positive)

    def test_no_mplhep_warning_without_scipy(self, mc_hists: list[Histogram]) -> None:
        with style_context() as st, warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, ax = plt.subplots()
            draw_histograms(mc_hists, ax, style=st, histtype="fill")
            draw_histograms(mc_hists, ax, style=st, stack=True)


class TestFitYlabel:
    """A lower panel's y label is bounded by the panel height, not the figure height."""

    @staticmethod
    def _panel(label: str, *, figsize: tuple[float, float] = (7.0, 7.0)) -> Axes:
        layout = make_figure(Style(), panel=True, figsize=figsize)
        assert layout.panel is not None
        layout.panel.set_ylabel(label, loc="center")
        return layout.panel

    @staticmethod
    def _overflow(ax: Axes) -> float:
        renderer = _renderer(ax.figure)
        label_height = ax.yaxis.label.get_window_extent(renderer).height
        return float(label_height / ax.get_window_extent(renderer).height)

    def test_short_label_is_untouched(self) -> None:
        with style_context():
            ax = self._panel("Data / MC")
            before = ax.yaxis.label.get_fontsize()
            fit_ylabel(ax)
            assert ax.yaxis.label.get_fontsize() == before
            assert ax.get_ylabel() == "Data / MC"
        plt.close(ax.figure)

    def test_long_label_is_shrunk_to_fit(self) -> None:
        with style_context():
            ax = self._panel("Ratio to Conformal seeding")
            before = ax.yaxis.label.get_fontsize()
            fit_ylabel(ax)
            assert ax.yaxis.label.get_fontsize() < before
            assert self._overflow(ax) <= 1.0
        plt.close(ax.figure)

    def test_wraps_only_when_shrinking_is_not_enough(self) -> None:
        with style_context():
            ax = self._panel("Ratio to Conformal seeding")
            fit_ylabel(ax)
            assert ax.get_ylabel() == "Ratio to\nConformal seeding"
        plt.close(ax.figure)

    def test_mathtext_is_never_wrapped(self) -> None:
        with style_context():
            ax = self._panel(r"$S/\sqrt{B}$ with a very long trailing description")
            fit_ylabel(ax, min_scale=0.9)  # force the wrapping branch
            assert "\n" not in ax.get_ylabel()
        plt.close(ax.figure)

    def test_floor_is_respected(self) -> None:
        with style_context():
            ax = self._panel("Ratio to Conformal seeding with ITk layout v2", figsize=(4, 3.2))
            before = ax.yaxis.label.get_fontsize()
            fit_ylabel(ax)  # cannot fit; must not shrink to nothing
            assert ax.yaxis.label.get_fontsize() >= before * PANEL_LABEL_MIN_SCALE
        plt.close(ax.figure)

    def test_empty_label_and_missing_renderer_are_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with style_context():
            ax = self._panel("")  # the right segment of a broken axis
            fit_ylabel(ax)
            assert ax.get_ylabel() == ""

            ax = self._panel("Ratio to Conformal seeding")
            before = ax.yaxis.label.get_fontsize()
            monkeypatch.setattr("rootfig.plotting.figure._renderer", lambda fig: None)
            fit_ylabel(ax)
            assert ax.yaxis.label.get_fontsize() == before
        plt.close(ax.figure)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Ratio to Conformal seeding", "Ratio to\nConformal seeding"),
            ("Ratio", "Ratio"),
            ("", ""),
        ],
    )
    def test_balanced_wrap(self, text: str, expected: str) -> None:
        assert _balanced_wrap(text) == expected


def comparison(
    values: Any,
    *,
    kind: ComparisonKind = "ratio",
    errors: Any = None,
    band: Any = None,
    syst_errors: Any = None,
) -> Comparison:
    """A comparison over the bins ``0, 1, ..., len(values)``; symmetric ``errors`` and ``band``
    may be given as one array."""
    values = np.asarray(values, dtype=float)
    return Comparison(
        kind=kind,
        label="N",
        reference="R",
        values=values,
        errors=_pair(np.zeros_like(values) if errors is None else errors),
        edges=np.arange(len(values) + 1.0),
        band=None if band is None else _pair(band),
        syst_errors=syst_errors,
    )


def _pair(errors: Any) -> tuple[np.ndarray, np.ndarray]:
    """``errors`` as ``(down, up)``: a pair as it is, one array on both sides."""
    if isinstance(errors, tuple):
        down, up = errors
        return np.asarray(down, dtype=float), np.asarray(up, dtype=float)
    symmetric = np.asarray(errors, dtype=float)
    return symmetric, symmetric.copy()


class TestPanel:
    def test_ratio_panel_propagate(self, mc_hists: list[Histogram]) -> None:
        fig, (ax, rax) = plt.subplots(2)
        # propagated error bars already hold the reference uncertainty: no band by default
        draw_panel([compare(mc_hists[1], mc_hists[0])], rax)
        assert rax.get_ylabel() == "Ratio to A"
        assert rax.get_xlim() == (0.0, 4.0)
        assert not [c for c in rax.collections if isinstance(c, PolyCollection)]  # no band
        (baseline,) = rax.lines[:1]
        assert baseline.get_ydata() == [1.0, 1.0]
        plt.close(fig)

    def test_the_band_follows_the_uncertainty_mode(self, mc_hists: list[Histogram]) -> None:
        propagated = compare(mc_hists[1], mc_hists[0])
        split = compare(mc_hists[1], mc_hists[0], uncertainty="numerator")
        assert (propagated.uncertainty, split.uncertainty) == ("propagate", "numerator")
        bands = []
        for comparison, band in (
            (propagated, None),
            (split, None),
            (propagated, True),
            (split, False),
        ):
            fig, rax = plt.subplots()
            draw_panel([comparison], rax, band=band)
            bands.append(len([c for c in rax.collections if isinstance(c, PolyCollection)]))
            plt.close(fig)
        assert bands == [0, 1, 1, 0]  # the default follows the mode, band= overrides it

    def test_ratio_panel_numerator_band(
        self, mc_hists: list[Histogram], data_hist: Histogram
    ) -> None:
        fig, (ax, rax) = plt.subplots(2)
        result = compare(data_hist, mc_hists[0], uncertainty="numerator")
        draw_panel([result], rax, observed=[True], ylim=(0, 2), ylabel="custom")
        assert rax.get_ylabel() == "custom"
        assert rax.get_ylim() == (0.0, 2.0)
        assert len([c for c in rax.collections if isinstance(c, PolyCollection)]) == 1  # band
        (container,) = rax.containers
        assert container.lines[0].get_markersize() == 5  # observed data: larger markers
        assert container.lines[0].get_color() == foreground()
        plt.close(fig)

    def test_references_varied_differently_are_not_one_reference(
        self, mc_hists: list[Histogram]
    ) -> None:
        # the panel draws one band, so the references must come to the same band as well
        fig, ax = plt.subplots()
        reference = mc_hists[0]
        varied = reference.replace(variations={"s": (reference.hist * 1.5, None)})
        others = reference.replace(variations={"other": (reference.hist * 1.2, None)})
        numerator = mc_hists[1]
        against = [
            compare(numerator, h, uncertainty="numerator") for h in (varied, others, reference)
        ]
        for pair in ((against[0], against[1]), (against[0], against[2])):
            with pytest.raises(ValueError, match="one reference"):
                draw_panel(list(pair), ax)
        # the same nominal contents and the same variations: one band, so one reference
        copied = Histogram(reference.hist.copy(), label=reference.label).replace(
            variations={"s": (reference.hist * 1.5, None)}
        )
        draw_panel([against[0], compare(numerator, copied, uncertainty="numerator")], ax)
        plt.close(fig)

    def test_points_references_must_agree_beyond_their_label(
        self, mc_hists: list[Histogram]
    ) -> None:
        edges = np.array([0.0, 1.0, 2.0])

        def eff(values: list[float], label: str = "MC") -> Efficiency:
            return Efficiency(
                values=np.array(values),
                lower=np.array(values) - 0.1,
                upper=np.array(values) + 0.05,
                edges=edges,
                label=label,
            )

        numerator, mc = eff([0.5, 0.6], "Data"), eff([0.4, 0.5])
        fig, ax = plt.subplots()
        # two references labelled "MC" that differ: not one reference to draw
        with pytest.raises(ValueError, match="one reference"):
            draw_panel([compare(numerator, mc), compare(numerator, eff([0.3, 0.5]))], ax)
        # the same reference, or an equal copy of it: one reference
        draw_panel([compare(numerator, mc), compare(eff([0.45, 0.6]), mc)], ax)
        draw_panel([compare(numerator, mc), compare(numerator, eff([0.4, 0.5]))], ax)

        def profile_(statistic: Any) -> Profile:
            values = np.array([0.4, 0.5])
            return Profile(values, values / 10, np.ones(2), edges, statistic=statistic)

        std = Profile(np.ones(2), np.ones(2), np.ones(2), edges, statistic="std")
        same = [compare(std, profile_("std")), compare(std, profile_("std"))]
        draw_panel(same, ax)
        with pytest.raises(ValueError, match="one reference"):  # a profile is not an efficiency
            draw_panel([compare(numerator, mc), same[0]], ax)
        # a histogram never stands for points with the same label (asymmetries: no band either)
        histogram = compare(mc_hists[1], mc_hists[0].replace(label="MC"), kind="asymmetry")
        with pytest.raises(ValueError, match="one reference"):
            draw_panel([compare(numerator, mc, kind="asymmetry"), histogram], ax)
        plt.close(fig)

    def test_category_references_must_list_the_same_categories(self) -> None:
        # a category axis reports numeric index edges, which say nothing about its categories
        def categories(names: str, value: float) -> Any:
            h = hist.Hist(hist.axis.StrCategory(list(names)), storage=hist.storage.Weight())
            h.view().value, h.view().variance = value, value
            return h

        fig, ax = plt.subplots()
        first = compare(categories("ab", 4.0), categories("ab", 2.0))
        other = compare(categories("cd", 4.0), categories("cd", 2.0))
        assert first.edges.tolist() == other.edges.tolist()  # the same indices, other categories
        with pytest.raises(ValueError, match="one reference"):
            draw_panel([first, other], ax)
        draw_panel([first, compare(categories("ab", 6.0), categories("ab", 2.0))], ax)
        plt.close(fig)

    @pytest.mark.parametrize(
        ("kind", "baseline"), [("ratio", 1.0), ("relative_difference", 0.0), ("difference", 0.0)]
    )
    def test_baselines_and_bands(
        self, mc_hists: list[Histogram], data_hist: Histogram, kind: Any, baseline: float
    ) -> None:
        fig, (ax, rax) = plt.subplots(2)
        result = compare(data_hist, mc_hists[0], kind=kind, uncertainty="numerator")
        draw_panel([result], rax)
        assert rax.lines[0].get_ydata() == [baseline, baseline]
        assert rax.lines[0].get_linestyle() == "--"
        (band,) = [c for c in rax.collections if isinstance(c, PolyCollection)]
        vertices = band.get_paths()[0].vertices
        assert result.band is not None
        down, up = result.band
        assert vertices[:, 1].max() == pytest.approx(
            baseline + np.nanmax(up)
        )  # around the baseline
        assert vertices[:, 1].min() == pytest.approx(baseline - np.nanmax(down))
        (container,) = rax.containers
        drawn = container.lines[0].get_ydata()
        np.testing.assert_allclose(drawn, result.values[np.isfinite(result.values)])
        plt.close(fig)

    def test_pull_draws_filled_steps_without_error_bars(self) -> None:
        fig, ax = plt.subplots()
        pull = comparison([1.5, np.nan, -2.0, 0.5], kind="pull", errors=[1, np.nan, 1, 1])
        draw_panel([pull], ax, colors=["red"])
        (steps,) = [p for p in ax.patches if isinstance(p, StepPatch)]
        assert steps.get_fill()
        assert steps.get_alpha() == pytest.approx(0.6)
        assert to_hex(steps.get_facecolor()) == to_hex("red")
        np.testing.assert_allclose(steps.get_data().values, [1.5, 0.0, -2.0, 0.5])  # nan: empty
        np.testing.assert_allclose(steps.get_data().edges, pull.edges)
        assert steps.get_data().baseline == 0.0
        assert not ax.containers  # no error bars: a pull's uncertainty is 1
        assert ax.lines[0].get_ydata() == [0.0, 0.0]
        assert ax.get_ylabel() == "Pull"
        plt.close(fig)

    def test_significance_draws_points_without_baseline(self) -> None:
        fig, ax = plt.subplots()
        draw_panel([comparison([1.0, 2.0], kind="s/sqrt(b)", errors=[0.1, 0.2])], ax)
        assert not [line for line in ax.lines if line.get_linestyle() == "--"]  # no baseline
        assert len(ax.containers) == 1
        assert ax.get_ylabel() == r"$S/\sqrt{B}$"
        plt.close(fig)

    def test_one_kind_per_panel(self) -> None:
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="one kind"):
            draw_panel([comparison([1.0]), comparison([0.0], kind="pull")], ax)
        with pytest.raises(ValueError, match="at least one"):
            draw_panel([], ax)
        plt.close(fig)

    def test_one_reference_and_one_binning_per_panel(
        self, mc_hists: list[Histogram], data_hist: Histogram
    ) -> None:
        # the label and the band come from the first comparison, so the rest must share them
        fig, ax = plt.subplots()
        against_a = compare(mc_hists[1], mc_hists[0])
        with pytest.raises(ValueError, match="one reference"):
            draw_panel([against_a, compare(mc_hists[1], data_hist)], ax)
        # the reference is the histogram, not its label: plain hists carry none at all
        plain = [compare(mc_hists[1].hist, h.hist) for h in (mc_hists[0], data_hist)]
        assert [c.reference for c in plain] == ["", ""]
        with pytest.raises(ValueError, match="one reference"):
            draw_panel(plain, ax)
        with pytest.raises(ValueError, match="one reference"):  # and labels alone, without one
            draw_panel([comparison([1.0]), replace(comparison([1.0]), reference="other")], ax)
        # one reference under another name, or copied: the same band, so the panel is drawn
        renamed = Histogram(mc_hists[0].hist, label="Renamed")
        copied = Histogram(mc_hists[0].hist.copy(), label="A")
        draw_panel([against_a, compare(mc_hists[1], renamed), compare(mc_hists[1], copied)], ax)
        # comparing with one histogram gives one binning; hand-built ones are checked too
        with pytest.raises(ValueError, match="one binning"):
            draw_panel([comparison([1.0, 2.0]), comparison([1.0])], ax)
        plt.close(fig)

    @pytest.mark.parametrize(
        ("kind", "data", "expected"),
        [
            ("ratio", True, "Data / MC"),
            ("ratio", False, "Ratio to MC"),
            ("relative_difference", True, "(Data \N{MINUS SIGN} MC) / MC"),
            ("relative_difference", False, "Rel. difference to MC"),
            ("difference", True, "Data \N{MINUS SIGN} MC"),
            ("difference", False, "Difference to MC"),
            ("pull", True, "Pull"),
            ("pull", False, "Pull"),
            ("asymmetry", True, "(Data \N{MINUS SIGN} MC) / (Data + MC)"),
            ("asymmetry", False, "Asymmetry to MC"),
            ("s/sqrt(b)", False, r"$S/\sqrt{B}$"),
            ("s/sqrt(s+b)", False, r"$S/\sqrt{S+B}$"),
        ],
    )
    def test_default_labels(self, kind: Any, data: bool, expected: str) -> None:
        assert comparison_label(kind, "MC", data=data) == expected

    def test_ratio_ylim(self) -> None:
        tight = comparison([0.9, 1.0, 1.1, 1.0])
        assert panel_ylim([tight]) == (0.5, 1.5)
        wide = comparison([0.2, 2.0, np.nan, 50.0])
        low, high = panel_ylim([wide])
        assert low < 0.2
        assert high == 3.0
        assert panel_ylim([]) == (0.5, 1.5)
        syst = comparison(np.ones(4), syst_errors=(np.full(4, 0.8),) * 2)
        low, high = panel_ylim([syst])  # a propagated systematic without a band
        assert low < 0.2
        assert high > 1.8
        signed = comparison([-0.5, 1.0, 1.0, 1.0])
        low, high = panel_ylim([signed])
        assert low < -0.5
        assert high == 1.5
        deep = comparison(np.ones(4), syst_errors=(np.full(4, 1.5),) * 2)
        low, _ = panel_ylim([deep])  # a systematic reaching below zero on positive ratios
        assert low == 0.0
        low, _ = panel_ylim([wide], band=(np.full(4, 0.9), np.full(4, 1.1)))
        assert low < 0.2  # a narrow band never narrows the range
        wide_band = (np.full(4, 0.2), np.full(4, 1.8))
        low, high = panel_ylim([tight], band=wide_band)
        assert low < 0.2
        assert high > 1.8

    def test_relative_difference_ylim_is_the_ratio_range_shifted(self) -> None:
        kind: ComparisonKind = "relative_difference"
        assert panel_ylim([comparison([-0.1, 0.0, 0.1, 0.0], kind=kind)]) == (-0.5, 0.5)
        _, high = panel_ylim([comparison([-0.8, 1.0, np.nan, 49.0], kind=kind)])
        assert high == 2.0  # clipped like a ratio at 3
        low, _ = panel_ylim([comparison([-9.0, 0.0, 0.0, 0.0], kind=kind)])
        assert low == -4.0  # a negative ratio: clipped at -3 as a ratio
        low, _ = panel_ylim([comparison([-0.9, 0.0, 0.0, 0.0], kind=kind)])
        assert low >= -1.0  # positive ratios stay above -1
        for values in ([0.3, 0.0, 0.1, 0.2], [-0.5, 0.2, 0.4, 0.1]):
            shifted = panel_ylim([comparison(np.add(values, 1.0))])
            low, high = panel_ylim([comparison(values, kind=kind)])
            assert (low, high) == pytest.approx((shifted[0] - 1.0, shifted[1] - 1.0))

    def test_difference_ylim_is_symmetric(self) -> None:
        values = np.linspace(-2.0, 10.0, 100)
        low, high = panel_ylim([comparison(values, kind="difference")])
        assert low == -high
        assert high == pytest.approx(1.1 * np.percentile(values, 95))
        assert panel_ylim([comparison(np.zeros(4), kind="difference")]) == (-1.0, 1.0)
        with_syst = comparison(
            np.zeros(4), kind="difference", syst_errors=(np.full(4, 2.0), np.full(4, 3.0))
        )
        assert panel_ylim([with_syst]) == pytest.approx((-3.3, 3.3))
        band = (np.full(4, -4.0), np.full(4, 4.0))
        assert panel_ylim([comparison(np.zeros(4), kind="difference")], band=band) == (
            pytest.approx((-4.4, 4.4))
        )

    def test_pull_ylim(self) -> None:
        assert panel_ylim([comparison([0.5, -1.0, 1.0, 0.2], kind="pull")]) == (-3.0, 3.0)
        values = np.linspace(-4.0, 1.0, 101)
        low, high = panel_ylim([comparison(values, kind="pull")])
        assert high == pytest.approx(1.1 * np.percentile(np.abs(values), 95))
        assert low == -high
        assert panel_ylim([comparison([-40.0, 30.0, 50.0], kind="pull")]) == (-5.0, 5.0)
        assert panel_ylim([comparison([np.nan], kind="pull")]) == (-3.0, 3.0)

    def test_asymmetry_ylim_keeps_its_bounds_in_the_frame(self) -> None:
        values = np.linspace(-0.2, 0.3, 100)
        low, high = panel_ylim([comparison(values, kind="asymmetry")])
        assert low == -high
        assert high == pytest.approx(1.1 * np.percentile(values, 95))
        # an empty side is -1 or 1: shown, never beyond
        bounds = comparison([-1.0, 1.0, 1.0, -1.0], kind="asymmetry")
        assert panel_ylim([bounds]) == pytest.approx((-1.1, 1.1))

    def test_asymmetry_panel(self, mc_hists: list[Histogram]) -> None:
        fig, rax = plt.subplots()
        draw_panel([compare(mc_hists[1], mc_hists[0], kind="asymmetry")], rax)
        assert rax.get_ylabel() == "Asymmetry to A"
        assert rax.lines[0].get_ydata() == [0.0, 0.0]
        assert rax.lines[0].get_linestyle() == "--"
        assert not [c for c in rax.collections if isinstance(c, PolyCollection)]  # no band
        plt.close(fig)


def _vertical_bars(ax: Axes) -> list[list[tuple[float, float]]]:
    """``(low, high)`` of every vertical error bar, per errorbar container of ``ax``."""
    found = []
    for container in ax.containers:
        for collection in container.lines[2]:
            segments = collection.get_segments()
            if segments and all(np.isclose(seg[0][0], seg[1][0]) for seg in segments):
                found.append([(float(seg[0][1]), float(seg[1][1])) for seg in segments])
    return found


def _asymmetric(histogram: Histogram, down: list[float], up: list[float]) -> Histogram:
    """``histogram`` reporting the given ``(down, up)`` statistical errors."""
    pair = (np.asarray(down, dtype=float), np.asarray(up, dtype=float))
    object.__setattr__(histogram, "errors", lambda *, flow=False: pair)
    return histogram


class TestAsymmetricErrorBars:
    """``(down, up)`` statistical errors reach the drawn error bars as they are."""

    DOWN = [0.5, 0.25, 1.0, 0.75]
    UP = [1.0, 2.0, 3.0, 4.0]

    def test_data_points(self, data_hist: Histogram) -> None:
        fig, ax = plt.subplots()
        draw_histograms([_asymmetric(data_hist, self.DOWN, self.UP)], ax, style=Style())
        (bars,) = _vertical_bars(ax)
        values = data_hist.values()
        np.testing.assert_allclose(bars, np.c_[values - self.DOWN, values + self.UP])
        plt.close(fig)

    def test_overlay_error_bars(self, mc_hists: list[Histogram]) -> None:
        fig, ax = plt.subplots()
        overlay = _asymmetric(mc_hists[0], self.DOWN, self.UP)
        drawn = draw_histograms([overlay], ax, style=Style(), errorbars=True)
        (bars,) = _vertical_bars(ax)
        values = overlay.values()
        np.testing.assert_allclose(bars, np.c_[values - self.DOWN, values + self.UP])
        assert drawn.ymax == pytest.approx(max(values + self.UP))
        plt.close(fig)

    def test_stack_band(self) -> None:
        # statistical errors only: the band must still reach down and up by different amounts
        h = hist.Hist(hist.axis.Regular(4, 0, 4), storage=hist.storage.Weight())
        h.fill([0.5, 1.5, 1.5, 2.5, 3.5, 3.5, 3.5])
        down, up = np.array(self.DOWN), np.array(self.UP)
        parts = [
            Histogram(h, label=label, stat_errors=(down / np.sqrt(2), up / np.sqrt(2)))
            for label in ("A", "B")
        ]
        fig, ax = plt.subplots()
        drawn = draw_histograms(parts, ax, style=Style(), stack=True)
        (band,) = [patch for patch in ax.patches if patch.get_hatch() == "////"]
        top, _, bottom = band.get_data()
        total = 2 * h.values()
        np.testing.assert_allclose(top, total + up)  # in quadrature: the sides of one part x sqrt 2
        np.testing.assert_allclose(bottom, total - down)
        assert drawn.ymax == pytest.approx(max(total + up))
        plt.close(fig)

    def test_panel_points(self) -> None:
        fig, ax = plt.subplots()
        values = [1.0, 1.2, 0.8, 1.1]
        draw_panel([comparison(values, errors=(self.DOWN, self.UP))], ax, ylim=(-5.0, 9.0))
        (bars,) = _vertical_bars(ax)
        np.testing.assert_allclose(
            bars, np.c_[np.subtract(values, self.DOWN), np.add(values, self.UP)]
        )
        plt.close(fig)


def _off_scale(ax: Axes) -> dict[str, list[float]]:
    """The x of the off-scale markers at the top (``^``) and bottom (``v``), once drawn."""
    ax.figure.canvas.draw()
    marked: dict[str, list[float]] = {"^": [], "v": []}
    for line in ax.lines:
        if line.get_marker() in marked and line.get_linestyle() == "None":
            marked[str(line.get_marker())] += [float(x) for x in line.get_xdata()]
    return marked


class TestOffScaleMarkers:
    def test_points_beyond_the_range_are_marked_at_their_edge(self) -> None:
        fig, ax = plt.subplots()
        values = [0.5, 1.0, 5.0, -2.0, np.nan]
        draw_panel([comparison(values)], ax, ylim=(0.0, 2.0), colors=["red"])
        assert _off_scale(ax) == {"^": [2.5], "v": [3.5]}
        markers = [line for line in ax.lines if line.get_marker() in ("^", "v")]
        assert {to_hex(line.get_color()) for line in markers} == {"#ff0000"}
        assert not any(line.get_in_layout() for line in markers)
        # clipped to the axes (a rectangular clip path becomes a clip box): x outside is hidden
        assert all(line.get_clip_on() and line.get_clip_box() is not None for line in markers)
        ax.set_ylim(-3.0, 6.0)  # the markers follow the range
        assert _off_scale(ax) == {"^": [], "v": []}
        ax.set_ylim(0.8, 1.2)
        assert _off_scale(ax) == {"^": [2.5], "v": [0.5, 3.5]}
        plt.close(fig)

    def test_the_automatic_range_marks_what_it_leaves_out(self) -> None:
        fig, ax = plt.subplots()
        draw_panel([comparison([1.0, 1.1, 0.9, 1.0, 50.0])], ax)
        assert ax.get_ylim()[1] == 3.0
        assert _off_scale(ax) == {"^": [4.5], "v": []}
        plt.close(fig)

    def test_nothing_inside_the_range_and_no_pull_is_marked(
        self, mc_hists: list[Histogram]
    ) -> None:
        fig, (ax, pull_ax) = plt.subplots(2)
        draw_panel([compare(mc_hists[1], mc_hists[0])], ax, ylim=(-100.0, 100.0))
        assert _off_scale(ax) == {"^": [], "v": []}
        draw_panel([comparison([0.0, 9.0], kind="pull")], pull_ax)  # a bar ends at the edge
        assert _off_scale(pull_ax) == {"^": [], "v": []}
        plt.close(fig)


class TestAnnotations:
    @pytest.mark.parametrize(
        ("stack", "expected"),
        [(True, ["Data", "B", "A", "Stat. unc."]), (["A"], ["Data", "A", "Stat. unc.", "B"])],
    )
    def test_legend_order(
        self,
        mc_hists: list[Histogram],
        data_hist: Histogram,
        stack: StackSpec,
        expected: list[str],
    ) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            draw_histograms([*mc_hists, data_hist], ax, style=st, stack=stack)
            legend = add_legend(ax, st)
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == expected

    def test_legend_disabled_or_empty(self) -> None:
        fig, ax = plt.subplots()
        assert add_legend(ax, Style(legend=False)) is None
        assert add_legend(ax, Style()) is None
        ax.plot([0, 1], label="x")
        legend = add_legend(ax, Style(legend="lower left", legend_kwargs={"ncol": 2}))
        assert legend is not None
        assert legend._loc == 3  # type: ignore[attr-defined]  # lower left

    def test_stats_box(self, mc_hists: list[Histogram]) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms(mc_hists, ax, style=st)
            legend = add_legend(ax, st, loc="upper right")
            texts = add_stats_box(ax, mc_hists, colors=drawn.colors, legend=legend)
        assert len(texts) == 2
        assert texts[0].get_color() == DEFAULT_COLORS[0]
        assert "A" in texts[0].get_text()
        assert "$\\mu$" in texts[0].get_text()
        # second block sits below the first, and both below the legend
        assert texts[1].get_position()[1] < texts[0].get_position()[1] < 0.96

    @pytest.mark.parametrize("loc", ["upper left", "lower right", "center right", "lower center"])
    def test_stats_box_locations(self, mc_hists: list[Histogram], loc: str) -> None:
        fig, ax = plt.subplots()
        texts = add_stats_box(ax, mc_hists, loc=loc)
        assert len(texts) == 2

    def test_stats_box_edge_cases(self, mc_hists: list[Histogram]) -> None:
        fig, ax = plt.subplots()
        assert add_stats_box(ax, [Histogram(mc_hists[0].hist, label="no stats")]) == []
        with pytest.raises(ValueError, match="location"):
            add_stats_box(ax, mc_hists, loc="nowhere")
        assert len(add_stats_box(ax, mc_hists[:1], include_entries=False)) == 1
        with pytest.raises(ValueError, match="1 colours for 2 histograms"):
            add_stats_box(ax, mc_hists, colors=["red"])
        duplicate = [h.replace(label="same") for h in mc_hists]
        texts = add_stats_box(ax, duplicate, colors=["red", "blue"])
        assert [text.get_color() for text in texts] == ["red", "blue"]

    def test_add_text(self) -> None:
        fig, ax = plt.subplots()
        add_text(ax, ["a", "b"], loc="lower left")
        add_text(ax, "c")
        assert len(ax.artists) == 2


class TestOtherDrawing:
    def test_hist2d(self) -> None:
        cols = Columns(
            arrays=(np.array([0.5, 1.5, 1.5]), np.array([0.5, 0.5, 2.5])),
            weights=None,
            n_events=3,
            n_selected_events=3,
        )
        h2 = Histogram(
            fill(
                [hist.axis.Regular(2, 0, 2, label="x"), hist.axis.Regular(3, 0, 3, label="y")], cols
            ),
            label="h",
        )
        fig, ax = plt.subplots()
        draw_hist2d(h2, ax, logz=True, zlabel="N")
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert len(fig.axes) == 2  # colour bar added
        fig, ax = plt.subplots()
        draw_hist2d(h2, ax, colorbar=False)
        assert len(fig.axes) == 1
        empty = Histogram(
            fill(
                [hist.axis.Regular(2, 0, 2), hist.axis.Regular(3, 0, 3)],
                Columns((np.array([]), np.array([])), None, 0, 0),
            ),
            label="e",
        )
        fig, ax = plt.subplots()
        with pytest.warns(RootfigWarning, match="no positive bins"):
            draw_hist2d(empty, ax, logz=True)  # falls back to a linear colour scale
        with pytest.raises(ValueError, match="two-dimensional"):
            draw_hist2d(make_hist([1.0], label="1d"), ax)

    def test_correlation(self) -> None:
        matrix = np.array([[1.0, -0.8], [-0.8, 1.0]])
        fig, ax = plt.subplots()
        draw_correlation(matrix, ["a", "b"], ax, percent=True)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
        assert any("%" in t.get_text() for t in ax.texts)
        fig, ax = plt.subplots()
        draw_correlation(
            np.array([[1.0, np.nan], [np.nan, np.nan]]),
            ["a", "b"],
            ax,
            annotate=True,
            colorbar=False,
        )
        assert len(ax.texts) == 1
        with pytest.raises(ValueError, match="square"):
            draw_correlation(np.zeros((2, 3)), ["a", "b"], ax)
        with pytest.raises(ValueError, match="labels"):
            draw_correlation(matrix, ["a"], ax)


class TestPlotResult:
    def test_save_variants(self, tmp_path: object, mc_hists: list[Histogram]) -> None:
        from pathlib import Path

        from rootfig.model import Variable

        directory = Path(str(tmp_path))
        fig, ax = plt.subplots()
        plot = Plot(fig=fig, ax=ax, histograms=mc_hists, variable=Variable("Muon_pt / 1000"))
        assert plot.axes == (ax,)
        assert plot.hists == [h.hist for h in mc_hists]
        assert "A" in repr(plot)
        written = plot.save(directory / "sub" / "out.png")
        assert written == [directory / "sub" / "out.png"]
        assert written[0].exists()
        # tight bounding box by default: the file is smaller than the nominal canvas
        from PIL import Image

        with Image.open(written[0]) as image:
            nominal = fig.get_size_inches() * fig.dpi
            assert image.size[0] < nominal[0]
        plot.save(directory / "loose.png", bbox_inches=None, dpi=fig.dpi)
        with Image.open(directory / "loose.png") as image:
            assert image.size[0] == int(nominal[0])
        written = plot.save(directory, formats=["pdf", "svg"])
        assert [p.name for p in written] == ["Muon_pt_1000.pdf", "Muon_pt_1000.svg"]
        assert all(p.exists() for p in written)
        written = plot.save(f"{directory}/")
        assert written[0].name == "Muon_pt_1000.pdf"
        plot.close()
        assert not plt.fignum_exists(fig.number)
        unnamed = Plot(fig=plt.figure(), ax=plt.gca())
        assert unnamed.save(directory)[0].name == "plot.pdf"


class TestFlowBins:
    def test_show_flow_bins_consistent(self) -> None:
        with_over = make_hist([0.5, 1.5, 9.0, 9.0], label="A")  # two entries overflow
        without = make_hist([0.5, 1.5, 2.5], label="B")
        shown, (under, over) = show_flow_bins([with_over, without])
        assert (under, over) == (False, True)
        assert len(shown[0].edges) == len(shown[1].edges) == 6
        np.testing.assert_allclose(shown[0].edges[-2:], [4.0, 5.0])  # mean width, not 5 % range
        np.testing.assert_allclose(shown[0].values(), [1, 1, 0, 0, 2])
        np.testing.assert_allclose(shown[1].values(), [1, 1, 1, 0, 0])
        assert shown[0].overflow == 0.0  # moved into the visible bin
        assert shown[0].label == with_over.label

    def test_show_flow_bins_keeps_axis_name(self) -> None:
        cols = Columns((np.array([-1.0, 5.0]),), None, 2, 2)
        h = Histogram(fill([hist.axis.Regular(4, 0, 4, name="met", label="MET")], cols), label="A")
        [shown], _ = show_flow_bins([h])
        assert (shown.axis.name, shown.axis.label) == ("met", "MET")
        assert shown.hist[{"met": slice(None)}].values().size == 6  # name-based indexing works

    def test_show_flow_bins_noop_and_underflow(self) -> None:
        plain = make_hist([0.5, 1.5], label="A")
        same, flags = show_flow_bins([plain])
        assert same == [plain]
        assert flags == (False, False)
        assert show_flow_bins([]) == ([], (False, False))
        both, flags = show_flow_bins([make_hist([-1.0, 5.0], label="A")])
        assert flags == (True, True)
        assert len(both[0].edges) == 7
        np.testing.assert_allclose(both[0].values(), [1, 0, 0, 0, 0, 1])

    def test_category_axis_with_unlisted_entries_is_refused(self) -> None:
        axis = hist.axis.StrCategory(["a", "b"], name="cut")
        listed = Histogram(hist.Hist(axis, storage=hist.storage.Weight()).fill(["a"]), label="L")
        unlisted = Histogram(
            hist.Hist(axis, storage=hist.storage.Weight()).fill(["a", "zzz"]), label="U"
        )
        assert show_flow_bins([listed])[0] == [listed]
        assert fold_flow_bins([listed]) == [listed]
        with pytest.raises(BinningError, match=r"flow='show' needs numeric bins.*'U'"):
            show_flow_bins([unlisted])
        with pytest.raises(BinningError, match="flow='sum' needs numeric bins"):
            fold_flow_bins([unlisted])

    def test_label_flow_bins(self) -> None:
        fig, ax = plt.subplots()
        edges = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ax.set_xlim(-1, 5)
        label_flow_bins(ax, edges, under=True, over=True)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels[0] == "<0"
        assert labels[-1] == ">4"
        assert "0" not in labels  # boundary ticks removed
        assert "4" not in labels
        plt.close(fig)

    def test_stack_with_mixed_overflow_draws(self) -> None:
        """mplhep alone fails on this combination (per-histogram flow bins)."""
        with_over = make_hist([0.5, 1.5, 9.0], label="A")
        without = make_hist([0.5, 1.5, 2.5], label="B")
        fig, ax = plt.subplots()
        drawn = draw_histograms([with_over, without], ax, style=Style(), stack=True, flow="show")
        assert drawn.labels[:2] == ["A", "B"]
        assert ">4" in [t.get_text() for t in ax.get_xticklabels()]
        plt.close(fig)

    def test_envelope(self) -> None:
        a = make_hist([0.5, 1.5, 1.5], label="A")
        b = make_hist([2.5, 2.5, 2.5, 3.5], label="B", is_data=True)
        edges, heights = envelope([a, b], stack=False)
        np.testing.assert_allclose(edges, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(heights, [2, 2 + np.sqrt(2), 3 + np.sqrt(3), 2])
        _, stacked = envelope([a, a], stack=True)
        np.testing.assert_allclose(stacked, [2 + np.sqrt(2), 4 + 2, 0, 0])
        _, partial_stack = envelope([a, a, b.replace(is_data=False)], stack=["A"])
        np.testing.assert_allclose(partial_stack, [2 + np.sqrt(2), 6, 3 + np.sqrt(3), 2])


class TestHeadroom:
    @staticmethod
    def _axes(*, logy: bool = False) -> tuple[Figure, Axes, np.ndarray, np.ndarray]:
        fig, ax = plt.subplots(figsize=(6, 4))
        edges = np.linspace(0, 10, 11)
        heights = np.full(10, 100.0)
        ax.stairs(heights, edges, fill=True, label="flat")
        for i in range(6):
            ax.plot([], [], label=f"entry {i}")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1, 145)
        else:
            ax.set_ylim(0, 120)  # the usual 1.20 headroom
        ax.set_xlim(0, 10)
        return fig, ax, edges, heights

    def test_anchored_legend_raises_top(self) -> None:
        fig, ax, edges, heights = self._axes()
        legend = ax.legend(loc="upper right")
        raise_ylim_above(
            [ax], overlay_artists(ax, legend), edges=edges, heights=heights, logy=False
        )
        assert ax.get_ylim()[1] > 120
        # the legend bottom now clears the histogram
        fig.canvas.draw()
        bottom = ax.transData.inverted().transform(legend.get_window_extent().get_points())[0, 1]
        assert bottom >= 100
        plt.close(fig)

    def test_low_legend_and_free_legend_untouched(self) -> None:
        fig, ax, edges, heights = self._axes()
        legend = ax.legend(loc="lower left")
        raise_ylim_above([ax], [legend], edges=edges, heights=heights, logy=False)
        assert ax.get_ylim()[1] == 120
        heights_low = np.full(10, 10.0)
        legend = ax.legend(loc="upper right")
        raise_ylim_above([ax], [legend], edges=edges, heights=heights_low, logy=False)
        assert ax.get_ylim()[1] == 120
        plt.close(fig)

    def test_floating_legend_and_log(self) -> None:
        fig, ax, edges, heights = self._axes(logy=True)
        legend = ax.legend(loc="best")
        raise_ylim_above([ax], [], edges=edges, heights=heights, logy=True, floating=[legend])
        assert ax.get_ylim()[1] > 145
        # a peak on one side only: the other corner is free, nothing to do
        fig2, ax2, edges2, _ = self._axes()
        peaked = np.r_[np.full(5, 140.0), np.zeros(5)]
        legend2 = ax2.legend(loc="best")
        raise_ylim_above([ax2], [], edges=edges2, heights=peaked, logy=False, floating=[legend2])
        assert ax2.get_ylim()[1] == 120
        plt.close(fig)
        plt.close(fig2)

    def test_label_resting_on_the_frame_is_not_an_obstacle(self) -> None:
        # a label above the frame sits on its top edge, up to rounding either way; the
        # error bars of the tallest bins can reach into the margin it would claim
        fig, ax, edges, _ = self._axes()
        heights = np.full(10, 115.0)
        for y in (1.0, 1.0 - 1e-15):
            text = ax.text(0.0, y, "CMS", transform=ax.transAxes, va="bottom")
            raise_ylim_above([ax], [text], edges=edges, heights=heights, logy=False)
            assert ax.get_ylim()[1] == 120
            text.remove()
        plt.close(fig)

    def test_degenerate_inputs(self) -> None:
        fig, ax, edges, heights = self._axes()
        raise_ylim_above([], [], edges=edges, heights=heights, logy=False)
        raise_ylim_above([ax], [], edges=edges, heights=np.array([]), logy=False)
        raise_ylim_above([ax], [ax.legend()], edges=edges, heights=heights * np.nan, logy=False)
        assert ax.get_ylim()[1] == 120
        plt.close(fig)


class TestCorrelationFormat:
    def test_percent_has_no_decimals_or_negative_zero(self) -> None:
        matrix = np.array([[1.0, -0.001], [-0.001, 1.0]])
        fig, ax = plt.subplots()
        draw_correlation(matrix, ["a", "b"], ax, percent=True)
        texts = {t.get_text() for t in ax.texts}
        assert "100%" in texts
        assert "0%" in texts
        assert not any(t.startswith("-") for t in texts)
        fig, ax = plt.subplots()
        draw_correlation(matrix, ["a", "b"], ax)
        assert "1.00" in {t.get_text() for t in ax.texts}
        plt.close("all")

    def test_figsize_grows_with_variables_and_the_style(self) -> None:
        with style_context():
            assert correlation_figsize(2) == pytest.approx((4.5 * 1.15, 4.5))
            assert correlation_figsize(5) == pytest.approx((6.25 * 1.15, 6.25))
        with style_context("CMS"):
            scale = matplotlib.rcParams["figure.figsize"][0] / 7.0
            assert scale > 1
            assert correlation_figsize(5) == pytest.approx((6.25 * 1.15 * scale, 6.25 * scale))


class TestExperimentLabelEnergy:
    def test_energy_only_when_given(self) -> None:
        fig, ax = plt.subplots()
        add_experiment_label(ax, Style(experiment="CMS"), has_data=False)
        assert not any("TeV" in t.get_text() for t in ax.texts)
        fig, ax = plt.subplots()
        add_experiment_label(ax, Style(experiment="CMS", com=13.6), has_data=False)
        assert any("13.6" in t.get_text() for t in ax.texts)
        plt.close("all")


class TestLabelUnits:
    def _texts(self, style: Style, *, has_data: bool = False) -> list[str]:
        fig, ax = plt.subplots()
        add_experiment_label(ax, style, has_data=has_data)
        texts = [t.get_text() for t in ax.texts]
        texts += [a.txt.get_text() for a in ax.artists if hasattr(a, "txt")]
        plt.close(fig)
        return texts

    def test_lepton_collider_units(self) -> None:
        texts = self._texts(
            Style(experiment="FCC-ee", status="Simulation", com="240 GeV", lumi="10.8 ab^-1")
        )
        joined = " ".join(texts)
        assert "ab^{-1}" in joined
        assert "240 GeV" in joined
        assert "fb" not in joined
        assert "TeV" not in joined
        assert joined.count("Simulation") == 1
        atlas = " ".join(self._texts(Style(experiment="ATLAS", com="240 GeV", lumi="10.8 ab")))
        assert "\\sqrt{s}" in atlas
        assert "ab^{-1}" in atlas
        cms = " ".join(
            self._texts(
                Style(experiment="CMS", lumi=3, lumi_unit="ab^{-1}", com=91, com_unit="GeV")
            )
        )
        assert "3 $\\mathrm{ab^{-1}}$ (91 GeV)" in cms

    def test_default_units_and_neutral_label(self) -> None:
        texts = " ".join(
            self._texts(Style(experiment="ATLAS", status="Internal", com=13.6, lumi=140))
        )
        assert "13.6" in texts
        assert "TeV" in texts
        assert "fb^{-1}" in texts
        neutral = " ".join(
            self._texts(Style(status="Simulation Preliminary", com="240 GeV", lumi="10.8 ab"))
        )
        assert neutral.count("Simulation") == 1
        assert "Preliminary" in neutral
        assert "240$ GeV" in neutral
        assert "ab^{-1}" in neutral
        # "Simulation" comes from the data flag, not the status
        assert "Simulation" in " ".join(self._texts(Style(status="Internal", com=13)))
        assert "Simulation" not in " ".join(
            self._texts(Style(status="Internal", com=13), has_data=True)
        )


class TestSignificancePanelAndPoints:
    def test_significance_panel(self) -> None:
        result = comparison([1.0, 2.0, np.nan], kind="s/sqrt(b)", errors=[0.1, 0.2, np.nan])
        fig, ax = plt.subplots()
        draw_panel([result], ax)
        assert ax.get_ylabel() == r"$S/\sqrt{B}$"
        assert ax.get_ylim() == (0.0, pytest.approx(1.25 * 2.2))
        assert ax.get_xlim() == (0.0, 3.0)
        large = comparison(result.values * 3, kind="s/sqrt(b)", errors=result.errors[1] * 2)
        draw_panel([result, large], ax, colors=["red", "blue"])
        assert ax.get_ylim() == (0.0, pytest.approx(1.25 * 6.4))
        assert [c.lines[0].get_color() for c in ax.containers[-2:]] == ["red", "blue"]
        other = comparison([1.0, 2.0, np.nan], kind="s/sqrt(s+b)", errors=[0.1, 0.2, np.nan])
        draw_panel([other], ax, ylim=(0, 5), ylabel="Z")
        assert ax.get_ylabel() == "Z"
        assert ax.get_ylim() == (0.0, 5.0)
        plt.close(fig)

    def test_points(self) -> None:
        from rootfig.histograms import Efficiency, Profile
        from rootfig.plotting import draw_efficiencies, draw_profiles

        eff = Efficiency(
            values=np.array([0.5, np.nan]),
            lower=np.array([0.4, np.nan]),
            upper=np.array([0.6, np.nan]),
            edges=np.array([0.0, 1.0, 2.0]),
            label="e",
        )
        fig, ax = plt.subplots()
        assert draw_efficiencies([eff], ax, style=Style()) == (0.0, 1.0)
        assert ax.get_legend_handles_labels()[1] == ["e"]
        prof = Profile(
            values=np.array([2.0, np.nan]),
            errors=np.array([0.5, np.nan]),
            counts=np.array([3.0, 0.0]),
            edges=np.array([0.0, 1.0, 2.0]),
        )
        assert draw_profiles([prof], ax, style=Style()) == (1.5, 2.5)
        empty = Profile(
            values=np.array([np.nan]),
            errors=np.array([np.nan]),
            counts=np.array([0.0]),
            edges=np.array([0.0, 1.0]),
        )
        assert draw_profiles([empty], ax, style=Style()) == (0.0, 1.0)
        plt.close(fig)


class TestFlowTransformations:
    def test_fold_flow_bins_moves_values_and_variances(self) -> None:
        from rootfig.plotting import fold_flow_bins

        h = make_hist([-1.0, 0.5, 9.0, 9.0], [2.0, 1.0, 1.0, 3.0], label="A")
        [folded] = fold_flow_bins([h])
        np.testing.assert_allclose(folded.values(), [3.0, 0.0, 0.0, 4.0])
        np.testing.assert_allclose(folded.variances(), [5.0, 0.0, 0.0, 10.0])
        assert folded.underflow == folded.overflow == 0.0
        np.testing.assert_allclose(h.values(), [1.0, 0.0, 0.0, 0.0])  # original untouched
        plain = make_hist([0.5], label="B")
        assert fold_flow_bins([plain]) == [plain]

    def test_fold_and_show_respect_axes_without_flow(self) -> None:
        from rootfig.plotting import fold_flow_bins

        axis = hist.axis.Regular(2, 0, 4, underflow=False, overflow=False)
        h = Histogram(hist.Hist(axis, storage=hist.storage.Weight()).fill([1.0, 3.0]), label="A")
        assert (h.underflow, h.overflow) == (0.0, 0.0)
        shown, flags = show_flow_bins([h])
        assert flags == (False, False)
        np.testing.assert_allclose(shown[0].values(), [1.0, 1.0])
        np.testing.assert_allclose(fold_flow_bins([h])[0].values(), [1.0, 1.0])

    def test_show_negative_and_cancelling_overflow(self) -> None:
        negative = make_hist([0.5, 9.0], [1.0, -2.0], label="A")
        shown, flags = show_flow_bins([negative])
        assert flags == (False, True)
        np.testing.assert_allclose(shown[0].values(), [1.0, 0.0, 0.0, 0.0, -2.0])
        cancelling = make_hist([0.5, 9.0, 9.0], [1.0, 1.0, -1.0], label="B")
        shown, flags = show_flow_bins([cancelling])
        assert flags == (False, True)
        assert shown[0].variances()[-1] == 2.0

    def test_flow_cell_with_given_errors_only(self) -> None:
        """An empty underflow cell with an error of its own is shown and folded with it."""
        from rootfig.plotting import fold_flow_bins

        empty = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Weight())
        empty.view(flow=True).value = [0.0, 4.0, 1.0, 0.0]
        errors = (np.array([3.0, 1.0, 1.0, 0.0]), np.array([4.0, 2.0, 1.0, 0.0]))
        h = Histogram(empty, label="A", stat_errors=errors)
        [shown], flags = show_flow_bins([h])
        assert flags == (True, False)
        np.testing.assert_allclose(shown.values(), [0.0, 4.0, 1.0])
        np.testing.assert_allclose(shown.errors()[0], [3.0, 1.0, 1.0])
        np.testing.assert_allclose(shown.errors()[1], [4.0, 2.0, 1.0])
        [folded] = fold_flow_bins([h])
        np.testing.assert_allclose(folded.values(), [4.0, 1.0])
        np.testing.assert_allclose(folded.errors()[0], [np.hypot(3.0, 1.0), 1.0])
        np.testing.assert_allclose(folded.errors()[1], [np.hypot(4.0, 2.0), 1.0])
        np.testing.assert_allclose(folded.errors(flow=True)[1][0], 0.0)

    def test_show_needs_identical_binning(self) -> None:
        from rootfig.errors import BinningError

        other = Histogram(
            fill([hist.axis.Regular(2, 0, 4)], Columns((np.array([9.0]),), None, 1, 1)), label="B"
        )
        with pytest.raises(BinningError, match="identical bin edges"):
            show_flow_bins([make_hist([9.0], label="A"), other])
        fig, ax = plt.subplots()
        with pytest.raises(BinningError, match="a stack"):
            draw_histograms([make_hist([0.5], label="A"), other], ax, style=Style(), stack=True)
        plt.close(fig)

    def test_draw_histograms_folds_for_sum(self) -> None:
        fig, ax = plt.subplots()
        drawn = draw_histograms([make_hist([9.0] * 5, label="A")], ax, style=Style(), flow="sum")
        assert drawn.ymax == 5.0
        assert drawn.colors == color_cycle(1, Style())
        plt.close(fig)


class TestEnvelopeAndColors:
    def test_envelope_with_different_binnings(self) -> None:
        coarse = make_hist([0.5, 0.5], label="A")
        fine = Histogram(
            fill([hist.axis.Regular(8, 0, 4)], Columns((np.array([0.75, 3.9]),), None, 2, 2)),
            label="B",
        )
        edges, heights = envelope([coarse, fine], stack=False)
        np.testing.assert_allclose(edges, np.linspace(0, 4, 9))
        assert heights[0] == pytest.approx(2 + np.sqrt(2))  # coarse bin 0 dominates
        assert heights[1] == pytest.approx(2 + np.sqrt(2))
        assert heights[-1] == pytest.approx(2.0)  # only the fine histogram is there

    def test_panel_reuses_main_colors(self) -> None:
        from matplotlib.colors import to_rgba

        hists = [
            make_hist([0.5], label="A"),
            make_hist([0.5], label="B"),
            make_hist([0.5], label="C"),
        ]
        fig, (ax, panel_ax) = plt.subplots(2)
        drawn = draw_histograms(hists, ax, style=Style())
        assert drawn.colors == color_cycle(3, Style())
        draw_panel([compare(h, hists[0]) for h in hists[1:]], panel_ax, colors=drawn.colors[1:])
        drawn_colors = [
            to_rgba(c.lines[0].get_color())
            for c in panel_ax.containers
            if isinstance(c, matplotlib.container.ErrorbarContainer)
        ]
        assert drawn_colors == [to_rgba(c) for c in drawn.colors[1:]]
        plt.close(fig)


class TestHist2DMask:
    def test_negative_bins_stay_visible_and_empty_bins_are_blank(self) -> None:
        h = hist.Hist(
            hist.axis.Regular(3, 0, 3), hist.axis.Regular(3, 0, 3), storage=hist.storage.Weight()
        )
        h.fill([0.5, 1.5, 2.5], [0.5, 1.5, 2.5], weight=[2.0, -1.0, 2.0])
        fig, ax = plt.subplots()
        draw_hist2d(Histogram(h, label="h"), ax, colorbar=False)
        drawn = ax.collections[0].get_array()
        mask = np.ma.getmaskarray(drawn)
        assert not mask[1, 1]
        assert drawn[1, 1] == -1.0
        assert mask[0, 1]
        assert mask[2, 0]
        plt.close(fig)
        fig, ax = plt.subplots()
        draw_hist2d(Histogram(h, label="h"), ax, colorbar=False, logz=True)
        assert np.ma.getmaskarray(ax.collections[0].get_array()).sum() == 0  # LogNorm masks
        plt.close(fig)


def with_variation(histogram: Histogram, factor: float) -> Histogram:
    return histogram.replace(variations={"s": (histogram.hist * factor, None)})


class TestSystematicDrawing:
    def test_stack_band_includes_systematics(
        self, mc_hists: list[Histogram], data_hist: Histogram
    ) -> None:
        varied = [with_variation(mc_hists[0], 1.5), mc_hists[1]]
        with style_context() as st:
            fig, ax = plt.subplots()
            plain = draw_histograms([*mc_hists, data_hist], ax, style=st, stack=True)
            fig, ax = plt.subplots()
            drawn = draw_histograms([*varied, data_hist], ax, style=st, stack=True)
        assert drawn.labels == ["A", "B", "Stat. + syst. unc.", "Data"]
        assert drawn.ymax > plain.ymax
        _, ax = plt.subplots()
        partial_stack = draw_histograms(varied, ax, style=Style(), stack=["B"])
        assert partial_stack.labels == ["B", "Stat. unc.", "A"]
        assert partial_stack.stack is not None
        assert partial_stack.stack.variations == {}
        bands = [
            artist
            for artist in partial_stack.artists
            if isinstance(artist, StepPatch) and artist.get_label() not in ("A", "B")
        ]
        assert len(bands) == 2  # stack uncertainty and the overlay's own light band
        for band, histogram in zip(bands, [varied[1], varied[0]], strict=True):
            expected = uncertainty(histogram)
            bounds = band.get_data()
            np.testing.assert_allclose(bounds.edges, histogram.edges)
            np.testing.assert_allclose(bounds.values, histogram.values() + expected.total_up)
            np.testing.assert_allclose(bounds.baseline, histogram.values() - expected.total_down)

    def test_overlay_band_and_envelope(self, mc_hists: list[Histogram]) -> None:
        varied = [mc_hists[0], with_variation(mc_hists[1], 2.0)]
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms(varied, ax, style=st)
        assert drawn.labels == ["A", "B"]
        assert len(ax.get_legend_handles_labels()[1]) == 2  # the band has no legend entry
        assert drawn.ymax >= 2.0 * mc_hists[1].values().max()
        _, heights = envelope(varied, stack=False)
        _, plain = envelope(mc_hists, stack=False)
        assert heights.max() > plain.max()

    def test_flow_bins_carry_variations(self) -> None:
        nominal = make_hist([1.5, 5.0], label="A")  # 5.0 is overflow
        up = make_hist([-1.0, 1.5], label="up")  # underflow only in the variation
        varied = nominal.replace(variations={"s": (up.hist, None)})
        (shown,), (under, over) = show_flow_bins([varied])
        assert (under, over) == (True, True)
        np.testing.assert_allclose(shown.variations["s"][0].values(), [1, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(shown.values(), [0, 0, 1, 0, 0, 1])
        (folded,) = fold_flow_bins([varied])
        np.testing.assert_allclose(folded.variations["s"][0].values(), [1, 1, 0, 0])
        np.testing.assert_allclose(folded.variations["s"][1].values(), [-1, 1, 0, 2])

    def test_ratio_panel_band_and_errors(
        self, mc_hists: list[Histogram], data_hist: Histogram
    ) -> None:
        reference = with_variation(mc_hists[0], 1.5)
        fig, ax = plt.subplots()
        result = compare(with_variation(mc_hists[1], 1.2), reference, uncertainty="numerator")
        draw_panel([result], ax)
        assert result.syst_band is not None
        assert result.syst_errors is not None
        band = next(c for c in ax.collections if isinstance(c, PolyCollection))
        assert band.get_label() == "A stat. + syst. unc."


class TestFinishing:
    """What is measured against the laid-out figure runs after drawing, once per figure."""

    @pytest.mark.parametrize("count", [1, 4])
    def test_headroom_survives_a_new_y_axis_offset(self, count: int) -> None:
        fig, axes = plt.subplots(1, count, figsize=(6 * count, 4), layout="constrained")
        plots = []
        legends = []
        for ax in np.atleast_1d(axes):
            for i in range(4):
                ax.plot([0, 1], [8000, 8000], label=f"entry {i}")
            ax.set_ylim(0, 9600)
            ax.ticklabel_format(axis="y", scilimits=(-3, 4))
            legend = ax.legend(loc="upper right")
            legends.append(legend)
            plots.append(
                Finish(
                    ax,
                    headroom=partial(
                        raise_ylim_above,
                        [ax],
                        [legend],
                        edges=np.array([0, 1]),
                        heights=np.array([8000]),
                        logy=False,
                    ),
                )
            )
        fig.canvas.draw()
        assert all(not plot.main.yaxis.get_offset_text().get_text() for plot in plots)
        draws: list[object] = []
        fig.canvas.mpl_connect("draw_event", draws.append)
        with finishing_together(fig):
            for plot in plots:
                finish_figure(fig, [plot])
        assert len(draws) == 2  # shared layout passes, independent of the number of plots
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for plot, legend in zip(plots, legends, strict=True):
            ax = plot.main
            assert ax.yaxis.get_offset_text().get_text() == "1e4"
            box = legend.get_window_extent(renderer)
            bottom = ax.transData.inverted().transform((box.x0, box.y0))[1]
            assert bottom >= 1.08 * 8000 - 1e-6

    def test_plots_drawn_together_are_finished_at_the_end_of_the_block(self) -> None:
        fig = plt.figure()
        calls: list[int] = []
        first = Finish(fig.add_subplot(1, 2, 1), headroom=lambda: calls.append(1))
        second = Finish(fig.add_subplot(1, 2, 2), headroom=lambda: calls.append(2))
        with finishing_together(fig):
            finish_figure(fig, [first])
            finish_figure(fig, [second])
            assert calls == []
        assert calls == [1, 2]
        finish_figure(fig, [first])  # collected only within the block
        assert calls == [1, 2, 1]
        plt.close(fig)

    def test_a_block_that_raises_finishes_nothing(self) -> None:
        fig = plt.figure()
        calls: list[int] = []
        finish = Finish(fig.add_subplot(), headroom=lambda: calls.append(1))

        def failing_page() -> None:
            with finishing_together(fig):
                finish_figure(fig, [finish])
                raise RuntimeError("drawing failed")

        with pytest.raises(RuntimeError, match="drawing failed"):
            failing_page()
        assert calls == []
        finish_figure(fig, [finish])
        assert calls == [1]
        plt.close(fig)

    def test_nested_blocks_finish_at_the_outermost_exit(self) -> None:
        fig, ax = plt.subplots()
        calls: list[int] = []
        with finishing_together(fig):
            finish_figure(fig, [Finish(ax, headroom=lambda: calls.append(1))])
            with finishing_together(fig):
                finish_figure(fig, [Finish(ax, headroom=lambda: calls.append(2))])
            assert calls == []
            finish_figure(fig, [Finish(ax, headroom=lambda: calls.append(3))])
        assert calls == [1, 2, 3]

    def test_failed_nested_block_preserves_outer_work(self) -> None:
        fig, ax = plt.subplots()
        calls: list[int] = []

        def failing_block() -> None:
            with finishing_together(fig):
                finish_figure(fig, [Finish(ax, headroom=lambda: calls.append(2))])
                raise RuntimeError("inner failed")

        with finishing_together(fig):
            finish_figure(fig, [Finish(ax, headroom=lambda: calls.append(1))])
            with pytest.raises(RuntimeError, match="inner failed"):
                failing_block()
            assert calls == []
            finish_figure(fig, [Finish(ax, headroom=lambda: calls.append(3))])
        assert calls == [1, 3]

    def test_draw_suppression_restores_custom_canvas_draw(self) -> None:
        fig = plt.figure()
        calls: list[int] = []

        def custom_draw() -> None:
            calls.append(1)

        def failing_block() -> None:
            with without_redraw(fig):
                fig.canvas.draw()
                raise RuntimeError("inner failed")

        fig.canvas.draw = custom_draw
        with without_redraw(fig):
            with pytest.raises(RuntimeError, match="inner failed"):
                failing_block()
            fig.canvas.draw()
            assert calls == []
        assert fig.canvas.draw is custom_draw
        fig.canvas.draw()
        assert calls == [1]


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (None, [True] * 4),
        ([(1, 3)], [False, True, True, False]),
        ([(0.5, 1.5), (3, 4)], [True, True, False, True]),
        ([(4, 5)], [False] * 4),
        ([], [False] * 4),
    ],
)
def test_bins_overlapping_view(
    view: list[tuple[float, float]] | None, expected: list[bool]
) -> None:
    from rootfig.plotting.hist1d import in_view

    np.testing.assert_array_equal(in_view(np.arange(5), view), expected)


def test_no_bins_in_view_is_empty() -> None:
    from rootfig.plotting import draw_histograms

    h = hist.Hist(hist.axis.Regular(4, 0, 4), storage=hist.storage.Weight()).fill([0.5])
    fig, ax = plt.subplots()
    drawn = draw_histograms([Histogram(h, label="h")], ax, style=Style(), view=[(5, 6)])
    assert (drawn.ymin, drawn.ymax) == (0, 0)
    assert np.isnan(drawn.ymin_positive)
    plt.close(fig)


@pytest.mark.parametrize("source", ["points", "band"])
def test_ratio_range_uses_only_visible_uncertainties(source: str) -> None:
    errors = np.array([10, 0.1, 0.1, 10])
    result = comparison(np.ones(4), syst_errors=(errors, errors) if source == "points" else None)
    band = (1 - errors, 1 + errors) if source == "band" else None
    assert panel_ylim([result], band=band) == (0, 3)
    assert panel_ylim([result], band=band, view=[(1, 3)]) == (0.5, 1.5)
    assert panel_ylim([result], band=band, view=[(5, 6)]) == (0.5, 1.5)
    if band is not None:
        assert panel_ylim([], band=band) == (0, 3)
