"""Tests for the plotting layer: styles, layout, drawing, ratio panel, annotations."""

from __future__ import annotations

import warnings

import hist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from rootfig.histograms import Histogram, fill, summarize
from rootfig.model import Style
from rootfig.plotting import (
    DEFAULT_COLORS,
    ROOTFIG_STYLE,
    Layout,
    Plot,
    add_experiment_label,
    add_legend,
    add_stats_box,
    add_text,
    apply_xbreak,
    break_segments,
    color_cycle,
    draw_correlation,
    draw_hist2d,
    draw_histograms,
    draw_ratio_panel,
    envelope,
    finish_axes,
    label_flow_bins,
    make_figure,
    overlay_artists,
    raise_ylim_above,
    ratio_ylim,
    resolve_rc,
    show_flow_bins,
    style_context,
    use_style,
    ylabel_for,
)
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


class TestFigure:
    def test_make_figure_single(self) -> None:
        with style_context():
            layout = make_figure(Style(), ratio=False)
        assert isinstance(layout, Layout)
        assert isinstance(layout.fig, Figure)
        assert isinstance(layout.main, Axes)
        assert layout.ratio is None
        assert not layout.is_broken
        assert layout.main_axes == (layout.main,)
        assert layout.ratio_axes == ()
        assert layout.legend_axes is layout.main
        assert layout.xlabel_axes is layout.main
        assert tuple(layout.fig.get_size_inches()) == pytest.approx(ROOTFIG_STYLE["figure.figsize"])

    def test_make_figure_ratio(self) -> None:
        with style_context():
            layout = make_figure(Style(), ratio=True, figsize=(5, 6))
        assert layout.ratio is not None
        assert layout.ratio.get_shared_x_axes().joined(layout.main, layout.ratio)
        assert layout.xlabel_axes is layout.ratio
        assert tuple(layout.fig.get_size_inches()) == (5, 6)

    def test_make_figure_broken(self) -> None:
        with style_context():
            layout = make_figure(Style(), ratio=True, break_widths=(0.7, 0.3))
        assert layout.is_broken
        assert layout.main_right is not None
        assert layout.ratio_right is not None
        assert len(layout.fig.axes) == 4
        assert layout.main_right.get_shared_y_axes().joined(layout.main, layout.main_right)
        assert layout.ratio_right.get_shared_x_axes().joined(layout.main_right, layout.ratio_right)
        assert layout.legend_axes is layout.main_right
        assert layout.xlabel_axes is layout.ratio_right
        left_width = layout.main.get_position().width
        right_width = layout.main_right.get_position().width
        assert left_width / right_width == pytest.approx(0.7 / 0.3, rel=0.05)

    def test_make_figure_existing_axes(self) -> None:
        fig, axes = plt.subplots(2)
        assert make_figure(Style(), ratio=False, ax=axes[0]).main is axes[0]
        layout = make_figure(Style(), ratio=True, ax=(axes[0], axes[1]))
        assert (layout.main, layout.ratio) == (axes[0], axes[1])
        assert make_figure(Style(), ratio=False, ax=(axes[0], axes[1])).ratio is None
        with pytest.raises(ValueError, match="two axes"):
            make_figure(Style(), ratio=True, ax=axes[0])
        with pytest.raises(ValueError, match="pair"):
            make_figure(Style(), ratio=True, ax=(axes[0],))
        with pytest.raises(ValueError, match="existing axes"):
            make_figure(Style(), ratio=False, ax=axes[0], break_widths=(0.5, 0.5))

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
        assert top == pytest.approx(14.5)

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
        assert drawn.colors == {"A": DEFAULT_COLORS[0], "B": "green"}
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

    def test_stack_with_data(self, mc_hists: list[Histogram], data_hist: Histogram) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            drawn = draw_histograms([*mc_hists, data_hist], ax, style=st, stack=True)
        assert drawn.labels == ["A", "B", "Stat. unc.", "Data"]
        total = mc_hists[0].values() + mc_hists[1].values()
        assert drawn.ymax >= total.max()
        assert drawn.colors["Data"] == "black"

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


class TestRatioPanel:
    def test_ratio_panel_propagate(self, mc_hists: list[Histogram]) -> None:
        with style_context() as st:
            fig, (ax, rax) = plt.subplots(2)
            ratios = draw_ratio_panel(
                [mc_hists[1]], mc_hists[0], rax, style=st, uncertainty="propagate"
            )
        assert len(ratios) == 1
        assert rax.get_ylabel() == "Ratio to A"
        assert rax.get_xlim() == (0.0, 4.0)
        assert not [c for c in rax.collections if isinstance(c, PolyCollection)]  # no band

    def test_ratio_panel_numerator_band(
        self, mc_hists: list[Histogram], data_hist: Histogram
    ) -> None:
        with style_context() as st:
            fig, (ax, rax) = plt.subplots(2)
            draw_ratio_panel(
                [data_hist],
                mc_hists[0],
                rax,
                style=st,
                uncertainty="numerator",
                ylim=(0, 2),
                ylabel="custom",
            )
        assert rax.get_ylabel() == "custom"
        assert rax.get_ylim() == (0.0, 2.0)
        assert len([c for c in rax.collections if isinstance(c, PolyCollection)]) == 1  # band

    def test_ratio_label_data_mc(self, mc_hists: list[Histogram], data_hist: Histogram) -> None:
        total = Histogram(mc_hists[0].hist + mc_hists[1].hist, label="Total")
        with style_context() as st:
            fig, (ax, rax) = plt.subplots(2)
            draw_ratio_panel([data_hist], total, rax, style=st, uncertainty="numerator")
        assert rax.get_ylabel() == "Data / MC"

    def test_ratio_ylim(self) -> None:
        from rootfig.histograms import Ratio

        edges = np.arange(5.0)
        tight = Ratio(np.array([0.9, 1.0, 1.1, 1.0]), np.zeros(4), np.zeros(4), edges)
        assert ratio_ylim([tight]) == (0.5, 1.5)
        wide = Ratio(np.array([0.2, 2.0, np.nan, 50.0]), np.zeros(4), np.zeros(4), edges)
        low, high = ratio_ylim([wide])
        assert low < 0.2
        assert high == 3.0
        assert ratio_ylim([]) == (0.5, 1.5)


class TestAnnotations:
    def test_legend_order(self, mc_hists: list[Histogram], data_hist: Histogram) -> None:
        with style_context() as st:
            fig, ax = plt.subplots()
            draw_histograms([*mc_hists, data_hist], ax, style=st, stack=True)
            legend = add_legend(ax, st)
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["Data", "B", "A", "Stat. unc."]  # top of the stack first

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
        draw_hist2d(empty, ax, logz=True)
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
            ax.set_ylim(0, 145)  # the usual 1.45 headroom
        ax.set_xlim(0, 10)
        return fig, ax, edges, heights

    def test_anchored_legend_raises_top(self) -> None:
        fig, ax, edges, heights = self._axes()
        legend = ax.legend(loc="upper right")
        raise_ylim_above(
            [ax], overlay_artists(ax, legend), edges=edges, heights=heights, logy=False
        )
        assert ax.get_ylim()[1] > 145
        # the legend bottom now clears the histogram
        fig.canvas.draw()
        bottom = ax.transData.inverted().transform(legend.get_window_extent().get_points())[0, 1]
        assert bottom >= 100
        plt.close(fig)

    def test_low_legend_and_free_legend_untouched(self) -> None:
        fig, ax, edges, heights = self._axes()
        legend = ax.legend(loc="lower left")
        raise_ylim_above([ax], [legend], edges=edges, heights=heights, logy=False)
        assert ax.get_ylim()[1] == 145
        heights_low = np.full(10, 10.0)
        legend = ax.legend(loc="upper right")
        raise_ylim_above([ax], [legend], edges=edges, heights=heights_low, logy=False)
        assert ax.get_ylim()[1] == 145
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
        assert ax2.get_ylim()[1] == 145
        plt.close(fig)
        plt.close(fig2)

    def test_degenerate_inputs(self) -> None:
        fig, ax, edges, heights = self._axes()
        raise_ylim_above([], [], edges=edges, heights=heights, logy=False)
        raise_ylim_above([ax], [], edges=edges, heights=np.array([]), logy=False)
        raise_ylim_above([ax], [ax.legend()], edges=edges, heights=heights * np.nan, logy=False)
        assert ax.get_ylim()[1] == 145
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
        from rootfig.histograms import Ratio
        from rootfig.plotting import draw_significance_panel

        result = Ratio(
            values=np.array([1.0, 2.0, np.nan]),
            errors=np.array([0.1, 0.2, np.nan]),
            band=np.full(3, np.nan),
            edges=np.array([0.0, 1.0, 2.0, 3.0]),
        )
        fig, ax = plt.subplots()
        draw_significance_panel(result, ax)
        assert ax.get_ylabel() == r"$S/\sqrt{B}$"
        assert ax.get_ylim() == (0.0, pytest.approx(1.25 * 2.2))
        assert ax.get_xlim() == (0.0, 3.0)
        draw_significance_panel(result, ax, kind="s/sqrt(s+b)", ylim=(0, 5), ylabel="Z")
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
        assert drawn.histogram_colors == color_cycle(1, Style())
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

    def test_ratio_panel_reuses_main_colors(self) -> None:
        from matplotlib.colors import to_rgba

        hists = [
            make_hist([0.5], label="A"),
            make_hist([0.5], label="B"),
            make_hist([0.5], label="C"),
        ]
        fig, (ax, ratio_ax) = plt.subplots(2)
        drawn = draw_histograms(hists, ax, style=Style())
        assert drawn.histogram_colors == color_cycle(3, Style())
        draw_ratio_panel(
            hists[1:],
            hists[0],
            ratio_ax,
            style=Style(),
            uncertainty="propagate",
            colors=drawn.histogram_colors[1:],
        )
        drawn_colors = [
            to_rgba(c.lines[0].get_color())
            for c in ratio_ax.containers
            if isinstance(c, matplotlib.container.ErrorbarContainer)
        ]
        assert drawn_colors == [to_rgba(c) for c in drawn.histogram_colors[1:]]
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
