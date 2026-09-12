"""Tests for the declarative model: Cut, Variable, binning, Sample, Style."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import hist
import numpy as np
import pytest

from rootfig.errors import BinningError, ExpressionError, LuminosityError, SourceError
from rootfig.io import ArraySource, FileSource
from rootfig.model import (
    DEFAULT_RANGE,
    Cut,
    Sample,
    Style,
    Variable,
    as_cut,
    as_samples,
    as_style,
    as_variable,
    auto_range,
    log_bins,
    resolve_axis,
)
from rootfig.model.binning import ROBUST_COVERAGE_BUDGET


class TestCut:
    def test_basic(self) -> None:
        cut = Cut("x > 1", label="positive")
        assert str(cut) == "x > 1"
        assert cut.parsed().names == ("x",)

    def test_invalid_expression(self) -> None:
        with pytest.raises(ExpressionError):
            Cut("x >")

    def test_composition(self) -> None:
        a, b = Cut("x > 1", label="A"), Cut("y < 2", label="B")
        assert (a & b).expression == "(x > 1) & (y < 2)"
        assert (a & b).label == "A & B"
        assert (a | "z == 0").expression == "(x > 1) | (z == 0)"
        assert ("z == 0" & a).expression == "(z == 0) & (x > 1)"
        assert ("z == 0" | a).expression == "(z == 0) | (x > 1)"
        assert (~a).expression == "~(x > 1)"
        assert (~a).label == "not A"
        assert (a & "y < 2").label is None

    def test_as_cut(self) -> None:
        assert as_cut(None) is None
        cut = Cut("x > 1")
        assert as_cut(cut) is cut
        assert as_cut("x > 1") == cut
        with pytest.raises(TypeError):
            as_cut(3)  # type: ignore[arg-type]


class TestVariable:
    def test_defaults(self) -> None:
        var = Variable("Muon_pt")
        assert var.bins == 50
        assert var.range == DEFAULT_RANGE == "robust"
        assert var.axis_label == "Muon_pt"
        assert var.safe_name == "Muon_pt"
        assert str(var) == "Muon_pt"

    def test_label_and_unit(self) -> None:
        var = Variable("Muon_pt / 1000", label=r"$p_T$", unit="GeV")
        assert var.axis_label == r"$p_T$ [GeV]"
        assert var.safe_name == "Muon_pt_1000"
        assert var.with_(name="pt").safe_name == "pt"

    def test_invalid_expression(self) -> None:
        with pytest.raises(ExpressionError):
            Variable("a +")

    @pytest.mark.parametrize(
        "bins",
        [
            0,
            -3,
            True,
            (0, 0.0, 1.0),
            (10, 1.0, 1.0),
            (10, 2.0, 1.0),
            [1.0],
            [1.0, 0.5],
            [0.0, np.inf],
            [[0, 1]],
        ],
    )
    def test_invalid_bins(self, bins: Any) -> None:
        with pytest.raises(BinningError):
            Variable("x", bins=bins)

    @pytest.mark.parametrize("range_", [(1.0, 1.0), (2.0, 1.0), (0.0, np.nan), "weird"])
    def test_invalid_range(self, range_: Any) -> None:
        with pytest.raises(BinningError):
            Variable("x", bins=10, range=range_)

    @pytest.mark.parametrize("name", ["/abs", "../x", "a/b", "a\\b", ".", ".."])
    def test_name_must_be_a_file_stem(self, name: str) -> None:
        with pytest.raises(ValueError, match="path separators"):
            Variable("x", name=name)
        with pytest.raises(ValueError, match="path separators"):
            Variable("x").with_(name=name)

    def test_name_keeps_plain_stems(self) -> None:
        assert Variable("x", name="pt-lead.window").safe_name == "pt-lead.window"
        assert Variable("Muon_pt / 1000").safe_name == "Muon_pt_1000"

    def test_as_variable(self) -> None:
        var = as_variable("x", bins=10, label=None)
        assert var == Variable("x", bins=10)
        assert as_variable(var) is var
        assert as_variable(var, bins=(5, 0.0, 1.0)).bins == (5, 0.0, 1.0)
        with pytest.raises(TypeError):
            as_variable(3)  # type: ignore[arg-type]


class TestBinning:
    def test_log_bins(self) -> None:
        assert log_bins(3, 1, 1000).tolist() == pytest.approx([1, 10, 100, 1000])
        with pytest.raises(BinningError):
            log_bins(0, 1, 10)
        with pytest.raises(BinningError):
            log_bins(3, 0, 10)

    def test_auto_range(self) -> None:
        low, high = auto_range([np.array([1.0, 2.0]), np.array([np.nan, 5.0])])
        assert low == 1.0
        assert high > 5.0
        assert high == pytest.approx(5.004)
        assert auto_range([np.array([]), np.array([np.nan])]) == (0.0, 1.0)
        assert auto_range([np.array([3.0, 3.0])]) == (pytest.approx(2.7), pytest.approx(3.3))
        assert auto_range([np.array([0.0])]) == (-0.5, 0.5)

    def test_robust_range_ignores_sentinels(self) -> None:
        rng = np.random.default_rng(0)
        for sentinel in (-999.0, -99.0):
            values = rng.normal(0, 1, 5000)
            values[:50] = sentinel
            low, high = auto_range([values], mode="robust")
            assert low > -6
            assert high < 6
            assert auto_range([values])[0] == sentinel  # "auto" keeps them

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(np.random.default_rng(2).uniform(0, 1, 20_000), id="uniform"),
            pytest.param(
                np.concatenate(
                    [
                        np.random.default_rng(4).normal(-5, 0.5, 10_000),
                        np.random.default_rng(5).normal(5, 0.5, 10_000),
                    ]
                ),
                id="bimodal",
            ),
            pytest.param(np.random.default_rng(6).poisson(2, 20_000).astype(float), id="poisson"),
            pytest.param(
                np.random.default_rng(7).integers(0, 2, 20_000).astype(float), id="binary"
            ),
            pytest.param(np.repeat([0.0, 1.0, 2.0, 3.0], [9000, 700, 250, 50]), id="counts"),
        ],
    )
    def test_robust_range_is_a_no_op_without_a_tail(self, values: np.ndarray) -> None:
        """Nothing to reject and nothing to cut: these must have identical edges.

        The distributions with a hard edge end where the data ends, and the
        categorical ones are excluded from tightening by their value count.
        """
        assert auto_range([values], mode="robust") == auto_range([values])

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(np.random.default_rng(1).normal(0, 1, 20_000), (-3.3, 3.3), id="gauss"),
            pytest.param(
                np.random.default_rng(3).exponential(30, 20_000), (0.0, 159.0), id="exponential"
            ),
            pytest.param(
                np.where(
                    np.random.default_rng(8).random(100_000) < 0.06,
                    np.random.default_rng(9).normal(0, 4, 100_000),
                    np.random.default_rng(10).normal(0, 1, 100_000),
                ),
                (-5.8, 5.8),
                id="gaussian-core-with-tails",
            ),
        ],
    )
    def test_robust_range_cuts_a_thin_tail_within_the_budget(
        self, values: np.ndarray, expected: tuple[float, float]
    ) -> None:
        """A tail the distance threshold keeps is cut while few entries leave the view."""
        low, high = auto_range([values], mode="robust")
        full_low, full_high = auto_range([values])
        assert (low, high) == pytest.approx(expected, abs=0.5)
        assert full_low <= low  # tightened, never widened
        assert high < full_high
        outside = float(((values < low) | (values > high)).mean())
        assert outside <= ROBUST_COVERAGE_BUDGET

    @pytest.mark.parametrize("n", [100_000, 10_000, 1_000, 500, 200, 50])
    def test_robust_range_keeps_a_small_distant_sample(self, n: int) -> None:
        """A signal far from a large background survives however few entries it has.

        The budget is charged per sample: measured against the pooled entries a
        signal of a few hundred beside a background of a hundred thousand would be
        under budget even when cut away completely.
        """
        rng = np.random.default_rng(0)
        background = rng.exponential(60, 100_000) + 50
        signal = rng.normal(800, 25, n)
        low, high = auto_range([background, signal], mode="robust")
        kept = float(((signal >= low) & (signal <= high)).mean())
        assert kept >= 1.0 - ROBUST_COVERAGE_BUDGET

    @pytest.mark.parametrize("reverse", [False, True])
    def test_robust_range_keeps_a_category_overlaid_with_a_continuous_sample(
        self, reverse: bool
    ) -> None:
        """Categorical samples are recognised one by one, not in the pooled values.

        Pooled with a large continuous sample this one looks continuous, and its
        rarest category is one entry in a thousand - under the budget, and cut,
        unless the sample is judged on its own.
        """
        background = np.random.default_rng(0).normal(0, 1, 100_000)
        categorical = np.r_[np.zeros(999), 10.0]
        samples = [categorical, background] if reverse else [background, categorical]
        low, high = auto_range(samples, mode="robust")
        assert low <= 10.0 <= high
        assert (low, high) == auto_range(samples)

    def test_robust_range_still_cuts_an_overlay_of_continuous_samples(self) -> None:
        """A zero budget is for categorical samples only, not for every overlay."""
        rng = np.random.default_rng(0)
        samples = [rng.exponential(30, 20_000), rng.exponential(30, 20_000)]
        assert auto_range(samples, mode="robust")[1] < auto_range(samples)[1]

    def test_robust_range_keeps_categorical_values(self) -> None:
        """A rare category is a bin of its own, not empty space at the edge."""
        counts = np.repeat([0.0, 1.0, 2.0, 3.0], [9000, 700, 250, 50])
        assert auto_range([counts], mode="robust")[1] == auto_range([counts])[1]
        # the same shape spread over enough distinct values is tightened
        spread = np.repeat(np.linspace(0.0, 30.0, 30), [1000] * 10 + [50] * 19 + [1])
        assert auto_range([spread], mode="robust")[1] < auto_range([spread])[1]

    def test_robust_range_never_extends_past_the_data(self) -> None:
        rng = np.random.default_rng(0)
        for values in (
            rng.exponential(30, 20_000),  # strictly positive, long tail
            rng.lognormal(3, 1, 20_000),
            np.where(rng.random(20_000) < 0.8, 0.0, rng.exponential(0.3, 20_000)),
        ):
            low, high = auto_range([values], mode="robust")
            assert low >= float(values.min()) >= 0.0  # no negative edge from the pad
            assert high <= auto_range([values])[1]

    @pytest.mark.parametrize(
        ("arrays", "expected"),
        [
            ([np.zeros(100)], (-0.5, 0.5)),
            ([np.array([42.0])], (37.8, 46.2)),
            ([np.array([]), np.array([np.nan])], (0.0, 1.0)),
            ([np.append(np.zeros(999), 500.0)], (-0.5, 0.5)),
        ],
    )
    def test_robust_range_degenerate_spread(
        self, arrays: list[np.ndarray], expected: tuple[float, float]
    ) -> None:
        """A zero median absolute deviation must not divide by zero."""
        low, high = auto_range(arrays, mode="robust")
        assert (low, high) == pytest.approx(expected)

    def test_robust_range_and_a_distant_sample(self) -> None:
        """Known property: the range is inferred from all samples at once.

        A signal in the tail of this broad background is kept at both tested
        yields. One tens of deviations away from a narrow bulk is rejected
        even when its yield is a fifth of the background's.
        """
        rng = np.random.default_rng(0)
        broad = rng.exponential(60, 100_000) + 50
        for n in (1_000, 100_000):
            signal = rng.normal(800, 25, n)
            assert auto_range([broad, signal], mode="robust") == auto_range([broad, signal])

        narrow = rng.normal(0, 1, 100_000)
        signal = rng.normal(50, 1, 20_000)
        assert auto_range([narrow, signal], mode="robust")[1] < 10
        assert auto_range([narrow, signal])[1] > 50  # "auto" keeps it on the axis

    def test_resolve_int_bins_explicit_range(self) -> None:
        axis = resolve_axis(Variable("x", bins=10, range=(0.0, 5.0), unit="GeV"))
        assert isinstance(axis, hist.axis.Regular)
        assert axis.size == 10
        assert axis.edges[0] == 0.0
        assert axis.edges[-1] == 5.0
        assert axis.label == "x [GeV]"

    def test_resolve_int_bins_auto(self) -> None:
        axis = resolve_axis(Variable("x", bins=4), [np.array([0.0, 2.0]), np.array([4.0])])
        assert axis.edges[0] == 0.0
        assert axis.edges[-1] == pytest.approx(4.004)

    def test_resolve_int_bins_without_data(self) -> None:
        with pytest.raises(BinningError, match="cannot infer"):
            resolve_axis(Variable("x", bins=4))

    def test_resolve_tuple(self) -> None:
        axis = resolve_axis(Variable("x", bins=(20, -1, 1)))
        assert isinstance(axis, hist.axis.Regular)
        assert axis.size == 20

    def test_resolve_edges(self) -> None:
        axis = resolve_axis(Variable("x", bins=[0, 1, 5, 20]))
        assert isinstance(axis, hist.axis.Variable)
        assert axis.edges.tolist() == [0, 1, 5, 20]
        axis = resolve_axis(Variable("x", bins=log_bins(2, 1, 100)))
        assert axis.edges.tolist() == pytest.approx([1, 10, 100])

    def test_resolve_hist_axis_is_copied(self) -> None:
        given = hist.axis.Regular(3, 1, 1000, transform=hist.axis.transform.log)
        axis = resolve_axis(Variable("x", bins=given, label="lab"))
        assert axis is not given  # the caller's axis is never modified
        assert axis.label == "lab"
        assert given.label == ""
        np.testing.assert_allclose(axis.edges, given.edges)
        assert axis.traits == given.traits
        assert type(axis.transform) is type(given.transform)
        # reusing one axis object for a second variable does not leak the first label
        other = resolve_axis(Variable("y", bins=given, label="other"))
        assert other.label == "other"
        own = hist.axis.Variable([0, 1, 5], overflow=False, label="own")
        copied = resolve_axis(Variable("x", bins=own, label="lab"))
        assert copied is not own
        assert copied.label == "own"
        assert copied.traits.overflow is False


class TestSample:
    def test_from_file(self, signal_file: Path) -> None:
        sample = Sample(signal_file, tree="events")
        assert isinstance(sample.source, FileSource)
        assert sample.label == "signal"
        assert sample.files == (str(signal_file),)
        assert "signal.root" in repr(sample)

    def test_full_options(self, signal_file: Path) -> None:
        sample = Sample(
            f"{signal_file}:events",
            label="Signal",
            selection="nMuon > 0",
            weight="weight",
            is_data=True,
            color="red",
            histtype="fill",
            scale=2,
            entry_stop=10,
        )
        assert sample.selection == Cut("nMuon > 0")
        assert sample.weight == "weight"
        assert sample.is_data
        assert sample.scale == 2.0
        assert sample.source.entry_stop == 10  # type: ignore[attr-defined]
        assert "is_data=True" in repr(sample)

    def test_blank_weight_is_none(self) -> None:
        assert Sample({"x": [1]}, weight="  ").weight is None

    def test_from_arrays(self) -> None:
        sample = Sample({"x": ak.Array([1, 2])}, label="mem")
        assert isinstance(sample.source, ArraySource)
        assert sample.files == ()

    def test_with(self, signal_file: Path) -> None:
        sample = Sample(signal_file, tree="events")
        other = sample.with_(label="new", selection="MET > 1")
        assert other.label == "new"
        assert other.selection == Cut("MET > 1")
        assert other.source is sample.source

    def test_with_validates_like_init(self) -> None:
        sample = Sample({"x": np.arange(2.0)})
        with pytest.raises(ValueError, match="scale must be"):
            sample.with_(scale=np.nan)
        with pytest.raises(LuminosityError, match="finite"):
            sample.with_(ngen=np.inf)
        with pytest.raises(LuminosityError):
            sample.with_(xsec="bad")
        with pytest.raises(SourceError, match="cannot interpret"):
            sample.with_(source=42)
        with pytest.raises(TypeError, match="weight must be"):
            sample.with_(weight=42)
        with pytest.raises(TypeError, match="label must be"):
            sample.with_(label=3)
        assert sample.with_(weight="  ").weight is None
        assert sample.with_(scale=2).scale == 2.0
        assert sample.with_(source=sample.source).source is sample.source
        assert sample.with_(xsec="1.2 fb").xsec == "1.2 fb"

    def test_existing_source_rejects_entry_range(self, signal_file: Path) -> None:
        source = FileSource(signal_file, tree="events")
        assert Sample(source, tree="events").source is source
        with pytest.raises(SourceError, match="entry_stop=100 cannot be applied"):
            Sample(source, entry_stop=100)

    def test_invalid_data(self) -> None:
        with pytest.raises(SourceError):
            Sample(42)


class TestAsSamples:
    def test_single_and_list(self, signal_file: Path, background_file: Path) -> None:
        assert len(as_samples(signal_file, tree="events")) == 1
        samples = as_samples([signal_file, background_file], tree="events")
        assert [s.label for s in samples] == ["signal", "background"]

    def test_labels(self, signal_file: Path, background_file: Path) -> None:
        samples = as_samples([signal_file, background_file], tree="events", labels=["S", "B"])
        assert [s.label for s in samples] == ["S", "B"]
        assert as_samples(signal_file, tree="events", labels="only")[0].label == "only"
        with pytest.raises(SourceError, match="labels"):
            as_samples([signal_file, background_file], tree="events", labels=["S"])

    def test_mixed_list(self, signal_file: Path, background_file: Path) -> None:
        samples = as_samples(
            [Sample(signal_file, tree="events", label="S"), background_file], tree="events"
        )
        assert [s.label for s in samples] == ["S", "background"]

    def test_label_mapping(self, signal_file: Path, background_file: Path) -> None:
        samples = as_samples(
            {"Signal": signal_file, "Background": [background_file]}, tree="events"
        )
        assert [s.label for s in samples] == ["Signal", "Background"]

    def test_column_mapping_is_one_sample(self) -> None:
        samples = as_samples({"x": ak.Array([1.0, 2.0]), "y": np.array([3.0, 4.0])})
        assert len(samples) == 1
        assert isinstance(samples[0].source, ArraySource)

    def test_empty(self) -> None:
        with pytest.raises(SourceError, match="no samples"):
            as_samples([])


class TestStyle:
    def test_defaults(self) -> None:
        style = Style()
        assert not style.has_label
        assert style.text_lines == ()
        assert style.legend is True

    def test_text_lines(self) -> None:
        assert Style(text="a\nb").text_lines == ("a", "b")
        assert Style(text=["a", "b"]).text_lines == ("a", "b")
        assert Style(text="x").has_label

    def test_as_style(self) -> None:
        assert as_style(None) == Style()
        assert as_style("atlas") == Style(experiment="ATLAS")
        assert as_style("CMS").experiment == "CMS"
        assert as_style("ggplot") == Style(base="ggplot")
        style = Style(lumi=140)
        assert as_style(style) is style
        assert style.with_(com=13.6).com == 13.6
        with pytest.raises(TypeError):
            as_style(3)  # type: ignore[arg-type]


class TestUnits:
    def test_split_quantity(self) -> None:
        from rootfig.model.units import split_quantity

        assert split_quantity(None, "TeV") is None
        assert split_quantity(13.6, "TeV") == ("13.6", "TeV")
        assert split_quantity("240 GeV", "TeV") == ("240", "GeV")
        assert split_quantity("240", "TeV") == ("240", "TeV")
        assert split_quantity("Run 2", "TeV") == ("Run 2", "")
        for text in ["10.8 ab^-1", "10.8 ab⁻¹", "10.8 ab$^{-1}$", "10.8 ab", "10.8 ab-1"]:
            assert split_quantity(text, "fb^{-1}", inverse=True) == ("10.8", "ab^{-1}"), text
        assert split_quantity(139, "fb^{-1}", inverse=True) == ("139", "fb^{-1}")

    def test_cross_section_and_luminosity(self) -> None:
        from rootfig.errors import LuminosityError
        from rootfig.model.units import cross_section_pb, luminosity_fb

        assert cross_section_pb(2.0) == 2.0
        assert cross_section_pb("1.5 fb") == pytest.approx(1.5e-3)
        assert cross_section_pb("3 nb") == pytest.approx(3e3)
        assert luminosity_fb(140) == 140
        assert luminosity_fb("10.8 ab^-1") == pytest.approx(10.8e3)
        assert luminosity_fb("5 ab") == pytest.approx(5e3)
        assert luminosity_fb("300 pb-1") == pytest.approx(0.3)
        with pytest.raises(LuminosityError, match="unknown unit"):
            luminosity_fb("3 furlongs")
        with pytest.raises(LuminosityError, match="inverse area"):
            cross_section_pb("1 pb^-1")
        with pytest.raises(LuminosityError, match="cannot read"):
            cross_section_pb("lots")


class TestSampleLuminosity:
    def test_lumi_scale(self, signal_file: Path) -> None:
        from rootfig.errors import LuminosityError

        sample = Sample(signal_file, tree="events", xsec="0.2 pb", ngen=20000)
        # 0.2 pb * 10.8 ab^-1 = 0.2e-12 b * 10.8e18 b^-1 = 2.16e6 events / 20000 generated
        assert sample.lumi_scale("10.8 ab^-1") == pytest.approx(2.16e6 / 20000)
        assert sample.lumi_scale(1.0) == pytest.approx(0.2 * 1e3 / 20000)  # pb x fb^-1
        assert Sample(signal_file, tree="events").lumi_scale(None) == 1.0
        assert Sample(signal_file, tree="events").lumi_scale(140) == 1.0  # no cross section
        with pytest.raises(LuminosityError, match="no luminosity"):
            sample.lumi_scale(None)
        with pytest.raises(LuminosityError):
            Sample(signal_file, tree="events", xsec="bad")
        assert "xsec" in repr(sample)

    def test_ngen_from_entries_and_file(self, signal_file: Path, tmp_path: Path) -> None:
        import uproot

        from rootfig.errors import LuminosityError

        # None: the number of entries in the source (2000 in the fixture file)
        assert Sample(signal_file, tree="events", xsec=1.0).generated_events() == 2000
        assert Sample(signal_file, tree="events", xsec=1.0, ngen=5).generated_events() == 5.0
        # a string: an object in the file (sum-of-weights histogram)
        path = tmp_path / "sumw.root"
        with uproot.recreate(path) as file:
            file.mktree("events", {"x": "float64"}).extend({"x": np.zeros(10)})
            file["sumw"] = np.histogram(np.zeros(50), bins=1)
        sample = Sample(path, tree="events", xsec=1.0, ngen="sumw")
        assert sample.generated_events() == 50.0
        assert sample.lumi_scale(1.0) == pytest.approx(1e3 / 50)
        with pytest.raises(SourceError, match="not found"):
            Sample(path, tree="events", xsec=1.0, ngen="nope").generated_events()
        # in-memory arrays have no objects to read from
        mem = Sample({"x": np.arange(4.0)}, xsec=1.0, ngen="sumw")
        with pytest.raises(LuminosityError, match="file source"):
            mem.generated_events()
        assert Sample({"x": np.arange(4.0)}, xsec=1.0).generated_events() == 4.0
        with pytest.raises(LuminosityError, match="generated events is 0"):
            Sample({"x": np.arange(4.0)}, xsec=1.0, ngen=0).lumi_scale(1.0)

    def test_style_units(self) -> None:
        assert Style(lumi=140, com=13.6).lumi_parts == ("140", "fb^{-1}")
        assert Style(lumi=140, com=13.6).com_parts == ("13.6", "TeV")
        fcc = Style(lumi=10.8, lumi_unit="ab^{-1}", com=240, com_unit="GeV")
        assert fcc.lumi_parts == ("10.8", "ab^{-1}")
        assert fcc.com_parts == ("240", "GeV")
        assert Style(lumi="10.8 ab", com="240 GeV").lumi_parts == ("10.8", "ab^{-1}")
        assert Style().lumi_parts is None


class TestInputDispatch:
    def test_mapping_of_columns_is_one_sample(self) -> None:
        [sample] = as_samples({"x": [1.0, 2.0, 3.0]})
        assert isinstance(sample.source, ArraySource)
        assert sample.source.branches() == ["x"]
        [sample] = as_samples({"A": [1.0, 2.0], "B": np.arange(2.0)})
        assert sample.source.branches() == ["A", "B"]

    def test_label_map_values(self, signal_file: Path) -> None:
        inner = Sample({"x": np.arange(3.0)}, weight="2", xsec=1.0)
        samples = as_samples(
            {
                "S": inner,
                "F": str(signal_file),
                "L": [str(signal_file)],
                "M": {"x": np.arange(2.0)},
                "R": ak.Array({"x": [1.0]}),
            },
            tree="events",
        )
        assert [s.label for s in samples] == ["S", "F", "L", "M", "R"]
        assert samples[0].weight == "2"
        assert samples[0].xsec == 1.0
        assert samples[0].source is inner.source
        assert isinstance(samples[1].source, FileSource)
        assert isinstance(samples[3].source, ArraySource)
        assert isinstance(samples[4].source, ArraySource)

    def test_numeric_list_is_not_a_file_list(self) -> None:
        with pytest.raises(SourceError, match="cannot interpret"):
            Sample([1.0, 2.0])


class TestFiniteScaling:
    def test_scale_and_cross_section_must_be_finite(self) -> None:
        from rootfig.errors import LuminosityError

        with pytest.raises(ValueError, match="scale must be"):
            Sample({"x": np.arange(2.0)}, scale=np.inf)
        with pytest.raises(LuminosityError, match="finite"):
            Sample({"x": np.arange(2.0)}, xsec=np.inf)
        with pytest.raises(LuminosityError, match="finite"):
            Sample({"x": np.arange(2.0)}, xsec=-1.0)
        with pytest.raises(LuminosityError, match="finite"):
            Sample({"x": np.arange(2.0)}, xsec=1.0, ngen=np.nan)
        sample = Sample({"x": np.arange(2.0)}, xsec=1.0)
        with pytest.raises(LuminosityError, match="finite"):
            sample.lumi_scale(np.inf)
        huge = Sample({"x": np.arange(2.0)}, xsec=1e300, ngen=1e-300)
        with pytest.raises(LuminosityError, match="not finite"):
            huge.lumi_scale(1e300)

    def test_units_keep_numeric_precision(self) -> None:
        from rootfig.model.units import cross_section_pb, luminosity_fb

        assert cross_section_pb(1.23456789) == 1.23456789
        assert luminosity_fb(1.23456789) == 1.23456789
        assert cross_section_pb("1.23456789 pb") == 1.23456789
