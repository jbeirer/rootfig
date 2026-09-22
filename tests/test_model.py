"""Tests for the declarative model: Cut, Variable, binning, Sample, Style."""

from __future__ import annotations

import copy
import pickle
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import hist
import numpy as np
import pytest

from rootfig.errors import (
    BinningError,
    ExpressionError,
    LuminosityError,
    SourceError,
    SystematicError,
)
from rootfig.io import ArraySource, FileSource
from rootfig.model import (
    Cut,
    Group,
    Sample,
    Style,
    Systematic,
    Variable,
    as_cut,
    as_plot_items,
    as_samples,
    as_style,
    as_systematics,
    as_variable,
    check_file_stem,
    leaf_samples,
    log_bins,
    map_samples,
    resolve_axis,
    safe_file_stem,
)
from rootfig.model.binning import DEFAULT_RANGE, ROBUST_COVERAGE_BUDGET, auto_range


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
        assert var.bins is None
        assert var.range == DEFAULT_RANGE == "robust"
        assert var.axis_label == "Muon_pt"
        assert var.safe_name == "Muon_pt"
        assert str(var) == "Muon_pt"

    def test_label_and_unit(self) -> None:
        var = Variable("Muon_pt / 1000", label=r"$p_T$", unit="GeV")
        assert var.axis_label == r"$p_T$ [GeV]"
        assert var.safe_name == "Muon_pt_1000"
        assert var.replace(name="pt").safe_name == "pt"

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

    @pytest.mark.parametrize(
        "name", ["/abs", "../x", "a/b", "a\\b", ".", "..", "a:b", "pt.", "CON", "nul.x"]
    )
    def test_name_must_be_a_file_stem(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"Variable name .* cannot be a file name component"):
            Variable("x", name=name)
        with pytest.raises(ValueError, match=r"Variable name .* cannot be a file name component"):
            Variable("x").replace(name=name)

    def test_name_keeps_plain_stems(self) -> None:
        assert Variable("x", name="pt-lead.window").safe_name == "pt-lead.window"
        assert Variable("Muon_pt / 1000").safe_name == "Muon_pt_1000"
        assert Variable("x", name="console").safe_name == "console"  # not a device name

    @pytest.mark.parametrize(
        ("expression", "stem"),
        [
            ("CON", "CON_"),
            ("nul", "nul_"),
            ("COM1 * 2", "COM1_2"),
        ],
    )
    def test_generated_name_is_always_a_file_stem(self, expression: str, stem: str) -> None:
        # Plot.save(directory) and PlotBook use safe_name without checking it again.
        assert Variable(expression).safe_name == stem
        check_file_stem(Variable(expression).safe_name, what="name")

    def test_as_variable(self) -> None:
        var = as_variable("x", bins=10, label=None)
        assert var == Variable("x", bins=10)
        assert as_variable(var) is var
        assert as_variable(var, bins=(5, 0.0, 1.0)).bins == (5, 0.0, 1.0)
        with pytest.raises(TypeError):
            as_variable(3)  # type: ignore[arg-type]


class TestFileStem:
    @pytest.mark.parametrize(
        ("value", "reason", "hint"),
        [
            ("", "more than dots and spaces", ""),
            (" .. ", "more than dots and spaces", ""),
            ("lin.", "end with a dot or a space", "; 'lin' would work"),
            ("lin ", "end with a dot or a space", "; 'lin' would work"),
            ("SR:high", "it holds ':'", "; 'SR_high' would work"),
            ('a"b<c>|?*', "it holds '\"', '*', '<', '>', '?', '|'", "; 'a_b_c' would work"),
            ("a\x00b\x7f", "it holds '\\x00', '\\x7f'", "; 'a_b' would work"),
            ("a/b\\c", "it holds '/', '\\\\'", "; 'a_b_c' would work"),
            ("CON", "'CON' is a reserved device name on Windows", "; 'CON_' would work"),
            ("com1", "'com1' is a reserved device name on Windows", "; 'com1_' would work"),
            ("Nul.mass", "'Nul' is a reserved device name on Windows", "; 'Nul_mass' would work"),
            ("CONOUT$", "'CONOUT$' is a reserved device name", "; 'CONOUT' would work"),
            ("COM¹", "'COM¹' is a reserved device name", "; 'COM' would work"),
            ("lpt³.x", "'lpt³' is a reserved device name", "; 'lpt_x' would work"),
            ("NUL .txt", "'NUL' is a reserved device name", "; 'NUL_txt' would work"),
            ("com1 .root", "'com1' is a reserved device name", "; 'com1_root' would work"),
            # The suggestion is itself checked: stripping the bad part must not leave a device.
            ("CON.", "end with a dot or a space", "; 'CON_' would work"),
            ("COM1:", "it holds ':'", "; 'COM1_' would work"),
            ("NUL ", "end with a dot or a space", "; 'NUL_' would work"),
        ],
    )
    def test_rejects_with_reason_and_suggestion(self, value: str, reason: str, hint: str) -> None:
        with pytest.raises(ValueError, match=re.escape(reason) + ".*" + re.escape(hint) + "$"):
            check_file_stem(value, what="selection name")

    @pytest.mark.parametrize(
        "value",
        [
            "mass",
            "pt-lead.window",
            "lin.2",
            "µ pt",
            "CONSOLE",
            "com",
            "lpt10",
            ".hidden",
            "COM0",
            "LPT0",
        ],
    )
    def test_accepts_portable_stems(self, value: str) -> None:
        # COM0 and LPT0 are not ports Windows reserves (ntpath.isreserved agrees).
        assert check_file_stem(value, what="name") is value

    @pytest.mark.parametrize(
        ("text", "stem"),
        [
            ("Muon_pt / 1000", "Muon_pt_1000"),
            ("CON.", "CON_"),
            ("COM1:", "COM1_"),
            ("com¹", "com"),
            ("Nul.mass", "Nul_mass"),
            ("?", ""),
            ("", ""),
        ],
    )
    def test_safe_file_stem(self, text: str, stem: str) -> None:
        assert safe_file_stem(text) == stem
        if stem:
            assert check_file_stem(stem, what="name") is stem


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
        kept = float(((signal >= low) & (signal < high)).mean())
        assert kept >= 1.0 - ROBUST_COVERAGE_BUDGET

    @pytest.mark.parametrize("reverse", [False, True])
    def test_robust_range_keeps_a_category_overlaid_with_a_continuous_sample(
        self, reverse: bool
    ) -> None:
        """Categorical samples are recognised one by one, not in the pooled values.

        Pooled with a large continuous sample this one looks continuous, and its
        rarest category is five entries in a thousand - inside the budget, and
        cut, unless the sample is judged on its own and given none.
        """
        background = np.random.default_rng(0).normal(0, 0.3, 100_000)
        categorical = np.repeat([0.0, 1.0, 2.0, 3.0, 4.0], [300, 300, 200, 195, 5])
        samples = [categorical, background] if reverse else [background, categorical]
        low, high = auto_range(samples, mode="robust")
        assert low <= 4.0 < high  # the upper edge is exclusive, as on the axis
        # the continuous sample alone is tightened to a fraction of that
        assert auto_range([background], mode="robust")[1] < 2.0

    def test_robust_range_judges_a_sample_the_same_way_alone_or_overlaid(self) -> None:
        """What a sample keeps does not depend on what it is plotted next to.

        Rejecting within each sample means a value is an outlier by its own
        sample's spread. A lone far value among near-identical ones is one either
        way; a whole sample somewhere else is one neither way.
        """
        rng = np.random.default_rng(0)
        sentinel_like = np.r_[np.zeros(999), 10.0]
        background = rng.normal(0, 1, 100_000)
        assert auto_range([sentinel_like], mode="robust")[1] < 10.0
        assert auto_range([background, sentinel_like], mode="robust")[1] < 10.0
        shifted = rng.normal(50, 1, 20_000)
        assert auto_range([shifted], mode="robust")[1] > 50.0
        assert auto_range([background, shifted], mode="robust")[1] > 50.0

    def test_robust_range_still_cuts_an_overlay_of_continuous_samples(self) -> None:
        """A zero budget is for categorical samples only, not for every overlay."""
        rng = np.random.default_rng(0)
        samples = [rng.exponential(30, 20_000), rng.exponential(30, 20_000)]
        assert auto_range(samples, mode="robust")[1] < auto_range(samples)[1]

    def test_robust_range_charges_the_budget_against_the_weights(self) -> None:
        """Rare entries carrying most of the content are not a cheap cut.

        The cluster is 0.25 percent of the entries -- inside the budget -- but
        carries most of the ``|weight|``, so the range must keep it. Unweighted,
        the same values are a thin tail and are cut.
        """
        rng = np.random.default_rng(0)
        values = np.r_[rng.normal(0, 1, 20_000), rng.normal(8, 0.2, 50)]
        weights = np.r_[np.ones(20_000), np.full(50, 1000.0)]
        low, high = auto_range([values], mode="robust", weights=[weights])
        assert high > 8.0
        kept = (values >= low) & (values < high)
        assert weights[kept].sum() / weights.sum() >= 1.0 - ROBUST_COVERAGE_BUDGET
        assert auto_range([values], mode="robust")[1] < 8.0  # unweighted: a thin tail

    def test_robust_range_weights_are_optional_and_per_sample(self) -> None:
        """``None`` weights fall back to counting entries, per sample."""
        rng = np.random.default_rng(0)
        plain = rng.normal(0, 1, 20_000)
        heavy = np.r_[rng.normal(0, 1, 5_000), rng.normal(9, 0.2, 20)]
        unweighted = auto_range([plain, heavy], mode="robust")
        assert auto_range([plain, heavy], mode="robust", weights=None) == unweighted
        assert auto_range([plain, heavy], mode="robust", weights=[None, None]) == unweighted
        # weighting only the second sample keeps its far cluster on the axis
        weights = [None, np.r_[np.ones(5_000), np.full(20, 5000.0)]]
        assert auto_range([plain, heavy], mode="robust", weights=weights)[1] > 9.0

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
        """Outliers are rejected within each sample, so a shifted sample is kept.

        A signal in the tail of this broad background is kept at both tested
        yields. One tens of deviations away from a narrow bulk is kept too, and
        by its own spread rather than by how much the other sample outnumbers it.
        """
        rng = np.random.default_rng(0)
        broad = rng.exponential(60, 100_000) + 50
        for n in (1_000, 100_000):
            signal = rng.normal(800, 25, n)
            assert auto_range([broad, signal], mode="robust") == auto_range([broad, signal])

        narrow = rng.normal(0, 1, 100_000)
        signal = rng.normal(50, 1, 20_000)
        low, high = auto_range([narrow, signal], mode="robust")
        assert high > 50.0  # the whole sample would otherwise be overflow
        assert float(((signal >= low) & (signal < high)).mean()) == 1.0
        # a twentieth of the yield is still kept: it is not a question of size
        tiny = rng.normal(50, 1, 1_000)
        assert auto_range([narrow, tiny], mode="robust")[1] > 50.0

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
        other = sample.replace(label="new", selection="MET > 1")
        assert other.label == "new"
        assert other.selection == Cut("MET > 1")
        assert other.source is sample.source

    def test_with_validates_like_init(self) -> None:
        sample = Sample({"x": np.arange(2.0)})
        with pytest.raises(ValueError, match="scale must be"):
            sample.replace(scale=np.nan)
        with pytest.raises(LuminosityError, match="finite"):
            sample.replace(ngen=np.inf)
        with pytest.raises(LuminosityError):
            sample.replace(xsec="bad")
        with pytest.raises(SourceError, match="cannot interpret"):
            sample.replace(source=42)
        with pytest.raises(TypeError, match="weight must be"):
            sample.replace(weight=42)
        with pytest.raises(TypeError, match="label must be"):
            sample.replace(label=3)
        assert sample.replace(weight="  ").weight is None
        assert sample.replace(scale=2).scale == 2.0
        assert sample.replace(source=sample.source).source is sample.source
        assert sample.replace(xsec="1.2 fb").xsec == "1.2 fb"

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
        assert style.replace(com=13.6).com == 13.6
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


class TestSystematic:
    def test_forms(self) -> None:
        systematics = as_systematics(
            {
                "one_sided": "w_up",
                "pair": ("w_up", "w_down"),
                "relative": 0.05,
                "factors": (1.1, 0.95),
                "branches": {"x": ("x_up", "x_down")},
                "one_sided_branches": {"x": "x_up", "y": "y_up"},
                "files": Systematic.samples("alt.root"),
            }
        )
        assert systematics["one_sided"] == Systematic("weight", "w_up")
        assert systematics["one_sided"].symmetric
        assert systematics["pair"] == Systematic("weight", "w_up", "w_down")
        assert systematics["relative"] == Systematic("norm", 1.05, 0.95)
        assert systematics["factors"] == Systematic("norm", 1.1, 0.95)
        assert systematics["branches"] == Systematic("replace", {"x": "x_up"}, {"x": "x_down"})
        assert systematics["one_sided_branches"].down is None
        assert systematics["files"] == Systematic("samples", "alt.root")
        assert repr(systematics["relative"]) == "Systematic(kind='norm', up=1.05, down=0.95)"
        assert as_systematics(None) == {}

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ([1, "a"], "cannot interpret"),
            (True, "cannot interpret"),
            ("w +", "not a valid expression"),
            (("w", "w +"), "not a valid expression"),
            ({}, "non-empty mapping"),
            ({"x": ("a", "b"), "y": "c"}, "every branch"),
            ({"x": 3}, "branch name or"),
            ({"x": ""}, "non-empty strings"),
            (float("nan"), "finite numbers"),
            ((1.1, float("inf")), "finite numbers"),
            (1.0, "magnitude from 0 to below 1"),
            (1.5, "magnitude from 0 to below 1"),
            (-0.05, "magnitude from 0 to below 1"),
            ((1.1, 0.0), "must be positive"),
            ((-1.1, 0.9), "must be positive"),
        ],
    )
    def test_invalid_forms(self, value: Any, message: str) -> None:
        with pytest.raises(SystematicError, match=message):
            as_systematics({"s": value})

    def test_invalid_mapping_and_kind(self) -> None:
        with pytest.raises(SystematicError, match="non-empty strings"):
            as_systematics({"": 0.1})
        with pytest.raises(SystematicError, match="must be a mapping"):
            as_systematics(["s"])  # type: ignore[arg-type]
        with pytest.raises(SystematicError, match="systematic kind"):
            Systematic("shape", "x")  # type: ignore[arg-type]
        with pytest.raises(SystematicError, match="needs an up"):
            Systematic("weight", None)

    def test_sample_field(self) -> None:
        sample = Sample({"x": [1.0, 2.0]}, label="S", systematics={"n": 0.1})
        assert sample.systematics == {"n": Systematic("norm", 1.1, 0.9)}
        assert "systematics=['n']" in repr(sample)
        assert sample.replace(systematics={"w": "x"}).systematics == {
            "w": Systematic("weight", "x")
        }
        assert Sample({"x": [1.0]}).systematics == {}
        with pytest.raises(SystematicError, match="sample 'S': systematic 'bad'"):
            sample.replace(systematics={"bad": object()})

    @pytest.mark.parametrize(
        ("kind", "up", "down"),
        [
            ("norm", float("nan"), None),
            ("norm", 1.1, float("inf")),
            ("norm", True, None),
            ("norm", "1.1", None),
            ("weight", 2.0, None),
            ("weight", "w", "w +"),
            ("replace", {}, None),
            ("replace", {"x": "up"}, {"x": 2.0}),
            ("replace", {"x": "up`"}, None),
        ],
    )
    def test_explicit_definitions_are_validated(self, kind: Any, up: Any, down: Any) -> None:
        with pytest.raises(SystematicError):
            Systematic(kind, up, down)

    def test_explicit_definitions_keep_the_short_form_invariants(self) -> None:
        with pytest.raises(SystematicError, match="same branches"):
            Systematic("replace", {"Jet_pt": "Jet_pt_up"}, {"MET": "MET_down"})
        assert Systematic("replace", {"x": "x_up"}, {"x": "x_down"}).down == {"x": "x_down"}
        for up in (2.0, 3.0):
            with pytest.raises(SystematicError, match="one-sided normalisation factor"):
                Systematic("norm", up)
        assert Systematic("norm", 1.9).symmetric
        assert Systematic("norm", 3.0, 0.5).up == 3.0

    def test_explicit_replacements_copy_input(self) -> None:
        replacements = {"x": "x_up"}
        systematic = Systematic("replace", replacements)
        replacements["x"] = "another"
        assert systematic.up == {"x": "x_up"}

    def test_data_samples_cannot_carry_systematics(self) -> None:
        columns = {"x": [1.0]}
        with pytest.raises(SystematicError, match="observed data"):
            Sample(columns, is_data=True, systematics={"s": 0.1})
        data = Sample(columns, is_data=True, systematics={})  # an empty mapping is fine
        with pytest.raises(SystematicError, match="observed data"):
            data.replace(systematics={"s": 0.1})
        mc = Sample(columns, systematics={"s": 0.1})
        with pytest.raises(SystematicError, match="observed data"):
            mc.replace(is_data=True)
        assert mc.replace(is_data=True, systematics={}).is_data
        assert data.replace(is_data=False, systematics={"s": 0.1}).systematics

    def test_systematic_mappings_are_read_only_snapshots(self) -> None:
        branches = {"x": ("x_up", "x_down")}
        given = {"shape": branches}
        sample = Sample({"x": [1.0]}, systematics=given)
        given.clear()
        branches["x"] = ("changed", "changed")
        systematic = sample.systematics["shape"]
        assert systematic.up == {"x": "x_up"}
        assert systematic.down == {"x": "x_down"}
        with pytest.raises(TypeError):
            sample.systematics["extra"] = Systematic("norm", 1.1)  # type: ignore[index]
        with pytest.raises(TypeError):
            systematic.up["x"] = "changed"
        with pytest.raises(TypeError):
            del systematic.down["x"]
        with pytest.raises(TypeError):
            Sample({"x": [1.0]}).systematics["extra"] = systematic  # type: ignore[index]

        updated = sample.replace(systematics={"norm": 0.1})
        assert list(updated.systematics) == ["norm"]
        assert list(sample.systematics) == ["shape"]
        with pytest.raises(TypeError):
            updated.systematics["extra"] = systematic  # type: ignore[index]

    @pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
    def test_immutable_systematics_support_copy_and_pickle(self, operation: str) -> None:
        sample = Sample({"x": [1.0]}, systematics={"shape": {"x": ("up", "down")}})
        restored = (
            pickle.loads(pickle.dumps(sample))
            if operation == "pickle"
            else getattr(copy, operation)(sample)
        )
        assert restored.systematics == sample.systematics
        assert repr(restored) == repr(sample)
        assert restored.source.arrays(["x"])["x"].to_list() == [1.0]
        with pytest.raises(TypeError):
            restored.systematics["new"] = Systematic("norm", 1.1)
        with pytest.raises(TypeError):
            restored.systematics["shape"].up["x"] = "changed"


class TestMergeTarget:
    def test_count_edges_or_nothing(self) -> None:
        from rootfig.model import merge_target

        assert merge_target(None) is None
        assert merge_target(None, "auto") is None
        assert merge_target(20) == 20
        assert merge_target(20, "robust") == 20
        np.testing.assert_allclose(merge_target(2, (0.0, 4.0)), [0.0, 2.0, 4.0])  # type: ignore[arg-type]
        np.testing.assert_allclose(merge_target((2, 0, 4)), [0.0, 2.0, 4.0])  # type: ignore[arg-type]
        np.testing.assert_allclose(merge_target([0, 1, 4]), [0.0, 1.0, 4.0])  # type: ignore[arg-type]
        np.testing.assert_allclose(merge_target(hist.axis.Regular(2, 0, 4)), [0.0, 2.0, 4.0])  # type: ignore[arg-type]
        assert merge_target(None, (0.0, 4.0)) == (0.0, 4.0)
        with pytest.raises(BinningError, match="strictly increasing"):
            merge_target([0, 4, 1])
        with pytest.raises(BinningError, match="positive"):
            merge_target(0)


class TestGroup:
    @staticmethod
    def _sample(label: str, **kwargs: Any) -> Sample:
        return Sample({"x": np.arange(3.0)}, label=label, **kwargs)

    def test_construction(self) -> None:
        a, b = self._sample("A"), self._sample("B")
        components = [a, b]
        group = Group(components, label="AB", color="C0", histtype="fill")
        components.append(a)  # the group keeps its own copy
        assert group.components == (a, b)
        assert group.samples == (a, b)
        assert (group.label, group.color, group.histtype) == ("AB", "C0", "fill")
        assert not group.is_data
        assert repr(group) == "Group(['A', 'B'], label='AB', color='C0', histtype='fill')"
        assert repr(Group([a], label="A")) == "Group(['A'], label='A')"

    def test_nested(self) -> None:
        ww, zz, qq, tautau = (self._sample(name) for name in ("WW", "ZZ", "qq", "tautau"))
        vv = Group([ww, zz], label="VV")
        other = Group([qq, tautau], label="Other")
        background = Group([vv, other], label="Background")
        assert background.components == (vv, other)
        assert background.samples == (ww, zz, qq, tautau)
        assert leaf_samples([background, ww]) == [ww, zz, qq, tautau, ww]

    def test_is_data(self) -> None:
        d1, d2 = self._sample("D1", is_data=True), self._sample("D2", is_data=True)
        data = Group([d1, d2], label="Data")
        assert data.is_data
        assert Group([data], label="outer").is_data
        with pytest.raises(
            ValueError, match=r"mixes observed data \(\['D1', 'D2'\]\) and simulated"
        ):
            Group([data, self._sample("A")], label="Mixed")

    @pytest.mark.parametrize(
        ("components", "error", "message"),
        [
            ([], ValueError, "group 'G' has no components"),
            (["a.root"], TypeError, r"component 0 of group 'G' is a str \('a.root'\).*rf.Sample"),
            ([Sample({"x": [1.0]}), 3], TypeError, "component 1 of group 'G' is a int"),
            ("a.root", TypeError, "Sample or Group objects, got str"),
            ({"A": Sample({"x": [1.0]})}, TypeError, "Sample or Group objects, got dict"),
            (Sample({"x": [1.0]}), TypeError, "Sample or Group objects, got Sample"),
        ],
    )
    def test_invalid_components(
        self, components: Any, error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=message):
            Group(components, label="G")

    def test_invalid_label(self) -> None:
        a = self._sample("A")
        with pytest.raises(TypeError, match="label must be a string, got int"):
            Group([a], label=3)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must not be blank"):
            Group([a], label="  ")

    def test_immutable(self) -> None:
        group = Group([self._sample("A")], label="G")
        with pytest.raises(FrozenInstanceError):
            group.label = "H"  # type: ignore[misc]
        assert isinstance(group.components, tuple)

    def test_replace(self) -> None:
        a, b, data = self._sample("A"), self._sample("B"), self._sample("D", is_data=True)
        group = Group([a], label="G", color="C1")
        changed = group.replace(label="H", components=[a, b])
        assert (changed.label, changed.components, changed.color) == ("H", (a, b), "C1")
        assert group.components == (a,)
        assert group.replace(components=[data]).is_data
        with pytest.raises(ValueError, match="mixes observed data"):
            group.replace(components=[a, data])
        with pytest.raises(TypeError, match="unknown Group field"):
            group.replace(colour="C2")
        with pytest.raises(ValueError, match="blank"):
            group.replace(label="")
        nested = group.replace(components=[group])  # holds the original: a tree, not a cycle
        assert nested.components == (group,)
        assert nested.samples == (a,)

    @pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
    def test_copy_and_pickle(self, operation: str) -> None:
        group = Group([Group([self._sample("A")], label="inner")], label="outer")
        restored = (
            pickle.loads(pickle.dumps(group))
            if operation == "pickle"
            else getattr(copy, operation)(group)
        )
        assert restored.label == "outer"
        assert [s.label for s in restored.samples] == ["A"]

    def test_map_samples(self) -> None:
        a, b = self._sample("A"), self._sample("B")
        group = Group([Group([a], label="inner"), b], label="outer", color="C0")
        marked = map_samples(group, lambda s: s.replace(is_data=True))
        assert isinstance(marked, Group)
        assert marked.is_data
        assert marked.color == "C0"
        assert [s.label for s in marked.samples] == ["A", "B"]
        assert not group.is_data
        assert map_samples(a, lambda s: s.replace(label="Z")).label == "Z"


class TestAsPlotItems:
    def test_group_free_input_matches_as_samples(
        self, signal_file: Path, background_file: Path
    ) -> None:
        for data in (
            signal_file,
            [signal_file, background_file],
            {"S": signal_file, "B": background_file},
            {"x": [1.0, 2.0]},
        ):
            items = as_plot_items(data, tree="events")
            samples = as_samples(data, tree="events")
            assert [(type(i), i.label) for i in items] == [(type(s), s.label) for s in samples]
        labelled = as_plot_items([signal_file, background_file], tree="events", labels=["S", "B"])
        assert [i.label for i in labelled] == ["S", "B"]

    def test_groups_in_lists_and_mappings(self, signal_file: Path) -> None:
        a = Sample({"x": [1.0]}, label="A")
        group = Group([a], label="G")
        assert as_plot_items(group) == [group]
        items = as_plot_items([group, a, signal_file], tree="events")
        assert items[:2] == [group, a]
        assert items[2].label == "signal"
        relabelled = as_plot_items({"VV": group, "S": a, "F": signal_file}, tree="events")
        assert [i.label for i in relabelled] == ["VV", "S", "F"]
        assert isinstance(relabelled[0], Group)
        assert relabelled[0].components == (a,)
        assert (group.label, a.label) == ("G", "A")  # the originals are untouched
        assert [i.label for i in as_plot_items([group, a], labels=["X", "Y"])] == ["X", "Y"]
        with pytest.raises(SourceError, match="got 1 labels for 2 samples"):
            as_plot_items([group, a], labels="X")

    def test_mapping_holding_a_group_is_a_label_map(self) -> None:
        group = Group([Sample({"x": [1.0]}, label="A")], label="G")
        with pytest.raises(SourceError, match="mapping holding groups"):
            as_plot_items({"G": group, "x": [1.0, 2.0]})
        with pytest.raises(SourceError, match="mapping holding groups"):
            as_plot_items({1: group})

    def test_as_samples_refuses_groups(self) -> None:
        a = Sample({"x": [1.0]}, label="A")
        group = Group([a], label="G")
        with pytest.raises(TypeError, match=r"not accepted here \(\['G'\]\).*group.samples"):
            as_samples([group, a])
        with pytest.raises(TypeError, match="not accepted here"):
            as_samples({"G": group})
        assert as_samples([a]) == [a]


@pytest.mark.parametrize("count", [np.int64(3), np.int32(3)])
def test_numpy_integer_counts(count: np.integer[Any]) -> None:
    import rootfig as rf
    from rootfig.model.binning import merge_target

    target = merge_target(count)  # type: ignore[arg-type]
    assert target == 3
    assert type(target) is int
    h = hist.Hist(hist.axis.Regular(6, 0, 6), storage=hist.storage.Weight())
    assert rf.Histogram(h, label="h").rebinned_to(count).axis.size == 3
    p = rf.plot(h, bins=count)  # type: ignore[arg-type]
    assert p.histograms[0].axis.size == 3
    p.close()
    x = {"x": np.arange(6.0)}
    assert rf.histogram(x, "x", bins=count, range=(0, 6)).axes[0].size == 3  # type: ignore[arg-type]
    # not the edges 3, 10, 20
    axis = rf.histogram(x, "x", bins=(count, 10, 20)).axes[0]  # type: ignore[arg-type]
    np.testing.assert_allclose(axis.edges, np.linspace(10, 20, 4))


@pytest.mark.parametrize(
    "axis",
    [
        hist.axis.IntCategory([0, 4, 5]),
        hist.axis.StrCategory(["a", "b"]),
        hist.axis.Integer(0, 6),
        hist.axis.Boolean(),
        bh.axis.Regular(3, 0, 6),
    ],
)
def test_bins_refuses_unsupported_axes(axis: Any) -> None:
    import rootfig as rf
    from rootfig.model.binning import validate_bins

    with pytest.raises(BinningError, match=type(axis).__name__):
        validate_bins(axis, None)
    with pytest.raises(BinningError, match=r"hist\.axis\.Regular and hist\.axis\.Variable"):
        validate_bins(axis, None)
    with pytest.raises(BinningError, match=type(axis).__name__):
        rf.histogram({"x": np.arange(6)}, "x", bins=axis)
    h = hist.Hist(hist.axis.Regular(6, 0, 6), storage=hist.storage.Weight())
    with pytest.raises(BinningError, match=type(axis).__name__):
        rf.plot(h, bins=axis)
