"""Static checks for the public Group contracts (run by mypy)."""

from typing import assert_type

import rootfig as rf
from rootfig.histograms import Histogram, build_histograms
from rootfig.model import PlotItem, as_plot_items, leaf_samples, map_samples


def groups(a: rf.Sample, b: rf.Sample) -> None:
    inner = rf.Group([a, b], label="AB", color="C0", histtype="fill")
    outer = rf.Group([inner, a], label="All")
    assert_type(outer.components, tuple[rf.Sample | rf.Group, ...])
    assert_type(outer.samples, tuple[rf.Sample, ...])
    assert_type(outer.is_data, bool)
    assert_type(outer.replace(label="X"), rf.Group)
    assert_type(as_plot_items([outer, a, "file.root"]), list[PlotItem])
    assert_type(leaf_samples([outer, a]), list[rf.Sample])
    assert_type(map_samples(outer, lambda s: s.replace(is_data=True)), PlotItem)
    assert_type(build_histograms([outer, a], "x"), list[Histogram])
    assert_type(rf.histograms(outer, "x"), list[Histogram])
    # Raw file specifications are not components; mypy reports an unused ignore if this loosens.
    rf.Group(["file.root"], label="raw")  # type: ignore[list-item]
