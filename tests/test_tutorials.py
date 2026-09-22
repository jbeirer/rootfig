"""Integration tests against files shipped with ROOT's tutorials, when available locally.

These are skipped in CI (no ROOT there); they guard against surprises in
real-world files: TNtuple, ``std::vector`` branches, many-branch trees with
odd names such as ``jet1_b-tag``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rootfig as rf

pytestmark = pytest.mark.root_tutorials


def _require(tutorials_dir: Path, relative: str) -> Path:
    path = tutorials_dir / relative
    if not path.is_file():
        pytest.skip(f"{relative} not present in {tutorials_dir}")
    return path


def test_hsimple_ntuple(tutorials_dir: Path) -> None:
    path = _require(tutorials_dir, "hsimple.root")
    # TNtuple, tree auto-detected among TH1/TH2/TProfile objects
    p = rf.plot(path, "px", selection="pz > 0", weight="random", bins=(50, -4, 4), normalize=True)
    assert p.histograms[0].integral == pytest.approx(1.0)
    arrays = rf.load(path, ["px", "py"], selection="pz > 0")
    assert len(arrays) == p.histograms[0].stats.n_selected_events  # type: ignore[union-attr]
    p2 = rf.plot2d(path, "px", "py", bins=(30, -4, 4))
    assert p2.histograms[0].ndim == 2


def test_vector_branches(tutorials_dir: Path) -> None:
    path = _require(tutorials_dir, "analysis/dataframe/df017_vecOpsHEP.root")
    h = rf.histogram(
        path, "sqrt(px**2 + py**2)", tree="myDataset", selection="E > 100", bins=(20, 0, 200)
    )
    arrays = rf.load(path, ["px", "py", "E"], tree="myDataset")
    pt = np.sqrt(arrays["px"] ** 2 + arrays["py"] ** 2)
    selected = pt[arrays["E"] > 100]
    import awkward as ak

    assert h.sum(flow=True).value == pytest.approx(len(ak.flatten(selected)))
    table = rf.summarize(path, "count(px)", tree="myDataset")
    assert table.get("count(px)").entries == 3


def test_higgs_data_two_trees(tutorials_dir: Path) -> None:
    path = _require(tutorials_dir, "machine_learning/data/Higgs_data.root")
    with pytest.raises(rf.SourceError, match="several trees"):
        rf.histogram(path, "lepton_pT")
    sig = rf.Sample(f"{path}:sig_tree", label="Signal")
    bkg = rf.Sample(f"{path}:bkg_tree", label="Background")
    p = rf.plot(
        [sig, bkg],
        "`jet1_b-tag`",
        selection="lepton_pT > 1",
        bins=10,
        normalize=True,
        panel="ratio",
    )
    assert p.panel_ax is not None
    assert all(h.integral == pytest.approx(1.0) for h in p.histograms)
