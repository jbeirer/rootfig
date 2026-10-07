"""Integration tests against files shipped with ROOT's tutorials, when available locally.

These are skipped in CI (no ROOT there); they guard against surprises in
real-world files: TNtuple, ``std::vector`` branches, many-branch trees with
odd names such as ``jet1_b-tag``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import uproot

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


_GOODNESS_OF_FIT_MACRO = """
void gof() {
   TFile f("%s");
   auto all = f.Get<TH1>("hpx");
   auto tree = f.Get<TTree>("ntuple");
   TH1D sub("sub", "", 100, -4, 4), weighted("weighted", "", 100, -4, 4);
   tree->Draw("py>>sub", "random < 0.5", "goff");
   tree->Draw("py>>weighted", "random", "goff");
   for (auto h : {(TH1*)&sub, (TH1*)&weighted}) {
      Double_t chi2 = 0; Int_t ndf = 0, igood = 0;
      Double_t p = all->Chi2TestX(h, chi2, ndf, igood, h == &sub ? "UU" : "UW");
      printf("%%.17g %%d %%.17g %%.17g\\n", chi2, ndf, p, all->KolmogorovTest(h));
   }
}
"""


def test_goodness_of_fit_matches_root(tutorials_dir: Path, tmp_path: Path) -> None:
    # TH1::Chi2Test and TH1::KolmogorovTest of a stored TH1F against tree-filled histograms
    path = _require(tutorials_dir, "hsimple.root")
    root = shutil.which("root")
    if root is None:
        pytest.skip("the root executable is not available")
    macro = tmp_path / "gof.C"
    macro.write_text(_GOODNESS_OF_FIT_MACRO % path)
    # a cold start of ROOT from CVMFS (Key4hep) can take minutes before the macro runs
    out = subprocess.run(
        [root, "-l", "-b", "-q", str(macro)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    expected = [line.split() for line in out.stdout.splitlines() if line[:1].isdigit()]
    stored = uproot.open(path)["hpx"].to_hist()
    # py is drawn independently from the distribution of px: p-values well inside (0, 1)
    sub = rf.histogram(path, "py", tree="ntuple", selection="random < 0.5", bins=(100, -4, 4))
    weighted = rf.histogram(path, "py", tree="ntuple", weight="random", bins=(100, -4, 4))
    for histogram, method, (chi2, ndf, p_value, ks) in zip(
        (sub, weighted), ("UU", "UW"), expected, strict=True
    ):
        result = rf.goodness_of_fit(stored, histogram)
        assert result.method == method
        assert result.statistic == pytest.approx(float(chi2), rel=1e-9)
        assert result.ndf == int(ndf)
        assert result.p_value == pytest.approx(float(p_value), rel=1e-9)
        assert rf.goodness_of_fit(stored, histogram, test="ks").p_value == pytest.approx(
            float(ks), rel=1e-9
        )
