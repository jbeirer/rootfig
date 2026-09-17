"""Shared fixtures: small deterministic ROOT files written with uproot.

Two physics-like samples ("signal" and "background") are generated with a
seeded random generator and written both as ``TTree`` and as ``RNTuple`` so
every file-based test can run against either format. No ROOT installation is
needed.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib
import numpy as np
import pytest
import uproot

matplotlib.use("Agg")

N_EVENTS = 2000


def make_events(seed: int, *, signal: bool) -> dict[str, Any]:
    """Generate one sample as a dict of column arrays (flat and jagged)."""
    rng = np.random.default_rng(seed)
    n = N_EVENTS
    n_muon = rng.poisson(1.6 if signal else 1.1, size=n)
    total = int(n_muon.sum())
    scale = 40.0 if signal else 25.0
    muon_pt = ak.unflatten(rng.exponential(scale, size=total) + 5.0, n_muon)
    muon_eta = ak.unflatten(rng.uniform(-2.7, 2.7, size=total), n_muon)
    muon_phi = ak.unflatten(rng.uniform(-np.pi, np.pi, size=total), n_muon)
    muon_charge = ak.unflatten(rng.choice([-1, 1], size=total).astype(np.int32), n_muon)
    muon_is_tight = ak.unflatten(rng.random(size=total) < 0.7, n_muon)
    met = rng.exponential(30.0 if signal else 20.0, size=n)
    weight = rng.normal(1.0, 0.1, size=n)
    # a few sentinel / pathological values for the robustness tests
    sentinel = np.where(rng.random(size=n) < 0.02, -999.0, rng.normal(0.0, 1.0, size=n))
    with_nan = rng.normal(100.0, 15.0, size=n)
    with_nan[:5] = np.nan
    with_nan[5:8] = np.inf
    return {
        "event": np.arange(n, dtype=np.int64),
        "run": np.full(n, 1 if signal else 2, dtype=np.int32),
        "nMuon": n_muon.astype(np.int32),
        "Muon_pt": muon_pt,
        "Muon_eta": muon_eta,
        "Muon_phi": muon_phi,
        "Muon_charge": muon_charge,
        "Muon_isTight": muon_is_tight,
        "MET": met,
        "weight": weight,
        "sentinel": sentinel,
        "with_nan": with_nan,
        "jet1_b-tag": rng.random(size=n),
        "is_signal": np.full(n, signal),
    }


def write_ttree(path: Path, tree: str, columns: dict[str, Any]) -> None:
    """Write ``columns`` as a TTree (uproot writes RNTuple by default, so use mktree)."""
    with uproot.recreate(path) as file:
        out = file.mktree(tree, {k: _uproot_type(v) for k, v in columns.items()})
        out.extend(
            {k: ak.Array(v) if not isinstance(v, ak.Array) else v for k, v in columns.items()}
        )


def _uproot_type(column: Any) -> Any:
    array = column if isinstance(column, ak.Array) else ak.Array(column)
    if array.layout.purelist_depth == 1:
        return ak.to_numpy(array).dtype
    return f"var * {ak.to_numpy(ak.flatten(array)).dtype}"


def write_rntuple(path: Path, tree: str, columns: dict[str, Any]) -> None:
    """Write ``columns`` as an RNTuple (uproot's default on assignment)."""
    with uproot.recreate(path) as file:
        file[tree] = {
            k: ak.Array(v) if not isinstance(v, ak.Array) else v for k, v in columns.items()
        }


@pytest.fixture(scope="session")
def signal_columns() -> dict[str, Any]:
    return make_events(1, signal=True)


@pytest.fixture(scope="session")
def background_columns() -> dict[str, Any]:
    return make_events(2, signal=False)


@pytest.fixture(scope="session")
def data_dir(
    tmp_path_factory: pytest.TempPathFactory,
    signal_columns: dict[str, Any],
    background_columns: dict[str, Any],
) -> Path:
    """Directory with signal.root / background.root (TTree) and *_rntuple.root (RNTuple)."""
    directory = tmp_path_factory.mktemp("rootfiles")
    write_ttree(directory / "signal.root", "events", signal_columns)
    write_ttree(directory / "background.root", "events", background_columns)
    write_rntuple(directory / "signal_rntuple.root", "events", signal_columns)
    write_rntuple(directory / "background_rntuple.root", "events", background_columns)
    # split background into two files to exercise multi-file inputs
    half = N_EVENTS // 2
    first = {k: v[:half] for k, v in background_columns.items()}
    second = {k: v[half:] for k, v in background_columns.items()}
    write_ttree(directory / "bkg_part1.root", "events", first)
    write_ttree(directory / "bkg_part2.root", "events", second)
    # a file with two trees and a histogram, to test tree detection
    with uproot.recreate(directory / "multi.root") as file:
        file["a"] = {"x": np.arange(10.0)}
        file["b"] = {"y": np.arange(5.0)}
        file["h"] = np.histogram(np.arange(10.0), bins=5)
    with uproot.recreate(directory / "no_tree.root") as file:
        file["h"] = np.histogram(np.arange(10.0), bins=5)
    return directory


@pytest.fixture(scope="session")
def signal_file(data_dir: Path) -> Path:
    return data_dir / "signal.root"


@pytest.fixture(scope="session")
def background_file(data_dir: Path) -> Path:
    return data_dir / "background.root"


@pytest.fixture(scope="session", params=["signal.root", "signal_rntuple.root"])
def signal_file_any_format(request: pytest.FixtureRequest, data_dir: Path) -> Path:
    """The signal sample as TTree and as RNTuple."""
    return data_dir / str(request.param)


@pytest.fixture
def signal_arrays(signal_columns: dict[str, Any]) -> dict[str, ak.Array]:
    return {k: ak.Array(v) if not isinstance(v, ak.Array) else v for k, v in signal_columns.items()}


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def root_tutorials_dir() -> Path | None:
    """Locate ROOT's tutorials directory via ``root-config``; None if unavailable."""
    exe = shutil.which("root-config")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--tutdir"], capture_output=True, text=True, check=True, timeout=20
        )
    except (subprocess.SubprocessError, OSError):
        return None
    path = Path(out.stdout.strip())
    return path if path.is_dir() else None


@pytest.fixture(scope="session")
def tutorials_dir() -> Path:
    directory = root_tutorials_dir()
    if directory is None:
        pytest.skip("ROOT tutorials directory not available")
    return directory


@pytest.fixture
def approx() -> Callable[..., Any]:
    return pytest.approx


def write_stored_histograms(directory: Path) -> None:
    """Write framework-style histogram files (one per process) with uproot.

    Every ``<process>_sel0_histo.root`` holds ``mz`` (``Weight`` storage, so the
    file carries ``Sumw2``), an unweighted ``mz_raw`` written as a plain count
    histogram, a 2D ``mz_recoil_2D`` and a one-bin ``eventsProcessed``. Further
    files exercise the inference rule: a tree with a branch ``mz`` next to a
    histogram ``mz``, several trees next to it, a tree without that branch, a
    histogram in a directory, and one with another binning. ``untitled_2D.root``
    holds 2D histograms whose axes carry only placeholder titles: ROOT's empty
    ones (``mz_recoil_2D``) and uproot's ``Axis 0``/``Axis 1`` (``hist_2D``).
    ``unsupported.root`` holds objects rootfig cannot plot (a ``TProfile``
    ``prof`` and a ``TH3`` ``h3``), ``unsupported_with_tree.root`` the same
    next to a tree whose branch ``prof`` takes that name.
    """
    import hist

    rng = np.random.default_rng(11)
    axis = hist.axis.Regular(100, 0.0, 250.0, name="mz", label="m_{Z} [GeV]")
    for process, mean, n in (("ZH", 91.0, 4000), ("WW", 80.0, 2000), ("ZZ", 91.0, 1000)):
        values = rng.normal(mean, 6.0, n)
        weighted = hist.Hist(axis, storage=hist.storage.Weight())
        weighted.fill(values, weight=0.5)
        two_d = hist.Hist(
            hist.axis.Regular(10, 80.0, 100.0, label="m_{Z} [GeV]"),
            hist.axis.Regular(12, 120.0, 140.0, label="recoil [GeV]"),
            storage=hist.storage.Weight(),
        )
        two_d.fill(rng.normal(91.0, 3.0, n), rng.normal(125.0, 4.0, n), weight=0.5)
        cutflow = hist.Hist(
            hist.axis.StrCategory(["all", "sel0", "sel1"], label="Selection"),
            storage=hist.storage.Weight(),
        )
        cutflow.fill(["all"] * n + ["sel0"] * (n // 2) + ["sel1"] * (n // 4), weight=0.5)
        with uproot.recreate(directory / f"{process}_sel0_histo.root") as file:
            file["mz"] = weighted
            file["mz_raw"] = np.histogram(values, bins=100, range=(0.0, 250.0))
            file["mz_recoil_2D"] = two_d
            file["cutflow"] = cutflow
            file["eventsProcessed"] = np.histogram(np.full(n, 0.5), bins=1, range=(0.0, 1.0))
    with uproot.recreate(directory / "untitled_2D.root") as file:
        file["mz_recoil_2D"] = np.histogram2d(
            rng.normal(91.0, 3.0, 300),
            rng.normal(130.0, 3.0, 300),
            bins=(10, 12),
            range=((80.0, 100.0), (120.0, 140.0)),
        )
        file["hist_2D"] = hist.Hist(
            hist.axis.Regular(10, 80.0, 100.0), hist.axis.Regular(12, 120.0, 140.0)
        )
    # the same binning as ``mz`` but no Sumw2 and ROOT's placeholder axis title
    with uproot.recreate(directory / "mixed_storage.root") as file:
        file["mz"] = np.histogram(rng.normal(91.0, 6.0, 500), bins=100, range=(0.0, 250.0))
    # negative contents without Sumw2: the counts it reports as variances are negative
    signed = hist.Hist(hist.axis.Regular(100, 0.0, 250.0, label="m_{Z} [GeV]"))
    signed[...] = np.where(np.arange(100) % 2, 2.0, -1.0)
    with uproot.recreate(directory / "negative.root") as file:
        file["mz"] = signed
    with uproot.recreate(directory / "other_binning.root") as file:
        file["mz"] = np.histogram(rng.normal(91.0, 6.0, 500), bins=50, range=(0.0, 250.0))
    with uproot.recreate(directory / "branch_and_histogram.root") as file:
        file.mktree("events", {"mz": np.float64})
        file["events"].extend({"mz": rng.normal(91.0, 6.0, 300)})
        file["mz"] = np.histogram(rng.normal(20.0, 1.0, 500), bins=100, range=(0.0, 250.0))
    profile = hist.Hist(hist.axis.Regular(3, 0.0, 1.0), storage=hist.storage.Mean())
    profile.fill([0.1, 0.5, 0.9], sample=[1.0, 2.0, 3.0])
    cube = hist.Hist(*[hist.axis.Regular(2, 0.0, 1.0) for _ in range(3)])
    with uproot.recreate(directory / "two_trees.root") as file:
        file.mktree("a", {"x": np.float64})
        file["a"].extend({"x": np.arange(3.0)})
        file.mktree("b", {"y": np.float64})
        file["b"].extend({"y": np.arange(3.0)})
        file["mz"] = np.histogram(rng.normal(91.0, 6.0, 500), bins=100, range=(0.0, 250.0))
        file["prof"] = profile
    with uproot.recreate(directory / "unsupported.root") as file:
        file["prof"] = profile
        file["h3"] = cube
    with uproot.recreate(directory / "unsupported_with_tree.root") as file:
        file.mktree("events", {"pt": np.float64, "prof": np.float64})
        file["events"].extend({"pt": np.arange(3.0), "prof": np.arange(3.0)})
        file["prof"] = profile
        file["h3"] = cube
    with uproot.recreate(directory / "tree_without_branch.root") as file:
        file.mktree("events", {"pt": np.float64})
        file["events"].extend({"pt": np.arange(3.0)})
        file["mz"] = np.histogram(rng.normal(91.0, 6.0, 500), bins=100, range=(0.0, 250.0))
    with uproot.recreate(directory / "in_directory.root") as file:
        file["sub/mz"] = np.histogram(rng.normal(91.0, 6.0, 500), bins=100, range=(0.0, 250.0))


@pytest.fixture(scope="session")
def stored_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory with framework-style histogram files (see :func:`write_stored_histograms`)."""
    directory = tmp_path_factory.mktemp("stored")
    write_stored_histograms(directory)
    return directory
