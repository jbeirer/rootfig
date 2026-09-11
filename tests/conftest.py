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
