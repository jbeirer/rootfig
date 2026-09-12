"""The toy dataset: three simulated processes and one "observed" sample.

Muons and jets are jagged per event, ``MET``, ``m_ll``, ``lep_iso`` and the
``weight`` are per event. Nothing here is specific to rootfig; it only produces
small ``TTree`` files the examples can read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import awkward as ak
import numpy as np
import uproot

N_EVENTS = 20_000

Kind = Literal["signal", "zjets", "diboson"]


def make_events(kind: Kind, n: int, seed: int) -> dict[str, Any]:
    """Generate ``n`` events of one process as a dict of flat and jagged columns."""
    rng = np.random.default_rng(seed)
    p = {
        "signal": {"mu": 2.0, "pt": 60.0, "tight": 0.85, "nj": 3.0, "jpt": 70.0, "met": 60.0},
        "zjets": {"mu": 1.9, "pt": 28.0, "tight": 0.75, "nj": 1.5, "jpt": 40.0, "met": 22.0},
        "diboson": {"mu": 1.6, "pt": 40.0, "tight": 0.75, "nj": 2.2, "jpt": 45.0, "met": 35.0},
    }[kind]

    n_muon = rng.poisson(p["mu"], size=n)
    n_mu_total = int(n_muon.sum())
    n_jet = rng.poisson(p["nj"], size=n)
    n_jet_total = int(n_jet.sum())
    jet_pt = ak.unflatten(rng.exponential(p["jpt"], n_jet_total) + 20.0, n_jet)
    ht = ak.to_numpy(ak.fill_none(ak.sum(jet_pt, axis=1), 0.0))

    if kind == "signal":
        m_ll = rng.normal(220.0, 5.0, n)
        btag = rng.beta(4.0, 1.5, n_jet_total)
    elif kind == "zjets":
        m_ll = np.where(
            rng.random(n) < 0.85,
            rng.normal(91.2, 3.5, n),
            50.0 + rng.exponential(45.0, n),
        )
        btag = rng.beta(1.0, 6.0, n_jet_total)
    else:
        m_ll = np.where(
            rng.random(n) < 0.35,
            rng.normal(91.2, 4.0, n),
            50.0 + rng.exponential(70.0, n),
        )
        btag = rng.beta(1.5, 4.0, n_jet_total)

    return {
        "event": np.arange(n, dtype=np.int64),
        "nMuon": n_muon.astype(np.int32),
        "Muon_pt": ak.unflatten(rng.exponential(p["pt"], n_mu_total) + 5.0, n_muon),
        "Muon_eta": ak.unflatten(rng.uniform(-2.7, 2.7, n_mu_total), n_muon),
        "Muon_phi": ak.unflatten(rng.uniform(-np.pi, np.pi, n_mu_total), n_muon),
        "Muon_charge": ak.unflatten(rng.choice([-1, 1], n_mu_total).astype(np.int32), n_muon),
        "Muon_isTight": ak.unflatten(rng.random(n_mu_total) < p["tight"], n_muon),
        "nJet": n_jet.astype(np.int32),
        "Jet_pt": jet_pt,
        "Jet_eta": ak.unflatten(rng.uniform(-4.5, 4.5, n_jet_total), n_jet),
        "Jet_btag": ak.unflatten(btag, n_jet),
        "MET": rng.exponential(p["met"], n) + rng.normal(0.25, 0.1, n).clip(0.0) * ht,
        "m_ll": m_ll,
        # an isolation-like variable with a -999 sentinel for "not computed"
        "lep_iso": np.where(rng.random(n) < 0.03, -999.0, rng.exponential(0.08, n)),
        "weight": rng.normal(1.0, 0.1, n),
    }


def concatenate(*parts: dict[str, Any]) -> dict[str, Any]:
    """Concatenate event dicts column by column."""
    return {k: ak.concatenate([ak.Array(part[k]) for part in parts]) for k in parts[0]}


def write_tree(path: Path, columns: dict[str, Any], tree: str = "events") -> None:
    """Write ``columns`` as a TTree (uproot writes RNTuple on assignment, so use mktree)."""
    arrays = {k: v if isinstance(v, ak.Array) else ak.Array(v) for k, v in columns.items()}
    types = {
        k: ak.to_numpy(v).dtype
        if v.layout.purelist_depth == 1
        else f"var * {ak.to_numpy(ak.flatten(v)).dtype}"
        for k, v in arrays.items()
    }
    with uproot.recreate(path) as file:
        out = file.mktree(tree, types)
        out.extend(arrays)


def write_dataset(out: Path) -> None:
    """Write signal.root, background.root, diboson.root and data.root into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    write_tree(out / "signal.root", make_events("signal", N_EVENTS, seed=1))
    write_tree(out / "background.root", make_events("zjets", N_EVENTS, seed=2))
    write_tree(out / "diboson.root", make_events("diboson", N_EVENTS, seed=3))
    # "data" is a Z+jets-like sample plus a diboson-like admixture, unweighted
    observed = concatenate(
        make_events("zjets", N_EVENTS, seed=4), make_events("diboson", N_EVENTS * 3 // 20, seed=5)
    )
    observed["weight"] = np.ones(len(observed["event"]))
    write_tree(out / "data.root", observed)
