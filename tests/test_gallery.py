"""The gallery as an end-to-end and image-regression suite.

Every example in ``examples/gallery`` is run against a freshly generated toy
dataset. With ``pytest --mpl`` the resulting figures are compared pixel-wise
(RMS tolerance) against the PNGs in ``docs/images/gallery/``, which are also the
images shown in the documentation. Without ``--mpl`` the examples still run, so
a crash in any of them fails the suite on every platform; the comparison itself
is only enabled on Linux in CI because font rendering differs across systems.

Regenerate the baselines after an intended visual change::

    uv run pytest tests/test_gallery.py --mpl-generate-path=docs/images/gallery
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent
GALLERY_DIR = ROOT / "examples" / "gallery"
BASELINE_DIR = ROOT / "docs" / "images" / "gallery"


def load_gallery() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rootfig_gallery",
        GALLERY_DIR / "__init__.py",
        submodule_search_locations=[str(GALLERY_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gallery = load_gallery()
EXAMPLES = list(gallery.EXAMPLES)
SAVEFIG = {"dpi": 150}  # as Plot.save: constrained layout, no cropping


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return gallery.make_dataset(tmp_path_factory.mktemp("gallery"))


def test_registry_is_consistent() -> None:
    names = [example.name for example in EXAMPLES]
    assert len(names) == len(set(names)), "example names must be unique"
    assert len(names) >= 15
    for example in EXAMPLES:
        assert example.title
        assert example.description, f"{example.name} needs a docstring"
        body = gallery.body_source(example.func)
        assert "rf." in body
        assert "return " not in body
        assert '"""' not in body


def test_every_example_has_a_baseline() -> None:
    missing = [ex.name for ex in EXAMPLES if not (BASELINE_DIR / f"{ex.name}.png").is_file()]
    assert not missing, (
        f"no baseline image for {missing}; run "
        "`uv run pytest tests/test_gallery.py --mpl-generate-path=docs/images/gallery`"
    )
    stale = sorted(
        p.stem for p in BASELINE_DIR.glob("*.png") if p.stem not in {e.name for e in EXAMPLES}
    )
    assert not stale, f"baseline images without an example: {stale}"


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(
            example,
            id=example.name,
            marks=pytest.mark.mpl_image_compare(
                baseline_dir="../docs/images/gallery",
                filename=f"{example.name}.png",
                savefig_kwargs=SAVEFIG,
                style="default",
                deterministic=True,
            ),
        )
        for example in EXAMPLES
    ],
)
def test_example(example: Any, dataset: Any) -> Figure:
    plot = example.run(dataset)
    assert plot.fig is not None
    assert plot.ax is not None
    return plot.fig


def test_main_writes_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command-line path: dataset, numbered PNGs, summary table (two examples suffice)."""
    monkeypatch.setattr(gallery, "EXAMPLES", EXAMPLES[:2])
    gallery.main(tmp_path)
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == [f"01_{EXAMPLES[0].name}.png", f"02_{EXAMPLES[1].name}.png"]
    assert (tmp_path / "data.root").is_file()
    out = capsys.readouterr().out
    assert "figures written to" in out
    assert "Muon_pt" in out  # the summary table
