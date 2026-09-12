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
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent
GALLERY_DIR = ROOT / "examples" / "gallery"
HOOK_PY = ROOT / "docs" / "hooks" / "gallery.py"
BASELINE_DIR = ROOT / "docs" / "images" / "gallery"


def load_module(name: str, path: Path, *, package: bool = False) -> ModuleType:
    """Import a file (or a package's ``__init__.py``) under ``name``, replacing any earlier load."""
    for cached in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
        del sys.modules[cached]
    locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gallery(name: str = "rootfig_gallery") -> ModuleType:
    return load_module(name, GALLERY_DIR / "__init__.py", package=True)


gallery = load_gallery()
EXAMPLES = list(gallery.EXAMPLES)
SAVEFIG = {"dpi": 150}  # as Plot.save: constrained layout, no cropping


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory with the toy files; the examples run inside it."""
    return tmp_path_factory.mktemp("gallery")


@pytest.fixture(scope="module")
def dataset(workdir: Path) -> Any:
    return gallery.make_dataset(workdir)


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


def test_loading_the_package_twice_does_not_duplicate_examples() -> None:
    """The registry lives in a submodule; a stale cached copy would collect every example twice."""
    twice = load_gallery("rootfig_gallery_again")
    twice = load_gallery("rootfig_gallery_again")
    assert [ex.name for ex in twice.EXAMPLES] == [ex.name for ex in EXAMPLES]


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
def test_example(example: Any, dataset: Any, workdir: Path) -> Figure:
    before = Path.cwd()
    plot = example.run(dataset, cwd=workdir)
    assert Path.cwd() == before, "Example.run must restore the working directory"
    assert plot.fig is not None
    assert plot.ax is not None
    return plot.fig


PAGE = """# Gallery

<!-- gallery: quick -->

## Setup

<!-- gallery-setup -->

<!-- gallery -->

## Beyond figures
"""


def test_docs_hook_renders_every_example_once_in_marker_order() -> None:
    """The docs hook: a named marker pulls an example forward, the bare marker takes the rest."""
    hook = load_module("rootfig_gallery_hook", HOOK_PY)
    page = hook.on_page_markdown(PAGE)
    titles = re.findall(r"^## (.+)$", page, flags=re.MULTILINE)
    expected = ["The one-liner", "Setup", *[ex.title for ex in EXAMPLES if ex.name != "quick"]]
    assert titles == [*expected, "Beyond figures"]
    assert page.count("![") == len(EXAMPLES)
    assert 'signal = rf.Sample("signal.root"' in page  # the Setup block
    assert "return " not in page
    # rendering again (mkdocs serve rebuilds) must give the same page, not a doubled registry
    assert hook.on_page_markdown(PAGE) == page
    assert hook.on_page_markdown("no markers here") == "no markers here"
    with pytest.raises(ValueError, match="unknown gallery example"):
        hook.on_page_markdown("<!-- gallery: no_such_example -->")


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
