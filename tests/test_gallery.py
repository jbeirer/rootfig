"""The gallery as an end-to-end and image-regression suite.

Every example in ``examples/gallery`` is run against a freshly generated toy
dataset. With ``pytest --mpl`` the resulting figures are compared pixel-wise
(RMS tolerance) against the PNGs in ``docs/images/gallery/``, which are also the
images shown in the documentation. Without ``--mpl`` the examples still run, so
a crash in any of them fails the suite on every platform; the comparison itself
is only enabled on Linux in CI because font rendering differs across systems.

An example that takes a ``style`` is compared in every gallery style
(``<name>-<style>.png``; the default style keeps ``<name>.png``), and every
rendering twice: as drawn and inside :func:`rootfig.dark_theme` (``-dark``), the
image shown on dark pages. The experiment styles are drawn only when images are
compared or generated, which keeps a plain ``pytest`` run fast; ``-n auto``
(pytest-xdist) spreads the comparison over all cores.

Regenerate the baselines after an intended visual change::

    uv run pytest tests/test_gallery.py -n auto --mpl-generate-path=docs/images/gallery
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from matplotlib.figure import Figure

import rootfig as rf
from rootfig.model.style import EXPERIMENT_STYLES

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
SAMPLE_ATTRIBUTES = {
    field.name for field in dataclasses.fields(gallery.Dataset) if "Sample" in str(field.type)
}
VARIANTS = [
    (example, style, dark)
    for example in EXAMPLES
    for style in gallery.styles_of(example)
    for dark in (False, True)
]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory with the toy files; the examples run inside it."""
    return tmp_path_factory.mktemp("gallery")


@pytest.fixture(scope="module")
def dataset(workdir: Path) -> Any:
    return gallery.make_dataset(workdir)


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    return load_module("rootfig_gallery_hook", HOOK_PY)


def test_registry_is_consistent() -> None:
    names = [example.name for example in EXAMPLES]
    assert len(names) == len(set(names)), "example names must be unique"
    assert len(names) >= 15
    sections = [example.section for example in EXAMPLES]
    assert sections == sorted(sections, key=sections.index), "keep a section's examples together"
    for example in EXAMPLES:
        assert example.title
        assert example.description, f"{example.name} needs a docstring"
        body = gallery.body_source(example.func)
        assert "rf." in body
        assert "return " not in body
        assert '"""' not in body
        if example.styled:  # the style it is given must reach the figure
            assert "style=style" in body, example.name
        # the metadata the docs rely on to explain the toy files matches what the code reads
        takes_samples = bool(set(example.parameters) & SAMPLE_ATTRIBUTES)
        assert example.toy_data == ('.root"' in body or takes_samples), example.name


def test_styles_are_shown_as_the_code_that_makes_them() -> None:
    assert next(iter(gallery.STYLES)) == gallery.DEFAULT_STYLE
    assert gallery.STYLES[gallery.DEFAULT_STYLE] == rf.Style()
    experiments = {style.experiment for style in gallery.STYLES.values()} - {None}
    assert experiments == set(EXPERIMENT_STYLES)
    for style in gallery.STYLES.values():
        assert eval(gallery.style_source(style), {"rf": rf}) == style
    assert gallery.style_source(rf.Style(experiment="CMS", lumi=3.5)) == (
        'rf.Style(experiment="CMS", lumi=3.5)'
    )


def test_loading_the_package_twice_does_not_duplicate_examples() -> None:
    """The registry lives in a submodule; a stale cached copy would collect every example twice."""
    twice = load_gallery("rootfig_gallery_again")
    twice = load_gallery("rootfig_gallery_again")
    assert [ex.name for ex in twice.EXAMPLES] == [ex.name for ex in EXAMPLES]


def test_every_example_has_a_baseline() -> None:
    expected = {example.image(style, dark=dark) for example, style, dark in VARIANTS}
    missing = sorted(name for name in expected if not (BASELINE_DIR / name).is_file())
    assert not missing, (
        f"no baseline image for {missing}; run `uv run pytest tests/test_gallery.py -n auto "
        "--mpl-generate-path=docs/images/gallery`"
    )
    stale = sorted(p.name for p in BASELINE_DIR.glob("*.png") if p.name not in expected)
    assert not stale, f"baseline images without an example: {stale}"


def compares_images(config: pytest.Config) -> bool:
    """Whether this run compares or generates the images (pytest-mpl)."""
    return bool(config.getoption("--mpl") or config.getoption("--mpl-generate-path"))


@pytest.mark.parametrize(
    ("example", "style", "dark"),
    [
        pytest.param(
            example,
            style,
            dark,
            id=example.image(style, dark=dark).removesuffix(".png"),
            marks=pytest.mark.mpl_image_compare(
                baseline_dir="../docs/images/gallery",
                filename=example.image(style, dark=dark),
                savefig_kwargs=SAVEFIG,
                style="default",
                deterministic=True,
            ),
        )
        for example, style, dark in VARIANTS
    ],
)
def test_example(
    example: Any,
    style: str,
    dark: bool,
    dataset: Any,
    workdir: Path,
    request: pytest.FixtureRequest,
) -> Figure:
    if style != gallery.DEFAULT_STYLE and not compares_images(request.config):
        pytest.skip("the experiment styles are drawn when images are compared (--mpl)")
    before = Path.cwd()
    with rf.dark_theme() if dark else contextlib.nullcontext():
        plot = example.run(dataset, cwd=workdir, style=gallery.STYLES[style])
    assert Path.cwd() == before, "Example.run must restore the working directory"
    assert plot.fig is not None
    assert plot.ax is not None
    return plot.fig


def test_example_pages(hook: ModuleType) -> None:
    """One page per example: a tab per style with its images and the complete code."""
    for example in EXAMPLES:
        styles = gallery.styles_of(example)
        page = hook.render_page(gallery, example)
        assert f"\n# {example.title}\n\n{example.description}\n" in page
        tabs = re.findall(r"^/// tab \| (.+)$", page, flags=re.MULTILINE)
        assert tabs == (list(styles) if example.styled else [])
        images = re.findall(r"\]\(\.\./images/gallery/([^)#]+)#only-(?:light|dark)\)", page)
        assert images == [example.image(s, dark=dark) for s in styles for dark in (False, True)]
        assert "return " not in page
        shared = any(name != "style" for name in example.parameters)
        assert ("/// details | Setup" in page) == (example.toy_data or shared)
        # a page read on its own says how to make the files its code opens
        assert ("python examples/gallery" in page) == example.toy_data
        assert ('signal = rf.Sample("signal.root"' in page) == shared
        code_blocks = re.findall(r'```python(?: hl_lines="([\d ]+)")?\n(.*?)```', page, re.DOTALL)
        for (numbers, code), (name, style) in zip(code_blocks, styles.items(), strict=False):
            lines = code.splitlines()
            assert "import rootfig as rf" in lines
            for alias in ("plt", "np"):
                assert (f" as {alias}" in code) == (f"{alias}." in code), (example.name, alias)
            highlighted = [lines[int(number) - 1] for number in numbers.split()]
            if example.styled:
                assert highlighted[0] == f"style = {gallery.style_source(style)}", name
                assert highlighted[1:] == [line for line in lines if "style=style" in line]
            else:
                assert highlighted == []


def test_overview(hook: ModuleType) -> None:
    """A tab per style, holding every example once as a card, section by section."""
    page = hook.on_page_markdown("# Gallery\n\n<!-- gallery-overview -->\n\n## Beyond figures\n")
    assert "<!-- gallery-overview -->" not in page
    assert page == hook.on_page_markdown(
        "# Gallery\n\n<!-- gallery-overview -->\n\n## Beyond figures\n"
    ), "rendering again (mkdocs serve rebuilds) must give the same page"
    assert hook.on_page_markdown("no markers here") == "no markers here"
    tabs = re.split(r"^//// tab \| (.+)$", page, flags=re.MULTILINE)[1:]
    assert tabs[::2] == list(gallery.STYLES)
    sections = list(dict.fromkeys(example.section for example in EXAMPLES))
    for style, content in zip(tabs[::2], tabs[1::2], strict=True):
        titles = re.findall(r"^///// html \| p\.gallery-section.*\n\n(.+)$", content, re.MULTILINE)
        assert titles == sections
        for example in EXAMPLES:
            assert content.count(f"]({example.name}.md)") == 1
            shown = style if example.styled else gallery.DEFAULT_STYLE
            assert f"](../images/gallery/{example.image(shown)}#only-light)" in content


def test_navigation(hook: ModuleType) -> None:
    """The gallery entry becomes a section: the overview, then the pages by section."""
    nav = [{"Home": "index.md"}, {"Gallery": "gallery/index.md"}, {"API": "api.md"}]
    home, section, api = hook.gallery_nav(nav, gallery)
    assert home == nav[0]
    assert api == nav[2]
    overview, *groups = section["Gallery"]
    assert overview == "gallery/index.md"
    assert [next(iter(group)) for group in groups] == list(
        dict.fromkeys(example.section for example in EXAMPLES)
    )
    pages = [page for group in groups for page in next(iter(group.values()))]
    assert pages == [f"gallery/{example.name}.md" for example in EXAMPLES]


def test_main_writes_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command-line path: dataset, numbered PNGs, summary table (two examples suffice)."""
    from PIL import Image

    monkeypatch.setattr(gallery, "EXAMPLES", EXAMPLES[:2])
    gallery.main(tmp_path, style="CMS")
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == [f"01_{EXAMPLES[0].name}.png", f"02_{EXAMPLES[1].name}.png"]
    assert (tmp_path / "data.root").is_file()
    width, _ = Image.open(tmp_path / written[0]).size
    assert width == 150 * 10  # the CMS style draws on a 10 inch wide figure
    out = capsys.readouterr().out
    assert "figures written to" in out
    assert "Muon_pt" in out  # the summary table
