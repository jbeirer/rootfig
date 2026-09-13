"""MkDocs hook that builds the gallery pages from the ``examples/gallery`` package.

The gallery is an overview page and one page per example, all made from the
registry:

``gallery/index.md``
    carries the ``<!-- gallery-overview -->`` marker, replaced by one tab per
    gallery style, each holding the examples section by section as a grid of
    cards (thumbnail and title, linking to the example's page);
``gallery/<name>.md``
    generated for every example (``on_files``), with no file in ``docs/``: the
    title, the description, one tab per style the example is drawn in with its
    image (``docs/images/gallery/<name>[-<style>].png``, and ``-dark`` for the
    dark palette, switched by Material's ``#only-light``/``#only-dark``) and the
    complete code, and the setup block the code relies on.

The Gallery entry of the navigation, ``gallery/index.md``, becomes a section with
the overview and the example pages, grouped like the overview (``on_config``).
Every tab set uses the same labels, so Material's ``content.tabs.link`` switches
all figures on all pages together. Tabs, cards and the setup block are
pymdownx blocks, which nest without indentation.

The gallery module is imported, never executed, so building the docs needs no
data and draws nothing. Because the images are also the baselines of
``tests/test_gallery.py``, picture and code cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # MkDocs runs the hook; the tests import it without MkDocs installed
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files

ROOT = Path(__file__).resolve().parents[2]
GALLERY_DIR = ROOT / "examples" / "gallery"
OVERVIEW = "gallery/index.md"
OVERVIEW_MARKER = "<!-- gallery-overview -->"
IMAGES = "../images/gallery"  # relative to the gallery pages


def _load_gallery(name: str = "rootfig_gallery_docs") -> ModuleType:
    """Import ``examples/gallery`` as a fresh package, without ``examples/`` on ``sys.path``.

    Every call re-executes the package (``mkdocs serve`` rebuilds on each change),
    so its submodules are evicted from ``sys.modules`` first: a cached
    ``registry`` would keep its ``EXAMPLES`` list and the re-run decorators would
    register every example a second time.
    """
    for cached in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
        del sys.modules[cached]
    spec = importlib.util.spec_from_file_location(
        name, GALLERY_DIR / "__init__.py", submodule_search_locations=[str(GALLERY_DIR)]
    )
    if spec is None or spec.loader is None:
        msg = f"cannot load {GALLERY_DIR}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# MkDocs events
# --------------------------------------------------------------------------------------


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Turn the navigation's gallery entry into the gallery section."""
    if config.nav is not None:
        config.nav = gallery_nav(config.nav, _load_gallery())
    return config


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add a generated page for every example."""
    from mkdocs.structure.files import File  # noqa: PLC0415 - only MkDocs calls this

    gallery = _load_gallery()
    for example in gallery.EXAMPLES:
        content = render_page(gallery, example)
        files.append(File.generated(config, page_path(example), content=content))
    return files


def on_page_markdown(markdown: str, **_: Any) -> str:
    """Expand the overview marker (MkDocs ``on_page_markdown`` event)."""
    if OVERVIEW_MARKER not in markdown:
        return markdown
    return markdown.replace(OVERVIEW_MARKER, render_overview(_load_gallery()))


# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------


def page_path(example: Any) -> str:
    """Path of an example's page, relative to ``docs/``."""
    return f"gallery/{example.name}.md"


def sections(examples: Iterable[Any]) -> dict[str, list[Any]]:
    """Group ``examples`` by section, sections in order of their first example."""
    grouped: dict[str, list[Any]] = {}
    for example in examples:
        grouped.setdefault(example.section, []).append(example)
    return grouped


def gallery_nav(nav: Sequence[Any], gallery: ModuleType) -> list[Any]:
    """Return ``nav`` with its ``gallery/index.md`` entry expanded into the gallery section."""
    pages = [
        {title: [page_path(example) for example in examples]}
        for title, examples in sections(gallery.EXAMPLES).items()
    ]

    def expand(item: Any) -> Any:
        if isinstance(item, dict) and list(item.values()) == [OVERVIEW]:
            return {title: [OVERVIEW, *pages] for title in item}
        return item

    return [expand(item) for item in nav]


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def render_overview(gallery: ModuleType) -> str:
    """One tab per gallery style; each holds every section's cards in that style."""
    grouped = sections(gallery.EXAMPLES)
    tabs = []
    for style in gallery.STYLES:
        content = "\n\n".join(
            _block("html", 'p.gallery-section[role="heading" aria-level="2"]', title, level=5)
            + "\n\n"
            + _block(
                "html", "div.grid.cards.gallery-cards", _cards(gallery, examples, style), level=5
            )
            for title, examples in grouped.items()
        )
        tabs.append((style, content))
    return _block("html", "div.gallery-overview", _tabs(tabs, level=4), level=3)


def _cards(gallery: ModuleType, examples: Iterable[Any], style: str) -> str:
    cards = []
    for example in examples:
        shown = style if style in gallery.styles_of(example) else gallery.DEFAULT_STYLE
        images = _images(example, shown, alt="", width=None)
        link = f"[{example.title}]({page_path(example).removeprefix('gallery/')})"
        card = textwrap.indent(f"{images}\n\n{link}", "    ")
        cards.append(f"-{card[1:]}")  # the list marker takes the first indent's place
    return "\n\n".join(cards)


def render_page(gallery: ModuleType, example: Any) -> str:
    """The page of one example: title, description, figure and code per style, setup."""
    figures = [
        (style, _figure(gallery, example, style if example.styled else None))
        for style in gallery.styles_of(example)
    ]
    shown = _tabs(figures, level=3) if example.styled else figures[0][1]
    # no table of contents (a page has one heading), so figure and code get its width
    parts = ["---\nhide:\n  - toc\n---", f"# {example.title}", example.description, shown]
    setup = _setup(gallery, example)
    return "\n\n".join([*parts, *([setup] if setup else [])]) + "\n"


def _figure(gallery: ModuleType, example: Any, style: str | None) -> str:
    """The image pair and the complete code of ``example`` in ``style`` (``None``: unstyled)."""
    alt = example.title if style is None else f"{example.title}, {style} style"
    images = _images(example, style or gallery.DEFAULT_STYLE, alt=alt, width=example.image_width)
    code, highlighted = example_code(gallery, example, style)
    options = f' hl_lines="{" ".join(map(str, highlighted))}"' if highlighted else ""
    figure = _block("html", "figure.gallery-figure", images, level=4)
    return f"{figure}\n\n```python{options}\n{code}```"


def _images(example: Any, style: str, *, alt: str, width: str | None) -> str:
    """Markdown for the light and the dark image of ``example`` in ``style``, lazily loaded."""
    attributes = f'width="{width}" loading=lazy' if width else "loading=lazy"
    return "\n".join(
        f"![{alt}]({IMAGES}/{example.image(style, dark=dark)}#only-{theme}){{ {attributes} }}"
        for dark, theme in ((False, "light"), (True, "dark"))
    )


def example_code(gallery: ModuleType, example: Any, style: str | None) -> tuple[str, list[int]]:
    """The code shown for ``example``, and the numbers of the lines to highlight.

    It starts with the imports the body uses and, for a style, the line that makes
    that style; the highlighted lines are that line and those passing it on, so
    switching tabs shows exactly what changes.
    """
    body = gallery.body_source(example.func).splitlines()
    imports = [
        f"import {module} as {alias}"
        for module, alias in (("matplotlib.pyplot", "plt"), ("numpy", "np"))
        if any(re.search(rf"\b{alias}\.", text) for text in body)
    ]
    lines = [*imports, "import rootfig as rf", ""]
    highlighted = []
    if style is not None:
        lines += [f"style = {gallery.style_source(gallery.STYLES[style])}", ""]
        highlighted.append(len(lines) - 1)
        highlighted += [len(lines) + n for n, text in enumerate(body, 1) if "style=style" in text]
    return "\n".join([*lines, *body]) + "\n", highlighted


def _setup(gallery: ModuleType, example: Any) -> str | None:
    """How to run ``example``: where the toy files come from, and the definitions it uses."""
    shared = any(name != "style" for name in example.parameters)
    parts = []
    if example.toy_data:
        parts.append(
            "The code reads the toy dataset. In a checkout of "
            "[rootfig](https://github.com/jbeirer/rootfig), `python examples/gallery` writes "
            "it to `examples/out/` in a few seconds; run the code inside that directory."
        )
    if shared:
        code = gallery.body_source(gallery.define, returns="omit")
        parts.append(
            "It uses these samples and variables, shared by the gallery examples (see "
            f"[Samples, variables, cuts and styles](../composable.md)):\n\n```python\n{code}```"
        )
    if not parts:
        return None
    return _block("details", "Setup", "\n\n".join(parts), level=3, options="type: abstract")


def _tabs(tabs: Iterable[tuple[str, str]], *, level: int) -> str:
    return "\n\n".join(_block("tab", label, content, level=level) for label, content in tabs)


def _block(kind: str, argument: str, content: str, *, level: int, options: str = "") -> str:
    """A pymdownx block; nested blocks need more slashes than the block around them."""
    fence = "/" * level
    header = f"{fence} {kind} | {argument}" + (f"\n    {options}" if options else "")
    return f"{header}\n\n{content}\n\n{fence}"
