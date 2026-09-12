"""MkDocs hook that fills ``docs/gallery.md`` from the ``examples/gallery`` package.

Two kinds of marker are expanded when the page is rendered:

``<!-- gallery-setup -->``
    the body of ``define()``: the samples, variables and style shared by the
    examples;
``<!-- gallery -->`` / ``<!-- gallery: name ... -->``
    one section per example with its title, description, image
    (``docs/images/gallery/<name>.png``) and the source of the example
    function. A marker naming examples renders exactly those, so a few can be
    shown before the setup section; a bare marker renders everything that no
    earlier marker on the page has shown yet, in registration order.

The gallery module is imported, never executed, so building the docs needs no
data and draws nothing. Because the images are also the baselines of
``tests/test_gallery.py``, picture and code cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GALLERY_DIR = ROOT / "examples" / "gallery"
SETUP_MARKER = "<!-- gallery-setup -->"
GALLERY_MARKER = re.compile(r"<!-- gallery(?::\s*(?P<names>[\w\s,]+?))?\s*-->")


def _load_gallery(name: str = "rootfig_gallery_docs") -> ModuleType:
    """Import ``examples/gallery`` as a fresh package, without ``examples/`` on ``sys.path``.

    Every call re-executes the package (``mkdocs serve`` renders the page on each
    rebuild), so its submodules are evicted from ``sys.modules`` first: a cached
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


def render_section(gallery: ModuleType, example: Any) -> str:
    """Markdown for one example: heading, description, image, code."""
    code = gallery.body_source(example.func)
    return (
        f"## {example.title}\n\n"
        f"{example.description}\n\n"
        f'![{example.title}](images/gallery/{example.name}.png){{ width="75%" }}\n\n'
        f"```python\n{code}```\n"
    )


def expand_markers(markdown: str, gallery: ModuleType) -> str:
    """Replace every gallery marker, each example going to the first marker that asks."""
    shown: set[str] = set()
    by_name = {example.name: example for example in gallery.EXAMPLES}

    def replace(match: re.Match[str]) -> str:
        names = match.group("names")
        if names is None:
            wanted = [ex.name for ex in gallery.EXAMPLES if ex.name not in shown]
        else:
            wanted = [name.strip() for name in names.replace(",", " ").split()]
            unknown = [name for name in wanted if name not in by_name]
            if unknown:
                msg = f"unknown gallery example(s) {unknown} in {match.group(0)!r}"
                raise ValueError(msg)
        shown.update(wanted)
        return "\n".join(render_section(gallery, by_name[name]) for name in wanted)

    return GALLERY_MARKER.sub(replace, markdown)


def on_page_markdown(markdown: str, **_: Any) -> str:
    """Expand the gallery markers (MkDocs ``on_page_markdown`` event)."""
    if SETUP_MARKER not in markdown and not GALLERY_MARKER.search(markdown):
        return markdown
    gallery = _load_gallery()
    setup = gallery.body_source(gallery.define, returns="omit")
    markdown = markdown.replace(SETUP_MARKER, f"```python\n{setup}```")
    return expand_markers(markdown, gallery)
