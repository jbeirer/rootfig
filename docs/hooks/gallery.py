"""MkDocs hook that fills ``docs/gallery.md`` from ``examples/gallery.py``.

Two markers are expanded when the page is rendered:

``<!-- gallery-setup -->``
    the body of ``define()``: the samples, variables, cuts and style shared by
    the examples;
``<!-- gallery -->``
    one section per registered example with its title, description, image
    (``docs/images/gallery/<name>.png``) and the source of the example function.

The gallery module is imported, never executed, so building the docs needs no
data and draws nothing. Because the images are also the baselines of
``tests/test_gallery.py``, picture and code cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GALLERY_PY = ROOT / "examples" / "gallery.py"
SETUP_MARKER = "<!-- gallery-setup -->"
GALLERY_MARKER = "<!-- gallery -->"


def _load_gallery() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rootfig_gallery_docs", GALLERY_PY)
    if spec is None or spec.loader is None:
        msg = f"cannot load {GALLERY_PY}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_sections(gallery: ModuleType) -> str:
    """Markdown for every example: heading, description, image, code."""
    parts: list[str] = []
    for example in gallery.EXAMPLES:
        code = gallery.body_source(example.func)
        parts.append(
            f"## {example.title}\n\n"
            f"{example.description}\n\n"
            f'![{example.title}](images/gallery/{example.name}.png){{ width="75%" }}\n\n'
            f"```python\n{code}```\n"
        )
    return "\n".join(parts)


def on_page_markdown(markdown: str, **_: Any) -> str:
    """Expand the gallery markers (MkDocs ``on_page_markdown`` event)."""
    if SETUP_MARKER not in markdown and GALLERY_MARKER not in markdown:
        return markdown
    gallery = _load_gallery()
    setup = gallery.body_source(gallery.define, returns="omit")
    markdown = markdown.replace(SETUP_MARKER, f"```python\n{setup}```")
    return markdown.replace(GALLERY_MARKER, render_sections(gallery))
