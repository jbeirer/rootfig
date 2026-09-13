"""Command line: ``python examples/gallery [--out DIR] [--style NAME]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:  # python -m gallery, from examples/
    from . import DEFAULT_OUT, DEFAULT_STYLE, STYLES, main
else:  # python examples/gallery: the directory itself is on sys.path, not its parent
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gallery import DEFAULT_OUT, DEFAULT_STYLE, STYLES, main

parser = argparse.ArgumentParser(description="Write the toy dataset and every gallery figure.")
parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
parser.add_argument(
    "--style", choices=list(STYLES), default=DEFAULT_STYLE, help="style of the figures"
)
arguments = parser.parse_args()
main(arguments.out, arguments.style)
