"""Validation of the names that become components of output file names."""

from __future__ import annotations

import re

__all__ = ["check_file_stem", "safe_file_stem"]

_UNSAFE_CHARACTERS = frozenset('<>:"|?*/\\')
"""Characters that no file name component may hold on Windows or on POSIX."""

_RESERVED_STEMS = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}) | {
    f"{device}{digit}" for device in ("COM", "LPT") for digit in "123456789¹²³"
}
"""Device names Windows refuses as the part of a file name before its first dot.

The set :func:`ntpath.isreserved` (Python 3.13) uses: ports 1 to 9, also spelt with
the superscript digits, which Windows reads as digits when it looks for a port.
"""

_SAFE_RE = re.compile(r"[^0-9A-Za-z_]+")


def _unusable(value: str) -> str | None:
    """Say why ``value`` cannot be a file name component on every platform, or ``None``."""
    if not value.strip(" ."):
        return "it must hold more than dots and spaces"
    if value[-1] in " .":
        return "it must not end with a dot or a space, which Windows drops"
    bad = sorted({c for c in value if c < " " or c == "\x7f" or c in _UNSAFE_CHARACTERS})
    if bad:
        shown = ", ".join(repr(c) for c in bad)
        return f'it holds {shown}; control characters, slashes and <>:"|?* are not allowed'
    device = value.split(".", 1)[0].rstrip(" ")  # "NUL .txt" is NUL too, as in ntpath
    if device.upper() in _RESERVED_STEMS:
        return f"{device!r} is a reserved device name on Windows"
    return None


def safe_file_stem(text: str) -> str:
    """Reduce ``text`` to a component that passes :func:`check_file_stem`, or to ``""``.

    Runs of anything but ASCII letters, digits and ``_`` become one ``_``, and a
    stem Windows reserves (``CON``, ``COM1``) gets a trailing ``_``. ``""`` means
    nothing usable was left, as for ``"?"``. :attr:`~rootfig.model.Variable.safe_name`
    builds the automatic name of a variable from its expression this way.
    """
    stem = _SAFE_RE.sub("_", text).strip("_")
    # A sanitised, non-empty stem can only still fail as a reserved device name.
    return f"{stem}_" if stem and _unusable(stem) is not None else stem


def check_file_stem(value: str, *, what: str) -> str:
    """Return ``value`` if it can be a file name component on every platform, else raise.

    Checked once, when a :class:`~rootfig.model.Variable` name or a
    :class:`~rootfig.PlotBook` selection or variant name is given, rather than
    when ``save()`` reaches a file system that refuses it. ``what`` names the value
    in the message, which also offers :func:`safe_file_stem` of the value when
    that leaves anything: a spelling that passes this check.
    """
    reason = _unusable(value)
    if reason is None:
        return value
    suggestion = safe_file_stem(value)
    hint = f"; {suggestion!r} would work" if suggestion else ""
    msg = f"{what} {value!r} cannot be a file name component: {reason}{hint}"
    raise ValueError(msg)
