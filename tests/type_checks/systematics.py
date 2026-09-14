"""Static checks for the public variation input and output contracts (run by mypy)."""

from collections.abc import Mapping
from typing import assert_type

from rootfig import Histogram
from rootfig._typing import Hist


def histogram_variations(nominal: Hist, up: Hist, down: Hist) -> None:
    given: Mapping[str, tuple[Hist, Hist | None]] = {"one": (up, None), "two": (up, down)}
    result = Histogram(nominal, label="MC", variations=given)
    Histogram(nominal, label="MC", variations={"one": (up, None)})
    Histogram(nominal, label="MC", variations={"two": (up, down)})
    # The up histogram is still required; mypy reports an unused ignore if this loosens.
    Histogram(nominal, label="MC", variations={"bad": (None, down)})  # type: ignore[dict-item]
    assert_type(result.variations, Mapping[str, tuple[Hist, Hist]])
    for varied_up, varied_down in result.variations.values():
        assert_type(varied_up, Hist)
        assert_type(varied_down, Hist)
