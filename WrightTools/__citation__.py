"""Define WrightTools citation."""

import pathlib

__all__ = ["__citation__"]

here = pathlib.Path(__file__).parent

with open(here / "CITATION") as f:
    # remove extra whitespace
    __citation__ = " ".join(f.read().split())
