"""Test transform."""


# --- import --------------------------------------------------------------------------------------


import pathlib
import pytest

import WrightTools as wt
from WrightTools import datasets

# --- tests ---------------------------------------------------------------------------------------


@pytest.mark.skip("don't want to write files")
def test_datasets_mos2():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie).at(w2=[18000, "wn"])
    d.translate_to_txt(pathlib.Path(__file__).resolve().parent / "translate.txt", verbose=True)
    d.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_datasets_mos2()
