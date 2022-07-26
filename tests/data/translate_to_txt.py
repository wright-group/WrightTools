"""Test transform."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets
import pathlib

# --- tests ---------------------------------------------------------------------------------------


@pytest.mark.skip("don't want to write files")
def test_datasets_mos2():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie).chop(
        "w1=wm", at={"d2":[0, "fs"], "w2":[18000, "wn"]}
    )[0]
    d.translate_to_txt(
        pathlib.Path(__file__).resolve().parent / "translate.txt",
        verbose=False
    )
    d.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_datasets_mos2()
