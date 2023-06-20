"""Test transform."""


# --- import --------------------------------------------------------------------------------------


import pathlib
import WrightTools as wt

from WrightTools import datasets
from tempfile import NamedTemporaryFile


# --- tests ---------------------------------------------------------------------------------------


def test_datasets_mos2():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie).at(w2=[18000, "wn"])
    tmp = NamedTemporaryFile(delete=False)
    d.translate_to_txt(tmp.name, verbose=True)
    d.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_datasets_mos2()
