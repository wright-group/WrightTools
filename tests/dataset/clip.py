"""Test clipping."""


# --- import --------------------------------------------------------------------------------------


import random

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_w1_wa():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    new_max = random.random() * 0.5 * data.array_signal.max() + data.array_signal.min()
    data.array_signal.clip(max=new_max)
    assert data.array_signal.max() <= new_max
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_w1_wa()
