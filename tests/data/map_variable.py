"""Test map_variable."""

# --- import --------------------------------------------------------------------------------------


import numpy as np
import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_array():
    p = datasets.JASCO.PbSe_batch_1
    data = wt.data.from_JASCO(p)
    assert data.shape == (1801,)
    new = np.linspace(6000, 8000, 55)
    mapped = data.map_variable("energy", new, "wn")
    assert np.allclose(mapped.axes[0][:], 1e7 / new)
    data.close()


def test_int():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)[::10]
    assert data.shape == (4, 11, 11)
    mapped = data.map_variable("w2", 5)
    assert mapped.shape == (4, 5, 11)
    mapped = data.map_variable("w1", 25)
    assert mapped.shape == (4, 11, 25)
    data.close()


def test_excess_data_kwarg_1D():
    p = datasets.wt5.v1p0p0_perovskite_TA
    data = wt.open(p).chop("w2", at={"w1=wm": [1.6, "eV"], "d2": [0, "fs"]})[0]
    mapped = data.map_variable("w2", 11)
    assert mapped.w2.size == 11
    data.close()


@pytest.mark.skip("It's a long test")
def test_v1p0p0():
    p = datasets.wt5.v1p0p0_perovskite_TA
    data = wt.open(p)
    mapped = data.map_variable("w2", 2)
    assert mapped.w2.size == 2
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_array()
    test_int()
    test_excess_data_kwarg_1D()
    test_v1p0p0()
