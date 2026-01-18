import numpy as np
import WrightTools as wt
from WrightTools import datasets


def test_3D():
    data = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)

    data.norm_for_each("w1", channel=0)
    assert np.all(data.channels[0][:].max(axis=(0, 2)) == 1)

    data.norm_for_each("d2", new_channel={"name": "ai0_d2_norm"})
    assert np.all(data.channels[-1][:].max(axis=(0, 1)) == 1)

    data.norm_for_each("d1", new_channel=True)
    assert data.channels[-1].natural_name == "ai0_v6_norm"

    data.norm_for_each("d1", new_channel={"name": "ai0_d1_norm"})
    data.channels[0].normalize()
    assert np.all(np.isclose(data.channels[-1][:], data.channels[0][:]))


def test_two_vars():
    data = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    data.norm_for_each("w1", "d2")
    assert np.all(data.channels[0][:].max(axis=0) == 1)


if __name__ == "__main__":
    test_3D()
    test_two_vars()
