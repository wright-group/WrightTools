import numpy as np
import WrightTools as wt
from WrightTools import datasets


def test_3D():
    data = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)

    data.norm_for_each("w1", 0)
    assert np.all(data.channels[0][:].max(axis=(0, 2)) == 1)

    data.norm_for_each("d2", 0, new_channel={"name": "ai0_d2_norm"})
    assert np.all(data.channels[-1][:].max(axis=(0, 1)) == 1)

    data.norm_for_each("d1", 0, new_channel=True)

    data.norm_for_each("d1", 0, new_channel={"name": "ai0_d1_norm1"})
    data.channels[0].normalize()
    assert np.all(np.isclose(data.ai0_d1_norm[:], data.channels[0][:]))


if __name__ == "__main__":
    test_3D()
