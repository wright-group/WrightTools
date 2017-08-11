import numpy as np

import WrightTools as wt
from WrightTools import datasets

def test_wm_w2_w1():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    p = datasets.PyCMDS.wm_w2_w1_001
    b = wt.data.from_PyCMDS(p)
    joined = wt.data.join([a, b])
    assert joined.shape == (63, 11, 11)
    assert not np.isnan(joined.channels[0].values).any()
