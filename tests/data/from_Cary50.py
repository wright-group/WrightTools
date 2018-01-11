"""test from_Cary50"""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


@pytest.mark.skip()
def test_CuPCtS_H2O_vis():
    p = datasets.Cary50.CuPCtS_H2O_vis
    datas = wt.data.from_Cary50(p)

    assert isinstance(datas, list)
    assert len(datas) == 1

    data = datas[0]
    assert data.shape == (141,)
    assert data.axis_names == ('wm',)
    datas.close()
