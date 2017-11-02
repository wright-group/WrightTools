"""test map_axis"""


# --- import --------------------------------------------------------------------------------------


import pytest

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


@pytest.mark.skip()
def test_array():
    p = datasets.JASCO.PbSe_batch_1
    data = wt.data.from_JASCO(p)
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    print(inspect.stack()[0][3], data.filepath)
    assert data.shape == (1801,)
    new = np.linspace(6000, 8000, 55)
    data.map_axis(0, new, 'wn')
    assert data.axes[0].points.all() == new.all()
    data.file.flush()
    data.close()


@pytest.mark.skip()
def test_edge_tolerance():
    ps = datasets.KENT.LDS821_TRSF
    data = wt.data.from_KENT(ps, ignore=['wm', 'd1', 'd2'])
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    print(inspect.stack()[0][3], data.filepath)
    new = np.linspace(1250, 1600, 101)
    data.map_axis('w2', new, edge_tolerance=1)
    assert data.w2.points.all() == new.all()
    assert not np.isnan(data.channels[0].values).any()
    data.file.flush()
    data.close()


@pytest.mark.skip()
def test_int():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    print(inspect.stack()[0][3], data.filepath)
    assert data.shape == (35, 11, 11)
    data.map_axis(1, 5)
    assert data.shape == (35, 5, 11)
    data.map_axis(2, 25)
    assert data.shape == (35, 5, 25)
    data.file.flush()
    data.close()
