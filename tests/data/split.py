"""test split"""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


@pytest.mark.skip()
def test_split():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    a.flip(0)
    split = a.split(0, [19700])
    assert len(split) == 2
    assert split[0].shape == (15, 11, 11)
    assert split[1].shape == (20, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_descending():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    split = a.split(0, [19700])
    assert len(split) == 2
    assert split[0].shape == (20, 11, 11)
    assert split[1].shape == (15, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_edge():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    split = a.split(0, [20700])
    assert len(split) == 2
    assert split[0] is None
    assert split[1].shape == (35, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_multiple():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    split = a.split(0, [20600, 19700])
    assert len(split) == 3
    assert split[0].shape == (2, 11, 11)
    assert split[1].shape == (18, 11, 11)
    assert split[2].shape == (15, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_close():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    a.flip(0)
    split = a.split(0, [19705, 19694])
    assert len(split) == 3
    assert split[0].shape == (15, 11, 11)
    assert split[1] is None
    assert split[2].shape == (20, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_above():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    a.flip(0)
    split = a.split(0, [19700], direction='above')
    assert len(split) == 2
    assert split[0].shape == (14, 11, 11)
    assert split[1].shape == (21, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_above_descending():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    split = a.split(0, [19700], direction='above')
    assert len(split) == 2
    assert split[0].shape == (21, 11, 11)
    assert split[1].shape == (14, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_units():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    a.flip(0)
    split = a.split(0, [507], units='nm')
    assert len(split) == 2
    assert split[0].shape == (15, 11, 11)
    assert split[1].shape == (20, 11, 11)
    a.close()


@pytest.mark.skip()
def test_split_axis_name():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    split = a.split('w2', [1500])
    assert len(split) == 2
    assert split[0].shape == (35, 10, 11)
    assert split[1].shape == (35, 11)
    a.close()


@pytest.mark.skip()
def test_split_constant():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    split = a.split(1, [1500])
    assert len(split) == 2
    assert split[1].shape == (35, 11)
    assert split[1].w2.is_constant()
    a.close()


@pytest.mark.skip()
def test_split_parent():
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    parent = wt.Collection()
    split = a.split(1, [1500], parent=parent)
    assert 'split' in parent
    assert split.filepath == parent.filepath
    assert len(split) == 2
    a.close()
