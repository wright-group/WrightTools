"""Test getattr and associated."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_axis_variable_namespace_collision():
    root = wt.Collection()
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p, parent=root, name='data')
    assert isinstance(data.wm, wt.data._axis.Axis)
    assert isinstance(data.w2, wt.data._axis.Axis)
    assert isinstance(data.w1, wt.data._axis.Axis)
    assert isinstance(data.d1, wt.data._variable.Variable)
    assert isinstance(data.d2, wt.data._variable.Variable)
    data.close()


def test_transform():
    root = wt.Collection()
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p, parent=root, name='data')
    data.transform(['wm-w1', 'w1', 'w2'])
    assert hasattr(data, 'wm__m__w1')
    assert hasattr(data, 'w1')
    assert hasattr(data, 'w2')
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_axis_variable_namespace_collision()
    test_transform()
