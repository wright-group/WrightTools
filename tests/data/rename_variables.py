"""Test rename_variable."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_simple():
    p = datasets.BrunoldrRaman.LDS821_514nm_80mW
    data = wt.data.from_BrunoldrRaman(p)
    data.rename_variables(energy='blaise')
    assert data.variable_names == ('blaise',)
    assert data.variables[0].natural_name == 'blaise'
    assert 'blaise' in data.variables[0].name
    data.close()


def test_implied():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    names = data.variable_names
    data.rename_variables(w1='w2', w2='w1')
    for old, new in zip(names, data.variable_names):
        if old.startswith('w1'):
            assert new.startswith('w2')
        elif old.startswith('w2'):
            assert new.startswith('w1')
        else:
            pass
    data.close()


def test_propagate_units():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    data.convert('meV')
    data.rename_variables(wm='mono')
    assert data.units == ('meV', 'meV', 'meV')
    data.close()


def test_updated_axes():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    assert data.axis_expressions == ('wm', 'd1',)
    data.rename_variables(wm='mono')
    assert data.axis_expressions == ('mono', 'd1',)
    data.close()
