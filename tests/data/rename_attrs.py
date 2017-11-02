"""test rename_attrs."""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


@pytest.mark.skip()
def test_rename():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    print(inspect.stack()[0][3], data.filepath)
    data.rename_attrs(w1='w2', w2='w1')
    assert data.shape == (35, 11, 11)
    assert data.axis_names == ['wm', 'w1', 'w2']
    data.file.flush()
    data.close()


@pytest.mark.skip()
def test_error():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    print(inspect.stack()[0][3], data.filepath)
    try:
        data.rename_attrs(w1='w2')
    except wt.exceptions.NameNotUniqueError:
        assert True
    else:
        assert False

    data.file.flush()
    data.close()
