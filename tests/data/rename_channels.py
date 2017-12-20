"""Test rename_channel."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_rename():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.channel_names[1] == 'signal_mean'
    data.rename_channels(signal_mean='sig')
    assert data.channel_names[1] == 'sig'
    assert data.channels[1].natural_name == 'sig'
    assert 'sig' in data.channels[1].name
    data.close()


def test_error():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    try:
        data.rename_channels(pyro1='pyro2')
    except wt.exceptions.NameNotUniqueError:
        assert True
    else:
        assert False
    data.close()
