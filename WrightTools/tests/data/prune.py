"""Test remove channel."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_prune_default():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.create_constant("d1")
    num_channels = len(data.channels)
    data.prune()
    assert len(data.variables) == 4
    assert set(data.variable_names) == {"wm", "w2", "w1", "d1"}
    assert len(data.channels) == num_channels
    data.close()


def test_prune_false():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.create_constant("d1")
    data.prune(False)
    assert len(data.variables) == 4
    assert set(data.variable_names) == {"wm", "w2", "w1", "d1"}
    assert len(data.channels) == 1
    assert data.channel_names[0] == "signal_diff"
    data.close()


def test_prune_number():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.create_constant("d1")
    data.prune(2)
    assert len(data.variables) == 4
    assert set(data.variable_names) == {"wm", "w2", "w1", "d1"}
    assert len(data.channels) == 1
    assert data.channel_names[0] == "pyro1"
    data.close()


def test_prune_string():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.create_constant("d1")
    data.prune("pyro1")
    assert len(data.variables) == 4
    assert set(data.variable_names) == {"wm", "w2", "w1", "d1"}
    assert len(data.channels) == 1
    assert data.channel_names[0] == "pyro1"
    data.close()


def test_prune_tuple():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    num_channels = len(data.channels)
    data.prune(("pyro1", 3, 4), verbose=False)
    assert len(data.variables) == 3
    assert set(data.variable_names) == {"wm", "w2", "w1"}
    assert len(data.channels) == 3
    assert set(data.channel_names) == {"pyro1", "pyro2", "pyro3"}
    data.close()


if __name__ == "__main__":
    test_prune_default()
    test_prune_false()
    test_prune_number()
    test_prune_string()
    test_prune_tuple()
