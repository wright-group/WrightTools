"""Test get_index."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_string():
    assert wt.kit.get_index(['w1', 'd1', 'd2'], 'd2') == 2


def test_int():
    assert wt.kit.get_index(['w1', 'w2', 'w3'], 1) == 1
