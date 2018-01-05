"""Test group copy."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------



def test_simple():
    original = wt.Collection(name='blaise')
    new = c.copy()
    assert original.fullpath != new.fullpath
    for k, v in original.attrs.items():
        assert new.attrs[k] == v
    for k, v in original.items():
        assert new[k] == v
    original.close()
    new.close()
