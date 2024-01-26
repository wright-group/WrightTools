"""Test guess_signed"""


# --- import -------------------------------------------------------------------------------------


import numpy as np
import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_many_ranges():
    for (minmax, signed) in [
        ([-0.05, 1], False),
        ([-1, 0.05], False), 
        ([-1, 1], True),
        ([0, 0], True),
        ([0, -1], False),
        ([-1, 0.5], True),
    ]:
        assert wt.kit.guess_signed(np.array(minmax)) == signed


def test_channel():
    d = wt.Data()
    chan = d.create_channel("chan", values=np.linspace(3,4,16).reshape(4,4))
    assert wt.kit.guess_signed(chan) == False
    chan.null = 3.5
    assert wt.kit.guess_signed(chan)

