"""test propagation of other channels"""


# --- import --------------------------------------------------------------------------------------


import pytest

import os

import WrightTools as wt


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- test ----------------------------------------------------------------------------------------


@pytest.mark.skip()
def test_propagate_channels():
    p = os.path.join(here, 'propagate_other_channels.data')
    data = wt.data.from_PyCMDS(p)
    function = wt.fit.Moments()
    function.subtract_baseline = False
    fitter = wt.fit.Fitter(function, data, 'w3_BBO', verbose=False)
    outs = fitter.run(4, verbose=False)
    assert outs.channel_names == ['integral', 'one', 'two', 'three', 'four', 'baseline',
                                  'signal_diff', 'signal_mean', 'pyro1', 'pyro2', 'PMT_voltage']
