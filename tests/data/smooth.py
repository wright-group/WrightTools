"""Test data.smooth."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt

import matplotlib.pyplot as plt


# --- test ----------------------------------------------------------------------------------------


def test_1():
    # create collection
    root = wt.Collection(name='root_coll')
    kwargs = {'name': 'test1'}
    data = root.create_data(**kwargs)
    # create test arrays
    x = np.linspace(0, 1, 1000)
    y = 1*x
    # create channels and variables
    data.create_variable(name='x', values=x, units=None)
    data.create_channel(name='y', values=y)
    data.create_channel(name='z', values=y)
    data.transform('x')
    # smooth
    data.smooth(5, channel='y')
    check_arr = data.y[:] - data.z[:]
    assert np.isclose(check_arr.all(), 0, rtol=1e-4, atol=1e-4)


def test_2():
    # create collection
    root = wt.Collection(name='root_coll')
    kwargs = {'name': 'test2'}
    data = root.create_data(**kwargs)
    # create test arrays
    x = np.linspace(0, 1, 100)
    z = np.random.rand(100, 100)
    # create channels and variables
    data.create_variable(name='x1', values=x, units=None)
    data.create_variable(name='x2', values=x, units=None)
    data.create_channel(name='z', values=z)
    data.transform('x1', 'x2')
    # smooth
    data.smooth(90, channel='z')
    assert np.allclose(data.z[:], .5, rtol=.1, atol=.4)


def test_3():
    # create collection
    root = wt.Collection(name='root_coll')
    kwargs = {'name': 'test3'}
    data = root.create_data(**kwargs)
    # create test arrays
    x = np.linspace(0, 4*np.pi, 1000)
    z_big = np.sin(x)
    z_small = np.sin(x*50)
    z_comb = z_big + .1*z_small
    # create channels and variables
    data.create_variable(name='x', values=x, units=None)
    data.create_channel(name='z_comb', values=z_comb)
    data.transform('x')
    # smooth
    data.smooth(10, channel='z_comb')
    assert np.allclose(data.z_comb[:], z_big, rtol=.1, atol=.05)
