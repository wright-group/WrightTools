#! /usr/bin/env python3
"""Test collapse."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt
import numpy as np


# --- tests --------------------------------------------------------------------------------------


def test_integrate():
    data = wt.Data()
    data.create_variable("v1", np.arange(0, 10)[None, :])
    data.create_variable("v2", np.arange(0, 10)[:, None])
    data.create_channel("ch", data.v1[:] * data.v2[:])
    data.transform("v2", "v1")
    data.collapse("v1", "integrate")
    data.collapse("v2", "int")

    assert data.ch_v2_int.shape == (1, 10)
    assert data.ch_v1_integrate.shape == (10, 1)
    assert data.ch_v1_integrate_v2_int.shape == (1, 1)
    assert np.allclose(
        data.ch_v2_int.points,
        np.array([0.0, 40.5, 81.0, 121.5, 162.0, 202.5, 243.0, 283.5, 324.0, 364.5]),
    )
    assert np.allclose(
        data.ch_v1_integrate.points,
        np.array([0.0, 40.5, 81.0, 121.5, 162.0, 202.5, 243.0, 283.5, 324.0, 364.5]),
    )
    assert np.allclose(data.ch_v1_integrate_v2_int.points, np.array(1640.25))
    assert len(data.channel_names) == 4


def test_average():
    data = wt.Data()
    data.create_variable("v1", np.arange(0, 10)[None, :])
    data.create_variable("v2", np.arange(0, 10)[:, None])
    data.create_channel("ch", data.v1[:] * data.v2[:])
    data.transform("v2", "v1")
    data.collapse(0, "average")
    data.collapse(1, "mean")

    assert data.ch_1_mean.shape == (10, 1)
    assert data.ch_0_average.shape == (1, 10)
    assert data.ch_0_average_1_mean.shape == (1, 1)
    assert np.allclose(
        data.ch_1_mean.points,
        np.array([0.0, 4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0, 40.5]),
    )
    assert np.allclose(
        data.ch_0_average.points,
        np.array([0.0, 4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0, 40.5]),
    )
    assert np.allclose(data.ch_0_average_1_mean.points, np.array(20.25))
    assert len(data.channel_names) == 4


def test_sum():
    data = wt.Data()
    data.create_variable("v1", np.arange(0, 10)[None, :])
    data.create_variable("v2", np.arange(0, 10)[:, None])
    data.create_channel("ch", data.v1[:] * data.v2[:])
    data.transform("v2", "v1")
    data.collapse("v1", "sum")
    data.collapse("v2", "sum")

    assert data.ch_v2_sum.shape == (1, 10)
    assert data.ch_v1_sum.shape == (10, 1)
    assert data.ch_v1_sum_v2_sum.shape == (1, 1)
    assert np.allclose(
        data.ch_v2_sum.points,
        np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0, 405.0]),
    )
    assert np.allclose(
        data.ch_v1_sum.points,
        np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0, 405.0]),
    )
    assert np.allclose(data.ch_v1_sum_v2_sum.points, np.array(2025))
    assert len(data.channel_names) == 4


def test_max():
    data = wt.Data()
    data.create_variable("v1", np.arange(0, 10)[None, :])
    data.create_variable("v2", np.arange(0, 10)[:, None])
    data.create_channel("ch", data.v1[:] * data.v2[:])
    data.transform("v2", "v1")
    data.collapse("v1", "maximum")
    data.collapse("v2", "max")

    assert data.ch_v2_max.shape == (1, 10)
    assert data.ch_v1_maximum.shape == (10, 1)
    assert data.ch_v1_maximum_v2_max.shape == (1, 1)
    assert np.allclose(
        data.ch_v2_max.points,
        np.array([0.0, 9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0]),
    )
    assert np.allclose(
        data.ch_v1_maximum.points,
        np.array([0.0, 9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0]),
    )
    assert np.allclose(data.ch_v1_maximum_v2_max.points, np.array(81))
    assert len(data.channel_names) == 4


def test_min():
    data = wt.Data()
    data.create_variable("v1", np.arange(0, 10)[None, :])
    data.create_variable("v2", np.arange(0, 10)[:, None])
    data.create_channel("ch", data.v1[:] * data.v2[:])
    data.transform("v2", "v1")
    data.collapse("v1", "minimum")
    data.collapse("v2", "min")

    assert data.ch_v2_min.shape == (1, 10)
    assert data.ch_v1_minimum.shape == (10, 1)
    assert data.ch_v1_minimum_v2_min.shape == (1, 1)
    assert np.allclose(
        data.ch_v2_min.points,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert np.allclose(
        data.ch_v1_minimum.points,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert np.allclose(data.ch_v1_minimum_v2_min.points, np.array(0))
    assert len(data.channel_names) == 4


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_integrate()
    test_average()
    test_sum()
    test_max()
    test_min()
