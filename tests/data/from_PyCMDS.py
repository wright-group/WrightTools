#! /usr/bin/env python3
"""test from_PyCMDS"""


# --- import --------------------------------------------------------------------------------------


import os
import numpy as np

import WrightTools as wt
from WrightTools import datasets

here = os.path.abspath(os.path.dirname(__file__))


# --- test ----------------------------------------------------------------------------------------


def test_w1_000():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (51,)
    assert data.axis_expressions == ("w1",)
    assert "w1_points" in data.variable_names
    data.close()


def test_w1_wa_000():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (25, 256)
    assert data.axis_expressions == ("w1=wm", "wa")
    assert data.wa_centers.shape == (25, 1)
    assert data.wa_points.shape == (1, 256)
    data.close()


def test_w2_w1_000():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (81, 81)
    assert data.axis_expressions == ("w2", "w1")
    data.close()


def test_wm_w2_w1_000():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (35, 11, 11)
    assert data.axis_expressions == ("wm", "w2", "w1")
    data.close()


def test_d1_d2_000():
    p = datasets.PyCMDS.d1_d2_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (101, 101)
    assert data.axis_expressions == ("d1", "d2")
    # Test for correction factor applied
    assert data.d1.max() < 230
    assert data.d1.min() > -230
    data.close()


def test_wm_w2_w1_001():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (29, 11, 11)
    assert data.axis_expressions == ("wm", "w2", "w1")
    data.close()


def test_incomplete():
    p = os.path.join(here, "test_data", "incomplete.data")
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (9, 9)
    assert data.axis_expressions == ("d1", "d2")
    assert np.allclose(
        data.d1.points, np.array([-1., -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875, -2.])
    )
    data.close()


def test_ps_delay():
    p = os.path.join(here, "test_data", "ps_delay.data")
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (11, 15, 15)
    assert data.axis_expressions == ("d1", "w2", "w1")
    data.close()


def test_ps_delay_together():
    p = os.path.join(here, "test_data", "ps_delay_together.data")
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (33, 21)
    assert data.axis_expressions == ("w3", "d1=d2")
    assert data.d1.shape == (1, 21)
    assert data.d2.shape == (1, 21)
    assert data.d1__e__d2.shape == (1, 21)
    data.close()


def test_tolerance():
    p = os.path.join(here, "test_data", "tolerance.data")
    data = wt.data.from_PyCMDS(p)
    assert data.d1.shape == (4, 1, 1)
    assert data.shape == (4, 36, 36)
    assert data.axis_expressions == ("d1", "w2", "w1=wm")
    data.close()


def test_autotune():
    p = os.path.join(here, "test_data", "autotune.data")
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (20, 21)
    assert data.axis_expressions == ("w2", "w2_BBO")
    assert "w2_BBO_points" in data.variable_names
    assert "w2_BBO_centers" in data.variable_names
    data.close()


def test_two_centers():
    p = os.path.join(here, "test_data", "two_centers.data")
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (11, 21, 51)
    assert data.axis_expressions == ("w2", "w2_Mixer_1", "wm")
    assert "wm_points" in data.variable_names
    assert "wm_centers" in data.variable_names
    assert "w2_Mixer_1_points" in data.variable_names
    assert "w2_Mixer_1_centers" in data.variable_names
    data.close()


def test_remote():
    data = wt.data.from_PyCMDS("https://osf.io/download/rdn7v")
    assert data.shape == (21, 81)
    assert data.axis_expressions == ("wm", "w2=w1")
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_w1_000()
    test_w1_wa_000()
    test_w2_w1_000()
    test_wm_w2_w1_000()
    test_wm_w2_w1_001()
    test_incomplete()
    test_ps_delay()
    test_tolerance()
    test_autotune()
    test_two_centers()
    test_remote()
