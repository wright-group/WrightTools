"""Test from_Aramis."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
import numpy as np

import pathlib

here = pathlib.Path(__file__).parent

# --- test ----------------------------------------------------------------------------------------


def test_circular_map():
    p = here / "test_data" / "circular_map.ngc"
    data = wt.data.from_Aramis(p)
    assert data.shape == (5, 4, 1024)
    assert data.variable_names == ("Spectr", "X", "Y")
    assert data.Spectr.shape == (1, 1, 1024)
    assert data.X.shape == (1, 4, 1)
    assert data.Y.shape == (5, 1, 1)
    assert data.channel_names == ("Intens",)
    assert np.sum(data.Intens[:] == 0) == 8192


def test_ellipsoidal_map():
    p = here / "test_data" / "ellipsoidal_map.ngc"
    data = wt.data.from_Aramis(p)
    assert data.shape == (8, 4, 1024)
    assert data.variable_names == ("Spectr", "X", "Y")
    assert data.Spectr.shape == (1, 1, 1024)
    assert data.X.shape == (1, 4, 1)
    assert data.Y.shape == (8, 1, 1)
    assert data.channel_names == ("Intens",)
    assert np.sum(data.Intens[:] == 0) == 10240


def test_irregular_map():
    p = here / "test_data" / "irregular_map.ngc"
    data = wt.data.from_Aramis(p)
    assert data.shape == (14, 13, 1024)
    assert data.variable_names == ("Spectr", "X", "Y")
    assert data.Spectr.shape == (1, 1, 1024)
    assert data.X.shape == (1, 13, 1)
    assert data.Y.shape == (14, 1, 1)
    assert data.channel_names == ("Intens",)
    assert np.sum(data.Intens[:] == 0) == 99328


def test_rectangular_map():
    p = here / "test_data" / "rectangular_map.ngc"
    data = wt.data.from_Aramis(p)
    assert data.shape == (6, 5, 1024)
    assert data.variable_names == ("Spectr", "X", "Y")
    assert data.Spectr.shape == (1, 1, 1024)
    assert data.X.shape == (1, 5, 1)
    assert data.Y.shape == (6, 1, 1)
    assert data.channel_names == ("Intens",)
    assert np.sum(data.Intens[:] == 0) == 0


def test_skew_line():
    p = here / "test_data" / "skew_line.ngc"
    data = wt.data.from_Aramis(p)
    assert data.shape == (8, 3328)
    assert data.variable_names == ("Spectr", "X", "Y")
    assert data.Spectr.shape == (1, 3328)
    assert data.X.shape == (8, 1)
    assert data.Y.shape == (8, 1)
    assert data.channel_names == ("Intens",)
    assert np.sum(data.Intens[:] == 0) == 0


def test_square_map():
    p = here / "test_data" / "square_map.ngc"
    data = wt.data.from_Aramis(p)
    assert data.shape == (5, 5, 1024)
    assert data.variable_names == ("Spectr", "X", "Y")
    assert data.Spectr.shape == (1, 1, 1024)
    assert data.X.shape == (1, 5, 1)
    assert data.Y.shape == (5, 1, 1)
    assert data.channel_names == ("Intens",)
    assert np.sum(data.Intens[:] == 0) == 0


def test_vertical_linescan():
    p = here / "test_data" / "vertical_linescan.ngc"
    data = wt.data.from_Aramis(p)
    assert data.shape == (5, 3328)
    assert data.variable_names == ("Spectr", "Y")
    assert data.Spectr.shape == (1, 3328)
    assert data.Y.shape == (5, 1)
    assert data.channel_names == ("Intens",)
    assert np.sum(data.Intens[:] == 0) == 0


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_circular_map()
