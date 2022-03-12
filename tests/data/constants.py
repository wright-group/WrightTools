"""Tests relating to constants."""

import WrightTools as wt
import numpy as np
import re


def test_set_remove():
    data = wt.Data()
    data.create_variable("x", np.linspace(0, 10))
    data.create_variable("y", np.linspace(0, 10))
    data.create_variable("z", np.zeros(50))
    data.set_constants("x-y", "z")
    assert data.constant_names == ("x__m__y", "z")
    data.remove_constant("z")
    assert data.constant_names == ("x__m__y",)
    data.remove_constant(0)
    assert data.constant_names == ()


def test_label():
    data = wt.Data()
    data.create_variable("x", np.linspace(0, 10))
    data.create_variable("y", np.linspace(0, 10))
    data.create_variable("z", np.full(50, 2.3), units="fs")
    data.z.label = "z"
    data.set_constants("x-y", "z")
    assert data.constants[1].label == r"$\mathsf{\tau_{z}\,=\,2.3\,fs}$"


def test_repr():
    data = wt.Data()
    data.create_variable("x", np.linspace(0, 10))
    data.create_variable("y", np.linspace(0, 10))
    data.create_variable("z", np.full(50, 2.3), units="fs")
    data.z.label = "z"
    data.set_constants("x-y", "z")
    assert (
        re.match(
            r"\<WrightTools\.Constant x-y = 0\.0 None at .*\>", repr(data.constants[0])
        )
        is not None
    )
