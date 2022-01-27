#! /usr/bin/env python3

import WrightTools as wt
import numpy as np


def test_short():
    data = wt.Data()
    data.create_variable("t", np.linspace(0, 10, 10), units="fs")
    data.transform("t")
    assert data.axes[0].label == r"$\mathsf{\tau\,\left(fs\right)}$"


if __name__ == "__main__":
    test_short()
