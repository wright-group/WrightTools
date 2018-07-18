"""Test svd."""


# --- import -------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------


def test_reconstruction():
    arr1 = np.linspace(0, 1, 20)
    arr2 = np.linspace(0, 1, 21)
    arr = arr1[:, None] * arr2[None, :]
    U, V, s = wt.kit.svd(arr, i=None)
    U = U.T
    S = np.diag(s)
    assert np.allclose(arr, np.dot(U, np.dot(S, V)))
