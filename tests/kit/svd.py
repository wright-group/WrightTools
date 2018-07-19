"""Test svd."""

import numpy as np
import WrightTools as wt


def test_int():
    arr1 = np.linspace(0, 1, 20)
    arr2 = np.linspace(0, 1, 21)
    arr = arr1[:, None] * arr2[None, :]
    u, v, s = wt.kit.svd(arr, i=0)
    assert u.shape == (20,)
    assert v.shape == (21,)
    assert s.shape == ()


def test_reconstruction():
    arr1 = np.linspace(0, 1, 20)
    arr2 = np.linspace(0, 1, 21)
    arr = arr1[:, None] * arr2[None, :]
    U, V, s = wt.kit.svd(arr, i=None)
    U = U.T
    S = np.diag(s)
    assert np.allclose(arr, np.dot(U, np.dot(S, V)))


def test_slice():
    arr1 = np.linspace(0, 1, 20)
    arr2 = np.linspace(0, 1, 21)
    arr = arr1[:, None] * arr2[None, :]
    u, v, s = wt.kit.svd(arr, i=slice(1, 5))
    assert u.shape == (4, 20)
    assert v.shape == (4, 21)
    assert s.shape == (4,)


if __name__ == "__main__":
    test_int()
    test_reconstruction()
    test_slice()
