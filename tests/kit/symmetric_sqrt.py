"""test symmetric_sqrt"""

# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_numbers():
    numbers = np.random.normal(size=100)
    for number in numbers:
        answer = wt.kit.symmetric_sqrt(number)
        assert answer == np.sign(number) * np.sqrt(np.abs(number))


def test_no_reallocation():
    a = np.linspace(-9, 9, 3)
    out = np.empty_like(a)
    ret = wt.kit.symmetric_sqrt(a, out=out)
    assert out is ret
    assert np.allclose(ret, [-3, 0, 3])
