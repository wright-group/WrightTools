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
