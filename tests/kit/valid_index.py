"""Test valid index function."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test__1_5__7():
    index = (1, 5)
    shape = (7,)
    assert wt.kit.valid_index(index, shape) == (5,)


def test__4_2_12__1_25_1():
    index = (4, 2, 12)
    shape = (1, 25, 1)
    assert wt.kit.valid_index(index, shape) == (0, 2, 0)


def test__s__23():
    index = (slice(None),)
    shape = (23,)
    assert wt.kit.valid_index(index, shape) == (slice(None),)


def test__s__1_25():
    index = (slice(None),)
    shape = (1, 25,)
    assert wt.kit.valid_index(index, shape) == (slice(None), slice(None))


def test__ss_ss__1_25():
    index = (slice(20, None, 1), slice(20, None, 1))
    shape = (1, 25,)
    assert wt.kit.valid_index(index, shape) == (slice(None), slice(20, None, 1))


def test__s__13_25_99():
    index = (slice(None),)
    shape = (13, 25, 99)
    assert wt.kit.valid_index(index, shape) == (slice(None), slice(None), slice(None))


def test__s_s__51():
    index = (slice(None), slice(None))
    shape = (51,)
    assert wt.kit.valid_index(index, shape) == (slice(None),)


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test__1_5__7()
    test__4_2_12__1_25_1()
    test__s__23()
    test__s__1_25()
    test__ss_ss__1_25()
    test__s__13_25_99()
    test__s_s__51()
