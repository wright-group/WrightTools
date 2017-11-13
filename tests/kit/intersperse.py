"""Test intersperse."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_abcd():
    lis = ['a', 'b', 'c', 'd']
    assert wt.kit.intersperse(lis, 'blaise') == ['a', 'blaise', 'b', 'blaise', 'c', 'blaise', 'd']


def test_empty():
    assert wt.kit.intersperse([], 'blaise') == []


def test_1():
    assert wt.kit.intersperse(['sprout'], 'potato') == ['sprout']
