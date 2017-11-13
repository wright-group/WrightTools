"""Test file_len."""


# --- import -------------------------------------------------------------------------------------


import os

import WrightTools as wt


# --- define -------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- test ---------------------------------------------------------------------------------------


def test_a():
    p = os.path.join(here, 'file_len', 'a.txt')
    assert wt.kit.file_len(p) == 6


def test_b():
    p = os.path.join(here, 'file_len', 'b.txt')
    assert wt.kit.file_len(p) == 21
