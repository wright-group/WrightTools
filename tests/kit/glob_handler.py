"""Test glob_handler."""


# --- import -------------------------------------------------------------------------------------


import os

import WrightTools as wt


# --- define -------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- test ---------------------------------------------------------------------------------------


def test_here():
    wt.kit.glob_handler('.py', here)  # exception will be raised if broken
