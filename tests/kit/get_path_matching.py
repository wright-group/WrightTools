"""Test get_path_matching."""


# --- import -------------------------------------------------------------------------------------


import os

import WrightTools as wt


# --- define -------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- test ---------------------------------------------------------------------------------------


def test_WrightTools():
    wt.kit.get_path_matching('WrightTools')  # exception will be raised if broken
