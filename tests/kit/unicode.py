"""Test unicode dictionary."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_print():
    for key, value in wt.kit.unicode_dictionary.items():
        print(key, value)
