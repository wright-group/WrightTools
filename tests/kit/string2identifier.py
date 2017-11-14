"""Test string2identifier."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_ω():
    assert wt.kit.string2identifier('blaiseω') == 'blaise_'


def test_numstart():
    assert wt.kit.string2identifier('2blaise') == '_2blaise'


def test_space():
    assert wt.kit.string2identifier('blaise thompson') == 'blaise_thompson'
