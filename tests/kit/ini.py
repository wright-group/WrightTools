#! /usr/bin/env python3
"""Test INI file creation and parsing."""
from WrightTools.kit import INI
import numpy as np
import os


def test_create():
    ini = INI('test.ini')
    ini.add_section('section0')
    ini.add_section('section1')
    ini.write('section0', 'integer', 39)
    ini.write('section1', 'float', np.e)
    ini.write('section0', 'string', 'hello world, this is an ini file')
    ini.write('section1', 'list', ['Linux', 'rocks'])
    ini.write('section1', 'tuple', ('Linux', 'rocks'))
    ini.write('section0', 'none', None)
    ini.write('section1', 'true', True)
    ini.write('section1', 'false', False)
    ini.write('section0', 'ndarray', np.linspace(0, 10, 21))


def test_has_option():
    ini = INI('test.ini')
    assert ini.has_option('section0', 'integer')
    assert not ini.has_option('section0', 'nooooooooooooo')


def test_has_section():
    ini = INI('test.ini')
    assert ini.has_section('section0')
    assert not ini.has_section('section9')


def test_dictionary():
    ini = INI('test.ini')
    dict_ = ini.dictionary
    assert isinstance(dict_, dict)
    assert dict_['section0']['integer'] == '39'


def test_get_options():
    ini = INI('test.ini')
    assert ini.get_options('section0') == ['integer', 'string', 'none', 'ndarray']


def test_sections():
    ini = INI('test.ini')
    assert ini.sections == ['section0', 'section1']


def test_read():
    ini = INI('test.ini')
    assert ini.read('section0', 'integer') == 39
    assert np.isclose(ini.read('section1', 'float'), np.e)
    assert ini.read('section0', 'string') == 'hello world, this is an ini file'
    assert ini.read('section1', 'list') == ['Linux', 'rocks']
    #assert ini.read('section1', 'tuple') == ('Linux', 'rocks')
    assert ini.read('section0', 'none') is None
    assert ini.read('section1', 'true') is True
    assert ini.read('section1', 'false') is False
    #print(ini.read('section0', 'ndarray'))
    #assert np.allclose(np.array(ini.read('section0', 'ndarray')), np.linspace(0,10,21))


def test_clear():
    ini = INI('test.ini')
    ini.clear()
    assert ini.sections == []
    os.remove('test.ini')


if __name__ == '__main__':
    test_create()
    test_has_option()
    test_has_section()
    test_dictionary()
    test_get_options()
    test_sections()
    test_read()
    test_clear()
