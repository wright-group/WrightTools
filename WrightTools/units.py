"""
unit and label handling in WrightTools
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


# --- units ---------------------------------------------------------------------------------------


# units are stored in dictionaries of like kind. format:
#     unit : to native, from native, units_symbol, units_label

# angle units (native: rad)
angle = {'kind': 'angle',
         'rad': ['x', 'x', r'rad'],
         'deg': ['x/57.2958', '57.2958*x', r'deg']}

# delay units (native: fs)
fs_per_mm = 3336.
delay = {'kind': 'delay',
         'fs': ['x', 'x', r'fs'],
         'ps': ['x*1e3', 'x/1e3', r'ps'],
         'ns': ['x*1e6', 'x/1e6', r'ns'],
         'mm_delay': ['x*2*fs_per_mm', 'x/(2*fs_per_mm)', r'mm']}

# energy units (native: nm)
energy = {'kind': 'energy',
          'nm': ['x', 'x', r'nm'],
          'wn': ['1e7/x', '1e7/x', r'cm^{-1}'],
          'eV': ['1240./x', '1240./x', r'eV'],
          'meV': ['1240000./x', '1240000./x', r'meV'],
          'Hz': ['2.99792458e17/x', '2.99792458e17/x', r'Hz'],
          'THz': ['2.99792458e5/x', '2.99792458e5/x', r'THz'],
          'GHz': ['2.99792458e8/x', '2.99792458e8/x', r'GHz']}

# fluence units (native: uJ per sq. cm)
fluence = {'kind': 'fluence',
           'uJ per sq. cm': ['x', 'x', r'\frac{\mu J}{cm^{2}}']}

# optical density units (native: od)
od = {'kind': 'od',
      'mOD': ['1e3*x', 'x/1e3', r'mOD'],
      'OD': ['x', 'x', r'OD']}

# position units (native: mm)
position = {'kind': 'position',
            # can't have same name as energy nm
            'nm_p': ['x/1e6', '1e6/x', r'nm'],
            'um': ['x/1000.', '1000/x.', r'um'],
            'mm': ['x', 'x', r'mm'],
            'cm': ['10.*x', 'x/10.', r'cm'],
            'in': ['x*0.039370', '0.039370*x', r'in']}

# pulse width units (native: FWHM)
pulse_width = {'kind': 'pulse_width',
               'FWHM': ['x', 'x', r'FWHM']}

# time units (native: s)
time = {'kind': 'time',
        'fs_t': ['x/1e15', 'x*1e15', r'fs'],
        'ps_t': ['x/1e12', 'x*1e12', r'ps'],
        'ns_t': ['x/1e9', 'x*1e9', r'ns'],
        'us_t': ['x/1e6', 'x*1e6', r'us'],
        'ms_t': ['x/1000.', 'x*1000.', r'ms'],
        's_t': ['x', 'x', r's'],
        'm_t': ['x*60.', 'x/60.', r'm'],
        'h_t': ['x*3600.', 'x/3600.', r'h'],
        'd_t': ['x*86400.', 'x/86400.', r'd']}

unit_dicts = [angle, delay, energy, time, position, pulse_width, fluence, od]


def converter(val, current_unit, destination_unit):
    x = val
    for dic in unit_dicts:
        if current_unit in dic.keys() and destination_unit in dic.keys():
            try:
                native = eval(dic[current_unit][0])
            except ZeroDivisionError:
                native = np.inf
            x = native
            try:
                out = eval(dic[destination_unit][1])
            except ZeroDivisionError:
                out = np.inf
            return out
    # if all dictionaries fail
    if current_unit is None and destination_unit is None:
        pass
    else:
        print('conversion {0} to {1} not valid: returning input'.format(
            current_unit, destination_unit))
    return val


def kind(units):
    """ Find the kind of given units.

    Parameters
    ----------
    units : string
        The units of interest

    Returns
    -------
    string
        The kind of the given units. If no match is found, returns None.
    """
    for d in unit_dicts:
        if units in d.keys():
            return str(d['kind'])


# --- symbol --------------------------------------------------------------------------------------


class symbol_dict(dict):
    # subclass dictionary to get at __missing__ method
    def __missing__(self, key):
        return self['default']


# color
color_symbols = symbol_dict()
color_symbols['default'] = r'E'
color_symbols['nm'] = r'\lambda'
color_symbols['wn'] = r'\bar\nu'
color_symbols['eV'] = r'\hslash\omega'
color_symbols['Hz'] = r'f'
color_symbols['THz'] = r'f'
color_symbols['GHz'] = r'f'

# delay
delay_symbols = symbol_dict()
delay_symbols['default'] = r'\tau'

# fluence
fluence_symbols = symbol_dict()
fluence_symbols['default'] = r'\mathcal{F}'

# pulse width
pulse_width_symbols = symbol_dict()
pulse_width_symbols['default'] = r'\sigma'

# catch all
none_symbols = symbol_dict()
none_symbols['default'] = ''


def get_default_symbol_type(units_str):
    if units_str in ['nm', 'wn', 'eV']:
        return 'color_symbols'
    elif units_str in ['fs', 'ps', 'ns']:
        return 'delay_symbols'
    elif units_str in ['uJ per sq. cm']:
        return 'fluence_symbols'
    elif units_str in ['FWHM']:
        return 'pulse_width_symbols'
    else:
        return 'none_symbols'
