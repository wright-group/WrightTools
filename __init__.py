### import ####################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

import os as _os
import sys as _sys

from . import artists
from . import calibration
from . import data
from . import fit
from . import google_drive
from . import kit
from . import units


### define ####################################################################


# WrightTools version format: a.b.c
# a - major release (potentially breaks old scripts)
# b - minor release
# c - bugfix
__version__ = '2.8.0'


### create temp folder if none exists #########################################


_temp_dir = _os.path.join(_os.path.dirname(__file__), 'temp')
if not _os.path.isdir(_temp_dir):
    _os.mkdir(_temp_dir)

