"""WrightTools init."""
# flake8: noqa


# --- import --------------------------------------------------------------------------------------


import sys as _sys

from .__version__ import *
from . import artists
from . import calibration
from . import collection
from . import data
from . import diagrams
from . import fit
from . import google_drive
from . import kit
from . import units

from .collection._collection import *
from .data._data import *
from .open import *


# --- rcparams ------------------------------------------------------------------------------------


if int(_sys.version.split('.')[0]) > 2:
    artists.apply_rcparams('fast')
