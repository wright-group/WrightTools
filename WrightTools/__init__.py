"""WrightTools init."""


# --- import --------------------------------------------------------------------------------------


import sys as _sys
import os as _os

import matplotlib as _matplotlib


# --- temp directory ------------------------------------------------------------------------------


_here = _os.path.abspath(_os.path.dirname(__file__))

_temp_dir = _os.path.join(_here, 'temp')
if not _os.path.isdir(_temp_dir):
    _os.mkdir(_temp_dir)


# --- import --------------------------------------------------------------------------------------


from . import artists           # noqa: F401
from . import calibration       # noqa: F401
from . import data              # noqa: F401
from . import diagrams          # noqa: F401
from . import fit               # noqa: F401
from . import google_drive      # noqa: F401
from . import kit               # noqa: F401
from . import tuning            # noqa: F401
from . import units             # noqa: F401


# --- version -------------------------------------------------------------------------------------


# read from VERSION file
with open(_os.path.join(_os.path.dirname(_here), 'VERSION')) as _version_file:
    __version__ = _version_file.read().strip()

# add git branch, if appropriate
_directory = _os.path.dirname(_os.path.dirname(__file__))
_p = _os.path.join(_directory, '.git', 'HEAD')
if _os.path.isfile(_p):
    with open(_p) as _f:
        __branch__ = _f.readline().rstrip().split(r'/')[-1]
    if __branch__ != 'master':
        __version__ += '-' + __branch__
else:
    __branch__ = None


# --- rcparams ------------------------------------------------------------------------------------

if int(_sys.version.split('.')[0]) > 2 and int(_matplotlib.__version__.split('.')[0]) > 1:
    artists.apply_rcparams('fast')
