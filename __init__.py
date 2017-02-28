### create temp folder if none exists #########################################

import os as _os
_temp_dir = _os.path.join(_os.path.dirname(__file__), 'temp')
if not _os.path.isdir(_temp_dir):
    _os.mkdir(_temp_dir)


### version ###################################################################


# MAJOR.MINOR.PATCH (semantic versioning)
# major version changes may break backwards compatibility
__version__ = '2.8.3'  


### import ####################################################################


from . import artists
from . import calibration
from . import data
from . import diagrams
from . import fit
from . import google_drive
from . import kit
from . import tuning
from . import units
