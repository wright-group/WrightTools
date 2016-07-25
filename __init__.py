'''
Import all subdirectories and modules.
'''


### import ####################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

import os as _os
import sys as _sys
import importlib as _importlib
_wt_dir = _os.path.dirname(__file__)

try:
    import configparser as _ConfigParser  # python 3
except ImportError:
    import ConfigParser as _ConfigParser  # python 2


### create temp folder if none exists #########################################


_temp_dir = _os.path.join(_wt_dir, 'temp')
if not _os.path.isdir(_temp_dir):
    _os.mkdir(_temp_dir)


### version information #######################################################


# get config
_ini_path = _os.path.join(_wt_dir, 'main.ini')
if _sys.version[0] == '3':
    _config = _ConfigParser.ConfigParser()
else:
    _config = _ConfigParser.SafeConfigParser()
_config.read(_ini_path)

# attempt get git sha
try:
    _HEAD_file = _os.path.join(_wt_dir, '.git', 'logs', 'HEAD')
    with open(_HEAD_file) as _f:
        for _line in _f.readlines():
            _sha = _line.split(' ')[1]  # most recent commit is last
except:
    _sha = '0000000'

# WrightTools version format: a.b.c.d
# a - major release
# b - minor release
# c - bugfix
# d - git sha key
__version__ = _config.get('main', 'version') + '.' + _sha[:7]


## import own modules #########################################################


__all__ = []
for _path in _os.listdir(_wt_dir):
    _full_path = _os.path.join(_wt_dir, _path)
    if _os.path.isdir(_full_path) and _path not in ['.git', 'examples', 'widgets', 'documentation', 'temp']:
        _importlib.import_module('.'+_path, __name__)
        __all__.append(_path)
    elif _path[-3:] == '.py' and _path not in ['__init__.py', 'gui.py']:
        _importlib.import_module('.'+_path[:-3], __name__)
        __all__.append(_path[:-3])
