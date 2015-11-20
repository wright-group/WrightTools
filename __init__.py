'''
Import all subdirectories and modules.
'''

### import ####################################################################

import os as _os
_wt_dir = _os.path.dirname(__file__)

### version information #######################################################

# get config
import ConfigParser as _ConfigParser
_config = _ConfigParser.SafeConfigParser()
_config.read(_os.path.join(_wt_dir, 'main.ini')) 

# attempt to update ini
try:
    _HEAD_file = _os.path.join(_wt_dir, '.git', 'logs', 'HEAD')
    with open(_HEAD_file) as _f:
        for _line in _f.readlines():
            _sha = _line.split(' ')[1]  # most recent commit is last
    _config.set('main', 'git sha', _sha)
except:
    pass

# WrightTools version format: a.b.c.d
# a - major release
# b - minor release
# c - bugfix
# d - git sha key
__version__ = '1.0.0.' + _config.get('main', 'git sha')[:7]

## iterate ####################################################################

__all__ = []
for _path in _os.listdir(_wt_dir):
    _full_path = _os.path.join(_wt_dir, _path)
    if _os.path.isdir(_full_path) and _path not in ['.git', 'examples', 'widgets']:
        __import__(_path, locals(), globals())
        __all__.append(_path)
    elif _path[-3:] == '.py' and _path not in ['__init__.py', 'gui.py']:
        __import__(_path[:-3], locals(), globals())
        __all__.append(_path[:-3])
