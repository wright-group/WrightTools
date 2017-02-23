'''
Import all subdirectories and modules.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os as _os
import importlib as _importlib

__all__ = []
for _path in _os.listdir(_os.path.dirname(__file__)):
    _full_path = _os.path.join(_os.path.dirname(__file__), _path)
    if _os.path.isdir(_full_path) and _path not in ['.git', 'examples','__pycache__']:
        _importlib.import_module('.'+_path, __name__)
        __all__.append(_path)
    elif _path[-3:] == '.py' and _path not in ['__init__.py', 'gui.py']:
        _importlib.import_module('.'+_path[:-3], __name__)
        __all__.append(_path[:-3])
