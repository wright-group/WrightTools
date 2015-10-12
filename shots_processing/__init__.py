'''
Import all subdirectories and modules.
'''

import os as _os

__all__ = []
for _path in _os.listdir(_os.path.dirname(__file__)):
    _full_path = _os.path.join(_os.path.dirname(__file__), _path)
    if _os.path.isdir(_full_path) and _path not in ['.git', 'examples']:
        __import__(_path, locals(), globals())
        __all__.append(_path)
    elif _path[-3:] == '.py' and _path != '__init__.py':
        __import__(_path[:-3], locals(), globals())
        __all__.append(_path[:-3])
