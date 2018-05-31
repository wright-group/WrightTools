"""Define WrightTools version."""


# --- import --------------------------------------------------------------------------------------


import os


# ---- define -------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


__all__ = ['__version__', '__branch__']


# --- version -------------------------------------------------------------------------------------


# read from VERSION file
with open(os.path.join(here, 'VERSION')) as f:
    __version__ = f.read().strip()


# add git branch, if appropriate
p = os.path.join(os.path.dirname(here), '.git', 'HEAD')
if os.path.isfile(p):
    with open(p) as f:
        __branch__ = f.readline().rstrip().split(r'/')[-1]
    if __branch__ != 'master':
        __version__ += '+' + __branch__
else:
    __branch__ = None
