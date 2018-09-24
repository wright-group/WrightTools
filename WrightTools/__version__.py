"""Define WrightTools version."""


# --- import --------------------------------------------------------------------------------------


import pathlib


# ---- define -------------------------------------------------------------------------------------


here = pathlib.Path(__file__).resolve().parent


__all__ = ["__version__", "__branch__"]


# --- version -------------------------------------------------------------------------------------


# read from VERSION file
with open(str(here / "VERSION")) as f:
    __version__ = f.read().strip()


# add git branch, if appropriate
p = here.parent / ".git"
if p.is_file():
    with open(str(p)) as f:
        p = p.parent / f.readline()[8:].strip()  # Strip "gitdir: "
p = p / "HEAD"
if p.exists():
    with open(str(p)) as f:
        __branch__ = f.readline().rstrip().split(r"/")[-1]
    if __branch__ != "master":
        __version__ += "+" + __branch__
else:
    __branch__ = None
