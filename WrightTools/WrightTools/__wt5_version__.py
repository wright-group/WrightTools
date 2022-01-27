"""Define wt5 version."""


# --- import --------------------------------------------------------------------------------------


import os


# ---- define -------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


__all__ = ["__wt5_version__"]


# --- version -------------------------------------------------------------------------------------


# read from VERSION file
with open(os.path.join(here, "WT5_VERSION")) as f:
    __wt5_version__ = f.read().strip()
