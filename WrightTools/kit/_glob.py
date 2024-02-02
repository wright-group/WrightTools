import pathlib
import os
from typing import Union
from ..data import Data
from ..collection import Collection
from .._open import open


__all__ = ["describe_wt5", "glob_wt5s"]


def describe_wt5(path: Union[str, os.PathLike])->dict:
    """report useful general information about a wt5 file"""
    wt5 = open(path)
    desc = dict()
    desc["name"] = wt5.natural_name
    try:
        desc["created"] = wt5.created.human
    except:  # likely an old timestamp that cannot be parsed
        desc["created"] = wt5.attrs["created"]

    if isinstance(wt5, Data):
        desc["shape"] = wt5.shape
        desc["axes"] = wt5.axis_expressions
        desc["nvars"] = len(wt5.variables)
        desc["nchan"] = len(wt5.channels)
    elif isinstance(wt5, Collection):
        for k in ["shape", "axes", "nvars", "nchan"]:
            desc[k] = "---"
    wt5.close()
    return desc


def glob_wt5s(directory: Union[str, os.PathLike]):
    """find all wt5 files in a directory"""
    return pathlib.Path(directory).glob("**/*.wt5")
