"""Brunold."""


# --- import --------------------------------------------------------------------------------------


import os
import pathlib

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_BrunoldrRaman"]


# --- from function -------------------------------------------------------------------------------


def from_BrunoldrRaman(filepath, name=None, parent=None, verbose=True) -> Data:
    """Create a data object from the Brunold rRaman instrument.

    Expects one energy (in wavenumbers) and one counts value.

    Parameters
    ----------
    filepath : path-like
        Path to .txt file.
        Can be either a local or remote file (http/ftp).
        Can be compressed with gz/bz2, decompression based on file name.
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    parent : WrightTools.Collection (optional)
        Collection to place new data object within. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object(s).
    """
    # parse filepath
    filestr = os.fspath(filepath)
    filepath = pathlib.Path(filepath)

    if not ".txt" in filepath.suffixes:
        wt_exceptions.WrongFileTypeWarning.warn(filepath, ".txt")
    # parse name
    if not name:
        name = filepath.name.split(".")[0]
    # create data
    kwargs = {"name": name, "kind": "BrunoldrRaman", "source": filestr}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)
    # array
    ds = np.DataSource(None)
    f = ds.open(filestr, "rt")
    arr = np.genfromtxt(f, delimiter="\t").T
    f.close()
    # chew through all scans
    data.create_variable(name="energy", values=arr[0], units="wn")
    data.create_channel(name="signal", values=arr[1])
    data.transform("energy")
    # finish
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  range: {0} to {1} (wn)".format(data.energy[0], data.energy[-1]))
        print("  size: {0}".format(data.size))
    return data
