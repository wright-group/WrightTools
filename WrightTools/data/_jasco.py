"""JASCO."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_JASCO"]


# --- from function -------------------------------------------------------------------------------


def from_JASCO(filepath, name=None, parent=None, verbose=True) -> Data:
    """Create a data object from JASCO UV-Vis spectrometers.

    Parameters
    ----------
    filepath : string, list of strings, or array of strings
        Path to .txt file.
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
    if not filepath.endswith("txt"):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, "txt")
    # parse name
    if not name:
        name = os.path.basename(filepath).split(".")[0]
    # create data
    kwargs = {"name": name, "kind": "JASCO", "source": filepath}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)
    # array
    arr = np.genfromtxt(filepath, skip_header=18).T
    # chew through all scans
    data.create_variable(name="energy", values=arr[0], units="nm")
    data.create_channel(name="signal", values=arr[1])
    data.transform("energy")
    # finish
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  range: {0} to {1} (nm)".format(data.energy[0], data.energy[-1]))
        print("  size: {0}".format(data.size))
    return data
