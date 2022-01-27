"""Ocean Optics."""


# --- import --------------------------------------------------------------------------------------


import os
import pathlib

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_ocean_optics"]


# --- from function -------------------------------------------------------------------------------


def from_ocean_optics(filepath, name=None, *, parent=None, verbose=True) -> Data:
    """Create a data object from an Ocean Optics brand spectrometer.

    Parameters
    ----------
    filepath : path-like
        Path to an ocean optics output file.
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
        New data object.
    """
    # parse filepath
    filestr = os.fspath(filepath)
    filepath = pathlib.Path(filepath)

    if not ".scope" in filepath.suffixes:
        wt_exceptions.WrongFileTypeWarning.warn(filepath, ".scope")
    # parse name
    if not name:
        name = filepath.name.split(".")[0]
    # create data
    kwargs = {"name": name, "kind": "Ocean Optics", "source": filestr}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)
    # array
    skip_header = 14
    skip_footer = 1
    ds = np.DataSource(None)
    f = ds.open(filestr, "rt")
    arr = np.genfromtxt(f, skip_header=skip_header, skip_footer=skip_footer, delimiter="\t").T
    f.close()
    # construct data
    data.create_variable(name="energy", values=arr[0], units="nm")
    data.create_channel(name="signal", values=arr[1])
    data.transform("energy")
    # finish
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  range: {0} to {1} (nm)".format(data.energy[0], data.energy[-1]))
        print("  size: {0}".format(data.size))
    return data
