"""Tensor 27."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_Tensor27"]


# --- from function -------------------------------------------------------------------------------


def from_Tensor27(filepath, name=None, parent=None, verbose=True) -> Data:
    """Create a data object from a Tensor27 FTIR file.

    .. plot::

        >>> import WrightTools as wt
        >>> import matplotlib.pyplot as plt
        >>> from WrightTools import datasets
        >>> p = datasets.Tensor27.CuPCtS_powder_ATR
        >>> data = wt.data.from_Tensor27(p)
        >>> artist = wt.artists.quick1D(data)
        >>> plt.xlim(1300,1700)
        >>> plt.ylim(-0.005,.02)

    Parameters
    ----------
    filepath : string
        Path to Tensor27 output file (.dpt).
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
    if not filepath.endswith("dpt"):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, "dpt")
    # parse name
    if not name:
        name = os.path.basename(filepath).split(".")[0]
    # create data
    kwargs = {"name": name, "kind": "Tensor27", "source": filepath}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)
    # array
    arr = np.genfromtxt(filepath, skip_header=0).T
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
