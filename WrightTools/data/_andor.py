"""Andor."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_andor"]


# --- from function -------------------------------------------------------------------------------


def from_andor(filepath, name=None, parent=None, verbose=True):
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
    if not filepath.endswith("asc"):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, "asc")
    # parse name
    if not name:
        name = os.path.basename(filepath).split(".")[0]
    # create data
    kwargs = {"name": name, "kind": "andor", "source": filepath}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)   

    with open(filepath) as f:
        axis0 = []
        arr = []
        while True:
            line = f.readline().strip()[:-1]
            if len(line) == 0:
                break
            else:
                line = line.split(',')
                line = [eval(x) for x in line]
                axis0.append(line.pop(0))
                arr.append(line)
        
        i = 0
        while i < 3:
            line = f.readline().strip()
            print(line)
            if len(line) == 0:
                i += 1
            else:
                try:
                    key, val = line.split(':', 1)
                except ValueError:
                    val = ''
                data.attrs[key.strip()] = val.strip()

    arr = np.array(arr, dtype=np.float)
    arr = data.create_channel(name='signal', values=arr, signed=False)

    axis0 = np.array(axis0)
    if axis0.dtype == int:
        axis0 = data.create_variable(name='xpos', values=axis0[:, None], units='um')
    else:
        axis0 = data.create_variable(name='wm', values=axis0[:, None], units='nm')

    data.create_variable(name='ypos', values=np.arange(arr.shape[1])[None, :], units='um')

    data.transform(axis0.name, 'ypos')
    # finish
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  range: {0} to {1} (nm)".format(data.wm[0], data.wm[-1]))
        print("  size: {0}".format(data.size))
    return data
