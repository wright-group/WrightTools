"""Andor."""


# --- import --------------------------------------------------------------------------------------


import os
import time

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions
from ..kit import _timestamp as timestamp


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_Solis"]


# --- from function -------------------------------------------------------------------------------


def from_Solis(filepath, name=None, parent=None, verbose=True) -> Data:
    """Create a data object from Andor Solis software (ascii exports).

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
        New data object.
    """
    # parse filepath
    if not filepath.endswith("asc"):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, "asc")
    # parse name
    if not name:
        name = os.path.basename(filepath).split(".")[0]
    # create data
    with open(filepath) as f:
        axis0 = []
        arr = []
        attrs = {}
        while True:
            line = f.readline().strip()[:-1]
            if len(line) == 0:
                break
            else:
                line = line.split(",")
                line = [float(x) for x in line]
                axis0.append(line.pop(0))
                arr.append(line)

        i = 0
        while i < 3:
            line = f.readline().strip()
            if len(line) == 0:
                i += 1
            else:
                try:
                    key, val = line.split(":", 1)
                except ValueError:
                    val = ""
                attrs[key.strip()] = val.strip()

    created = attrs["Date and Time"]  # is this UTC?
    created = time.strptime(created, "%a %b %d %H:%M:%S %Y")
    created = timestamp.TimeStamp(time.mktime(created)).RFC3339

    kwargs = {"name": name, "kind": "Solis", "source": filepath, "created": created}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)
    arr = np.array(arr)
    arr /= float(attrs["Exposure Time (secs)"])
    # signal has units of Hz because time normalized
    arr = data.create_channel(name="signal", values=arr, signed=False, units="Hz")
    axis0 = np.array(axis0)
    if float(attrs["Grating Groove Density (l/mm)"]) == 0:
        xname = "xindex"
        xunits = None
    else:
        xname = "wm"
        xunits = "nm"
    data.create_variable(name=xname, values=axis0[:, None], units=xunits)
    data.create_variable(name="yindex", values=np.arange(arr.shape[1])[None, :], units=None)
    data.transform(data.variables[0].natural_name, "yindex")

    for key, val in attrs.items():
        data.attrs[key] = val

    # finish
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  axes: {0}".format(data.axis_names))
        print("  shape: {0}".format(data.shape))
    return data
