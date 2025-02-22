"""COLORS."""

# --- import --------------------------------------------------------------------------------------


import os
import pathlib
import collections

import numpy as np

from scipy.interpolate import griddata

from ._data import Data
from .. import kit as wt_kit
from numpy.lib.npyio import DataSource


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_COLORS"]


# --- from function -------------------------------------------------------------------------------


def from_COLORS(
    filepaths,
    name=None,
    cols=None,
    invert_d1=True,
    ignore=["w3", "wa", "dref", "m0", "m1", "m2", "m3", "m4", "m5", "m6"],
    parent=None,
    verbose=True,
):
    """Create data object from COLORS file(s).

    Parameters
    ----------
    filepaths : path-like or list of path-like
        Filepath(s).
        Can be either a local or remote file (http/ftp).
        Can be compressed with gz/bz2, decompression based on file name.
    name : string (optional)
        Unique dataset identifier. If None (default), autogenerated.
    cols : {'v0', 'v1', 'v2'} (optional)
        Format of COLORS dat file. If None, autorecognized. Default is None.
    invert_d1 : boolean (optional)
        Toggle inversion of D1 at import time. Default is True.
    ignore : list of strings (optional)
        Columns to ignore.
    parent : WrightTools.Collection (optional)
        Collection to place new data object within. Default is None.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    WrightTools.Data
        Data from COLORS.
    """
    # do we have a list of files or just one file? ------------------------------------------------
    if isinstance(filepaths, list):
        filestrs = [os.fspath(f) for f in filepaths]
        filepaths = [pathlib.Path(f) for f in filepaths]
    else:
        filestrs = [os.fspath(filepaths)]
        filepaths = [pathlib.Path(filepaths)]
    ds = DataSource(None)
    # define format of dat file -------------------------------------------------------------------
    if cols:
        pass
    else:
        f = ds.open(filestrs[0], "rt")
        num_cols = len(np.genfromtxt(f).T)
        f.close()
        if num_cols in [28, 35, 41]:
            cols = "v2"
        elif num_cols in [20]:
            cols = "v1"
        elif num_cols in [15, 16, 19]:
            cols = "v0"
        if verbose:
            print("cols recognized as", cols, "(%d)" % num_cols)
    if cols == "v2":
        axes = collections.OrderedDict()
        axes["w1"] = {"idx": 1, "units": "nm", "tolerance": 0.5, "label": "1"}
        axes["w2"] = {"idx": 3, "units": "nm", "tolerance": 0.5, "label": "2"}
        axes["w3"] = {"idx": 5, "units": "nm", "tolerance": 0.5, "label": "3"}
        axes["wm"] = {"idx": 7, "units": "nm", "tolerance": 0.5, "label": "m"}
        axes["wa"] = {"idx": 8, "units": "nm", "tolerance": 1.0, "label": "a"}
        axes["d0"] = {"idx": 10, "units": "fs", "tolerance": 25.0, "label": "0"}
        axes["d1"] = {"idx": 12, "units": "fs", "tolerance": 4.0, "label": "1"}
        axes["d2"] = {"idx": 14, "units": "fs", "tolerance": 4.0, "label": "2"}
        axes["m0"] = {"idx": 22, "units": None, "tolerance": 10.0, "label": 0}
        axes["m1"] = {"idx": 23, "units": None, "tolerance": 10.0, "label": 1}
        axes["m2"] = {"idx": 24, "units": None, "tolerance": 10.0, "label": 2}
        axes["m3"] = {"idx": 25, "units": None, "tolerance": 10.0, "label": 3}
        axes["m4"] = {"idx": 26, "units": None, "tolerance": 15.0, "label": 4}
        axes["m5"] = {"idx": 27, "units": None, "tolerance": 15.0, "label": 5}
        axes["m6"] = {"idx": 28, "units": None, "tolerance": 15.0, "label": 6}
        channels = collections.OrderedDict()
        channels["ai0"] = {"idx": 16, "label": "0"}
        channels["ai1"] = {"idx": 17, "label": "1"}
        channels["ai2"] = {"idx": 18, "label": "2"}
        channels["ai3"] = {"idx": 19, "label": "3"}
        channels["ai4"] = {"idx": 20, "label": "4"}
        channels["mc"] = {"idx": 21, "label": "a"}
    elif cols == "v1":
        axes = collections.OrderedDict()
        axes["w1"] = {"idx": 1, "units": "nm", "tolerance": 0.5, "label": "1"}
        axes["w2"] = {"idx": 3, "units": "nm", "tolerance": 0.5, "label": "2"}
        axes["wm"] = {"idx": 5, "units": "nm", "tolerance": 0.5, "label": "m"}
        axes["d1"] = {"idx": 6, "units": "fs", "tolerance": 3.0, "label": "1"}
        axes["d2"] = {"idx": 7, "units": "fs", "tolerance": 3.0, "label": "2"}
        channels = collections.OrderedDict()
        channels["ai0"] = {"idx": 8, "label": "0"}
        channels["ai1"] = {"idx": 9, "label": "1"}
        channels["ai2"] = {"idx": 10, "label": "2"}
        channels["ai3"] = {"idx": 11, "label": "3"}
    elif cols == "v0":
        axes = collections.OrderedDict()
        axes["w1"] = {"idx": 1, "units": "nm", "tolerance": 0.5, "label": "1"}
        axes["w2"] = {"idx": 3, "units": "nm", "tolerance": 0.5, "label": "2"}
        axes["wm"] = {"idx": 5, "units": "nm", "tolerance": 0.5, "label": "m"}
        axes["d1"] = {"idx": 6, "units": "fs", "tolerance": 3.0, "label": "1"}
        axes["d2"] = {"idx": 8, "units": "fs", "tolerance": 3.0, "label": "2"}
        channels = collections.OrderedDict()
        channels["ai0"] = {"idx": 10, "label": "0"}
        channels["ai1"] = {"idx": 11, "label": "1"}
        channels["ai2"] = {"idx": 12, "label": "2"}
        channels["ai3"] = {"idx": 13, "label": "3"}
    # import full array ---------------------------------------------------------------------------
    arr = []
    for f in filestrs:
        ff = ds.open(f, "rt")
        arr.append(np.genfromtxt(ff).T)
        ff.close()
    arr = np.concatenate(arr, axis=1)
    if invert_d1:
        idx = axes["d1"]["idx"]
        arr[idx] = -arr[idx]
    # recognize dimensionality of data ------------------------------------------------------------
    axes_discover = axes.copy()
    for key in ignore:
        if key in axes_discover:
            axes_discover.pop(key)  # remove dimensions that mess up discovery
    scanned = wt_kit.discover_dimensions(arr, axes_discover)
    # create data object --------------------------------------------------------------------------
    if name is None:
        name = wt_kit.string2identifier(filepaths[0].name)
    kwargs = {"name": name, "kind": "COLORS", "source": filestrs}
    if parent is not None:
        data = parent.create_data(**kwargs)
    else:
        data = Data(**kwargs)
    # grid and fill data --------------------------------------------------------------------------
    # variables
    ndim = len(scanned)
    for i, key in enumerate(scanned.keys()):
        for name in key.split("="):
            shape = [1] * ndim
            a = scanned[key]
            shape[i] = a.size
            a.shape = tuple(shape)
            units = axes[name]["units"]
            label = axes[name]["label"]
            data.create_variable(name=name, values=a, units=units, label=label)
    for key, dic in axes.items():
        if key not in data.variable_names:
            c = np.mean(arr[dic["idx"]])
            if not np.isnan(c):
                shape = [1] * ndim
                a = np.array([c])
                a.shape = tuple(shape)
                units = dic["units"]
                label = dic["label"]
                data.create_variable(name=key, values=a, units=units, label=label)
    # channels
    points = tuple(arr[axes[key.split("=")[0]]["idx"]] for key in scanned.keys())
    if len(scanned) == 1:  # 1D data
        (xi,) = scanned.values()
        for key in channels.keys():
            channel = channels[key]
            zi = arr[channel["idx"]]
            grid_i = griddata(points, zi, xi, method="nearest")
            data.create_channel(name=key, values=grid_i)
    else:  # all other dimensionalities
        xi = tuple(np.meshgrid(*scanned.values(), indexing="ij"))
        for key in channels.keys():
            channel = channels[key]
            zi = arr[channel["idx"]]
            fill_value = min(zi)
            grid_i = griddata(points, zi, xi, method="linear", fill_value=fill_value)
            data.create_channel(name=key, values=grid_i)
    # axes
    data.transform(*scanned.keys())
    # return --------------------------------------------------------------------------------------
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  axes: {0}".format(data.axis_names))
        print("  shape: {0}".format(data.shape))
    return data
