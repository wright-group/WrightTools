

# --- import --------------------------------------------------------------------------------------


import os
import pathlib
import warnings
import re

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_LabRAM"]


# --- from function -------------------------------------------------------------------------------


def from_LabRAM(filepath, name=None, parent=None, verbose=True) -> Data:
    """Create a data object from Horiba LabRAM txt file.

    Parameters
    ----------
    filepath : path-like
        Path to txt file.
        Can be either a local or remote file (http/ftp).
        Can be compressed with gz/bz2, decompression based on file name.
    name : string (optional)
        Name to give to the created data object. If None, name is extracted from file.
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

    kwargs = {"name": name, "kind": "Horiba", "source": filestr}
    # create data
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)

    # header
    header = {}
    spectral_units = None
    with open(filepath) as f:
        while True:
            line:str = f.readline()
            if not line.startswith("#"):          
                break
            key, val = [s.strip() for s in line[1:].split("=", 1)]
            if "Acq. time" in key:
                val = float(val)
            elif "Accumulations" in line:
                val = int(val)
            elif "Range (" in line:
                if 'cm' in line:
                    spectral_units='wn'
                else:
                    spectral_units='nm'
            elif 'Spectro' in line:
                if 'cm' in line:
                    spectral_units='wn'
                else:
                    spectral_units='nm'
            header[key] = val
    acquisition_time = header["Acq. time (s)"] * header["Accumulations"]

    # array
    ds = np.DataSource(None)
    f = ds.open(filestr, "rt")
    arr = np.genfromtxt(f, delimiter="\t")
    f.close()

    wm = arr[0]
    arr = arr[1:]
    spatial_ndim = np.isnan(wm).sum()
    wm = wm[spatial_ndim:]

    if spatial_ndim == 0:  # spectrum
        data.create_variable("wm", values=wm, units=spectral_units)
        data.create_channel("signal", values=arr / acquisition_time, units="cps")
        data.transform("wm")
    elif spatial_ndim == 1:  # spectrum vs (x or xindex)
        data.create_variable("wm", values=wm[:, None], units=spectral_units)
        data.create_channel("signal", values=arr[:,1:].T / acquisition_time, units="cps")
        x = arr[:, 0]
        if np.all(np.diff(x) == 1):  # xindex
            data.create_variable("xindex", values=x[None, :])
            data.transform("wm", "xindex")
        else:  # x
            data.create_variable("x", values=x[None, :], units="um")
            data.transform("wm", "x")
    elif spatial_ndim == 2:  # spectrum vs x vs y
        # fold to 3D
        x = list(set(arr[:, 0]))  # 0th column is stepped always (?)
        x = x.reshape(1, -1, 1)
        y = arr[:, 1].reshape(1, x.size, -1)
        ypts = y.mean(axis=1).reshape(1, 1, -1)
        sig = arr[:, 2:].T.rehsape(wm.size, x.size, -1)  # TODO: test fold
        data.create_variable("wm", values=wm[:, None, None], units=spectral_units)
        data.create_variable("x", values=x, units="um")
        data.create_variable("y", values=y, units="um")
        data.create_variable("y_points", values=ypts, units="um")
        data.create_channel("signal", values=sig / acquisition_time, units="cps")
        data.transform("wm", "x", "ypts")

    return data

