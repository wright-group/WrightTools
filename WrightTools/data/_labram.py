# --- import --------------------------------------------------------------------------------------


import os
import pathlib
import warnings
import time

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions
from ..kit import _timestamp as timestamp


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

    ds = np.DataSource(None)
    f = ds.open(filestr, "rt", encoding="ISO-8859-1")

    # header
    header = {}
    while True:
        line = f.readline()
        if not line.startswith("#"):
            wm = np.array([np.nan if i == "" else float(i) for i in line.split("\t")])
            break
        key, val = [s.strip() for s in line[1:].split("=", 1)]
        header[key] = val

    if not header:
        raise NotImplementedError(
            "At this time, we require metadata to parse LabRAM data. \
            Consider manually importing this data."
        )

    # extract key metadata
    created = header["Acquired"]
    created = time.strptime(created, "%d.%m.%Y %H:%M:%S")
    created = timestamp.TimeStamp(time.mktime(created)).RFC3339
    data.attrs["created"] = created
    data.attrs.update()

    try:
        acquisition_time = float(header["Acq. time (s)"]) * int(header["Accumulations"])
        channel_units = "cps"
    except KeyError:
        warnings.warn(f"{filepath.name}: could not determine signal acquisition time.")
        acquisition_time = 1
        channel_units = None

    # spectral units
    k_spec = [k for k in header.keys() if k.startswith("Spectro") or k.startswith("Range (")][0]
    if "cm-¹" in k_spec:
        spectral_units = "wn"
    elif "nm" in k_spec:
        spectral_units = "nm"
    else:
        warnings.warn(f"spectral units are unrecognized: {k_spec}")
        spectral_units = None

    # dimensionality
    extra_dims = np.isnan(wm).sum()

    if extra_dims == 0:  # single spectrum; we extracted wm wrong, so go back in file
        f.seek(0)
        wm, arr = np.genfromtxt(f, delimiter="\t", unpack=True)
        f.close()
        data.create_variable("wm", values=wm, units=spectral_units)
        data.create_channel("signal", values=arr / acquisition_time, units=channel_units)
        data.transform("wm")
    else:
        arr = np.genfromtxt(f, delimiter="\t")
        f.close()
        wm = wm[extra_dims:]

        if extra_dims == 1:  # spectrum vs (x or survey)
            data.create_variable("wm", values=wm[:, None], units=spectral_units)
            data.create_channel(
                "signal", values=arr[:, 1:].T / acquisition_time, units=channel_units
            )
            x = arr[:, 0]
            if np.all(x == np.arange(x.size) + 1):  # survey
                data.create_variable("index", values=x[None, :])
                data.transform("wm", "index")
            else:  # x
                data.create_variable("x", values=x[None, :], units="um")
                data.transform("wm", "x")
        elif extra_dims == 2:  # spectrum vs x vs y
            # fold to 3D
            x = sorted(
                set(arr[:, 0]), reverse=arr[0, 0] > arr[-1, 0]
            )  # 0th column is stepped always (?)
            x = np.array(list(x))
            x = x.reshape(1, -1, 1)
            y = arr[:, 1].reshape(1, x.size, -1)
            ypts = y.mean(axis=1).reshape(1, 1, -1)
            sig = arr[:, 2:].T.reshape(wm.size, x.size, -1)  # TODO: test fold
            data.create_variable("wm", values=wm[:, None, None], units=spectral_units)
            data.create_variable("x", values=x, units="um")
            data.create_variable("y", values=y, units="um")
            data.create_variable("y_points", values=ypts, units="um")
            data.create_channel("signal", values=sig / acquisition_time, units=channel_units)
            data.transform("wm", "x", "y_points")

    if verbose:
        data.print_tree()
        print("  kind: {0}".format(data.kind))

    return data
