"""Andor."""


# --- import --------------------------------------------------------------------------------------


import os
import pathlib
import time
import warnings

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
    filepath : path-like
        Path to file (should be .asc format).
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
    data : WrightTools.Data
        New data object.
        Channels: `signal`.  If exposure time is in metadata, signal is given as a count rate (Hz).
        Variables, Axes: `yindex` and `xindex` (no grating) or `wm` (grating)

    Notes
    -----
    When exporting as ascii, including metadata is optional.
    It is _strongly recommended_ that you include metadata in exports.
    Metadata informs the image creation date, exposure time, and axes.
    However, if metadata is not present, this importer will make its best guesses to populate these fields accurately.

    Saving processed data (e.g. vertically-binned data) in Solis software can remove/omit important metadata, so we advise exporting the raw camera images.

    """
    # parse filepath
    filestr = os.fspath(filepath)
    filepath = pathlib.Path(filepath)

    if not ".asc" in filepath.suffixes:
        wt_exceptions.WrongFileTypeWarning.warn(filepath, ".asc")
    # parse name
    if not name:
        name = filepath.name.split(".")[0]
    # create data
    ds = np.DataSource(None)
    f = ds.open(filestr, "rt")
    axis0 = []
    arr = []
    attrs = {}

    line0 = f.readline().strip()[:-1]
    line0 = [float(x) for x in line0.split(",")]  # TODO: robust to space, tab, comma
    axis0.append(line0.pop(0))
    arr.append(line0)

    def get_frames(f, arr, axis0):
        axis0_written = False
        while True:
            line = f.readline().strip()[:-1]
            if len(line) == 0:
                break
            else:
                line = [float(x) for x in line.split(",")]
                # signature of new frames is restart of axis0
                if not axis0_written and (line[0] == axis0[0]):
                    axis0_written = True
                if axis0_written:
                    line.pop(0)
                else:
                    axis0.append(line.pop(0))
                arr.append(line)
        return arr, axis0

    arr, axis0 = get_frames(f, arr, axis0)
    nframes = len(arr) // len(axis0)

    i = 0
    while i < 3:
        line = f.readline().strip()
        if len(line) == 0:
            i += 1
        else:
            try:
                key, val = line.split(":", 1)
            except ValueError:
                pass
            else:
                attrs[key.strip()] = val.strip()

    f.close()

    try:
        created = attrs["Date and Time"]  # is this UTC?
        created = time.strptime(created, "%a %b %d %H:%M:%S %Y")
        created = timestamp.TimeStamp(time.mktime(created)).RFC3339
    except KeyError:  # use file creation time
        created = os.stat(filepath).st_mtime
        created = timestamp.TimeStamp(created).RFC3339
        warnings.warn(
            f"{filepath.name} has no 'Date and Time' field: using file modified time instead: {created}"
        )

    kwargs = {"name": name, "kind": "Solis", "source": filestr, "created": created}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)

    axis0 = np.array(axis0)
    try:
        groove_density = float(attrs["Grating Groove Density (l/mm)"])
    except KeyError:  # assume no grating
        warnings.warn(
            f"{filepath.name} has no 'Grating Groove Density (1/mm)' field: guessing x axis units."
        )
        groove_density = isinstance(axis0[0], float)

    if groove_density == 0:
        xname = "xindex"
        xunits = None
    else:
        xname = "wm"
        xunits = "nm"
    axes = [xname, "yindex"]

    if nframes == 1:
        arr = np.array(arr)
        data.create_variable(name=xname, values=axis0[:, None], units=xunits)
        data.create_variable(name="yindex", values=np.arange(arr.shape[-1])[None, :], units=None)
    else:
        arr = np.array(arr).reshape(nframes, len(axis0), len(arr[0]))
        data.create_variable(name="frame", values=np.arange(nframes)[:, None, None], units=None)
        data.create_variable(name=xname, values=axis0[None, :, None], units=xunits)
        data.create_variable(
            name="yindex", values=np.arange(arr.shape[-1])[None, None, :], units=None
        )
        axes = ["frame"] + axes

    data.transform(*axes)
    try:
        exposure_time = float(attrs["Exposure Time (secs)"])
        if exposure_time == 0:
            raise ZeroDivisionError
        arr /= exposure_time
    except (KeyError, ZeroDivisionError) as e:  # do not normalize
        warnings.warn(f"{filepath.name} camera signal cannot be given as a count rate.")
        data.create_channel(name="signal", values=arr, signed=False)
    else:
        # signal has units of Hz because time normalized
        data.create_channel(name="signal", values=arr, signed=False, units="Hz")

    for key, val in attrs.items():
        data.attrs[key] = val

    # finish
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  axes: {0}".format(data.axis_names))
        print("  shape: {0}".format(data.shape))
    return data
