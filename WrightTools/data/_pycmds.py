"""PyCMDS."""


# --- import --------------------------------------------------------------------------------------


import itertools
import os
import pathlib

import h5py
import numpy as np

import tidy_headers

from ._data import Data
from .. import kit as wt_kit
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_PyCMDS"]


# --- from function -------------------------------------------------------------------------------


def from_PyCMDS(filepath, name=None, parent=None, verbose=True, *, collapse=True) -> Data:
    """Create a data object from a single PyCMDS output file.

    Parameters
    ----------
    filepath : path-like
        Path to the .data file
        Can be either a local or remote file (http/ftp).
        Can be compressed with gz/bz2, decompression based on file name.
    name : str or None (optional)
        The name to be applied to the new data object. If None, name is read
        from file.
    parent : WrightTools.Collection (optional)
        Collection to place new data object within. Default is None.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        A Data instance.
    """
    filestr = os.fspath(filepath)
    filepath = pathlib.Path(filepath)

    # header
    ds = np.DataSource(None)
    file_ = ds.open(filestr, "rt")
    headers = tidy_headers.read(file_)
    file_.seek(0)
    # name
    if name is None:  # name not given in method arguments
        data_name = headers["data name"]
    else:
        data_name = name
    if data_name == "":  # name not given in PyCMDS
        data_name = headers["data origin"]
    # create data object
    kwargs = {
        "name": data_name,
        "kind": "PyCMDS",
        "source": filestr,
        "created": headers["file created"],
    }
    if parent is not None:
        data = parent.create_data(**kwargs)
    else:
        data = Data(**kwargs)
    if collapse:
        # array
        arr = np.genfromtxt(file_).T
    # get axes and scanned variables
    axes = []
    for name, identity, units in zip(
        headers["axis names"], headers["axis identities"], headers["axis units"]
    ):
        # points and centers
        points = np.array(headers[name + " points"])
        if name + " centers" in headers.keys():
            centers = headers[name + " centers"]
        else:
            centers = None
        # create
        axis = {
            "points": points,
            "units": units,
            "name": name,
            "identity": identity,
            "centers": centers,
        }
        axes.append(axis)
    shape = tuple([a["points"].size for a in axes])
    for i, ax in enumerate(axes):
        sh = [1] * len(shape)
        sh[i] = len(ax["points"])
        data.create_variable(
            name=ax["name"] + "_points", values=np.array(ax["points"]).reshape(sh)
        )
        if ax["centers"] is not None:
            centers = np.array(ax["centers"])
            sh = list(shape)
            sh[i] = 1
            for j, s in enumerate(sh):
                if centers.size % s:
                    sh[j] = 1
            data.create_variable(
                name=ax["name"] + "_centers", values=np.array(centers.reshape(sh))
            )
    # get assorted remaining things
    # variables and channels
    try:
        signed = iter(headers["channel signed"])
    except KeyError:
        signed = itertools.repeat(False)
    for index, (kind, name) in enumerate(zip(headers["kind"], headers["name"])):
        if collapse:
            _collapse_read_in(data, headers, axes, arr, signed, index, kind, name, shape)
        else:
            _no_collapse_create(data, headers, signed, index, kind, name, shape)
    if not collapse:
        _no_collapse_fill(data, headers, file_, shape, verbose)
    file_.close()
    # axes
    for a in axes:
        expression = a["identity"]
        if expression.startswith("D"):
            expression = expression[1:]
        expression.replace("=D", "=")
        a["expression"] = expression
    data.transform(*[a["expression"] for a in axes])
    for a, u in zip(data.axes, headers["axis units"]):
        if u is not None:
            a.convert(u)
    if (
        headers["system name"] == "fs"
        and int(headers["PyCMDS version"].split(".")[0]) == 0
        and int(headers["PyCMDS version"].split(".")[1]) < 10
    ):
        # in versions of PyCMDS up to (and including) 0.9.0
        # there was an incorrect hard-coded conversion factor between mm and fs
        # this ONLY applied to Newport MFA stages
        # we apply this correction knowing that Newport MFAs were only used on the "fs" system
        # and knowing that the Newport MFAs were always assigned as "d1", "d2" and "d3"
        # ---Blaise 2019-04-09
        for delay in ("d1", "d2", "d3", "d1_points", "d2_points", "d3_points"):
            if delay not in data.variable_names:
                continue
            data[delay][:] *= 6000.671281903963041 / 6671.281903963041
            if verbose:
                print(f"Correction factor applied to {delay}")
    # return
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  axes: {0}".format(data.axis_names))
        print("  shape: {0}".format(data.shape))
    return data


def _collapse_read_in(data, headers, axes, arr, signed, index, kind, name, shape):
    values = np.full(np.prod(shape), np.nan)
    values[: len(arr[index])] = arr[index]
    values.shape = shape
    if name == "time":
        for i in range(len(shape)):
            tolerance = 1e-6
            mean = np.nanmean(values, axis=i)
            mean = np.expand_dims(mean, i)
            values, meanexp = wt_kit.share_nans(values, mean)
            if np.allclose(meanexp, values, atol=tolerance, rtol=0, equal_nan=True):
                values = mean
        data.create_variable(name="labtime", values=values)
    if kind == "hardware":
        # sadly, recorded tolerances are not reliable
        # so a bit of hard-coded hacking is needed
        # if this ends up being too fragile, we might have to use the points arrays
        # ---Blaise 2018-01-09
        units = headers["units"][index]
        label = headers["label"][index]
        if (
            "w" in name
            and name.startswith(tuple(data.variable_names))
            and name not in headers["axis names"]
        ):
            inherited_shape = data[name.split("_")[0]].shape
            for i, s in enumerate(inherited_shape):
                if s == 1:
                    values = np.mean(values, axis=i)
                    values = np.expand_dims(values, i)
        else:
            tolerance = headers["tolerance"][index]
            units = headers["units"][index]
            for i in range(len(shape)):
                if tolerance is None:
                    break
                if "d" in name:
                    # This is a hack because delay is particularly
                    # unreliable in tolerance. And 3 fs vs 3 ps is a huge
                    # difference... KFS 2019-2-27
                    if units == "fs":
                        tolerance = 3.0
                    else:
                        tolerance = 0.1
                if "zero" in name:
                    tolerance = 1e-10
                if name in headers["axis names"]:
                    if (
                        i == headers["axis names"].index(name)
                        or f"{name}_centers" in data.variable_names
                    ):
                        tolerance = 1e-10
                    else:
                        tolerance = np.inf
                mean = np.nanmean(values, axis=i)
                mean = np.expand_dims(mean, i)
                values, meanexp = wt_kit.share_nans(values, mean)
                if np.allclose(meanexp, values, atol=tolerance, rtol=0, equal_nan=True):
                    values = mean
        if name in headers["axis names"]:
            points = np.array(headers[name + " points"])
            pointsshape = [1] * values.ndim
            for i, ax in enumerate(axes):
                if ax["name"] == name:
                    pointsshape[i] = len(points)
                    break
            points.shape = pointsshape
            points = wt_units.converter(points, headers["axis units"][i], units)
            for i in range(points.ndim):
                if points.shape[i] == 1:
                    points = np.repeat(points, values.shape[i], axis=i)
            if points.size <= values.size:
                values[np.isnan(values)] = points[np.isnan(values)]
        data.create_variable(name, values=values, units=units, label=label)
    if kind == "channel":
        data.create_channel(name=name, values=values, shape=values.shape, signed=next(signed))


def _no_collapse_create(data, headers, signed, index, kind, name, shape):
    sh = shape
    if "wa" in headers["name"] and name not in ("wa", "array", "array_signal"):
        sh = list(sh)
        sh[-1] = 1
        sh = tuple(sh)
    if name == "time":
        data.create_variable(name="labtime", dtype=np.dtype(np.float64), shape=sh)
    if kind == "hardware":
        units = headers["units"][index]
        label = headers["label"][index]
        if (
            "w" in name
            and name.startswith(tuple(data.variable_names))
            and name not in headers["axis names"]
        ):
            inherited_shape = data[name.split("_")[0]].shape
        else:
            units = headers["units"][index]
        data.create_variable(name, shape=sh, dtype=np.dtype(np.float64), units=units, label=label)
    if kind == "channel":
        data.create_channel(name=name, shape=sh, dtype=np.dtype(np.float64), signed=next(signed))


def _no_collapse_fill(data, headers, file_, shape, verbose):
    frame_size = shape[-1]
    file_.seek(0)
    arr = np.genfromtxt(file_, max_rows=frame_size)
    while arr.size > 0:
        index = tuple(arr[0, 0 : len(shape) - 1].astype(np.int))
        if verbose:
            print(index)
        for i, (kind, name) in enumerate(zip(headers["kind"], headers["name"])):
            if kind is None and name != "time":
                continue
            if name == "time":
                name = "labtime"
            if "wa" not in headers["name"] or name in ("wa", "array", "array_signal"):
                h5py.Group.__getitem__(data, name)[index + (...,)] = arr[:, i]
            else:
                h5py.Group.__getitem__(data, name)[index + (...,)] = arr[0, i]
        arr = np.genfromtxt(file_, max_rows=frame_size)
