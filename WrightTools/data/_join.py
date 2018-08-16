"""Join multiple data objects together."""


# --- import --------------------------------------------------------------------------------------


import collections
import warnings

import numpy as np

from .. import kit as wt_kit
from .. import exceptions as wt_exceptions
from ._data import Data


# --- define --------------------------------------------------------------------------------------


__all__ = ["join"]


# --- functions -----------------------------------------------------------------------------------


def join(datas, *, name="join", parent=None, verbose=True):
    """Join a list of data objects together.

    For now datas must have identical dimensionalities (order and identity).
    Currently only supports 'last' method of handling overlapping data objects.
    Currently limited to joining with datas that have grid-aligned, orthogonal axes.

    Parameters
    ----------
    datas : list of data
        The list of data objects to join together.
    name : str (optional)
        The name for the data object which is created. Default is 'join'.
    parent : WrightTools.Collection (optional)
        The location to place the joined data object. Default is new temp file at root.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    WrightTools.data.Data
        A new Data instance.
    """
    warnings.warn("join", category=wt_exceptions.EntireDatasetInMemoryWarning)
    # TODO: fill value
    datas = list(datas)
    # check if variables are valid
    axis_expressions = datas[0].axis_expressions
    variable_names = set(datas[0].variable_names)
    channel_names = set(datas[0].channel_names)
    for d in datas:
        variable_names &= set(d.variable_names)
        channel_names &= set(d.channel_names)
    variable_names = list(variable_names)
    channel_names = list(channel_names)
    variable_units = []
    channel_units = []
    for v in variable_names:
        variable_units.append(datas[0][v].units)
    for c in channel_names:
        channel_units.append(datas[0][c].units)
    # axis variables
    axis_variable_names = []
    axis_variable_units = []
    for a in datas[0].axes:
        for v in a.variables:
            axis_variable_names.append(v.natural_name)
            axis_variable_units.append(v.units)

    vs = collections.OrderedDict()
    for n, units in zip(axis_variable_names, axis_variable_units):
        values = np.concatenate([d[n][:].flat for d in datas])
        rounded = values.round(8)
        _, idxs = np.unique(rounded, True)
        values = values.flat[idxs]
        vs[n] = {"values": values, "units": units}
    # TODO: the following should become a new from method

    def from_dict(d, parent=None):
        ndim = len(d)
        i = 0
        out = Data(name=name, parent=parent)
        for k, v in d.items():
            values = v["values"]
            shape = [1] * ndim
            shape[i] = values.size
            values.shape = tuple(shape)
            # **attrs passes the name and units as well
            out.create_variable(values=values, **datas[0][k].attrs)
            i += 1
        return out

    out = from_dict(vs, parent=parent)
    for channel_name, units in zip(channel_names, channel_units):
        # **attrs passes the name and units as well
        out.create_channel(**datas[0][channel_name].attrs)
    for variable_name in variable_names:
        if variable_name not in vs.keys():
            shape = tuple(
                1 if i == 1 else n for i, n in zip(datas[0][variable_name].shape, out.shape)
            )
            # **attrs passes the name and units as well
            out.create_variable(shape=shape, **datas[0][variable_name].attrs)
    # channels
    for data in datas:
        new_idx = []
        for variable_name in vs.keys():
            p = data[variable_name][:][np.newaxis, ...]
            arr = out[variable_name][:][..., np.newaxis]
            i = np.argmin(np.abs(arr - p), axis=np.argmax(arr.shape))
            new_idx.append(i)
        for variable_name in out.variable_names:
            if variable_name not in vs.keys():
                old = data[variable_name]
                new = out[variable_name]
                # These lines are needed because h5py doesn't support advanced indexing natively
                vals = new[:]
                vals[wt_kit.valid_index(new_idx, new.shape)] = old[:]
                new[:] = vals
        for channel_name in channel_names:
            old = data[channel_name]
            new = out[channel_name]
            # These lines are needed because h5py doesn't support advanced indexing natively
            vals = new[:]
            vals[wt_kit.valid_index(new_idx, new.shape)] = old[:]
            new[:] = vals
    # axes
    out.transform(*axis_expressions)
    # finish
    if verbose:
        print(len(datas), "datas joined to create new data:")
        print("  axes:")
        for axis in out.axes:
            points = axis[:]
            print(
                "    {0} : {1} points from {2} to {3} {4}".format(
                    axis.expression, points.size, np.min(points), np.max(points), axis.units
                )
            )
        print("  channels:")
        for channel in out.channels:
            percent_nan = np.around(
                100. * (np.isnan(channel[:]).sum() / float(channel.size)), decimals=2
            )
            print(
                "    {0} : {1} to {2} ({3}% NaN)".format(
                    channel.name, channel.min(), channel.max(), percent_nan
                )
            )
    return out
