"""Join multiple data objects together."""


# --- import --------------------------------------------------------------------------------------


import collections
import warnings

import numpy as np

from .. import kit as wt_kit
from .. import exceptions as wt_exceptions
from ._data import Data
from ..collection import Collection


# --- define --------------------------------------------------------------------------------------


__all__ = ["join"]


# --- functions -----------------------------------------------------------------------------------


def join(
    datas, *, atol=None, rtol=None, name="join", parent=None, method="first", verbose=True
) -> Data:
    """Join a list of data objects together.

    Joined datas must have the same transformation applied.
    This transformation will define the array order for the joined dataset.
    All axes in the applied transformation must be a single variable,
    the result will have sorted numbers.

    Join does not perform any interpolation.
    For that, look to ``Data.map_variable`` or ``Data.heal``

    Parameters
    ----------
    datas : list of data or WrightTools.Collection
        The list or collection of data objects to join together.
    atol : numeric or list of numeric
        The absolute tolerance to use (in ``np.isclose``) to consider points overlapped.
        If given as a single number, applies to all axes.
        If given as a list, must have same length as the data transformation.
        ``None`` in the list invokes default behavior.
        Default is 10% of the minimum spacing between consecutive points in any
        input data file.
    rtol : numeric or list of numeric
        The relative tolerance to use (in ``np.isclose``) to consider points overlapped.
        If given as a single number, applies to all axes.
        If given as a list, must have same length as the data transformation.
        ``None`` in the list invokes default behavior.
        Default is ``4 * np.finfo(dtype).resolution`` for floating point types,
        ``0`` for integer types.
    name : str (optional)
        The name for the data object which is created. Default is 'join'.
    parent : WrightTools.Collection (optional)
        The location to place the joined data object. Default is new temp file at root.
    method : {'first', 'last', 'min', 'max', 'sum', 'mean'}
        Mode to use for merged points in the joined space.
        Default is 'first'.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    WrightTools.data.Data
        A new Data instance.
    """
    warnings.warn("join", category=wt_exceptions.EntireDatasetInMemoryWarning)
    if isinstance(datas, Collection):
        datas = datas.values()
    datas = list(datas)
    if not isinstance(atol, collections.abc.Iterable):
        atol = [atol] * len(datas[0].axes)
    if not isinstance(rtol, collections.abc.Iterable):
        rtol = [rtol] * len(datas[0].axes)
    # check if variables are valid
    axis_expressions = datas[0].axis_expressions
    variable_names = set(datas[0].variable_names)
    channel_names = set(datas[0].channel_names)
    for d in datas[1:]:
        if d.axis_expressions != axis_expressions:
            raise wt_exceptions.ValueError("Joined data must have same axis_expressions")
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
    axis_variable_names = []
    axis_variable_units = []
    for a in datas[0].axes:
        if len(a.variables) > 1:
            raise wt_exceptions.ValueError("Applied transform must have single variable axes")
        for v in a.variables:
            axis_variable_names.append(v.natural_name)
            axis_variable_units.append(v.units)

    vs = collections.OrderedDict()
    for n, units, atol_, rtol_ in zip(axis_variable_names, axis_variable_units, atol, rtol):
        dtype = np.result_type(*[d[n].dtype for d in datas])
        if atol_ is None:
            try:
                # 10% of the minimum spacing between consecutive points in any singular input data
                atol_ = min(np.min(np.abs(np.diff(d[n][:]))) for d in datas if d[n].size > 1) * 0.1
            except ValueError:
                atol_ = 0
        if rtol_ is None:
            # Ignore floating point precision rounding, if dtype is floting
            rtol_ = 4 * np.finfo(dtype).resolution if dtype.kind in "fcmM" else 0
        values = np.concatenate([d[n][:].flat for d in datas])
        values = np.sort(values)
        filtered = []
        i = 0
        # Filter out consecutive values that are "close"
        while i < len(values):
            sum_ = values[i]
            count = 1
            i += 1
            if i < len(values):
                while np.isclose(values[i - 1], values[i], atol=atol_, rtol=rtol_):
                    sum_ += values[i]
                    count += 1
                    i += 1
                    if i >= len(values):
                        break
            filtered.append(sum_ / count)
        vs[n] = {"values": np.array(filtered), "units": units}
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
        out.transform(*list(d.keys()))
        return out

    def get_shape(out, datas, item_name):
        shape = [1] * out.ndim
        for i, s in enumerate(out.shape):
            idx = [np.argmax(d[out.axes[i].expression].shape) for d in datas]
            if any(d[item_name].shape[j] != 1 for d, j in zip(datas, idx)) or all(
                d[out.axes[i].expression].size == 1 for d in datas
            ):
                shape[i] = s
        return shape

    out = from_dict(vs, parent=parent)
    count = {}
    for channel_name in channel_names:
        # **attrs passes the name and units as well
        out.create_channel(
            shape=get_shape(out, datas, channel_name),
            **datas[0][channel_name].attrs,
            dtype=datas[0][channel_name].dtype
        )
        count[channel_name] = np.zeros_like(out[channel_name], dtype=int)
    for variable_name in variable_names:
        if variable_name not in vs.keys():
            # **attrs passes the name and units as well
            out.create_variable(
                shape=get_shape(out, datas, variable_name),
                **datas[0][variable_name].attrs,
                dtype=datas[0][variable_name].dtype
            )
            count[variable_name] = np.zeros_like(out[variable_name], dtype=int)

    def combine(data, out, item_name, new_idx, transpose, slice_):
        old = data[item_name]
        new = out[item_name]
        vals = np.empty_like(new)
        # Default fill value based on whether dtype is floating or not
        if vals.dtype.kind in "fcmM":
            vals[:] = np.nan
        else:
            vals[:] = 0
        # Use advanced indexing to populate vals, a temporary array with same shape as out
        valid_index = tuple(wt_kit.valid_index(new_idx, new.shape))
        vals[valid_index] = old[:].transpose(transpose)[slice_]

        # Overlap methods are accomplished by adding the existing array with the one added
        # for this particular data. Thus locations which should be set, but conflict by
        # the method chosen are set to 0. Handling for floating point vs. integer types may vary.
        # For floating types, nan indicates invalid, and must be explicitly allowed to add in.
        if method == "first":
            # Set any locations which have already been populated
            vals[~np.isnan(new[:])] = 0
            if not vals.dtype.kind in "fcmM":
                vals[count[item_name] > 0] = 0
        elif method == "last":
            # Reset points which are to be overwritten
            new[~np.isnan(vals)] = 0
            if not vals.dtype.kind in "fcmM":
                new[valid_index] = 0
        elif method == "min":
            rep_new = new > vals
            rep_vals = vals > new
            new[rep_new] = 0
            vals[rep_vals] = 0
        elif method == "max":
            rep_new = new < vals
            rep_vals = vals < new
            new[rep_new] = 0
            vals[rep_vals] = 0
        # Ensure that previously NaN points which have values are written
        new[np.isnan(new) & ~np.isnan(vals)] = 0
        # Ensure that new data does not overwrite any previous data with nan
        vals[np.isnan(vals)] = 0
        # Track how many times each point is set (for mean)
        count[item_name][valid_index] += 1
        new[:] += vals

    for data in datas:
        new_idx = []
        transpose = []
        slice_ = []
        for variable_name in vs.keys():
            # p is at most 1-D by precondition to join
            p = data[variable_name].points
            # If p not scalar, append the proper transposition to interop with out
            # And do not add new axis
            if np.ndim(p) > 0:
                transpose.append(np.argmax(data[variable_name].shape))
                slice_.append(slice(None))
            # If p is scalar, a new axis must be added, no transpose needed
            else:
                slice_.append(np.newaxis)
            # Triple subscripting needed because newaxis only applys to numpy array
            # New axis added so that subtracting p will broadcast
            arr = out[variable_name][:][..., np.newaxis]
            i = np.argmin(np.abs(arr - p), axis=np.argmax(arr.shape))
            # Reshape i, to match with the output shape
            sh = [1] * i.ndim
            sh[np.argmax(arr.shape)] = i.size
            i.shape = sh
            new_idx.append(i)
        slice_ = tuple(slice_)
        for variable_name in out.variable_names:
            if variable_name not in vs.keys():
                combine(data, out, variable_name, new_idx, transpose, slice_)
        for channel_name in channel_names:
            combine(data, out, channel_name, new_idx, transpose, slice_)

    if method == "mean":
        for name, c in count.items():
            if out[name].dtype.kind in "fcmM":
                out[name][:] /= c
            else:
                out[name][:] //= c
    out.transform(*axis_expressions)
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
