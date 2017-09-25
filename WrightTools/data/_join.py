"""Join multiple data objects together."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

from ._data import Channel, Data


# --- define --------------------------------------------------------------------------------------

__all__ = ['join']

# --- functions -----------------------------------------------------------------------------------


def join(datas, method='first', verbose=True, **kwargs):
    """Join a list of data objects together.

    For now datas must have identical dimensionalities (order and identity).

    Parameters
    ----------
    datas : list of data
        The list of data objects to join together.
    method : {'first', 'sum', 'max', 'min', 'mean'} (optional)
        The method for how overlapping points get treated. Default is first,
        meaning that the data object that appears first in datas will take
        precedence.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Keyword Arguments
    -----------------
    axis objects
        The axes of the new data object. If not supplied, the points of the
        new axis will be guessed from the given datas.

    Returns
    -------
    data
        A Data instance.
    """
    # TODO: a proper treatment of joining datas that have different dimensions
    # with intellegent treatment of their constant dimensions. perhaps changing
    # map_axis would be good for this. - Blaise 2015.10.31

    # copy datas so original objects are not changed
    datas = [d.copy() for d in datas]
    # get scanned dimensions
    axis_names = []
    axis_units = []
    axis_objects = []
    for data in datas:
        for i, axis in enumerate(data.axes):
            if axis.name in kwargs.keys():
                axis.convert(kwargs[axis.name].units)
            if axis.points[0] > axis.points[-1]:
                data.flip(i)
            if axis.name not in axis_names:
                axis_names.append(axis.name)
                axis_units.append(axis.units)
                axis_objects.append(axis)
    # TODO: transpose to same dimension orders
    # convert into same units
    for data in datas:
        for axis_name, axis_unit in zip(axis_names, axis_units):
            for axis in data.axes:
                if axis.name == axis_name:
                    axis.convert(axis_unit)
    # get axis points
    axis_points = []  # list of 1D arrays
    for axis_name in axis_names:
        if axis_name in kwargs.keys():
            axis_points.append(kwargs[axis_name].points)
            continue
        all_points = np.array([])
        step_sizes = []
        for data in datas:
            for axis in data.axes:
                if axis.name == axis_name:
                    all_points = np.concatenate([all_points, axis.points])
                    this_axis_min = np.nanmin(axis.points)
                    this_axis_max = np.nanmax(axis.points)
                    this_axis_number = float(axis.points.size) - 1
                    step_size = (this_axis_max - this_axis_min) / this_axis_number
                    step_sizes.append(step_size)
        axis_min = np.nanmin(all_points)
        axis_max = np.nanmax(all_points)
        axis_step_size = min(step_sizes)
        axis_n_points = np.ceil((axis_max - axis_min) / axis_step_size)
        points = np.linspace(axis_min, axis_max, axis_n_points + 1)
        axis_points.append(points)
    # map datas to new points
    for axis_index, axis_name in enumerate(axis_names):
        for data in datas:
            for axis in data.axes:
                if axis.name == axis_name:
                    if not np.array_equiv(axis.points, axis_points[axis_index]):
                        data.map_axis(axis_name, axis_points[axis_index])
    # make new channel objects
    channel_objects = []
    n_channels = min([len(d.channels) for d in datas])
    for channel_index in range(n_channels):
        full = np.array([d.channels[channel_index].values for d in datas])
        if method == 'first':
            zis = np.full(full.shape[1:], np.nan)
            for idx in np.ndindex(*full.shape[1:]):
                for data_index in range(len(full)):
                    value = full[data_index][idx]
                    if not np.isnan(value):
                        zis[idx] = value
                        break
        elif method == 'sum':
            zis = np.nansum(full, axis=0)
            zis[zis == 0.] = np.nan
        elif method == 'max':
            zis = np.nanmax(full, axis=0)
        elif method == 'min':
            zis = np.nanmin(full, axis=0)
        elif method == 'mean':
            zis = np.nanmean(full, axis=0)
        else:
            raise ValueError("method %s not recognized" % method)
        zis[np.isnan(full).all(axis=0)] = np.nan  # if all datas NaN, zis NaN
        channel = Channel(zis, null=0.,
                          signed=datas[0].channels[channel_index].signed,
                          name=datas[0].channels[channel_index].name)
        channel_objects.append(channel)
    # make new data object
    out = Data(axis_objects, channel_objects)
    # finish
    if verbose:
        print(len(datas), 'datas joined to create new data:')
        print('  axes:')
        for axis in out.axes:
            points = axis.points
            print('    {0} : {1} points from {2} to {3} {4}'.format(
                axis.name, points.size, min(points), max(points), axis.units))
        print('  channels:')
        for channel in out.channels:
            percent_nan = np.around(100. * (np.isnan(channel.values).sum() /
                                            float(channel.values.size)), decimals=2)
            print('    {0} : {1} to {2} ({3}% NaN)'.format(
                channel.name, channel.min(), channel.max(), percent_nan))
    return out
