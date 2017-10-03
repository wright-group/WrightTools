"""COLORS."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import numpy as np

from scipy.interpolate import griddata

from ._data import Axis, Channel, Data
from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_COLORS']


# --- helpers -------------------------------------------------------------------------------------


def discover_dimensions(arr, dimension_cols, verbose=True):
    """Discover the dimensions of array arr.

    Watches the indicies contained in dimension_cols. Returns dictionaries of
    axis objects [scanned, constant].
    Constant objects have their points object initialized. Scanned dictionary is
    in order of scanning (..., zi, yi, xi). Both dictionaries are condensed
    into coscanning / setting.
    """
    input_cols = dimension_cols
    # import values -------------------------------------------------------------------------------
    dc = dimension_cols
    di = [dc[key].file_idx for key in dc.keys()]
    dt = [dc[key].tolerance for key in dc.keys()]
    du = [dc[key].units for key in dc.keys()]
    dk = [key for key in dc.keys()]
    dims = list(zip(di, dt, du, dk))
    # remove nan dimensions and bad dimensions ----------------------------------------------------
    to_pop = []
    for i in range(len(dims)):
        if np.all(np.isnan(arr[dims[i][0]])):
            to_pop.append(i)

    to_pop.reverse()
    for i in to_pop:
        dims.pop(i)
    # which dimensions are equal ------------------------------------------------------------------
    # find
    d_equal = np.zeros((len(dims), len(dims)), dtype=bool)
    d_equal[:, :] = True
    for i in range(len(dims)):  # test
        for j in range(len(dims)):  # against
            for k in range(len(arr[0])):
                upper_bound = arr[dims[i][0], k] + dims[i][1]
                lower_bound = arr[dims[i][0], k] - dims[i][1]
                test_point = arr[dims[j][0], k]
                if upper_bound > test_point > lower_bound:
                    pass
                else:
                    d_equal[i, j] = False
                    break
    # condense
    dims_unaccounted = list(range(len(dims)))
    dims_condensed = []
    while dims_unaccounted:
        dim_current = dims_unaccounted[0]
        index = dims[dim_current][0]
        tolerance = [dims[dim_current][1]]
        units = dims[dim_current][2]
        key = [dims[dim_current][3]]
        dims_unaccounted.pop(0)
        indicies = list(range(len(dims_unaccounted)))
        indicies.reverse()
        for i in indicies:
            dim_check = dims_unaccounted[i]
            if d_equal[dim_check, dim_current]:
                tolerance.append(dims[dim_check][1])
                key.append(dims[dim_check][3])
                dims_unaccounted.pop(i)
        tolerance = max(tolerance)
        dims_condensed.append([index, tolerance, units, key])
    dims = dims_condensed
    # which dimensions are scanned ----------------------------------------------------------------
    # find
    scanned = []
    constant_list = []
    for dim in dims:
        name = dim[3]
        index = dim[0]
        vals = arr[index]
        tolerance = dim[1]
        if vals.max() - vals.min() > tolerance:
            scanned.append([name, index, tolerance, None])
        else:
            constant_list.append([name, index, tolerance, arr[index, 0]])
    # order scanned dimensions (..., zi, yi, xi)
    first_change_indicies = []
    for axis in scanned:
        first_point = arr[axis[1], 0]
        for i in range(len(arr[0])):
            upper_bound = arr[axis[1], i] + axis[2]
            lower_bound = arr[axis[1], i] - axis[2]
            if upper_bound > first_point > lower_bound:
                pass
            else:
                first_change_indicies.append(i)
                break
    scanned_ordered = [scanned[i] for i in np.argsort(first_change_indicies)]
    scanned_ordered.reverse()
    # return --------------------------------------------------------------------------------------
    # package back into ordered dictionary of objects
    scanned = collections.OrderedDict()
    for axis in scanned_ordered:
        key = axis[0][0]
        obj = input_cols[key]
        obj.label_seed = [input_cols[_key].label_seed[0] for _key in axis[0]]
        scanned[key] = obj
    constant = collections.OrderedDict()
    for axis in constant_list:
        key = axis[0][0]
        obj = input_cols[key]
        obj.label_seed = [input_cols[_key].label_seed[0] for _key in axis[0]]
        obj.points = axis[3]
        constant[key] = obj
    return list(scanned.values()), list(constant.values())


# --- from function -------------------------------------------------------------------------------


def from_COLORS(
        filepaths,
        null=None,
        name=None,
        cols=None,
        invert_d1=True,
        color_steps_as='energy',
        ignore=[
            'num',
            'w3',
            'wa',
            'dref',
            'm0',
            'm1',
            'm2',
            'm3',
            'm4',
            'm5',
            'm6'],
        even=True,
        verbose=True):
    """Read a Data object from a COLORS filetype.

    color_steps_as one in 'energy', 'wavelength'
    """
    # do we have a list of files or just one file? ------------------------------------------------
    if isinstance(filepaths, list):
        file_example = filepaths[0]
    else:
        file_example = filepaths
        filepaths = [filepaths]
    # define format of dat file -------------------------------------------------------------------
    if cols:
        pass
    else:
        num_cols = len(np.genfromtxt(file_example).T)
        if num_cols in [28, 35]:
            cols = 'v2'
        elif num_cols in [20]:
            cols = 'v1'
        elif num_cols in [15, 16, 19]:
            cols = 'v0'
        if verbose:
            print('cols recognized as', cols, '(%d)' % num_cols)
    if cols == 'v2':
        axes = collections.OrderedDict()
        axes['num'] = Axis(None, None, tolerance=0.5, file_idx=0,
                           name='num', label_seed=['num'])
        axes['w1'] = Axis(None, 'nm', tolerance=0.5, file_idx=1, name='w1', label_seed=['1'])
        axes['w2'] = Axis(None, 'nm', tolerance=0.5, file_idx=3, name='w2', label_seed=['2'])
        axes['w3'] = Axis(None, 'nm', tolerance=5.0, file_idx=5, name='w3', label_seed=['3'])
        axes['wm'] = Axis(None, 'nm', tolerance=1.0, file_idx=7, name='wm', label_seed=['m'])
        axes['wa'] = Axis(None, 'nm', tolerance=1.0, file_idx=8, name='wm', label_seed=['a'])
        axes['dref'] = Axis(None, 'fs', tolerance=25.0, file_idx=10,
                            name='dref', label_seed=['ref'])
        axes['d1'] = Axis(None, 'fs', tolerance=4.0, file_idx=12,
                          name='d1', label_seed=['22\''])
        axes['d2'] = Axis(None, 'fs', tolerance=4.0, file_idx=14, name='d2', label_seed=['21'])
        axes['m0'] = Axis(None, None, tolerance=10.0, file_idx=22, name='m0', label_seed=['0'])
        axes['m1'] = Axis(None, None, tolerance=10.0, file_idx=23, name='m1', label_seed=['1'])
        axes['m2'] = Axis(None, None, tolerance=10.0, file_idx=24, name='m2', label_seed=['2'])
        axes['m3'] = Axis(None, None, tolerance=10.0, file_idx=25, name='m3', label_seed=['3'])
        axes['m4'] = Axis(None, None, tolerance=15.0, file_idx=26, name='m4', label_seed=['4'])
        axes['m5'] = Axis(None, None, tolerance=15.0, file_idx=27, name='m5', label_seed=['5'])
        axes['m6'] = Axis(None, None, tolerance=15.0, file_idx=28, name='m6', label_seed=['6'])
        channels = collections.OrderedDict()
        channels['ai0'] = Channel(None, file_idx=16, name='ai0', label_seed=['0'])
        channels['ai1'] = Channel(None, file_idx=17, name='ai1', label_seed=['1'])
        channels['ai2'] = Channel(None, file_idx=18, name='ai2', label_seed=['2'])
        channels['ai3'] = Channel(None, file_idx=19, name='ai3', label_seed=['3'])
        channels['ai4'] = Channel(None, file_idx=20, name='ai4', label_seed=['4'])
        channels['mc'] = Channel(None, file_idx=21, name='array', label_seed=['a'])
    elif cols == 'v1':
        axes = collections.OrderedDict()
        axes['num'] = Axis(None, None, tolerance=0.5, file_idx=0,
                           name='num', label_seed=['num'])
        axes['w1'] = Axis(None, 'nm', tolerance=0.5, file_idx=1, name='w1', label_seed=['1'])
        axes['w2'] = Axis(None, 'nm', tolerance=0.5, file_idx=3, name='w2', label_seed=['2'])
        axes['wm'] = Axis(None, 'nm', tolerance=0.5, file_idx=5, name='wm', label_seed=['m'])
        axes['d1'] = Axis(None, 'fs', tolerance=3.0, file_idx=6, name='d1', label_seed=['1'])
        axes['d2'] = Axis(None, 'fs', tolerance=3.0, file_idx=7, name='d2', label_seed=['2'])
        channels = collections.OrderedDict()
        channels['ai0'] = Channel(None, file_idx=8, name='ai0', label_seed=['0'])
        channels['ai1'] = Channel(None, file_idx=9, name='ai1', label_seed=['1'])
        channels['ai2'] = Channel(None, file_idx=10, name='ai2', label_seed=['2'])
        channels['ai3'] = Channel(None, file_idx=11, name='ai3', label_seed=['3'])
    elif cols == 'v0':
        axes = collections.OrderedDict()
        axes['num'] = Axis(None, None, tolerance=0.5, file_idx=0,
                           name='num', label_seed=['num'])
        axes['w1'] = Axis(None, 'nm', tolerance=0.5, file_idx=1, name='w1', label_seed=['1'])
        axes['w2'] = Axis(None, 'nm', tolerance=0.5, file_idx=3, name='w2', label_seed=['2'])
        axes['wm'] = Axis(None, 'nm', tolerance=0.5, file_idx=5, name='wm', label_seed=['m'])
        axes['d1'] = Axis(None, 'fs', tolerance=3.0, file_idx=6, name='d1', label_seed=['1'])
        axes['d2'] = Axis(None, 'fs', tolerance=3.0, file_idx=8, name='d2', label_seed=['2'])
        channels = collections.OrderedDict()
        channels['ai0'] = Channel(None, file_idx=10, name='ai0', label_seed=['0'])
        channels['ai1'] = Channel(None, file_idx=11, name='ai1', label_seed=['1'])
        channels['ai2'] = Channel(None, file_idx=12, name='ai2', label_seed=['2'])
        channels['ai3'] = Channel(None, file_idx=13, name='ai3', label_seed=['3'])
    # import full array ---------------------------------------------------------------------------
    for i in range(len(filepaths)):
        dat = np.genfromtxt(filepaths[i]).T
        if verbose:
            print('dat imported:', dat.shape)
        if i == 0:
            arr = dat
        else:
            arr = np.append(arr, dat, axis=1)
    if invert_d1:
        idx = axes['d1'].file_idx
        arr[idx] = -arr[idx]
    # recognize dimensionality of data ------------------------------------------------------------
    axes_discover = axes.copy()
    for key in ignore:
        if key in axes_discover:
            axes_discover.pop(key)  # remove dimensions that mess up discovery

    scanned, constant = discover_dimensions(arr, axes_discover)
    # get axes points -----------------------------------------------------------------------------
    for axis in scanned:
        # generate lists from data
        lis = sorted(arr[axis.file_idx])
        tol = axis.tolerance
        # values are binned according to their averages now, so min and max
        #  are better represented
        xstd = []
        xs = []
        # check to see if unique values are sufficiently unique
        # deplete to list of values by finding points that are within
        #  tolerance
        while len(lis) > 0:
            # find all the xi's that are like this one and group them
            # after grouping, remove from the list
            set_val = lis[0]
            xi_lis = [xi for xi in lis if np.abs(set_val - xi) < tol]
            # the complement of xi_lis is what remains of xlis, then
            lis = [xi for xi in lis if not np.abs(xi_lis[0] - xi) < tol]
            xi_lis_average = sum(xi_lis) / len(xi_lis)
            xs.append(xi_lis_average)
            xstdi = sum(np.abs(xi_lis - xi_lis_average)) / len(xi_lis)
            xstd.append(xstdi)
        # create uniformly spaced x and y lists for gridding
        # infinitesimal offset used to properly interpolate on bounds; can
        #   be a problem, especially for stepping axis
        tol = sum(xstd) / len(xstd)
        tol = max(tol, 0.3)
        if even:
            if axis.units_kind == 'energy' and color_steps_as == 'energy':
                min_wn = 1e7 / max(xs) + tol
                max_wn = 1e7 / min(xs) - tol
                axis.units = 'wn'
                axis.points = np.linspace(min_wn, max_wn, num=len(xs))
                axis.convert('nm')
            else:
                axis.points = np.linspace(min(xs) + tol, max(xs) - tol, num=len(xs))
        else:
            axis.points = np.array(xs)
    # grid data -----------------------------------------------------------------------------------
    if len(scanned) == 1:
        # 1D data
        axis = scanned[0]
        axis.points = arr[axis.file_idx]
        scanned[0] = axis
        for key in channels.keys():
            channel = channels[key]
            zi = arr[channel.file_idx]
            channel.give_values(zi)
    else:
        # all other dimensionalities
        points = tuple(arr[axis.file_idx] for axis in scanned)
        # beware, meshgrid gives wrong answer with default indexing
        # this took me many hours to figure out... - blaise
        xi = tuple(np.meshgrid(*[axis.points for axis in scanned], indexing='ij'))
        for key in channels.keys():
            channel = channels[key]
            zi = arr[channel.file_idx]
            fill_value = min(zi)
            grid_i = griddata(points, zi, xi,
                              method='linear', fill_value=fill_value)
            channel.give_values(grid_i)
    # create data object --------------------------------------------------------------------------
    data = Data(list(scanned), list(channels.values()), list(constant))
    if color_steps_as == 'energy':
        try:
            data.convert('wn', verbose=False)
        except BaseException:
            pass
    for axis in data.axes:
        axis.get_label()
    for axis in data.constants:
        axis.get_label()
    # add extra stuff to data object --------------------------------------------------------------
    data.source = filepaths
    if not name:
        name = wt_kit.filename_parse(file_example)[1]
    data.name = name
    # return --------------------------------------------------------------------------------------
    if verbose:
        print('data object succesfully created')
        print('axis names:', data.axis_names)
        print('values shape:', data.channels[0].values.shape)
    return data
