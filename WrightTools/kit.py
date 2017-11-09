"""
A collection of small, general purpose objects and methods.

.. _HDF5: https://www.hdfgroup.org/HDF5/doc/H5.intro.html
.. _intersperse: http://stackoverflow.com/a/5921708
.. _suppress:
    http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import ast
import sys
import copy
import h5py
import string
import warnings
import collections
from time import clock

from scipy import ndimage

try:
    import configparser as configparser  # python 3
except ImportError:
    import ConfigParser as configparser  # python 2

import numpy as np

from . import units as wt_units


# --- file processing -----------------------------------------------------------------------------


# --- array and math ------------------------------------------------------------------------------


def closest_pair(arr, give='indicies'):
    """Find the pair of indices corresponding to the closest elements in an array.

    If multiple pairs are equally close, both pairs of indicies are returned.
    Optionally returns the closest distance itself.

    I am sure that this could be written as a cheaper operation. I
    wrote this as a quick and dirty method because I need it now to use on some
    relatively small arrays. Feel free to refactor if you need this operation
    done as fast as possible. - Blaise 2016-02-07

    Parameters
    ----------
    arr : numpy.ndarray
        The array to search.
    give : {'indicies', 'distance'} (optional)
        Toggle return behavior. If 'distance', returns a single float - the
        closest distance itself. Default is indicies.

    Returns
    -------
    list of lists of two tuples
        List containing lists of two tuples: indicies the nearest pair in the
        array.

        >>> arr = np.array([0, 1, 2, 3, 3, 4, 5, 6, 1])
        >>> closest_pair(arr)
        [[(1,), (8,)], [(3,), (4,)]]

    """
    idxs = [idx for idx in np.ndindex(arr.shape)]
    outs = []
    min_dist = arr.max() - arr.min()
    for idxa in idxs:
        for idxb in idxs:
            if idxa == idxb:
                continue
            dist = abs(arr[idxa] - arr[idxb])
            if dist == min_dist:
                if not [idxb, idxa] in outs:
                    outs.append([idxa, idxb])
            elif dist < min_dist:
                min_dist = dist
                outs = [[idxa, idxb]]
    if give == 'indicies':
        return outs
    elif give == 'distance':
        return min_dist
    else:
        raise KeyError('give not recognized in closest_pair')


def diff(xi, yi, order=1):
    """Take the numerical derivative of a 1D array.

    Output is mapped onto the original coordinates  using linear interpolation.

    Parameters
    ----------
    xi : 1D array-like
        Coordinates.
    yi : 1D array-like
        Values.
    order : positive integer (optional)
        Order of differentiation.

    Returns
    -------
    1D numpy array
        Numerical derivative. Has the same shape as the input arrays.
    """
    xi = np.array(xi).copy()
    yi = np.array(yi).copy()
    arg = np.argsort(xi)
    xi = xi[arg]
    yi = yi[arg]
    midpoints = (xi[1:] + xi[:-1]) / 2
    for _ in range(order):
        d = np.diff(yi)
        d /= np.diff(xi)
        yi = np.interp(xi, midpoints, d)
    return yi[arg]


def fft(xi, yi, axis=0):
    """Take the 1D FFT of an N-dimensional array and return "sensible" properly shifted arrays.

    Parameters
    ----------
    xi : numpy.ndarray
        1D array over which the points to be FFT'ed are defined
    yi : numpy.ndarray
        ND array with values to FFT
    axis : int
        axis of yi to perform FFT over

    Returns
    -------
    xi : 1D numpy.ndarray
        1D array. Conjugate to input xi.
        Example: if input xi is in the time domain, output xi is in frequency domain.
    yi : ND numpy.ndarray
        FFT. Has the same shape as the input array (yi).
    """
    yi = np.fft.fft(yi, axis=axis)
    d = (xi.max() - xi.min()) / (xi.size - 1)
    xi = np.fft.fftfreq(xi.size, d=d)
    # shift
    xi = np.fft.fftshift(xi)
    yi = np.fft.fftshift(yi, axes=axis)
    return xi, yi


def mono_resolution(grooves_per_mm, slit_width, focal_length, output_color, output_units='wn'):
    """Calculate the resolution of a monochromator.

    Parameters
    ----------
    grooves_per_mm : number
        Grooves per millimeter.
    slit_width : number
        Slit width in microns.
    focal_length : number
        Focal length in mm.
    output_color : number
        Output color in nm.
    output_units : string (optional)
        Output units. Default is wn.

    Returns
    -------
    float
        Resolution.
    """
    d_lambda = 1e6 * slit_width / (grooves_per_mm * focal_length)  # nm
    upper = output_color + d_lambda / 2  # nm
    lower = output_color - d_lambda / 2  # nm
    return abs(wt_units.converter(upper, 'nm', output_units) -
               wt_units.converter(lower, 'nm', output_units))


def nm_width(center, width, units='wn'):
    """Given a center and width, in energy units, get back a width in nm.

    Parameters
    ----------
    center : number
        Center (in energy units).
    width : number
        Width (in energy units).
    units : string (optional)
        Input units. Default is wn.

    Returns
    -------
    number
        Width in nm.
    """
    red = wt_units.converter(center - width / 2., units, 'nm')
    blue = wt_units.converter(center + width / 2., units, 'nm')
    return red - blue


def remove_nans_1D(arrs):
    """Remove nans in a list of 1D arrays.

    Removes indicies in all arrays if any array is nan at that index.
    All input arrays must have the same size.

    Parameters
    ----------
    arrs : list of 1D arrays
        The arrays to remove nans from

    Returns
    -------
    list
        List of 1D arrays in same order as given, with nan indicies removed.
    """
    # find all indicies to keep
    bads = np.array([])
    for arr in arrs:
        bad = np.array(np.where(np.isnan(arr))).flatten()
        bads = np.hstack((bad, bads))
    if hasattr(arrs, 'shape') and len(arrs.shape) == 1:
        goods = [i for i in np.arange(arrs.shape[0]) if i not in bads]
    else:
        goods = [i for i in np.arange(len(arrs[0])) if i not in bads]
    # apply
    return [a[goods] for a in arrs]


def share_nans(arrs1):
    """Take a list of nD arrays and return a new list of nD arrays.

    The new list is in the same order as the old list.
    If one indexed element in an old array is nan then every element for that
    index in all new arrays in the list is then nan.

    Parameters
    ----------
    arrs1 : list of nD arrays
        The arrays to syncronize nans from

    Returns
    -------
    list
        List of nD arrays in same order as given, with nan indicies syncronized.
    """
    nans = np.zeros((arrs1[0].shape))
    for arr in arrs1:
        nans *= arr
    arrs2 = [a + nans for a in arrs1]
    return arrs2


def smooth_1D(arr, n=10):
    """Smooth 1D data by 'running average'.

    Parameters
    ----------
    n : int
        number of points to average
    """
    for i in range(n, len(arr) - n):
        window = arr[i - n:i + n].copy()
        arr[i] = window.mean()
    return arr


class Spline:
    """Spline."""

    def __call__(self, *args, **kwargs):
        """Evaluate."""
        return self.true_spline(*args, **kwargs)

    def __init__(self, xi, yi, k=3, s=1000, ignore_nans=True):
        """Initialize.

        Parameters
        ----------
        xi : 1D array
            x points.
        yi : 1D array
            y points.
        k : integer (optional)
            Degree of smoothing. Must be between 1 and 5 (inclusive). Default
            is 3.
        s : integer (optional)
            Positive smoothing factor used to choose the number of knots.
            Number of knots will be increased until the smoothing condition is
            satisfied::

            ``sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s``

            If 0, spline will interpolate through all data points. Default is
            1000.
        ignore_nans : boolean (optional)
            Toggle removle of nans. Default is True.


        .. note:: Use k=1 and s=0 for a linear interplation.

        """
        # import
        from scipy.interpolate import UnivariateSpline
        xi_internal = np.array(xi).copy()
        yi_internal = np.array(yi).copy()
        # nans
        if ignore_nans:
            lis = [xi_internal, yi_internal]
            xi_internal, yi_internal = remove_nans_1D(lis)
        # UnivariateSpline needs ascending xi
        sort = np.argsort(xi_internal)
        xi_internal = xi_internal[sort]
        yi_internal = yi_internal[sort]
        # create true spline
        self.true_spline = UnivariateSpline(xi_internal, yi_internal, k=k, s=s)


def symmetric_sqrt(x, out=None):
    """Compute the 'symmetric' square root: sign(x) * sqrt(abs(x)).

    Parameters
    ----------
    x : array_like or number
        Input array.
    out : ndarray, None, or tuple of ndarray and None (optional)
        A location into which the result is stored. If provided, it must
        have a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.

    Returns
    -------
    np.ndarray
        Symmetric square root of arr.
    """
    factor = np.sign(x)
    out = np.sqrt(np.abs(x), out=out)
    return out * factor


def unique(arr, tolerance=1e-6):
    """Return unique elements in 1D array, within tolerance.

    Parameters
    ----------
    arr : array_like
        Input array. This will be flattened if it is not already 1D.
    tolerance : number (optional)
        The tolerance for uniqueness.

    Returns
    -------
    array
        The sorted unique values.
    """
    arr = sorted(arr.flatten())
    unique = []
    while len(arr) > 0:
        current = arr[0]
        lis = [xi for xi in arr if np.abs(current - xi) < tolerance]
        arr = [xi for xi in arr if not np.abs(lis[0] - xi) < tolerance]
        xi_lis_average = sum(lis) / len(lis)
        unique.append(xi_lis_average)
    return np.array(unique)


def zoom2D(xi, yi, zi, xi_zoom=3., yi_zoom=3., order=3, mode='nearest',
           cval=0.):
    """Zoom a 2D array, with axes.

    Parameters
    ----------
    xi : 1D array
        x axis points.
    yi : 1D array
        y axis points.
    zi : 2D array
        array values. Shape of (x, y).
    xi_zoom : float (optional)
        Zoom factor along x axis. Default is 3.
    yi_zoom : float (optional)
        Zoom factor along y axis. Default is 3.
    order : int (optional)
        The order of the spline interpolation, between 0 and 5. Default is 3.
    mode : {'constant', 'nearest', 'reflect', or 'wrap'}
        Points outside the boundaries of the input are filled according to the
        given mode. Default is constant.
    cval : Value used for
    """
    xi = ndimage.interpolation.zoom(xi, xi_zoom, order=order, mode='nearest')
    yi = ndimage.interpolation.zoom(yi, yi_zoom, order=order, mode='nearest')
    zi = ndimage.interpolation.zoom(zi, (xi_zoom, yi_zoom), order=order, mode=mode)
    return xi, yi, zi


# --- uncategorized -------------------------------------------------------------------------------


def flatten_list(items, seqtypes=(list, tuple), in_place=True):
    """Flatten an irregular sequence.

    Works generally but may be slower than it could
    be if you can make assumptions about your list.

    `Source`__

    __ https://stackoverflow.com/a/10824086

    Parameters
    ----------
    items : iterable
        The irregular sequence to flatten.
    seqtypes : iterable of types (optional)
        Types to flatten. Default is (list, tuple).
    in_place : boolean (optional)
        Toggle in_place flattening. Default is True.

    Returns
    -------
    list
        Flattened list.

    Examples
    --------

    >>> l = [[[1, 2, 3], [4, 5]], 6]
    >>> wt.kit.flatten_list(l)
    [1, 2, 3, 4, 5, 6]
    """
    if not in_place:
        items = items[:]
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i + 1] = items[i]
    return items


def get_methods(the_class, class_only=False, instance_only=False,
                exclude_internal=True):
    """Get a list of strings corresponding to the names of the methods of an object."""
    import inspect

    def acceptMethod(tup):
        # internal function that analyzes the tuples returned by getmembers
        # tup[1] is the actual member object
        is_method = inspect.ismethod(tup[1])
        if is_method:
            bound_to = tup[1].im_self
            internal = (tup[1].im_func.func_name[:2] == '__' and
                        tup[1].im_func.func_name[-2:] == '__')
            if internal and exclude_internal:
                include = False
            else:
                include = (bound_to == the_class and not instance_only) or (
                    bound_to is None and not class_only)
        else:
            include = False
        return include

    # filter to return results according to internal function and arguments
    tups = filter(acceptMethod, inspect.getmembers(the_class))
    return [tup[0] for tup in tups]


def intersperse(lst, item):
    """Put item between each existing item in list.

    `Source`__

    __ intersperse_
    """
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

def get_index(lis, argument):
    """Find the index of an item, given either the item or index as an
    argument.

    Particularly useful as a wrapper for arguments like channel or axis.

    Parameters
    ----------
    lis : list
        List to parse.
    argument : int or object
        Argument.

    Returns
    -------
    int
        Index of chosen object.
    """
    # get channel
    if isinstance(argument, int):
        if -len(lis) <= argument < len(lis):
            return argument
        else:
            raise IndexError('index {0} incompatible with length {1}'.format(argument, len(lis)))
    else:
        return lis.index(argument)


class Suppress(object):
    """Context manager for doing a "deep suppression" of stdout and stderr in Python.

    i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    `Source`__

    __ suppress_


    >>> with WrightTools.kit.Supress():
    ...     rogue_function()

    """

    def __init__(self):
        """init."""
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        """enter."""
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        """exit."""
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def string2identifier(s):
    """Turn a string into a valid python identifier.

    Parameters
    ----------
    s : string
        string to convert

    Returns
    -------
    str
        valid python identifier.
    """
    # https://docs.python.org/3/reference/lexical_analysis.html#identifiers
    if s[0] not in string.ascii_letters:
        s = '_' + s
    valids = string.ascii_letters + string.digits + '_'
    out = ''
    for i, char in enumerate(s):
        if char in valids:
            out += char
        else:
            out += '_'
    return out


class Timer:
    """Context manager for timing code.

    >>> with Timer():
    ...     your_code()
    """

    def __init__(self, verbose=True):
        """init."""
        self.verbose = verbose

    def __enter__(self, progress=None):
        """enter."""
        self.start = clock()

    def __exit__(self, type, value, traceback):
        """exit."""
        self.end = clock()
        self.interval = self.end - self.start
        if self.verbose:
            print('elapsed time: {0} sec'.format(self.interval))
