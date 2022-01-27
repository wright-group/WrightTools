"""Discover dimensions of a flattened ND array."""


# --- import --------------------------------------------------------------------------------------


import warnings
import collections

import numpy as np

from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ["discover_dimensions"]


# --- function ------------------------------------------------------------------------------------


def discover_dimensions(arr, cols) -> collections.OrderedDict:
    """Discover the dimensions of a flattened multidimensional array.

    Parameters
    ----------
    arr : 2D numpy ndarray
        Array in [col, value].
    cols : dictionary
        Dictionary with column names as keys, and idx, tolerance and units
        as values.

    Returns
    -------
    dictionary
        expression: points
    """
    # import values -------------------------------------------------------------------------------
    di = [cols[key]["idx"] for key in cols.keys()]
    dt = [cols[key]["tolerance"] for key in cols.keys()]
    du = [cols[key]["units"] for key in cols.keys()]
    dk = [key for key in cols.keys()]
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
    # shape ---------------------------------------------------------------------------------------
    out = collections.OrderedDict()
    for a in scanned_ordered:
        key = a[0][0]
        axis = cols[key]
        # generate lists from data
        lis = sorted(arr[axis["idx"]])
        tol = axis["tolerance"]
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
        tol = sum(xstd) / len(xstd)
        tol = max(tol, 1e-4)
        if axis["units"] == "nm":
            min_wn = 1e7 / max(xs) + tol
            max_wn = 1e7 / min(xs) - tol
            points = np.linspace(min_wn, max_wn, num=len(xs))
            points = wt_units.converter(points, "wn", "nm")
        else:
            points = np.linspace(min(xs) + tol, max(xs) - tol, num=len(xs))
        key = "=".join(a[0])
        out[key] = points
    # warn if data doesn't seem like the right shape ----------------------------------------------
    length = len(arr[0])
    size = 1
    for a in out.values():
        size *= a.size
    if not size == length:
        message = "array length ({0}) inconsistent with data size ({1})".format(length, size)
        warnings.warn(message)
    # return --------------------------------------------------------------------------------------
    return out
