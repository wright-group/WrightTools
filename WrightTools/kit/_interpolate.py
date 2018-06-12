"""Interpolation tools."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

from scipy import ndimage
from scipy.interpolate import UnivariateSpline

from ._array import remove_nans_1D


# --- define --------------------------------------------------------------------------------------


__all__ = ["zoom2D", "Spline"]


# --- functions -----------------------------------------------------------------------------------


def zoom2D(xi, yi, zi, xi_zoom=3., yi_zoom=3., order=3, mode="nearest", cval=0.):
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
        given mode. Default is nearest.
    cval : scalar (optional)
        Value used for constant mode. Default is 0.0.
    """
    xi = ndimage.interpolation.zoom(xi, xi_zoom, order=order, mode="nearest")
    yi = ndimage.interpolation.zoom(yi, yi_zoom, order=order, mode="nearest")
    zi = ndimage.interpolation.zoom(zi, (xi_zoom, yi_zoom), order=order, mode=mode, cval=cval)
    return xi, yi, zi


# --- classes -------------------------------------------------------------------------------------


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
