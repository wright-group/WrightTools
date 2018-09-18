"""Array interaction tools."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = [
    "closest_pair",
    "diff",
    "fft",
    "joint_shape",
    "orthogonal",
    "remove_nans_1D",
    "share_nans",
    "smooth_1D",
    "svd",
    "unique",
    "valid_index",
    "mask_reduce",
    "enforce_mask_shape",
]


# --- functions -----------------------------------------------------------------------------------


def closest_pair(arr, give="indicies"):
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
    if give == "indicies":
        return outs
    elif give == "distance":
        return min_dist
    else:
        raise KeyError("give not recognized in closest_pair")


def diff(xi, yi, order=1) -> np.ndarray:
    """Take the numerical derivative of a 1D array.

    Output is mapped onto the original coordinates  using linear interpolation.
    Expects monotonic xi values.

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
    yi = np.array(yi).copy()
    flip = False
    if xi[-1] < xi[0]:
        xi = np.flipud(xi.copy())
        yi = np.flipud(yi)
        flip = True
    midpoints = (xi[1:] + xi[:-1]) / 2
    for _ in range(order):
        d = np.diff(yi)
        d /= np.diff(xi)
        yi = np.interp(xi, midpoints, d)
    if flip:
        yi = np.flipud(yi)
    return yi


def fft(xi, yi, axis=0) -> tuple:
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
        1D array. Conjugate to input xi. Example: if input xi is in the time
        domain, output xi is in frequency domain.
    yi : ND numpy.ndarray
        FFT. Has the same shape as the input array (yi).
    """
    # xi must be 1D
    if xi.ndim != 1:
        raise wt_exceptions.DimensionalityError(1, xi.ndim)
    # xi must be evenly spaced
    spacing = np.diff(xi)
    if not np.allclose(spacing, spacing.mean()):
        raise RuntimeError("WrightTools.kit.fft: argument xi must be evenly spaced")
    # fft
    yi = np.fft.fft(yi, axis=axis)
    d = (xi.max() - xi.min()) / (xi.size - 1)
    xi = np.fft.fftfreq(xi.size, d=d)
    # shift
    xi = np.fft.fftshift(xi)
    yi = np.fft.fftshift(yi, axes=axis)
    return xi, yi


def joint_shape(*args) -> tuple:
    """Given a set of arrays, return the joint shape.

    Parameters
    ----------
    args : array-likes

    Returns
    -------
    tuple of int
        Joint shape.
    """
    if len(args) == 0:
        return ()
    shape = []
    shapes = [a.shape for a in args]
    ndim = args[0].ndim
    for i in range(ndim):
        shape.append(max([s[i] for s in shapes]))
    return tuple(shape)


def orthogonal(*args) -> bool:
    """Determine if a set of arrays are orthogonal.

    Parameters
    ----------
    args : array-likes or array shapes

    Returns
    -------
    bool
        Array orthogonality condition.
    """
    for i, arg in enumerate(args):
        if hasattr(arg, "shape"):
            args[i] = arg.shape
    for s in zip(*args):
        if np.product(s) != max(s):
            return False
    return True


def remove_nans_1D(*args) -> tuple:
    """Remove nans in a set of 1D arrays.

    Removes indicies in all arrays if any array is nan at that index.
    All input arrays must have the same size.

    Parameters
    ----------
    args : 1D arrays

    Returns
    -------
    tuple
        Tuple of 1D arrays in same order as given, with nan indicies removed.
    """
    vals = np.isnan(args[0])
    for a in args:
        vals |= np.isnan(a)
    return tuple(np.array(a)[~vals] for a in args)


def share_nans(*arrs) -> tuple:
    """Take a list of nD arrays and return a new list of nD arrays.

    The new list is in the same order as the old list.
    If one indexed element in an old array is nan then every element for that
    index in all new arrays in the list is then nan.

    Parameters
    ----------
    *arrs : nD arrays.

    Returns
    -------
    list
        List of nD arrays in same order as given, with nan indicies syncronized.
    """
    nans = np.zeros(joint_shape(*arrs))
    for arr in arrs:
        nans *= arr
    return tuple([a + nans for a in arrs])


def smooth_1D(arr, n=10, smooth_type="flat") -> np.ndarray:
    """Smooth 1D data using a window function.
    
    Edge effects will be present. 

    Parameters
    ----------
    arr : array_like
        Input array, 1D.
    n : int (optional)
        Window length.
    smooth_type : {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'} (optional)
        Type of window function to convolve data with.
        'flat' window will produce a moving average smoothing.
        
    Returns
    -------
    array_like
        Smoothed 1D array.
    """

    # check array input
    if arr.ndim != 1:
        raise wt_exceptions.DimensionalityError(1, arr.ndim)
    if arr.size < n:
        message = "Input array size must be larger than window size."
        raise wt_exceptions.ValueError(message)
    if n < 3:
        return arr
    # construct window array
    if smooth_type == "flat":
        w = np.ones(n, dtype=arr.dtype)
    elif smooth_type == "hanning":
        w = np.hanning(n)
    elif smooth_type == "hamming":
        w = np.hamming(n)
    elif smooth_type == "bartlett":
        w = np.bartlett(n)
    elif smooth_type == "blackman":
        w = np.blackman(n)
    else:
        message = "Given smooth_type, {0}, not available.".format(str(smooth_type))
        raise wt_exceptions.ValueError(message)
    # convolve reflected array with window function
    out = np.convolve(w / w.sum(), arr, mode="same")
    return out


def svd(a, i=None) -> tuple:
    """Singular Value Decomposition.

    Factors the matrix `a` as ``u * np.diag(s) * v``, where `u` and `v`
    are unitary and `s` is a 1D array of `a`'s singular values.

    Parameters
    ----------
    a : array_like
        Input array.
    i : int or slice (optional)
        What singular value "slice" to return.
        Default is None which returns unitary 2D arrays.

    Returns
    -------
    tuple
        Decomposed arrays in order `u`, `v`, `s`
    """
    u, s, v = np.linalg.svd(a, full_matrices=False, compute_uv=True)
    u = u.T
    if i is None:
        return u, v, s
    else:
        return u[i], v[i], s[i]


def unique(arr, tolerance=1e-6) -> np.ndarray:
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


def valid_index(index, shape) -> tuple:
    """Get a valid index for a broadcastable shape.

    Parameters
    ----------
    index : tuple
        Given index.
    shape : tuple of int
        Shape.

    Returns
    -------
    tuple
        Valid index.
    """
    # append slices to index
    index = list(index)
    while len(index) < len(shape):
        index.append(slice(None))
    # fill out, in reverse
    out = []
    for i, s in zip(index[::-1], shape[::-1]):
        if s == 1:
            if isinstance(i, slice):
                out.append(slice(None))
            else:
                out.append(0)
        else:
            out.append(i)
    return tuple(out[::-1])


def mask_reduce(mask):
    """Reduce a boolean mask, removing all false slices in any dimension.

    Parameters
    ----------
    mask : ndarray with bool dtype
        The mask which is to be reduced

    Returns
    -------
        A boolean mask with no all False slices.
    """
    mask = mask.copy()
    for i in range(len(mask.shape)):
        a = mask.copy()
        j = list(range(len(mask.shape)))
        j.remove(i)
        j = tuple(j)
        a = a.max(axis=j, keepdims=True)
        idx = [slice(None)] * len(mask.shape)
        a = a.flatten()
        idx[i] = [k for k in range(len(a)) if a[k]]
        mask = mask[tuple(idx)]
    return mask


def enforce_mask_shape(mask, shape):
    """Reduce a boolean mask to fit a given shape.

    Parameters
    ----------
    mask : ndarray with bool dtype
        The mask which is to be reduced
    shape : tuple of int
        Shape which broadcasts to the mask shape.

    Returns
    -------
        A boolean mask, collapsed along axes where the shape given has one element.
    """
    red = tuple([i for i in range(len(shape)) if shape[i] == 1])
    return mask.max(axis=red, keepdims=True)
