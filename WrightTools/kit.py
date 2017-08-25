"""
A collection of small, general purpose objects and methods.

.. _RFC3339: https://www.ietf.org/rfc/rfc3339.txt
.. _RFC5322: https://tools.ietf.org/html/rfc5322#section-3.3
.. _HDF5: https://www.hdfgroup.org/HDF5/doc/H5.intro.html
.. _intersperse: http://stackoverflow.com/a/5921708
.. _flatten: http://stackoverflow.com/questions/2158395
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
import time
import pytz
import h5py
import string
import warnings
import dateutil
import datetime
import linecache
import collections
from time import clock

from scipy import ndimage

try:
    import configparser as configparser  # python 3
except ImportError:
    import ConfigParser as configparser  # python 2

import numpy as np

from . import units  # legacy
from . import units as wt_units


# --- define --------------------------------------------------------------------------------------


if sys.version[0] == '2':
    # recognize unicode and string types
    string_type = basestring  # noqa: F821
else:
    string_type = str  # newer versions of python don't have unicode type


# --- time and date -------------------------------------------------------------------------------


def get_timestamp(style='RFC3339', at=None, hms=True, frac=False,
                  timezone='here', filename_compatible=False):
    """Get the current time as a string.

    .. note: Deprecated
             `get_timestamp` is no longer supported
             Use `Timestamp` objects instead

    Parameters
    ----------
    style : {'RFC3339', 'short', 'display', 'legacy'} (optional)
        The format of the returned string. legacy is the old WrightTools
        format. Default is RFC3339. All other arguments control RFC3339
        behavior.
    at : local seconds since epoch (optional)
        Time at-which to generate timestamp. If None, use current time. Default
        is None. Use time.time() to get seconds.
    hms : bool (optional)
        Toggle inclusion of current time (hours:minutes:seconds) in returned
        string. Default is True. Does not effect legacy timestamp.
    frac : bool (optional)
        Toggle inclusion of fractional seconds in returned string. Default is
        False. Does not effect legacy timestamp. Only appears if hms is
        present.
    timezone : {'here', 'utc'}
        Timezone. Default is here.
    filename_compatible : bool
        Remove special charachters. Default is False.
    """
    warnings.warn('get_timestamp is depreciated---use TimeStamp objects',
                  DeprecationWarning, stacklevel=2)
    # get timezone
    if timezone == 'here':
        tz = dateutil.tz.tzlocal()
    elif timezone == 'utc':
        tz = pytz.utc
    else:
        raise Exception('timezone not recognized in kit.get_timestamp')
    # get now
    if at is None:
        now = datetime.datetime.now(tz)
    else:
        now = datetime.datetime.fromtimestamp(at, tz)
    # generate string
    if style == 'RFC3339':
        # get timezone offset
        delta_obj = tz.utcoffset(datetime.datetime.now(tz))
        delta_sec = delta_obj.total_seconds()
        m, s = divmod(delta_sec, 60)
        h, m = divmod(m, 60)
        # create output
        format_string = '%Y-%m-%d'
        if hms:
            format_string += 'T%H:%M:%S'
            if frac:
                format_string += '.%f'
        out = now.strftime(format_string)
        if hms:
            # add timezone information
            if delta_sec == 0.:
                out += 'Z'
            else:
                if delta_sec > 0:
                    sign = '+'
                elif delta_sec < 0:
                    sign = '-'

                def as_string(num):
                    return str(np.abs(int(num))).zfill(2)
                out += sign + as_string(h) + ':' + as_string(m)
    elif style == 'short':
        ssm = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        out = now.strftime('%Y-%m-%d')
        out += ' '
        out += str(int(ssm)).zfill(5)
    elif style == 'display':
        # get timezone offset
        delta_obj = tz.utcoffset(datetime.datetime.now(tz))
        delta_sec = delta_obj.total_seconds()
        m, s = divmod(delta_sec, 60)
        h, m = divmod(m, 60)
        # create output
        format_string = '%Y-%m-%d'
        if hms:
            format_string += ' %H:%M:%S'
            if frac:
                format_string += '.%f'
        out = now.strftime(format_string)
    elif style == 'legacy':
        out = now.strftime('%Y.%m.%d %H_%M_%S')
    else:
        raise Exception('format not recognized in kit.get_timestamp')
    if filename_compatible:
        illegal_characters = [':']
        for char in illegal_characters:
            out = out.replace(char, '')
    return out


class TimeStamp:
    """Class for representing a moment in time."""

    def __init__(self, at=None, timezone='local'):
        """Create a ``TimeStamp`` object.

        Parameters
        ----------
        at : float (optional)
            Seconds since epoch (unix time). If None, current time will be
            used. Default is None.
        timezone : string or integer (optional)
            String (one in {'local', 'utc'} or seconds offset from UTC. Default
            is local.

        Attributes
        ----------
        unix : float
            Seconds since epoch (unix time).
        date : string
            Date.
        hms : string
            Hours, minutes, seconds.
        human : string
            Representation of the timestamp meant to be human readable.
        legacy : string
            Legacy WrightTools timestamp representation.
        RFC3339 : string
            `RFC3339`_ representation (recommended for most applications).
        RFC5322 : string
            `RFC5322`_ representation.
        path : string
            Representation of the timestamp meant for inclusion in filepaths.


        """
        # get timezone
        if timezone == 'local':
            self.tz = dateutil.tz.tzlocal()
        elif timezone == 'utc':
            self.tz = pytz.utc
        elif type(timezone) in [int, float]:
            self.tz = dateutil.tz.tzoffset(None, timezone)
        else:
            raise KeyError
        # get unix timestamp
        if at is None:
            self.unix = time.time()
        else:
            self.unix = at
        # get now
        if at is None:
            self.datetime = datetime.datetime.now(self.tz)
        else:
            self.datetime = datetime.datetime.fromtimestamp(at, self.tz)

    def __repr__(self):
        """Unambiguous representation."""
        return str(self.unix)

    def __str__(self):
        """Readable representation."""
        return self.RFC3339

    @property
    def date(self):
        """year-month-day."""
        return self.datetime.strftime('%Y-%m-%d')

    @property
    def hms(self):
        """Get time formated.

        ``HH:MM:SS``"""
        return self.datetime.strftime('%H:%M:%S')

    @property
    def human(self):
        """Human-readable timestamp."""
        # get timezone offset
        delta_sec = time.timezone
        m, s = divmod(delta_sec, 60)
        h, m = divmod(m, 60)
        # create output
        format_string = '%Y-%m-%d %H:%M:%S'
        out = self.datetime.strftime(format_string)
        return out

    @property
    def legacy(self):
        """Legacy timestamp format."""
        return self.datetime.strftime('%Y.%m.%d %H_%M_%S')

    @property
    def RFC3339(self):
        """RFC3339_."""
        # get timezone offset
        delta_sec = time.timezone
        m, s = divmod(delta_sec, 60)
        h, m = divmod(m, 60)
        # timestamp
        format_string = '%Y-%m-%dT%H:%M:%S.%f'
        out = self.datetime.strftime(format_string)
        # timezone
        if delta_sec == 0.:
            out += 'Z'
        else:
            if delta_sec > 0:
                sign = '+'
            elif delta_sec < 0:
                sign = '-'

            def as_string(num):
                return str(np.abs(int(num))).zfill(2)
            out += sign + as_string(h) + ':' + as_string(m)
        return out

    @property
    def RFC5322(self):
        """RFC5322_."""
        return self.datetime.astimezone(tz=pytz.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')

    @property
    def path(self):
        """Timestamp for placing into filepaths."""
        out = self.datetime.strftime('%Y-%m-%d')
        out += ' '
        ssm = (
            self.datetime -
            self.datetime.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0)).total_seconds()
        out += str(int(ssm)).zfill(5)
        return out


def timestamp_from_RFC3339(RFC3339):
    """Generate a Timestamp object from a RFC3339_ formatted string.

    Parameters
    ----------
    RFC3339 : string
        RFC3339 formatted string.

    Returns
    -------
    WrightTools.kit.TimeStamp
    """
    dt = dateutil.parser.parse(RFC3339)
    timezone = dt.tzinfo._offset.total_seconds()
    unix = (dt - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
            ).total_seconds()  # could use .timestamp() in 3.3 forwards
    timestamp = TimeStamp(at=unix, timezone=timezone)
    return timestamp


# --- file processing -----------------------------------------------------------------------------


def filename_parse(fstr):
    """Parse a filepath string into it's path, name, and suffix."""
    folder, filename = os.path.split(fstr)

    split = filename.split('.', 1)
    file_name = split[0]
    if len(split) == 1:
        file_suffix = None
    else:
        file_suffix = split[1]
    return folder, file_name, file_suffix


def file_len(fname):
    """Cheaply get the number of lines in a file.

    File is not entirely loaded into memory at once.
    """
    # adapted from http://stackoverflow.com/questions/845058
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


class FileSlicer:
    """Access groups of lines from a file quickly, without loading the entire file into memory.

    Mostly a convinient wrapper around the standard library linecache
    module.
    """

    def __init__(self, path, skip_headers=True, header_charachter='#'):
        """Create a ``FileSlicer`` object.

        Parameters
        ----------
        path : string
            Path to the file.
        skip_headers : bool (optional)
            Toggle to skip headers at beginning of file. Default is True.
        header_charachter : string (optional)
            Charachter that appears at the beginning of header lines for this
            file. Default is '#'.
        """
        self.path = path
        self.n = 0
        self.length = file_len(path)
        if skip_headers:
            with open(path) as f:
                while f.readline()[0] == header_charachter:
                    self.n += 1

    def close(self):
        """Clear the cache, attempting to free as much memory as possible."""
        linecache.clearcache()

    def get(self, line_count):
        """Get the next group of lines from the file.

        Parameters
        ----------
        line_count : int
            The number of lines to read from the file.

        Returns
        -------
        list
            List of lines as strings.
        """
        # calculate indicies
        start = self.n
        stop = self.n + line_count
        if stop > self.length:
            raise IndexError('there are no more lines in the slicer: ' +
                             '(file length {})'.format(self.length))
        # get lines using list comprehension
        # for some reason linecache is 1 indexed >:-(
        out = [linecache.getline(self.path, i) for i in range(start + 1, stop + 1)]
        # finish
        self.n += line_count
        return out

    def skip(self, line_count):
        """Skip the next group of lines from the file.

        Parameters
        ----------
        line_count : int
            The number of lines to skip.
        """
        if self.n + line_count > self.length:
            raise IndexError('there are no more lines in the slicer: ' +
                             '(file length {})'.format(self.length))
        # finish
        self.n += line_count


def get_path_matching(name):
    """Get path matching a name.

    Parameters
    ----------
    name : string
        Name to search for.

    Returns
    -------
    string
        Full filepath.
    """
    # first try looking in the user folder
    p = os.path.join(os.path.expanduser('~'), name)
    # then try expanding upwards from cwd
    if not os.path.isdir(p):
        p = None
        drive, folders = os.path.splitdrive(os.getcwd())
        folders = folders.split(os.sep)
        folders.insert(0, os.sep)
        if name in folders:
            p = os.path.join(drive, *folders[:folders.index(name) + 1])
    # TODO: something more robust to catch the rest of the cases?
    return p


def glob_handler(extension, folder=None, identifier=None):
    """Return a list of all files matching specified inputs.

    Parameters
    ----------
    extension: string
    folder: string
    identifier: string
    """
    import glob

    filepaths = []

    if folder:
        # comment out [ and ]...
        folder = folder.replace('[', '?')
        folder = folder.replace(']', '*')
        folder = folder.replace('?', '[[]')
        folder = folder.replace('*', '[]]')
        glob_str = os.path.join(folder, '*' + extension)
    else:
        glob_str = '*' + extension + '*'

    for filepath in glob.glob(glob_str):
        if identifier:
            if identifier in filepath:
                filepaths.append(filepath)
        else:
            filepaths.append(filepath)

    return filepaths


class INI():
    """Handle communication with an INI file."""

    def __init__(self, filepath):
        """Create an INI handler object.

        Parameters
        ----------
        filepath : string
            Filepath.
        """
        self.filepath = filepath
        if sys.version[0] == '3':
            self.config = configparser.ConfigParser()
        else:
            self.config = configparser.SafeConfigParser()

    def add_section(self, section):
        """Add section.

        Parameters
        ----------
        section : string
            Section to add.
        """
        self.config.read(self.filepath)
        self.config.add_section(section)
        with open(self.filepath, 'w') as f:
            self.config.write(f)

    def clear(self):
        """Remove all contents from file. Use with extreme caution.

        .. warning:: This is a destructive action.
        """
        with open(self.filepath, "w"):
            pass
        if sys.version[0] == '3':
            self.config = configparser.ConfigParser()
        else:
            self.config = configparser.SafeConfigParser()

    @property
    def dictionary(self):
        """Get a python dictionary of contents."""
        self.config.read(self.filepath)
        return self.config._sections

    def get_options(self, section):
        """List the options in a section.

        Parameters
        ----------
        section : string
            The section to investigate.

        Returns
        -------
        list of strings
            The options within the given section.
        """
        return list(self.dictionary[section].keys())

    def has_option(self, section, option):
        """Test if file has option.

        Parameters
        ----------
        section : string
            Section.
        option : string
            Option.

        Returns
        -------
        boolean
        """
        self.config.read(self.filepath)
        return self.config.has_option(section, option)

    def has_section(self, section):
        """Test if file has section.

        Parameters
        ----------
        section : string
            Section.

        Returns
        -------
        boolean
        """
        self.config.read(self.filepath)
        return self.config.has_section(section)

    def read(self, section, option):
        """Read from file.

        Parameters
        ----------
        section : string
            Section.
        option : string
            Option.

        Returns
        -------
        string
            Value.
        """
        self.config.read(self.filepath)
        raw = self.config.get(section, option)
        out = string2item(raw, sep=', ')
        return out

    @property
    def sections(self):
        """List of sections."""
        self.config.read(self.filepath)
        return self.config.sections()

    def write(self, section, option, value):
        """Write to file.

        Parameters
        ----------
        section : string
            Section.
        option : string
            Option.
        value : string
            Value.
        """
        self.config.read(self.filepath)
        string = item2string(value, sep=', ')
        self.config.set(section, option, string)
        with open(self.filepath, 'w') as f:
            self.config.write(f)


def read_data_column(path, name):
    """Read a named column of a PyCMDS data file as a single array.

    Parameters
    ----------
    path : string
        Path of PyCMDS data file.
    name : string
        Name of column to read.

    Returns
    -------
    1D numpy.ndarray
        The column of the data file, as an array.
    """
    headers = read_headers(path)
    index = headers['name'].index(name)
    arr = np.genfromtxt(path).T
    return arr[index]


def read_h5(filepath):
    """Read from a `HDF5`_ file, returning the data within as a python dictionary.

    Returns
    -------
    OrderedDict
        Dictionary containing data from HDF5 file.

    See Also
    --------
    kit.write_h5
    """
    d = collections.OrderedDict()
    h5f = h5py.File(filepath, mode='r')
    for key in h5f.keys():
        d[key] = np.array(h5f[key])
    h5f.close()
    return d


def read_headers(filepath):
    """Read 'WrightTools formatted' headers from given path.

    Parameters
    ----------
    filepath : str
        Path of file.

    Returns
    -------
    OrderedDict
        Dictionary containing header information.

    See Also
    --------
    kit.write_headers
    """
    headers = collections.OrderedDict()
    for line in open(filepath):
        if line[0] == '#':
            split = re.split('\: |\:\t', line)
            key = split[0][2:]
            headers[key] = string2item(split[1])
        else:
            break  # all header lines are at the beginning
    return headers


def write_h5(filepath, dictionary):
    """Save a python dictionary into an `HDF5`_ file.

    Right now it only works to store numpy arrays of numbers.

    Parameters
    ----------
    filepath : str
        Filepath to HDF5 file to create. The .hdf5 extension will be appended
        to the filename if it is not already there.
    dictionary : python dictionary-like
        The content to store to the HDF5 file.

    Returns
    -------
    str
        The full filepath to the created HDF5 file.

    See Also
    --------
    kit.read_h5
    """
    # get full filepath
    if filepath[-5:] == '.hdf5':
        filepath = filepath
    else:
        filepath += '.hdf5'
    filepath = os.path.abspath(filepath)
    # create h5f object
    h5f = h5py.File(filepath, 'w')
    # fill h5f object
    for name, data in dictionary.items():
        if isinstance(data, np.ndarray):
            h5f.create_dataset(name, data=data, compression="gzip")
        else:
            # TODO: store it as a string
            data = str(data)
            dt = h5py.special_dtype(vlen=str)
            h5f.create_dataset(name, data=data, dtype=dt)
    # finish
    h5f.close()
    return filepath


def write_headers(filepath, dictionary):
    """Write 'WrightTooTools formatted' headers to given file.

    Headers written can be read again using read_headers.

    Parameters
    ----------
    filepath : str
        Path of file. File must not exist.
    dictionary : dict or OrderedDict
        Dictionary of header items.

    Returns
    -------
    str
        Filepath of file.

    See Also
    --------
    kit.read_headers
    """
    dictionary = copy.deepcopy(dictionary)
    # write header
    for key, value in dictionary.items():
        dictionary[key] = item2string(value)
    lines = []
    for key, value in dictionary.items():
        if '\t' in value:
            joiner = ''
        else:
            joiner = '\t'
        lines.append(joiner.join([key + ':', value]))
    header_str = '\n'.join(lines)
    np.savetxt(filepath, [], header=header_str)
    # return
    return filepath


# --- array and math ------------------------------------------------------------------------------


def closest_pair(arr, give='indicies'):
    """Find the pair of indices corresponding to the closest elements in an array.

    If multiple pairs are equally close, both pairs of indicies are returned.
    Optionally returns the closest distance itself.

    I am sure that this could be written as a cheaper operation. I
    wrote this as a quick and dirty method because I need it now to use on some
    relatively small arrays. Feel free to refactor if you need this operation
    done as fast as possible. - Blaise 2016.02.07

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
    return abs(units.converter(upper, 'nm', output_units) -
               units.converter(lower, 'nm', output_units))


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
    # Written by DJM. darienmorrow@gmail.com. January 15, 2016.
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
            l = [xi_internal, yi_internal]
            xi_internal, yi_internal = remove_nans_1D(l)
        # UnivariateSpline needs ascending xi
        sort = np.argsort(xi_internal)
        xi_internal = xi_internal[sort]
        yi_internal = yi_internal[sort]
        # create true spline
        self.true_spline = UnivariateSpline(xi_internal, yi_internal, k=k, s=s)


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


def array2string(array, sep='\t'):
    """Generate a string from an array with useful formatting.

    Great for writing arrays into single lines in files.

    See Also
    --------
    string2array
    """
    np.set_printoptions(threshold=array.size)
    string = np.array2string(array, separator=sep)
    string = string.replace('\n', sep)
    string = re.sub(r'({})(?=\1)'.format(sep), '', string)
    return string


def flatten_list(l):
    """Flatten an irregular list.

    Works generally but may be slower than it could
    be if you can make assumptions about your list.


    `Source`__

    __ flatten_


        >>> l = [[[1, 2, 3], [4, 5]], 6]
        >>> wt.kit.flatten_list(l)
        [1, 2, 3, 4, 5, 6]

    """
    listIsNested = True
    while listIsNested:  # outer loop
        keepChecking = False
        Temp = []
        for element in l:  # inner loop
            if isinstance(element, list):
                Temp.extend(element)
                keepChecking = True
            else:
                Temp.append(element)
        listIsNested = keepChecking  # determine if outer loop exits
        l = Temp[:]
    return l


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


def item2string(item, sep='\t'):
    r"""Generate string from item.

    Parameters
    ----------
    item : object
        Item.
    sep : string (optional)
        Separator. Default is '\t'.

    Returns
    -------
    string

    See Also
    --------
    string2item
    """
    out = ''
    if isinstance(item, string_type):
        out += '\'' + item + '\''
    elif isinstance(item, list):
        for i in range(len(item)):
            if isinstance(item[i], string_type):
                item[i] = '\'' + item[i] + '\''
            else:
                item[i] = str(item[i])
        out += ' [' + sep.join(item) + ']'
    elif type(item).__module__ == np.__name__:  # anything from numpy
        if hasattr(item, 'shape'):
            out = ' ' + array2string(item, sep=sep)
        else:
            out += ' [' + sep.join([str(i) for i in item]) + ']'
    else:
        out = str(item)
    return out


identity_operators = ['=', '+', '-', '*', '/', 'F']


def parse_identity(string):
    """Parse an identity string into its components.

    Returns
    -------
    tuple of lists
        (names, operators)
    """
    names = re.split("[=F]+", string)
    operators = [c for c in list(string) if c in identity_operators]
    return names, operators


class suppress_stdout_stderr(object):
    """Context manager for doing a "deep suppression" of stdout and stderr in Python.

    i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    `Source`__

    __ suppress_


    >>> with wt.kit.suppress_stdout_stderr():
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


def string2array(string, sep='\t'):
    """Generate an array from a string created using array2string.

    See Also
    --------
    array2string
    """
    # discover size
    size = string.count('\t') + 1
    # discover dimensionality
    dimensionality = 0
    while string[dimensionality] == '[':
        dimensionality += 1
    # discover shape
    shape = []
    for i in range(1, dimensionality + 1)[::-1]:
        to_match = '[' * (i - 1)
        count_positive = string.count(to_match + ' ')
        count_negative = string.count(to_match + '-')
        shape.append(count_positive + count_negative)
    shape[-1] = size / shape[-2]
    for i in range(1, dimensionality - 1)[::-1]:
        shape[i] = shape[i] / shape[i - 1]
    shape = tuple([int(s) for s in shape])
    # import list of floats
    l = string.split(' ')
    # annoyingly series of negative values get past previous filters
    l = flatten_list([i.split('-') for i in l])
    for i, item in enumerate(l):
        bad_chars = ['[', ']', '\t', '\n']
        for bad_char in bad_chars:
            item = item.replace(bad_char, '')
        l[i] = item
    for i in range(len(l))[::-1]:
        try:
            l[i] = float(l[i])
        except ValueError:
            l.pop(i)
    # create and reshape array
    arr = np.array(l)
    arr.shape = shape
    # finish
    return arr


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


def string2item(string, sep='\t'):
    r"""Turn a string into a python object.

    Parameters
    ----------
    string : string
        String.
    sep : string (optional)
        Seperator. Default is '\t'.

    Returns
    -------
    object

    See Also
    --------
    item2string
    """
    if string[0] == '\'' and string[-1] == '\'':
        out = string[1:-1]
    else:
        split = string.split(sep)
        if split[0][0:2] == '[[':  # case of multidimensional arrays
            out = string2array(sep.join(split))
        else:
            split = [i.strip() for i in split]  # remove dumb things
            split = [i if i is not '' else 'None' for i in split]  # handle empties
            # handle lists
            is_list = False
            list_chars = ['[', ']']
            for item_index, item_string in enumerate(split):
                if item_string == '[]':
                    continue
                if item_string[0] == '\'' and item_string[-1] == '\'':  # this is a string
                    continue
                for char in item_string:
                    if char in list_chars:
                        is_list = True
                for char in list_chars:
                    item_string = split[item_index]
                    split[item_index] = item_string.replace(char, '')
            # eval contents
            split = [i.strip() for i in split]  # remove dumb things
            split = [ast.literal_eval(i) for i in split]
            if len(split) == 1 and not is_list:
                split = split[0]
            out = split
    return out


unicode_dictionary = collections.OrderedDict()
unicode_dictionary['Alpha'] = u'\u0391'
unicode_dictionary['Beta'] = u'\u0392'
unicode_dictionary['Gamma'] = u'\u0392'
unicode_dictionary['Delta'] = u'\u0394'
unicode_dictionary['Epsilon'] = u'\u0395'
unicode_dictionary['Zeta'] = u'\u0396'
unicode_dictionary['Eta'] = u'\u0397'
unicode_dictionary['Theta'] = u'\u0398'
unicode_dictionary['Iota'] = u'\u0399'
unicode_dictionary['Kappa'] = u'\u039A'
unicode_dictionary['Lamda'] = u'\u039B'
unicode_dictionary['Mu'] = u'\u039C'
unicode_dictionary['Nu'] = u'\u039D'
unicode_dictionary['Xi'] = u'\u039E'
unicode_dictionary['Omicron'] = u'\u039F'
unicode_dictionary['Pi'] = u'\u03A0'
unicode_dictionary['Rho'] = u'\u03A1'
unicode_dictionary['Sigma'] = u'\u03A3'
unicode_dictionary['Tau'] = u'\u03A4'
unicode_dictionary['Upsilon'] = u'\u03A5'
unicode_dictionary['Phi'] = u'\u03A6'
unicode_dictionary['Chi'] = u'\u03A7'
unicode_dictionary['Psi'] = u'\u03A8'
unicode_dictionary['Omega'] = u'\u03A9'
unicode_dictionary['alpha'] = u'\u03B1'
unicode_dictionary['beta'] = u'\u03B2'
unicode_dictionary['gamma'] = u'\u03B3'
unicode_dictionary['delta'] = u'\u03B4'
unicode_dictionary['epsilon'] = u'\u03B5'
unicode_dictionary['zeta'] = u'\u03B6'
unicode_dictionary['eta'] = u'\u03B7'
unicode_dictionary['theta'] = u'\u03B8'
unicode_dictionary['iota'] = u'\u03B9'
unicode_dictionary['kappa'] = u'\u03BA'
unicode_dictionary['lamda'] = u'\u03BB'
unicode_dictionary['mu'] = u'\u03BC'
unicode_dictionary['nu'] = u'\u03BD'
unicode_dictionary['xi'] = u'\u03BE'
unicode_dictionary['omicron'] = u'\u03BF'
unicode_dictionary['pi'] = u'\u03C0'
unicode_dictionary['rho'] = u'\u03C1'
unicode_dictionary['sigma'] = u'\u03C3'
unicode_dictionary['tau'] = u'\u03C4'
unicode_dictionary['upsilon'] = u'\u03C5'
unicode_dictionary['phi'] = u'\u03C6'
unicode_dictionary['chi'] = u'\u03C7'
unicode_dictionary['psi'] = u'\u03C8'
unicode_dictionary['omega'] = u'\u03C9'


def update_progress(progress, carriage_return=False, length=50):
    """Print a pretty progress bar to the console.

    accepts 'progress' as a percentage
    bool carriage_return toggles overwrite behavior
    """
    # make progress bar string
    text = '\r'
    num_oct = int(progress * (length / 100.))
    text += '[{0}{1}]'.format('#' * num_oct, ' ' * (length - num_oct))
    text += ' {}%'.format(np.round(progress, decimals=2))
    if carriage_return:
        text += '\r\n'
    sys.stdout.write(text)
    if progress == 100.:
        print('\n')
    else:
        sys.stdout.flush()


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
