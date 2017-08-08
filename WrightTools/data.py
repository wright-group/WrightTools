"""
Central data class and associated.
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import ast
import copy
import time
import collections
import warnings
import pickle

import numpy as np

import scipy
from scipy.interpolate import griddata, interp1d

from . import exceptions as wt_exceptions
from . import kit as wt_kit
from . import units as wt_units


# --- define --------------------------------------------------------------------------------------


debug = False

# string types
if sys.version[0] == '2':
    string_type = basestring  # recognize unicode and string types
else:
    string_type = str  # newer versions of python don't have unicode type


# --- data class ----------------------------------------------------------------------------------


class Axis:

    def __init__(self, points, units, symbol_type=None,
                 tolerance=None, file_idx=None,
                 name='', label_seed=[''], **kwargs):
        self.name = name
        self.tolerance = tolerance
        self.points = np.asarray(points)
        self.units = units
        self.file_idx = file_idx
        self.label_seed = label_seed
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        # get units kind
        self.units_kind = None
        for dic in wt_units.unit_dicts:
            if self.units in dic.keys():
                self.units_kind = dic['kind']
        # get symbol type
        if symbol_type:
            self.symbol_type = symbol_type
        else:
            self.symbol_type = wt_units.get_default_symbol_type(self.units)
        self.get_label()

    def __repr__(self):
        return 'WrightTools.data.Axis object \'{0}\' at {1}'.format(self.name, str(id(self)))

    def convert(self, destination_units):
        self.points = wt_units.converter(self.points, self.units,
                                         destination_units)
        self.units = destination_units

    def get_label(self, show_units=True, points=False, decimals=2):
        label = r'$\mathsf{'
        # label
        for part in self.label_seed:
            if self.units_kind is not None:
                units_dictionary = getattr(wt_units, self.units_kind)
                label += getattr(wt_units, self.symbol_type)[self.units]
                if part is not '':
                    label += r'_{' + str(part) + r'}'
            else:
                label += self.name.replace('_', '\,\,')
            label += r'='
        # remove the last equals sign
        label = label[:-1]
        if points:
            if self.points is not None:
                label += r'=\,' + str(np.round(self.points, decimals=decimals))
        # units
        if show_units:
            if self.units_kind:
                units_dictionary = getattr(wt_units, self.units_kind)
                label += r'\,'
                if not points:
                    label += r'\left('
                label += units_dictionary[self.units][2]
                if not points:
                    label += r'\right)'
            else:
                pass
        label += r'}$'
        return label

    @property
    def info(self):
        info = collections.OrderedDict()
        info['name'] = self.name
        info['id'] = id(self)
        if self.is_constant:
            info['point'] = self.points
        else:
            info['range'] = '{0} - {1} ({2})'.format(self.points.min(),
                                                     self.points.max(), self.units)
            info['number'] = len(self.points)
        return info

    def is_constant(self):
        try:
            len(self.points)
        except TypeError:
            return False
        finally:
            return True

    @property
    def label(self):
        return self.get_label()

    def max(self):
        return self.points.max()

    def min(self):
        return self.points.min()

    def min_max_step(self):
        _min = self.points.min()
        _max = self.points.max()
        _step = (_max - _min) / (len(self.points) - 1)
        return _min, _max, _step


class Channel:

    def __init__(self, values, units=None,
                 file_idx=None,
                 znull=None, zmin=None, zmax=None, signed=None,
                 name='channel', label=None, label_seed=None):
        # import
        self.name = name
        self.label = label
        self.label_seed = label_seed
        self.units = units
        self.file_idx = file_idx
        # values
        if values is not None:
            self.give_values(np.asarray(values), znull, zmin, zmax, signed)
        else:
            self.znull = znull
            self.zmin = zmin
            self.zmax = zmax
            self.signed = signed

    def __repr__(self):
        return 'WrightTools.data.Channel object \'{0}\' at {1}'.format(self.name, str(id(self)))

    def _update(self):
        self.zmin = np.nanmin(self.values)
        self.zmax = np.nanmax(self.values)
        self.znull = self.znull = max(self.zmin, min(self.znull, self.zmax))

    def _pupdate(self, *args, **kwargs):
        return self._update(*args, **kwargs)

    def clip(self, zmin=None, zmax=None, replace='nan'):
        """
        clip (limit) the values in a channel
        replace one in ['val', 'nan', 'mask']
        """
        # decide what zmin and zmax will actually be
        if zmax is not None:
            pass
        else:
            zmax = np.nanmax(self.values)
        if zmin is not None:
            pass
        else:
            zmin = np.nanmin(self.values)
        # replace values
        if replace == 'val':
            self.values.clip(zmin, zmax, out=self.values)
        elif replace == 'nan':
            self.values[self.values < zmin] = np.nan
            self.values[self.values > zmax] = np.nan
        elif replace == 'mask':
            self.values = np.ma.masked_outside(self.values,
                                               zmin, zmax,
                                               copy=False)
        else:
            print('replace not recognized in channel.clip')
        # recalculate zmin and zmax of channel object
        self._update()

    def give_values(self, values, znull=None, zmin=None, zmax=None,
                    signed=None):
        self.values = values
        # znull
        if znull is not None:
            self.znull = znull
        elif hasattr(self, 'znull'):
            if self.znull:
                pass
            else:
                self.znull = np.nanmin(self.values)
        else:
            self.znull = np.nanmin(self.values)
        # zmin
        if zmin is not None:
            self.zmin = zmin
        elif hasattr(self, 'zmin'):
            if self.zmin:
                pass
            else:
                self.zmin = np.nanmin(self.values)
        else:
            self.zmin = np.nanmin(self.values)
        # zmax
        if zmax is not None:
            self.zmax = zmax
        elif hasattr(self, 'zmax'):
            if self.zmax:
                pass
            else:
                self.zmax = np.nanmax(self.values)
        else:
            self.zmax = np.nanmax(self.values)
        # signed
        if signed is not None:
            self.signed = signed
        elif hasattr(self, 'signed'):
            if self.signed is None:
                if self.zmin < self.znull:
                    self.signed = True
                else:
                    self.signed = False
        else:
            if self.zmin < self.znull:
                self.signed = True
            else:
                self.signed = False

    @property
    def info(self):
        info = collections.OrderedDict()
        info['name'] = self.name
        info['id'] = id(self)
        info['zmin'] = self.zmin
        info['zmax'] = self.zmax
        info['znull'] = self.znull
        info['signed'] = self.signed
        return info

    def invert(self):
        self.values = - self.values

    def max(self):
        """ Maximum, ignorning nans.  """
        return np.nanmax(self.values)

    def min(self):
        """ Minimum, ignoring nans.  """
        return np.nanmin(self.values)

    def normalize(self, axis=None):
        """ Normalizes a Channel, setting z-null to 0 and the max to 1.  """
        # process axis argument
        if axis is not None:
            if hasattr(axis, '__contains__'):  # list, tuple or similar
                axis = tuple((int(i) for i in axis))
            else:  # presumably a simple number
                axis = int(axis)
        # subtract off znull
        self.values -= self.znull
        self.znull = 0.
        # create dummy array
        dummy = self.values.copy()
        dummy[np.isnan(dummy)] = 0  # nans are propagated in np.amax
        if self.signed:
            dummy = np.absolute(dummy)
        # divide through by max
        self.values /= np.amax(dummy, axis=axis, keepdims=True)
        # finish
        self._update()

    def trim(self, neighborhood, method='ztest', factor=3, replace='nan',
             verbose=True):
        """ Remove outliers from the dataset

        Identifies outliers by comparing each point to its
        neighbors using a statistical test.

        Parameters
        ----------
        neighborhood : list of integers
            Size of the neighborhood in each dimension. Length of the list must
            be equal to the dimensionality of the channel.
        method : {'ztest'} (optional)
            Statistical test used to detect outliers. Default is ztest.

            ztest
                Compare point deviation from neighborhood mean to neighborhood
                standard deviation.

        factor : number (optional)
            Tolerance factor.  Default is 3.
        replace : {'nan', 'mean', 'mask', number} (optional)
            Behavior of outlier replacement. Default is nan.

            nan
                Outliers are replaced by numpy nans.

            mean
                Outliers are replaced by the mean of its neighborhood.

            mask
                Array is masked at outliers.

            number
                Array becomes given number.

        Returns
        -------
        list of tuples
            Indicies of trimmed outliers.

        See Also
        --------
        clip
            Remove pixels outside of a certain range.
        """
        outliers = []
        means = []
        # find outliers
        for idx in np.ndindex(self.values.shape):
            slices = []
            for i, di, size in zip(idx, neighborhood, self.values.shape):
                start = max(0, i - di)
                stop = min(size, i + di + 1)
                slices.append(slice(start, stop, 1))
            neighbors = self.values[slices]
            mean = np.nanmean(neighbors)
            limit = np.nanstd(neighbors) * factor
            if np.abs(self.values[idx] - mean) > limit:
                outliers.append(idx)
                means.append(mean)
        # replace outliers
        i = tuple(zip(*outliers))
        if replace == 'nan':
            self.values[i] = np.nan
        elif replace == 'mean':
            self.values[i] = means
        elif replace == 'mask':
            self.values = np.ma.array(self.values)
            self.values[i] = np.ma.masked
        elif type(replace) in [int, float]:
            self.values[i] = replace
        else:
            raise KeyError('replace must be one of {nan, mean, mask} or some number')
        # finish
        self._update()
        if verbose:
            print('%i outliers removed' % len(outliers))
        return outliers

    @ property
    def zmag(self):
        return max((self.zmax - self.znull, self.znull - self.zmin))


class Data:

    def __init__(self, axes, channels, constants=[],
                 name='', source=None):
        """ Central class for data in the Wright Group.

        Parameters
        ----------
        channels : list
            A list of Channel objects. Channels are also inherited as
            attributes using the channel name: ``data.ai0``, for example.
        axes : list
            A list of Axis objects. Axes are also inherited as attributes using
            the axis name: ``data.w1``, for example.
        constants : list
            A list of Axis objects, each with exactly one point.
        """
        # record version
        from . import __version__
        self.__version__ = __version__
        # assign
        self.axes = axes
        self.constants = constants
        self.channels = channels
        self.name = name
        self.source = source
        # update
        self._update()
        # reserve a copy of own self at this stage
        self._original = self.copy()

    def __repr__(self):
        return 'WrightTools.data.Data object \'{0}\' {1} at {2}'.format(
            self.name, str(self.axis_names), str(id(self)))

    def _update(self):
        """
        Ensure that the ``axis_names``, ``constant_names``, ``channel_names``,
        and ``shape`` attributes are correct.
        """
        self.axis_names = [axis.name for axis in self.axes]
        self.constant_names = [axis.name for axis in self.constants]
        self.channel_names = [channel.name for channel in self.channels]
        all_names = self.axis_names + self.channel_names + self.constant_names
        if len(all_names) == len(set(all_names)):
            pass
        else:
            print('axis, constant, and channel names must all be unique - your data object is now broken!!!!')
            return
        for obj in self.axes + self.channels + self.constants:
            setattr(self, obj.name, obj)
        self.shape = self.channels[0].values.shape

    def _pupdate(self, *args, **kwargs):
        return self._update(*args, **kwargs)

    def bring_to_front(self, channel):
        """ Bring a specific channel to the zero-indexed position in channels.

        All other channels get pushed back but remain in order.

        Parameters
        ----------
        channel : int or str
            Channel index or name.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        # bring to front
        self.channels.insert(0, self.channels.pop(channel_index))
        self._update()

    def chop(self, *args, **kwargs):
        """ Divide the dataset into its lower-dimensionality components.

        Parameters
        ----------
        axis : str or int
            Axes of the returned data objects. Strings refer to the names of
            axes in this object, integers refer to their index.
        at : dict
            Dictonary. Keys are axis names, values are lists
            ``[position, input units]``.
        verbose : bool, optional
            Toggle talkback. Default is True.

        Returns
        -------
        list
            A list of data objects.

        See Also
        --------
        collapse
            Collapse the dataset along one axis.
        split
            Split the dataset while maintaining its dimensionality.

            >>> data.chop('w1', 'w2', at={'d2': [0, 'fs']})
            [data]

        """
        # organize arguments recieved -------------------------------------------------------------
        axes_args = list(args)
        keys = ['at', 'verbose']
        defaults = [{}, True]
        at, verbose = [kwargs.pop(k) if k in kwargs.keys() else d for k, d in zip(keys, defaults)]
        chopped_constants = at
        # interpret arguments recieved ------------------------------------------------------------
        for i in range(len(axes_args)):
            arg = axes_args[i]
            if isinstance(arg, string_type):
                pass
            elif isinstance(arg, int):
                arg = self.axis_names[arg]
            else:
                message = 'argument {arg} not recognized in Data.chop'.format(arg)
                raise TypeError(message)
            axes_args[i] = arg
        for arg in axes_args:
            if arg not in self.axis_names:
                raise Exception('axis {} not in data'.format(arg))
        # iterate! --------------------------------------------------------------------------------
        print(axes_args, chopped_constants)
        # find iterated dimensions
        iterated_dimensions = []
        iterated_shape = [1]
        for name in self.axis_names:
            if name not in axes_args and name not in chopped_constants.keys():
                iterated_dimensions.append(name)
                length = len(getattr(self, name).points)
                iterated_shape.append(length)
        # make copies of channel objects for handing out
        channels_chopped = copy.deepcopy(self.channels)
        chopped_constants_everywhere = chopped_constants
        out = []
        for index in np.ndindex(tuple(iterated_shape)):
            # get chopped_constants correct for this iteration
            chopped_constants = chopped_constants_everywhere.copy()
            for i in range(len(index[1:])):
                idx = index[1:][i]
                name = iterated_dimensions[i]
                axis_units = getattr(self, name).units
                position = getattr(self, name).points[idx]
                chopped_constants[name] = [position, axis_units]
            # re-order array: [all_chopped_constants, channels, all_chopped_axes]
            transpose_order = []
            constant_indicies = []
            # handle constants first
            constants = list(self.constants)  # copy
            for dim in chopped_constants.keys():
                idx = [idx for idx in range(len(self.axes)) if self.axes[idx].name == dim][0]
                transpose_order.append(idx)
                # get index of nearest value
                val = chopped_constants[dim][0]
                val = wt_units.converter(val, chopped_constants[dim][1], self.axes[idx].units)
                c_idx = np.argmin(abs(self.axes[idx].points - val))
                constant_indicies.append(c_idx)
                obj = copy.copy(self.axes[idx])
                obj.points = self.axes[idx].points[c_idx]
                constants.append(obj)
            # now handle axes
            axes_chopped = []
            for dim in axes_args:
                idx = [idx for idx in range(len(self.axes)) if self.axes[idx].name == dim][0]
                transpose_order.append(idx)
                axes_chopped.append(self.axes[idx])
            # ensure that everything is kosher
            if len(transpose_order) == len(self.channels[0].values.shape):
                pass
            else:
                print('chop failed: not enough dimensions specified')
                print(len(transpose_order))
                print(len(self.channels[0].values.shape))
                return
            if len(transpose_order) == len(set(transpose_order)):
                pass
            else:
                print('chop failed: same dimension used twice')
                return
            # chop
            for i in range(len(self.channels)):
                values = self.channels[i].values
                values = values.transpose(transpose_order)
                for idx in constant_indicies:
                    values = values[idx]
                channels_chopped[i].values = values
            # finish iteration
            data_out = Data(axes_chopped, copy.deepcopy(channels_chopped),
                            constants=constants,
                            name=self.name, source=self.source)
            out.append(data_out)
        # return ----------------------------------------------------------------------------------
        if verbose:
            print('chopped data into %d piece(s)' % len(out), 'in', axes_args)
        return out

    def clip(self, channel=0, *args, **kwargs):
        """ Wrapper method for ``Channel.clip``.

        Parameters
        ----------
        channel : int or str
            The channel to call clip on.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # call clip on channel object
        channel.clip(*args, **kwargs)

    def collapse(self, axis, method='integrate'):
        """
        Collapse the dataset along one axis.

        Parameters
        ----------
        axis : int or str
            The axis to collapse along.
        method : {'integrate', 'average', 'sum', 'max', 'min'} (optional)
            The method of collapsing the given axis. Method may also be list
            of methods corresponding to the channels of the object. Default
            is integrate.

        See Also
        --------
        chop
            Divide the dataset into its lower-dimensionality components.
        split
            Split the dataset while maintaining its dimensionality.
        """
        # get axis index --------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            print('axis type', type(axis), 'not valid')
        # methods ---------------------------------------------------------------------------------
        if isinstance(method, list):
            if len(method) == len(self.channels):
                methods = method
            else:
                print('method argument incompatible in data.collapse')
        elif isinstance(method, string_type):
            methods = [method for _ in self.channels]
        # collapse --------------------------------------------------------------------------------
        for method, channel in zip(methods, self.channels):
            if method in ['int', 'integrate']:
                channel.values = np.trapz(
                    y=channel.values, x=self.axes[axis_index].points, axis=axis_index)
            elif method == 'sum':
                channel.values = channel.values.sum(axis=axis_index)
            elif method in ['max', 'maximum']:
                channel.values = np.nanmax(channel.values, axis=axis_index)
            elif method in ['min', 'minimum']:
                channel.values = np.nanmin(channel.values, axis=axis_index)
            elif method in ['ave', 'average']:
                channel.values = np.average(channel.values, axis=axis_index)
            else:
                print('method not recognized in data.collapse')
            channel._update()
        # cleanup ---------------------------------------------------------------------------------
        self.axes.pop(axis_index)
        self._update()

    def convert(self, destination_units, verbose=True):
        """
        Converts all compatable constants and axes to given units.

        Parameters
        ----------
        destination_units : str
            Destination units.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        See Also
        --------
        Axis.convert
            Convert a single axis object to compatable units. Call on an
            axis object in data.axes or data.constants.
        """
        # get kind of units
        for dic in wt_units.unit_dicts:
            if destination_units in dic.keys():
                units_kind = dic['kind']
        # apply to all compatible axes
        for axis in self.axes + self.constants:
            if axis.units_kind == units_kind:
                axis.convert(destination_units)
                if verbose:
                    print('axis', axis.name, 'converted')

    def copy(self):
        """
        Copy the object.

        Returns
        -------
        data
            A deep copy of the data object.
        """
        return copy.deepcopy(self)

    @property
    def dimensionality(self):
        return len(self.axes)

    def divide(self, divisor, channel=0, divisor_channel=0):
        """
        Divide a given channel by another data object. Divisor may be self.
        All axes in divisor must be contained in self.

        Parameters
        ----------
        divisor : data
            The denominator in the division.
        channel : int or str
            The channel to divide into. The result will be written into this
            channel.
        divisor_channel : int or str
            The channel in the divisor object to use.
        """
        divisor = divisor.copy()
        # map points
        for name in divisor.axis_names:
            if name in self.axis_names:
                axis = getattr(self, name)
                divisor_axis = getattr(divisor, name)
                divisor_axis.convert(axis.units)
                divisor.map_axis(name, axis.points)
            else:
                raise RuntimeError('all axes in divisor must be contained in self')
        # divide
        # transpose so axes of divisor are last (in order)
        axis_indicies = [self.axis_names.index(name) for name in divisor.axis_names]
        axis_indicies.reverse()
        transpose_order = list(range(len(self.axes)))
        for i in range(len(axis_indicies)):
            ai = axis_indicies[i]
            ri = list(range(len(self.axes)))[-(i + 1)]
            transpose_order[ri], transpose_order[ai] = transpose_order[ai], transpose_order[ri]
        self.transpose(transpose_order, verbose=False)
        # get own channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # get divisor channel
        if isinstance(divisor_channel, int):
            divisor_channel_index = divisor_channel
        elif isinstance(divisor_channel, string_type):
            divisor_channel_index = divisor.channel_names.index(divisor_channel)
        else:
            print('divisor channel type', type(channel), 'not valid')
        divisor_channel = divisor.channels[divisor_channel_index]
        # do division
        channel.values /= divisor_channel.values
        channel._update()
        # transpose out
        self.transpose(transpose_order, verbose=False)

    def dOD(self, signal_channel, reference_channel,
            method='digital'):
        r"""
        For transient absorption,  convert zi signal from dI to dOD.

        Parameters
        ----------
        signal_channel : int or str
            Index or name of signal (dI) channel.
        reference_channel : int or str
            Index or name of reference (I) channel.
        method : {'digital', 'boxcar'} (optional)
            Shots processing method. Default is digital.

        Notes
        -----
        dOD is calculated as

        .. math::
             -\log_{10}\left(\frac{I+dI}{I}\right)

        where I is the reference channel and dI is the signal channel.
        """
        # get signal channel
        if isinstance(signal_channel, int):
            signal_channel_index = signal_channel
        elif isinstance(signal_channel, string_type):
            signal_channel_index = self.channel_names.index(signal_channel)
        else:
            print('channel type', type(signal_channel), 'not valid')
        # get reference channel
        if isinstance(reference_channel, int):
            reference_channel_index = reference_channel
        elif isinstance(reference_channel, string_type):
            reference_channel_index = self.channel_names.index(reference_channel)
        else:
            print('channel type', type(reference_channel), 'not valid')
        # process
        I = self.channels[reference_channel_index].values.copy()
        dI = self.channels[signal_channel_index].values.copy()
        if method == 'digital':
            out = -np.log10((I + dI) / I)
        elif method == 'boxcar':
            # assume data collected with boxcar i.e.
            # sig = 1/2 dT
            # ref = T + 1/2 dT
            dI *= 2
            out = -np.log10((I + dI) / I)
        else:
            print('method not recognized in dOD, returning')
            return
        # finish
        self.channels[signal_channel_index].give_values(out)
        self.channels[signal_channel_index].signed = True
        self.channels[signal_channel_index].znull = 0
        self.channels[signal_channel_index]._update()

    def flip(self, axis):
        """
        Flip direction of arrays along an axis. Changes the index of elements
        without changing their correspondance to axis positions.

        Parameters
        ----------
        axis : int or str
            The axis to flip.
        """
        # axis ------------------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            print('axis type', type(axis), 'not valid')
        axis = self.axes[axis_index]
        # flip ------------------------------------------------------------------------------------
        # axis
        axis.points = axis.points[::-1]
        # data
        for channel in self.channels:
            values = channel.values
            # transpose so the axis of interest is last
            transpose_order = range(len(values.shape))
            # replace axis_index with zero
            transpose_order = [len(values.shape) - 1 if i ==
                               axis_index else i for i in transpose_order]
            transpose_order[len(values.shape) - 1] = axis_index
            values = values.transpose(transpose_order)
            values = values[..., ::-1]
            # transpose out
            values = values.transpose(transpose_order)
            channel.values = values

    def get_nadir(self, channel=0):
        """
        Get the coordinates in units of the minimum in a channel.

        Parameters
        ----------
        channel : int or str (optional)
            Channel. Default is 0.

        Returns
        -------
        list of numbers
            Coordinates in units for each axis.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # get indicies
        arr = channel.values
        idxs = np.unravel_index(arr.argmin(), arr.shape)
        # finish
        return [a.points[i] for a, i in zip(self.axes, idxs)]

    def get_zenith(self, channel=0):
        """
        Get the coordinates in units of the maximum in a channel.

        Parameters
        ----------
        channel : int or str (optional)
            Channel. Default is 0.

        Returns
        -------
        list of numbers
            Coordinates in units for each axis.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # get indicies
        arr = channel.values
        idxs = np.unravel_index(arr.argmax(), arr.shape)
        # finish
        return [a.points[i] for a, i in zip(self.axes, idxs)]

    def heal(self, channel=0, method='linear', fill_value=np.nan,
             verbose=True):
        """
        Remove nans from channel using interpolation.

        Parameters
        ----------
        channel : int or str (optional)
            Channel to heal. Default is 0.
        method : {'linear', 'nearest', 'cubic'} (optional)
            The interpolation method. Note that cubic interpolation is only
            possible for 1D and 2D data. See `griddata <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_
            for more information. Default is linear.
        fill_value : number-like (optional)
            The value written to pixels that cannot be filled by interpolation.
            Default is nan.
        verbose : bool (optional)
            Toggle talkback. Default is True.


        .. note:: Healing may take several minutes for large datasets. Interpolation time goes as nearest, linear, then cubic.


        """
        timer = wt_kit.Timer(verbose=False)
        with timer:
            # channel
            if isinstance(channel, int):
                channel_index = channel
            elif isinstance(channel, string_type):
                channel_index = self.channel_names.index(channel)
            else:
                print('channel type', type(channel), 'not valid')
            channel = self.channels[channel_index]
            values = self.channels[channel_index].values
            points = [axis.points for axis in self.axes]
            xi = tuple(np.meshgrid(*points, indexing='ij'))
            # 'undo' gridding
            arr = np.zeros((len(self.axes) + 1, values.size))
            for i in range(len(self.axes)):
                arr[i] = xi[i].flatten()
            arr[-1] = values.flatten()
            # remove nans
            arr = arr[:, ~np.isnan(arr).any(axis=0)]
            # grid data wants tuples
            tup = tuple([arr[i] for i in range(len(arr) - 1)])
            # grid data
            out = griddata(tup, arr[-1], xi, method=method, fill_value=fill_value)
            self.channels[channel_index].values = out
            self.channels[channel_index]._update()
        # print
        if verbose:
            print('channel {0} healed in {1} seconds'.format(
                channel.name, np.around(timer.interval, decimals=3)))

    @property
    def info(self):
        info = collections.OrderedDict()
        info['name'] = self.name
        info['id'] = id(self)
        info['axes'] = self.axis_names
        info['channels'] = self.channel_names
        info['shape'] = self.shape
        info['version'] = self.__version__
        return info

    def level(self, channel, axis, npts, verbose=True):
        """
        For a channel, subtract the average value of several points at the edge
        of a given axis.

        Parameters
        ----------
        channel : int or str
            Channel to level.
        axis : int or str
            Axis to level along.
        npts : int
            Number of points to average for each slice. Positive numbers
            take leading points and negative numbers take trailing points.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        # channel ---------------------------------------------------------------------------------
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # axis ------------------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            print('axis type', type(axis), 'not valid')
        # verify npts not zero --------------------------------------------------------------------
        npts = int(npts)
        if npts == 0:
            print('cannot level if no sampling range is specified')
            return
        # level -----------------------------------------------------------------------------------
        channel = self.channels[channel_index]
        values = channel.values
        # transpose so the axis of interest is last
        transpose_order = range(len(values.shape))
        # replace axis_index with zero
        transpose_order = [len(values.shape) - 1 if i ==
                           axis_index else i for i in transpose_order]
        transpose_order[len(values.shape) - 1] = axis_index
        values = values.transpose(transpose_order)
        # subtract
        for index in np.ndindex(values[..., 0].shape):
            if npts > 0:
                offset = np.nanmean(values[index][:npts])
            elif npts < 0:
                offset = np.nanmean(values[index][npts:])
            values[index] = values[index] - offset
        # transpose back
        values = values.transpose(transpose_order)
        # return
        channel.values = values
        channel.znull = 0.
        channel.zmax = np.nanmax(values)
        channel.zmin = np.nanmin(values)
        # print
        if verbose:
            axis = self.axes[axis_index]
            if npts > 0:
                points = axis.points[:npts]
            if npts < 0:
                points = axis.points[npts:]
            print('channel', channel.name, 'offset by', axis.name, 'between',
                  int(points.min()), 'and', int(points.max()), axis.units)

    def m(self, abs_data, channel=0, this_exp='TG',
          indices=None, m=None,
          bounds_error=True, verbose=True):
        """
        placeholder docstring because the existing one needs numpydoc formatting and is causing errors
        """
        docsatring = """
        normalize channel by absorptive effects given by absorption data object
            'abs_data'

        indices can be used to override default assignments for normalization
        m can be used to override default assignments for functional forms
         --> better to just add to the dictionary, though!
        assumes all abs fxns are independent of each axis, so we can normalize
            each axis individually
        need to be ready that:
            1.  not all axes that m accepts may be present--in this case,
                assumes abs of 0
        currently in alpha testing...so be careful
        known issues:
            --requires unique, integer (0<x<10) numbering for index
                identification
        """
        # TODO: numpydoc format docstring
        # exp_name: [i], [m_i]
        exp_types = {
            'TG': [['1', '2'],
                   [lambda a1: 10**-a1,
                    lambda a2: ((1 - 10**-a2) / (a2 * np.log(10)))**2
                    ]
                   ],
            'TA': [['2'],
                   [lambda a2: 1 - 10**(-a2)]
                   ]
        }
        # try to figure out the experiment or adopt the imported norm functions
        if this_exp in exp_types.keys():
            if indices is None:
                indices = exp_types[this_exp][0]
            m = exp_types[this_exp][1]
        elif m is not None and indices is not None:
            pass
        else:
            print('m-factors for this experiment have not yet been implemented')
            print('currently available experiments:')
            for key in exp_types.keys():
                print(key)
            print('no m-factor normalization was performed')
            return
        # find which axes have m-factor dependence; move to the inside and
        # operate
        m_axes = [axi for axi in self.axes if axi.units_kind == 'energy']
        # loop through 'indices' and find axis whole label_seeds contain indi
        for i, indi in enumerate(indices):
            t_order = range(len(self.axes))
            # find axes indices that have the correct label seed
            # and also belong to the list of axes under consideration
            # ni = [j for j in range(len(m_axes)) if indi in
            ni = [j for j in range(len(self.axes)) if indi in
                  self.axes[j].label_seed and self.axes[j] in m_axes]
            # m_axes[j].label_seed]
            if verbose:
                print(ni)
            # there should never be more than one axis that agrees
            if len(ni) > 1:
                raise ValueError()
            elif len(ni) > 0:
                ni = ni[0]
                axi = self.axes[ni]
                mi = m[i]
                # move index of interest to inside
                t_order.pop(ni)
                t_order.append(ni)
                if verbose:
                    print(t_order)
                self.transpose(axes=t_order, verbose=verbose)
                # evaluate ai ---------------------------------------------------------------------
                abs_data.axes[0].convert(axi.units)
                Ei = abs_data.axes[0].points
                Ai = interp1d(Ei, abs_data.channels[0].values,
                              bounds_error=bounds_error)
                ai = Ai(axi.points)
                Mi = mi(ai)
                # apply Mi to channel -------------------------------------------------------------
                self.channels[i].values /= Mi
                # invert back out of the transpose
                t_inv = [t_order.index(j) for j in range(len(t_order))]
                if verbose:
                    print(t_inv)
                self.transpose(axes=t_inv, verbose=verbose)
            else:
                print('{0} label_seed not found'.format(indi))
        return

    def map_axis(self, axis, points, input_units='same', verbose=True):
        """
        Map points of an axis to new points using linear interpolation. Out-
        of-bounds points are written nan.

        Parameters
        ----------
        axis : int or str
            The axis to map onto.
        points : 1D array-like
            The new points.
        input_units : str (optional)
            The units of the new points. Default is same, which assumes
            the new points have the same units as the axis.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        # get axis index --------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            print('axis type', type(axis), 'not valid')
        axis = self.axes[axis_index]
        # transform points to axis units ----------------------------------------------------------
        if input_units == 'same':
            pass
        else:
            points = wt_units.converter(points, input_units, axis.units)
        # points must be ascending ----------------------------------------------------------------
        flipped = np.zeros(len(self.axes), dtype=np.bool)
        for i in range(len(self.axes)):
            if self.axes[i].points[0] > self.axes[i].points[-1]:
                self.flip(i)
                flipped[i] = True
        # interpn data ----------------------------------------------------------------------------
        old_points = [a.points for a in self.axes]
        new_points = [a.points if a is not axis else points for a in self.axes]
        if len(self.axes) == 1:
            for channel in self.channels:
                function = scipy.interpolate.interp1d(self.axes[0].points, channel.values)
                channel.values = function(new_points[0])
        else:
            xi = tuple(np.meshgrid(*new_points, indexing='ij'))
            for channel in self.channels:
                values = channel.values
                channel.values = scipy.interpolate.interpn(old_points, values, xi,
                                                           method='linear',
                                                           bounds_error=False,
                                                           fill_value=np.nan)
        # cleanup ---------------------------------------------------------------------------------
        for i in range(len(self.axes)):
            if not i == axis_index:
                if flipped[i]:
                    self.flip(i)
        axis.points = points
        self._update()

    def normalize(self, channel=0, axis=None):
        """
        Normalize data in given channel so that null=0 and zmax=1.

        Parameters
        ----------
        channel : str or int (optional)
            Channel to normalize. Default is 0.
        axis : str, int, or 1D list-like of str and int or None
            Axis/axes to normalize against. If None, normalizes by the entire
            dataset. Default is None.
        """
        # process channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # process axes

        def process(i):
            if isinstance(channel, string_type):
                return self.axis_names.index(i)
            else:
                return int(i)
        if axis is not None:
            if not hasattr(axis, '__contains__'):  # NOT list, tuple or similar
                axis = [axis]
            axis = [process(i) for i in axis]
        # call normalize on channel
        channel.normalize(axis=axis)

    def offset(self, points, offsets, along, offset_axis,
               units='same', offset_units='same', mode='valid',
               method='linear', verbose=True):
        """
        Offset one axis based on another axis' values. Useful for correcting
        instrumental artifacts such as zerotune.

        Parameters
        ----------
        points : 1D array-like
            Points.
        offsets : 1D array-like
            Offsets.
        along : str or int
            Axis that points array lies along.
        offset_axis : str or int
            Axis to offset using offsets.
        units : str (optional)
            Units of points array.
        offset_units : str (optional)
            Units of offsets aray.
        mode : {'valid', 'full', 'old'} (optional)
            Define how far the new axis will extend. Points outside of valid
            interpolation range will be written nan.
        method : {'linear', 'nearest', 'cubic'} (optional)
            The interpolation method. Note that cubic interpolation is only
            possible for 1D and 2D data. See `griddata <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_
            for more information. Default is linear.
        verbose : bool (optional)
            Toggle talkback. Default is True.

            >>> points  # an array of w1 points
            >>> offsets  # an array of d1 corrections
            >>> data.offset(points, offsets, 'w1', 'd1')

        """

        # axis ------------------------------------------------------------------------------------

        if isinstance(along, int):
            axis_index = along
        elif isinstance(along, string_type):
            axis_index = self.axis_names.index(along)
        else:
            print('axis type', type(along), 'not valid')
        axis = self.axes[axis_index]

        # values & points -------------------------------------------------------------------------

        # get values, points, units
        if units == 'same':
            input_units = axis.units
        else:
            input_units = units

        # check offsets is 1D or 0D
        if len(offsets.shape) == 1:
            pass
        else:
            raise RuntimeError('values must be 1D or 0D in offset!')

        # check if units is compatible, convert
        dictionary = getattr(wt_units, axis.units_kind)
        if input_units in dictionary.keys():
            pass
        else:
            raise RuntimeError('units incompatible in offset!')
        points = wt_units.converter(points, input_units, axis.units)

        # create correction array
        function = interp1d(points, offsets, bounds_error=False)
        corrections = function(axis.points)

        # remove nans
        finite_indicies = np.where(np.isfinite(corrections))[0]
        left_pad_width = finite_indicies[0]
        right_pad_width = len(corrections) - finite_indicies[-1] - 1
        corrections = np.pad(corrections[np.isfinite(corrections)],
                             (int(left_pad_width), int(right_pad_width)), mode='edge')

        # do correction ---------------------------------------------------------------------------

        # transpose so axis is last
        transpose_order = np.arange(len(self.axes))
        transpose_order[axis_index] = len(self.axes) - 1
        transpose_order[-1] = axis_index
        self.transpose(transpose_order, verbose=False)

        # get offset axis index
        if isinstance(offset_axis, int):
            offset_axis_index = offset_axis
        elif isinstance(offset_axis, string_type):
            offset_axis_index = self.axis_names.index(offset_axis)
        else:
            print('offset_axis type', type(offset_axis), 'not valid')

        # new points
        new_points = [a.points for a in self.axes]
        old_offset_axis_points = self.axes[offset_axis_index].points
        spacing = abs((old_offset_axis_points.max() - old_offset_axis_points.min()) /
                      float(len(old_offset_axis_points)))
        if mode == 'old':
            new_offset_axis_points = old_offset_axis_points
        elif mode == 'valid':
            _max = old_offset_axis_points.max() + corrections.min()
            _min = old_offset_axis_points.min() + corrections.max()
            n = int(abs(np.ceil((_max - _min) / spacing)))
            new_offset_axis_points = np.linspace(_min, _max, n)
        elif mode == 'full':
            _max = old_offset_axis_points.max() + corrections.max()
            _min = old_offset_axis_points.min() + corrections.min()
            n = np.ceil((_max - _min) / spacing)
            new_offset_axis_points = np.linspace(_min, _max, n)
        new_points[offset_axis_index] = new_offset_axis_points
        new_xi = tuple(np.meshgrid(*new_points, indexing='ij'))

        xi = tuple(np.meshgrid(*[a.points for a in self.axes], indexing='ij'))
        for channel in self.channels:

            # 'undo' gridding
            arr = np.zeros((len(self.axes) + 1, channel.values.size))
            for i in range(len(self.axes)):
                arr[i] = xi[i].flatten()
            arr[-1] = channel.values.flatten()

            # do corrections
            corrections = list(corrections)
            corrections = corrections * int((len(arr[0]) / len(corrections)))
            arr[offset_axis_index] += corrections

            # grid data
            tup = tuple([arr[i] for i in range(len(arr) - 1)])
            # note that rescale is crucial in this operation
            out = griddata(tup, arr[-1], new_xi, method=method,
                           fill_value=np.nan, rescale=True)
            channel.values = out
            channel._update()

        self.axes[offset_axis_index].points = new_offset_axis_points

        # transpose out
        self.transpose(transpose_order, verbose=False)
        self._update()

    def remove_channel(self, channel):
        """
        Remove channel from data.

        Parameters
        ----------
        channel : int (index) or str (name)
            Channel to remove.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        # remove
        self.channels.pop(channel_index)
        # finish
        self._update()

    def revert(self):
        """
        Revert this data object back to its original state.
        """
        for attribute_name in dir(self):
            if attribute_name not in ['_original'] + wt_kit.get_methods(self):
                # if attribute does not exist in original, delete it
                try:
                    original_attribute = getattr(self._original, attribute_name)
                    setattr(self, attribute_name, original_attribute)
                except AttributeError:
                    delattr(self, attribute_name)
        self._update()

    def save(self, filepath=None, verbose=True):
        """
        Save using the `pickle <https://docs.python.org/2/library/pickle.html>`_
        module.

        Parameters
        ----------
        filepath : str (optional)
            The savepath. '.p' extension must be included. If not defined,
            the pickle will be saved in the current working directory with a
            timestamp.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        str
            The filepath of the saved pickle.

        See Also
        --------
        from_pickle
            Generate a data object from a saved pickle.
        """
        # get filepath
        if not filepath:
            chdir = os.getcwd()
            timestamp = wt_kit.get_timestamp()
            filepath = os.path.join(chdir, timestamp + ' data.p')
        # save
        pickle.dump(self, open(filepath, 'wb'))
        # return
        if verbose:
            print('data saved at', filepath)
        return filepath

    def scale(self, channel=0, kind='amplitude', verbose=True):
        """
        Scale a channel.

        Parameters
        ----------
        channel : int or str (optional)
            The channel to scale. Default is 0.
        kind : {'amplitude', 'log', 'invert'} (optional)
            The scaling operation to perform.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # do scaling
        if kind in ['amp', 'amplitude']:
            channel_data = channel.values
            channel_data_abs = np.sqrt(np.abs(channel_data))
            factor = np.ones(channel_data.shape)
            factor[channel_data < 0] = -1
            channel_data_out = channel_data_abs * factor
            channel.values = channel_data_out
        if kind in ['log']:
            channel.values = np.log10(channel.values)
        if kind in ['invert']:
            channel.values *= -1.
        channel._update()

    def share_nans(self):
        """
        Share not-a-numbers between all channels. If any channel is nan at a
        given index, all channels will be nan at that index after this
        operation.

        Uses the share_nans method found in wt.kit.
        """
        arrs = [c.values for c in self.channels]
        outs = wt_kit.share_nans(arrs)
        for c, a, in zip(self.channels, outs):
            c.values = a

    def smooth(self, factors, channel=None, verbose=True):
        """
        Smooth a channel using an n-dimenional `kaiser window <https://en.wikipedia.org/wiki/Kaiser_window>`_.

        Parameters
        ----------
        factors : int or list of int
            The smoothing factor. You may provide a list of smoothing factors
            for each axis.
        channel : int or str or None (optional)
            The channel to smooth. If None, all channels will be smoothed.
            Default is None.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        # get factors -----------------------------------------------------------------------------

        if isinstance(factors, list):
            pass
        else:
            dummy = np.zeros(len(self.axes))
            dummy[::] = factors
            factors = list(dummy)
        # get channels ----------------------------------------------------------------------------
        if channel is None:
            channels = self.channels
        else:
            if isinstance(channel, int):
                channel_index = channel
            elif isinstance(channel, string_type):
                channel_index = self.channel_names.index(channel)
            else:
                print('channel type', type(channel), 'not valid')
            channels = [self.channels[channel_index]]
        # smooth ----------------------------------------------------------------------------------
        for channel in channels:
            values = channel.values
            for axis_index in range(len(factors)):
                factor = factors[axis_index]
                # transpose so the axis of interest is last
                transpose_order = range(len(values.shape))
                # replace axis_index with zero
                transpose_order = [len(values.shape) - 1 if i ==
                                   axis_index else i for i in transpose_order]
                transpose_order[len(values.shape) - 1] = axis_index
                values = values.transpose(transpose_order)
                # get kaiser window
                beta = 5.0
                w = np.kaiser(2 * factor + 1, beta)
                # for all slices...
                for index in np.ndindex(values[..., 0].shape):
                    current_slice = values[index]
                    temp_slice = np.pad(current_slice, int(factor), mode=str('edge'))
                    values[index] = np.convolve(temp_slice, w / w.sum(), mode=str('valid'))
                # transpose out
                values = values.transpose(transpose_order)
            # return array to channel object
            channel.values = values
        if verbose:
            print('smoothed data')

    def split(self, axis, positions, units='same',
              direction='below', verbose=True):
        """
        Split the data object along a given axis, in units.

        Parameters
        ----------
        axis : int or str
            The axis to split along.
        positions : number-type or 1D array-type
            The position(s) to split at, in units. If a non-exact position is
            given, the closest valid axis position will be used.
        units : str (optional)
            The units of the given positions. Default is same, which assumes
            input units are identical to axis units.
        direction : {'below', 'above'} (optional)
            Choose which group of data the points at positions remains with.
            Consider points [0, 1, 2, 3, 4, 5] and positions [3]. If direction
            is above the returned objects are [0, 1, 2] and [3, 4, 5]. If
            direction is below the returned objects are [0, 1, 2, 3] and
            [4, 5]. Default is below.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        list
            A list of data objects.

        See Also
        --------
        chop
            Divide the dataset into its lower-dimensionality components.
        collapse
            Collapse the dataset along one axis.
        """
        # axis ------------------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            print('axis type', type(axis), 'not valid')
        axis = self.axes[axis_index]
        # indicies --------------------------------------------------------------------------------
        # positions must be iterable and should be a numpy array
        if type(positions) in [int, float]:
            positions = [positions]
        positions = np.array(positions)
        # positions should be in the data units
        if not units == 'same':
            positions = wt_units.converter(positions, units, axis.units)
        # get indicies of split
        indicies = []
        for position in positions:
            idx = np.argmin(abs(axis.points - position))
            indicies.append(idx)
        indicies.sort()
        # indicies must be unique
        if len(indicies) == len(set(indicies)):
            pass
        else:
            print('some of your positions are too close together to split!')
            indicies = list(set(indicies))
        # set direction according to units
        if axis.points[-1] < axis.points[0]:
            directions = ['above', 'below']
            direction = [i for i in directions if i is not direction][0]
        if direction == 'above':
            indicies = [i - 1 for i in indicies]
        # process ---------------------------------------------------------------------------------
        outs = []
        start = 0
        stop = -1
        for i in range(len(indicies) + 1):
            # get start and stop
            start = stop + 1  # previous value
            if i == len(indicies):
                stop = len(axis.points)
            else:
                stop = indicies[i]
            # new data object prepare
            new_data = self.copy()
            # axis of interest will be FIRST
            transpose_order = range(len(new_data.axes))
            # replace axis_index with zero
            transpose_order = [0 if i == axis_index else i for i in transpose_order]
            transpose_order[0] = axis_index
            new_data.transpose(transpose_order, verbose=False)
            # axis
            new_data.axes[0].points = new_data.axes[0].points[start:stop]
            # channels
            for channel in new_data.channels:
                channel.values = channel.values[start:stop]
            # transpose out
            new_data.transpose(transpose_order, verbose=False)
            outs.append(new_data)
        # post process ----------------------------------------------------------------------------
        if verbose:
            print('split data into {0} pieces along {1}:'.format(len(indicies) + 1, axis.name))
            for i in range(len(outs)):
                new_data = outs[i]
                new_axis = new_data.axes[axis_index]
                print('  {0} : {1} to {2} {3} (length {4})'.format(
                    i, new_axis.points[0], new_axis.points[-1], new_axis.units, len(new_axis.points)))
        # deal with cases where only one element is left
        for new_data in outs:
            if len(new_data.axes[axis_index].points) == 1:
                # remove axis
                new_data.axis_names.pop(axis_index)
                axis = new_data.axes.pop(axis_index)
                new_data.constants.append(axis)
                # reshape channels
                shape = [i for i in new_data.channels[0].values.shape if not i == 1]
                for channel in new_data.channels:
                    channel.values.shape = shape
                new_data.shape = shape
        return outs

    def subtract(self, subtrahend, channel=0, subtrahend_channel=0):
        """
        Subtract a given channel by another data object. Subtrahend smay be self.
        All axes in divisor must be contained in self.

        Parameters
        ----------
        subtrahend : data
            The data being subtracted by. Can be self.
        channel : int or str
            The channel to subtract into. The result will be written into this
            channel.
        subtrahend_channel : int or str
            The channel in the subtrahend object to use.
        """
        subtrahend = subtrahend.copy()
        # map points
        for name in subtrahend.axis_names:
            if name in self.axis_names:
                axis = getattr(self, name)
                subtrahend_axis = getattr(subtrahend, name)
                subtrahend_axis.convert(axis.units)
                subtrahend.map_axis(name, axis.points)
            else:
                raise RuntimeError('all axes in divisor must be contained in self')
        # divide
        # transpose so axes of divisor are last (in order)
        axis_indicies = [self.axis_names.index(name) for name in subtrahend.axis_names]
        axis_indicies.reverse()
        transpose_order = range(len(self.axes))
        for i in range(len(axis_indicies)):
            ai = axis_indicies[i]
            ri = range(len(self.axes))[-(i + 1)]
            transpose_order[ri], transpose_order[ai] = transpose_order[ai], transpose_order[ri]
        self.transpose(transpose_order, verbose=False)
        # get own channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = self.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = self.channels[channel_index]
        # get subtrahend channel
        if isinstance(subtrahend_channel, int):
            subtrahend_channel_index = subtrahend_channel
        elif isinstance(subtrahend_channel, string_type):
            subtrahend_channel_index = subtrahend.channel_names.index(subtrahend_channel)
        else:
            print('divisor channel type', type(channel), 'not valid')
        subtrahend_channel = subtrahend.channels[subtrahend_channel_index]
        # do division
        channel.values -= subtrahend_channel.values
        channel._update()
        # transpose out
        self.transpose(transpose_order, verbose=False)

    def trim(self, channel, **kwargs):
        """ Wrapper method for ``Channel.trim``.

        Parameters
        ----------
        channel : int or str
            The channel index (or name) to trim.
        """
        # channel
        if type(channel) in [int, float]:
            channel = self.channels[channel]
        elif isinstance(channel, string_type):
            index = self.channel_names.index(channel)
            channel = self.channels[index]
        # neighborhood
        inputs = {}
        neighborhood = [0] * self.dimensionality
        for key, value in kwargs.items():
            if key in self.axis_names:
                index = self.axis_names.index(key)
                neighborhood[index] = value
            elif key in ['method', 'factor', 'replace', 'verbose']:
                inputs[key] = value
            else:
                raise KeyError(
                    'Keyword arguments to trim must be either an axis name or one of {method, factor, replace, verbose}')
        # call trim
        return channel.trim(neighborhood=neighborhood, **inputs)

    def transform(self, transform=None):
        """ Transforms the dataset using arbitrary coordinates, then regrids the data

        Parameters
        ----------
        transform: str
            The tranformation to perform. Str must use axis names. Only handles
            two axes at a time.
        """
        # TODO: interpret strings into a function
        # TODO: use tranform string to make new axis lables
        # TODO: Expand to larger than 2D tranforms (dream feature)
        # use np.griddata
        # find code used to deal with constants
        # possibly use to plot vs constants?
        print('not yet implemented.')

    def transpose(self, axes=None, verbose=True):
        """ Transpose the dataset.

        Parameters
        ----------
        axes : None or list of int (optional)
            The permutation to perform. If None axes are simply reversed.
            Default is None.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        if axes is not None:
            pass
        else:
            axes = range(len(self.channels[0].values.shape))[::-1]
        self.axes = [self.axes[i] for i in axes]
        self.axis_names = [self.axis_names[i] for i in axes]
        for channel in self.channels:
            channel.values = np.transpose(channel.values, axes=axes)
        if verbose:
            print('data transposed to', self.axis_names)
        self.shape = self.channels[0].values.shape

    def zoom(self, factor, order=1, verbose=True):
        """ Zoom the data array using spline interpolation of the requested order.

        The number of points along each axis is increased by factor.
        See `scipy.ndimage.interpolation.zoom <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.interpolation.zoom.html>`_
        for more info.

        Parameters
        ----------
        factor : float
            The number of points along each axis will increase by this factor.
        order : int (optional)
            The order of the spline used to interpolate onto new points.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        import scipy.ndimage
        # axes
        for axis in self.axes:
            axis.points = scipy.ndimage.interpolation.zoom(axis.points,
                                                           factor,
                                                           order=order)
        # channels
        for channel in self.channels:
            channel.values = scipy.ndimage.interpolation.zoom(channel.values,
                                                              factor,
                                                              order=order)
        # return
        if verbose:
            print('data zoomed to new shape:', self.channels[0].values.shape)


# --- data creation methods -----------------------------------------------------------------------


def from_Cary50(filepath, verbose=True):
    """ Create a data object from a Cary 50 UV VIS absorbance file.

    Parameters
    ----------
    filepath : string
        Path to Tensor27 output file (.dpt).
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object.
    """
    # check filepath
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'csv':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'csv')
    # import array
    lines = []
    with open(filepath, 'r') as f:
        header = f.readline()
        units = f.readline()
        while True:
            line = f.readline()
            if line == '\n' or line == '':
                break
            else:
                clean = line[:-2]  # lines end with ',/n'
                lines.append(np.fromstring(clean, sep=','))
    header = header.split(',')
    arr = np.array(lines).T
    # chew through all scans
    datas = []
    indicies = np.arange(len(header) // 2) * 2
    for i in indicies:
        axis = Axis(arr[i], 'nm', name='wm')
        signal = Channel(arr[i + 1], name='absorbance', label='absorbance', signed=False)
        data = Data([axis], [signal], source='Cary 50', name=header[i])
        datas.append(data)
    # finish
    if verbose:
        print('{0} data objects successfully created from Cary 50 file:'.format(len(indicies)))
        for i, data in enumerate(datas):
            print('  {0}: {1}'.format(i, data.name))
    return datas


def from_text(filepath, name=None, verbose=True):
    """ Create a data object from plaintext tab deliminated file

    Expects one energy and one intensity value.

    Parameters
    ----------
    filepath : string, list of strings, or array of strings
        Path to .txt file.
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object(s).
    """
    if isinstance(filepath, type([])) or isinstance(filepath, type(np.array([]))):
        return [from_rRaman(f) for f in filepath]

    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    if not filepath.endswith('txt') and not filepath.endswith('dpt'):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt or dpt')
    # import array
    lines = []
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if line == '\n' or line == '':
                break
            else:
                clean = line[:-2]  # lines end with ',/n'
                lines.append(np.fromstring(clean, sep=' '))

    arr = np.array(lines).T
    # chew through all scans
    indicies = np.arange(1)
    for i in indicies:
        axis = Axis(arr[i], 'wn', name='wm')
        signal = Channel(arr[i + 1], name='signal', label='counts', signed=False)
        if name:
            data = Data([axis], [signal], source='Brunold rRaman', name=name)
        else:
            name = filepath.split('//')[-1].split('.')[0]
            data = Data([axis], [signal], source='Brunold rRaman', name=name)
    # finish
    if verbose:
        print('{0} data objects successfully created from file:'.format(len(indicies)))
        for i, data in enumerate(datas):
            print('  {0}: {1}'.format(i, data.name))
    return data


def from_rRaman(*args, **kwargs):
    return from_text(*args, **kwargs)


def from_COLORS(
        filepaths,
        znull=None,
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
    """
    filepaths may be string or list
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
        channels['ai0'] = Channel(None, 'V', file_idx=16, name='ai0', label_seed=['0'])
        channels['ai1'] = Channel(None, 'V', file_idx=17, name='ai1', label_seed=['1'])
        channels['ai2'] = Channel(None, 'V', file_idx=18, name='ai2', label_seed=['2'])
        channels['ai3'] = Channel(None, 'V', file_idx=19, name='ai3', label_seed=['3'])
        channels['ai4'] = Channel(None, 'V', file_idx=20, name='ai4', label_seed=['4'])
        channels['mc'] = Channel(None, None, file_idx=21, name='array', label_seed=['a'])
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
        channels['ai0'] = Channel(None, 'V', file_idx=8, name='ai0', label_seed=['0'])
        channels['ai1'] = Channel(None, 'V', file_idx=9, name='ai1', label_seed=['1'])
        channels['ai2'] = Channel(None, 'V', file_idx=10, name='ai2', label_seed=['2'])
        channels['ai3'] = Channel(None, 'V', file_idx=11, name='ai3', label_seed=['3'])
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
        channels['ai0'] = Channel(None, 'V', file_idx=10, name='ai0', label_seed=['0'])
        channels['ai1'] = Channel(None, 'V', file_idx=11, name='ai1', label_seed=['1'])
        channels['ai2'] = Channel(None, 'V', file_idx=12, name='ai2', label_seed=['2'])
        channels['ai3'] = Channel(None, 'V', file_idx=13, name='ai3', label_seed=['3'])

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
            if debug:
                print(key)

    # create data object --------------------------------------------------------------------------

    data = Data(list(scanned), list(channels.values()), list(constant), znull)

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


def from_JASCO(filepath, name=None, kind='absorbance', verbose=True):
    """ Create a data object from a JASCO UV-VIS NIR file.

    Parameters
    ----------
    filepath : string
        Path to JASCO output file (.txt).
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    kind : {'absorbance', 'diffuse reflectance'} (optional)
        Kind of data taken. Default is absorbance.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object.
    """
    # check filepath
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'txt':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt')
    # import array
    arr = np.genfromtxt(filepath, skip_header=18).T
    # name
    if not name:
        name = filepath
    # construct data
    axis = Axis(arr[0], 'nm', name='wm')
    signal = Channel(arr[1], kind, signed=False)
    data = Data([axis], [signal], source='JASCO', name=name)
    # finish
    if verbose:
        print(data)
    return data


def from_KENT(filepaths, znull=None, name=None, ignore=['wm'], use_norm=False,
              delay_tolerance=0.1, frequency_tolerance=0.5, verbose=True):
    """
    filepaths may be string or list
    """
    # do we have a list of files or just one file? ------------------------------------------------
    if isinstance(filepaths, list):
        file_example = filepaths[0]
    else:
        file_example = filepaths
        filepaths = [filepaths]
    # define format of dat file -------------------------------------------------------------------
    # axes
    axes = collections.OrderedDict()
    axes['w1'] = Axis(None, 'wn', tolerance=frequency_tolerance,
                      file_idx=0, name='w1', label_seed=['1'])
    axes['w2'] = Axis(None, 'wn', tolerance=frequency_tolerance,
                      file_idx=1, name='w2', label_seed=['2'])
    axes['wm'] = Axis(None, 'wn', tolerance=frequency_tolerance,
                      file_idx=2, name='wm', label_seed=['m'])
    axes['d1'] = Axis(None, 'ps', tolerance=delay_tolerance,
                      file_idx=3, name='d1', label_seed=['1'])
    axes['d2'] = Axis(None, 'ps', tolerance=delay_tolerance,
                      file_idx=4, name='d2', label_seed=['2'])
    # channels
    channels = collections.OrderedDict()
    channels['signal'] = Channel(None, 'V', file_idx=5, name='signal', label_seed=['0'])
    channels['OPA2'] = Channel(None, 'V', file_idx=6, name='OPA2', label_seed=['1'])
    channels['OPA1'] = Channel(None, 'V', file_idx=7, name='OPA1', label_seed=['2'])
    # import full array ---------------------------------------------------------------------------
    for i in range(len(filepaths)):
        dat = np.genfromtxt(filepaths[i]).T
        if verbose:
            print('file imported:', dat.shape)
        if i == 0:
            arr = dat
        else:
            arr = np.append(arr, dat, axis=1)
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
        tol = max(tol, 1e-4)
        axis.points = np.linspace(min(xs) + tol, max(xs) - tol, num=len(xs))
    # grid data -----------------------------------------------------------------------------------
    # May not need, but doesnt hurt to include
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
        xi = tuple(np.meshgrid(*[axis.points for axis in scanned], indexing='ij'))
        for key in channels.keys():
            channel = channels[key]
            zi = arr[channel.file_idx]
            fill_value = min(zi)
            grid_i = griddata(points, zi, xi,
                              method='linear', fill_value=fill_value)
            channel.give_values(grid_i)
            if debug:
                print(key)
    # create data object --------------------------------------------------------------------------
    data = Data(list(scanned), list(channels.values()), list(constant), znull)
    for axis in data.axes:
        axis.get_label()
    for axis in data.constants:
        axis.get_label()
    # add extra stuff to data object --------------------------------------------------------------
    data.source = filepaths
    if not name:
        name = wt_kit.filename_parse(file_example)[1]
    data.name = name
    # normalize the data --------------------------------------------------------------------------
    if use_norm:
        # normalize the OPAs
        OPA1 = data.channels[2].values / data.axes[0].points
        OPA2 = data.channels[1].values / data.axes[1].points
        # Signal normalization
        #data_norm = data.channels[0].values*data.axes[0].points*data.axes[1].points/(OPA1*OPA2)
        data_norm = data.channels[0].values / (OPA1 * OPA2)  # I think this is correct.
        data.channels[0].values = data_norm
        data.channels[0].zmax = data_norm.max()
        data.channels[0].zmin = data_norm.min()
    # return --------------------------------------------------------------------------------------
    if verbose:
        print('data object succesfully created')
        print('axis names:', data.axis_names)
        print('values shape:', data.channels[0].values.shape)
    return data


def from_NISE(measure_object, name='simulation', ignore_constants=['A', 'p'],
              flip_delays=True, verbose=True):
    try:
        import NISE
    except BaseException:
        print('NISE is required to import scans, returning')
        return
    # axes ----------------------------------------------------------------------------------------
    NISE_axes = measure_object.scan_obj.axis_objs
    axes = []
    for NISE_axis in NISE_axes:
        axis_name = NISE_axis.pulse_var + str(NISE_axis.pulse_ind)
        points = NISE_axis.points
        units = NISE_axis.default_units
        label_seed = NISE_axis.also
        axis = Axis(points, units, name=axis_name, label_seed=label_seed)
        axes.append(axis)
    # constants -----------------------------------------------------------------------------------
    NISE_units = {'A': 'uJ per sq. cm',
                  's': 'FWHM',
                  'd': 'fs',
                  'w': 'wn',
                  'p': 'rad'}
    scan_object = measure_object.scan_obj
    positions_array = scan_object.positions.T
    pulse_class = getattr(NISE.lib.pulse, scan_object.pulse_class_name)
    constants = []
    for idx in range(len(positions_array)):
        key = list(pulse_class.cols.keys())[list(pulse_class.cols.values()).index(idx)]
        axes_sametype = [NISE_axis for NISE_axis in NISE_axes if NISE_axis.pulse_var == key]
        # get values that were not scanned
        indicies_scanned = []
        for axis in axes_sametype:
            indicies_scanned.append(axis.also)
        vals = np.delete(positions_array[idx], [
                         item for sublist in indicies_scanned for item in sublist])
        # find values that are co-set
        equal = np.zeros((len(vals), len(vals)), dtype=bool)
        for i in range(len(vals)):  # test
            for j in range(len(vals)):  # against
                if vals[i] == vals[j]:
                    equal[i, j] = True
        # create constant Axis objects
        vals_accounted_for = np.zeros(len(vals), dtype=bool)
        while not all(vals_accounted_for):
            for i in range(len(vals)):
                if vals_accounted_for[i]:
                    pass
                else:
                    cname = key + str(i)
                    value = np.array(vals[i])
                    units = NISE_units[key]
                    label_seed = list(np.where(equal[i])[0])
                    for j in label_seed:
                        vals_accounted_for[j] = True
                    axis = Axis(value, units, name=cname, label_seed=label_seed)
                    if key not in ignore_constants:
                        constants.append(axis)
    # channels ------------------------------------------------------------------------------------
    zi = measure_object.pol
    channel = Channel(zi, 'au', label='amplitude', name='simulation')
    channels = [channel]
    # data object ---------------------------------------------------------------------------------
    if flip_delays:
        for lis in [axes, constants]:
            for axis in lis:
                if axis.units_kind == 'delay':
                    axis.points *= -1.
    data = Data(axes, channels, constants=constants, name=name, source='NISE')
    return data


def from_pickle(filepath, verbose=True):
    data = pickle.load(open(filepath, 'rb'))
    if hasattr(data, '__version__'):
        from . import __version__
        if data.__version__.split('.')[0] == __version__.split('.')[0]:   # major versions agree
            pass
        else:
            print('pickled data is from different major version - consider remaking:')
            print('  current:', __version__)
            print('  pickle:', data.__version__)
    else:
        print('this pickle was made before November 2015 - you MUST remake your data object')
        return
    if verbose:
        print('data opened from', filepath)
    return data


def from_PyCMDS(filepath, name=None,
                shots_processing_module='mean_and_std', verbose=True):
    """ Create a data object from a single PyCMDS output file.

    Parameters
    ----------
    filepath : str
        The file to load. Can accept .data, .fit, or .shots files.
    name : str or None (optional)
        The name to be applied to the new data object. If None, name is read
        from file.
    shots_processing_module : str (optional)
        The module used to process .shots files, if provided. Must be the name
        of a module in the shots_processing directory.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        A Data instance.
    """
    # header
    headers = wt_kit.read_headers(filepath)
    # name
    if name is None:  # name not given in method arguments
        data_name = headers['data name']
    else:
        data_name = name
    if data_name == '':  # name not given in PyCMDS
        data_name = headers['data origin']
    # array
    arr = np.genfromtxt(filepath).T
    # get axes
    axes = []
    for name, identity, units in zip(headers['axis names'],
                                     headers['axis identities'],
                                     headers['axis units']):
        points = np.array(headers[name + ' points'])
        # label seed (temporary implementation)
        try:
            index = headers['name'].index(name)
            label = headers['label'][index]
            label_seed = [label]
        except ValueError:
            label_seed = ['']
        # create axis
        kwargs = {'identity': identity}
        if 'D' in identity:
            kwargs['centers'] = headers[name + ' centers']
        axis = Axis(points, units, name=name, label_seed=label_seed, **kwargs)
        axes.append(axis)
    # get indicies arrays
    indicies = arr[:len(axes)].T
    indicies = indicies.astype('int64')
    # get interpolation toggles
    if 'axis interpolate' in headers.keys():
        interpolate_toggles = headers['axis interpolate']
    else:
        # old data files may not have interpolate toggles in headers
        # assume no interpolation, unless the axis is the array detector
        interpolate_toggles = [True if name == 'wa' else False for name in headers['axis names']]
    # get assorted remaining things
    shape = tuple([a.points.size for a in axes])
    tols = [wt_kit.closest_pair(a.points, give='distance') / 2. for a in axes]
    # prepare points for interpolation
    points_dict = collections.OrderedDict()
    for i, axis in enumerate(axes):
        # TODO: math and proper full recognition...
        axis_col_name = [name for name in headers['name'][::-1] if name in axis.identity][0]
        axis_index = headers['name'].index(axis_col_name)
        # shape array acording to recorded coordinates
        points = np.full(shape, np.nan)
        for j, idx in enumerate(indicies):
            lis = arr[axis_index]
            points[tuple(idx)] = lis[j]
        # convert array
        points = wt_units.converter(points, headers['units'][axis_index], axis.units)
        # take case of scan about center
        if axis.identity[0] == 'D':
            # transpose so this axis is first
            transpose_order = list(range(len(axes)))
            transpose_order.insert(0, transpose_order.pop(i))
            points = points.transpose(transpose_order)
            # subtract out centers
            centers = np.array(headers[axis.name + ' centers'])
            points -= centers
            # transpose out
            transpose_order = list(range(len(axes)))
            transpose_order.insert(i, transpose_order.pop(0))
            points = points.transpose(transpose_order)
        points = points.flatten()
        points_dict[axis.name] = points
        # check, coerce non-interpolated axes
        if not interpolate_toggles[i]:
            for j, idx in enumerate(indicies):
                actual = points[j]
                expected = axis.points[idx[i]]
                if abs(actual - expected) > tols[i]:
                    warnings.warn('at least one point exceded tolerance ' +
                                  'in axis {}'.format(axis.name))
                points[j] = expected
    all_points = tuple(points_dict.values())
    # prepare values for interpolation
    values_dict = collections.OrderedDict()
    for i, kind in enumerate(headers['kind']):
        if kind == 'channel':
            values_dict[headers['name'][i]] = arr[i]
    # create grid to interpolate onto
    if len(axes) == 1:
        meshgrid = tuple([axes[0].points])
    else:
        meshgrid = tuple(np.meshgrid(*[a.points for a in axes], indexing='ij'))
    if any(interpolate_toggles):
        # create channels through linear interpolation
        channels = []
        for i in range(len(arr)):
            if headers['kind'][i] == 'channel':
                # unpack
                units = headers['units'][i]
                signed = headers['channel signed'][len(channels)]
                name = headers['name'][i]
                label = headers['label'][i]
                # interpolate
                values = values_dict[name]
                zi = griddata(all_points, values, meshgrid, rescale=True,
                              method='linear', fill_value=np.nan)
                # assemble
                channel = Channel(zi, units, signed=signed, name=name, label=label)
                channels.append(channel)
    else:
        # if none of the axes are interpolated onto,
        # simply fill zis based on recorded axis index
        num_channels = headers['kind'].count('channel')
        channel_indicies = [i for i, kind in enumerate(headers['kind']) if kind == 'channel']
        zis = [np.full(shape, np.nan) for _ in range(num_channels)]
        # iterate through each row of the array, filling zis
        for i in range(len(arr[0])):
            idx = tuple(indicies[i])  # yes, this MUST be a tuple >:(
            for zi_index, arr_index in enumerate(channel_indicies):
                zis[zi_index][idx] = arr[arr_index, i]
        # assemble channels
        channels = []
        for zi_index, arr_index in zip(range(len(zis)), channel_indicies):
            zi = zis[zi_index]
            units = headers['units'][arr_index]
            signed = headers['channel signed'][zi_index]
            name = headers['name'][arr_index]
            label = headers['label'][arr_index]
            channel = Channel(zi, units, signed=signed, name=name, label=label)
            channels.append(channel)
    # get constants
    constants = []
    for name, identity in zip(headers['constant names'], headers['constant identities']):
        # TODO: handle PyCMDS constants
        pass
    # create data object
    data = Data(axes, channels, constants, name=data_name, source=filepath)
    # return
    if verbose:
        print('data object succesfully created')
        print('  axes:', data.axis_names)
        print('  shape:', data.shape)
    return data


def from_scope(filepath, name=None, verbose=True):
    # check filepath
    if os.path.isfile(filepath):
        if verbose:
            print('found the file!')
    else:
        raise wt_exceptions.FileNotFound('{0}'.format(filepath))
    # import
    skip_header = 14
    skip_footer = 1
    arr = np.genfromtxt(filepath, skip_header=skip_header,
                        skip_footer=skip_footer, delimiter='\t').T
    # construct data
    a = Axis(arr[0], 'nm', name='wm')
    c = Channel(arr[1], name='intensity', signed=False)
    data = Data([a], [c], source='scope', name=name)
    # finish
    return data


def from_shimadzu(filepath, name=None, verbose=True):

    # check filepath ------------------------------------------------------------------------------

    if os.path.isfile(filepath):
        if verbose:
            print('found the file!')
    else:
        print('Error: filepath does not yield a file')
        return

    # is the file suffix one that we expect?  warn if it is not!
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'txt':
        should_continue = raw_input(
            'Filetype is not recognized and may not be supported.  Continue (y/n)?')
        if should_continue == 'y':
            pass
        else:
            print('Aborting')
            return

    # import data ---------------------------------------------------------------------------------

    # now import file as a local var--18 lines are just txt and thus discarded
    data = np.genfromtxt(filepath, skip_header=2, delimiter=',').T

    # construct data
    x_axis = Axis(data[0], 'nm', name='wm')
    signal = Channel(data[1], 'sig', file_idx=1, signed=False)
    data = Data([x_axis], [signal], source='Shimadzu', name=name)

    # return --------------------------------------------------------------------------------------

    return data


def from_Tensor27(filepath, name=None, verbose=True):
    """ Create a data object from a Tensor27 FTIR file.

    Parameters
    ----------
    filepath : string
        Path to Tensor27 output file (.dpt).
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object.
    """
    # check filepath
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'dpt':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'dpt')
    # import array
    arr = np.genfromtxt(filepath, skip_header=0).T
    # name
    if not name:
        name = os.path.basename(filepath)
    # construct data
    axis = Axis(arr[0], 'wn', name='w')
    signal = Channel(arr[1], name='absorbance', label='absorbance', signed=False)
    data = Data([axis], [signal], source='Tensor 27', name=name)
    # finish
    if verbose:
        print('data object successfully created from Tensor 27 file')
    return data


def join(datas, method='first', verbose=True):
    """ Join a list of data objects together.

    For now datas must have identical dimensionalities (order and identity).

    Parameters
    ----------
    datas : list of data
        The list of data objects to join together.
    method : {'first', 'sum', 'max', 'min', 'mean'} (optional)
        The method for how overlapping points get treated. Default is first,
        meaning that the data object that appears first in data will take
        precedence.
    verbose : bool (optional)
        Toggle talkback. Default is True.

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
        for axis in data.axes:
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
        all_points = np.array([])
        step_sizes = []
        for data in datas:
            for axis in data.axes:
                if axis.name == axis_name:
                    all_points = np.concatenate([all_points, axis.points])
                    this_axis_min = np.nanmin(axis.points)
                    this_axis_max = np.nanmax(axis.points)
                    this_axis_number = float(axis.points.size)
                    step_size = (this_axis_max - this_axis_min) / this_axis_number
                    step_sizes.append(step_size)
        axis_min = np.nanmin(all_points)
        axis_max = np.nanmax(all_points)
        axis_step_size = min(step_sizes)
        axis_n_points = np.ceil((axis_max - axis_min) / axis_step_size)
        points = np.linspace(axis_min, axis_max, axis_n_points)
        axis_points.append(points)
    # map datas to new points
    for axis_index, axis_name in enumerate(axis_names):
        for data in datas:
            for axis in data.axes:
                if axis.name == axis_name:
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
            print('method', method, 'not recognized in join')
            return
        zis[np.isnan(full).all(axis=0)] = np.nan  # if all datas NaN, zis NaN
        channel = Channel(zis, 'V', znull=0.,
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
                channel.name, channel.zmin, channel.zmax, percent_nan))
    return out


# --- other ---------------------------------------------------------------------------------------


def discover_dimensions(arr, dimension_cols, verbose=True):
    """
    Discover the dimensions of array arr.
    Watches the indicies contained in dimension_cols. Returns dictionaries of
    axis objects [scanned, constant].
    Constant objects have their points object initialized. Scanned dictionary is
    in order of scanning (..., zi, yi, xi). Both dictionaries are condensed
    into coscanning / setting.
    """

    # sorry that this method is so convoluted and unreadable - blaise

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
    if debug:
        print(d_equal)

    # condense
    dims_unaccounted = list(range(len(dims)))
    dims_condensed = []
    while dims_unaccounted:
        if debug:
            print(dims_unaccounted)
        dim_current = dims_unaccounted[0]
        index = dims[dim_current][0]
        tolerance = [dims[dim_current][1]]
        units = dims[dim_current][2]
        key = [dims[dim_current][3]]
        dims_unaccounted.pop(0)
        indicies = list(range(len(dims_unaccounted)))
        indicies.reverse()
        if debug:
            print(indicies)
        for i in indicies:
            dim_check = dims_unaccounted[i]
            if d_equal[dim_check, dim_current]:
                tolerance.append(dims[dim_check][1])
                key.append(dims[dim_check][3])
                dims_unaccounted.pop(i)
        tolerance = max(tolerance)
        dims_condensed.append([index, tolerance, units, key])
    dims = dims_condensed
    if debug:
        print(dims)

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
