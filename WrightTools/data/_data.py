"""Central data class and associated."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import copy
import collections
import warnings
import pickle

import numpy as np

import scipy
from scipy.interpolate import griddata, interp1d

from .. import exceptions as wt_exceptions
from .. import kit as wt_kit
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


# string types
if sys.version[0] == '2':
    # recognize unicode and string types
    string_type = basestring  # noqa: F821
else:
    string_type = str  # newer versions of python don't have unicode type


__all__ = ['Axis', 'Channel', 'Data']


# --- classes -------------------------------------------------------------------------------------


class Axis:
    """Axis class."""

    def __init__(self, points, units, name, symbol_type=None, label_seed=[''], **kwargs):
        """Create an `Axis` object.

        Parameters
        ----------
        points : 1D array-like
            Axis points.
        units : string
            Axis units.
        name : string
            Axis name. Must be unique.
        symbol_type : string (optional)
            Axis symbol type. If None, symbol_type is automatically
            generated. Default is None.
        label_seed : list of strings
            Axis label subscripts. Default is [''].
        **kwargs
            Additional keyword arguments are added to the attrs dictionary
            and to the natural namespace of the object (if possible).
        """
        self.name = wt_kit.string2identifier(name)
        self.points = np.asarray(points)
        # units
        self.units = units
        self.units_kind = None
        for dic in wt_units.unit_dicts:
            if self.units in dic.keys():
                self.units_kind = dic['kind']
        # label
        self.label_seed = label_seed
        if symbol_type:
            self.symbol_type = symbol_type
        else:
            self.symbol_type = wt_units.get_default_symbol_type(self.units)
        self.get_label()
        # attrs
        self.attrs = kwargs
        for key, value in self.attrs.items():
            identifier = wt_kit.string2identifier(key)
            if not hasattr(self, identifier):
                setattr(self, identifier, value)

    def __repr__(self):
        """Return an unambiguous representation of the Axis.

        Returns
        -------
        string
            Representation.
        """
        return 'WrightTools.data.Axis object \'{0}\' at {1}'.format(self.name, str(id(self)))

    def convert(self, destination_units):
        """Convert axis to destination units.

        Parameters
        ----------
        destination_units : string
            Destination units.
        """
        self.points = wt_units.converter(self.points, self.units,
                                         destination_units)
        self.units = destination_units

    def get_label(self, show_units=True, points=False, decimals=2):
        """Get a LaTeX formatted label.

        Parameters
        ----------
        show_units : boolean (optional)
            Toggle showing units. Default is True.
        points : boolean (optional)
            Toggle showing points. Default is False.
        decimals : integer (optional)
            Number of decimals to show for numbers. Default is 2.
        """
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
        """Axis info dictionary."""
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
        """Axis constant flag."""
        try:
            len(self.points)
        except TypeError:
            return False
        finally:
            return True

    @property
    def label(self):  # noqa: D403
        """LaTeX formatted label."""
        return self.get_label()

    def max(self):
        """Axis max, ignoring nans."""
        return self.points.max()

    def min(self):
        """Axis min, ignoring nans."""
        return self.points.min()


class Channel:
    """Channel."""

    def __init__(self, values, name, units=None, null=None, signed=None,
                 label=None, label_seed=None, **kwargs):
        """Construct a channel object.

        Parameters
        ----------
        values : array-like
            Values.
        name : string
            Channel name.
        units : string (optional)
            Channel units. Default is None.
        null : number (optional)
            Channel null. Default is None (0).
        signed : booelan (optional)
            Channel signed flag. Default is None (guess).
        label : string.
            Label. Default is None.
        label_seed : list of strings
            Label seed. Default is None.
        **kwargs
            Additional keyword arguments are added to the attrs dictionary
            and to the natural namespace of the object (if possible).
        """
        self.name = wt_kit.string2identifier(name)
        self.label = label
        self.label_seed = label_seed
        self.units = units
        # values
        if values is not None:
            self.give_values(np.asarray(values), null, signed)
        else:
            self._null = null
            self.signed = signed
        # attrs
        self.attrs = kwargs
        for key, value in self.attrs.items():
            identifier = wt_kit.string2identifier(key)
            if not hasattr(self, identifier):
                setattr(self, identifier, value)

    def __repr__(self):
        """Retrun an unambiguous representation of the Channel.

        Returns
        -------
        string
            Representation.
        """
        return 'WrightTools.data.Channel object \'{0}\' at {1}'.format(self.name, str(id(self)))

    def _update(self):
        """Update channel."""
        message = '_update is no longer necessary, and will be removed in future versions'
        warnings.warn(message, wt_exceptions.VisibleDeprecationWarning)

    def clip(self, min=None, max=None, replace='nan'):
        """Clip (limit) the values in a channel.

        Parameters
        ----------
        min : number (optional)
            New channel minimum. Default is None.
        max : number (optional)
            New channel maximum. Default is None.
        replace : {'val', 'nan', 'mask'} (optional)
           Replace behavior. Default is nan.
        """
        # decide what min and max will actually be
        if max is None:
            max = self.max()
        if min is None:
            min = self.min()
        # replace values
        if replace == 'val':
            self.values.clip(min, max, out=self.values)
        elif replace == 'nan':
            self.values[self.values < min] = np.nan
            self.values[self.values > max] = np.nan
        elif replace == 'mask':
            self.values = np.ma.masked_outside(self.values, min, max, copy=False)
        else:
            print('replace not recognized in channel.clip')

    def give_values(self, values, null=None, signed=None):
        """Give values.

        Parameters
        ----------
        values : array-like
            Values.
        null : number (optional)
            Null. Default is None (0).
        signed : boolean (optional)
            Signed flag. Default is None (guess).
        """
        self.values = values
        # null
        if null is not None:
            self._null = null
        elif hasattr(self, '_null'):
            if self._null:
                pass
            else:
                self._null = 0.
        else:
            self._null = 0.
        # signed
        if signed is not None:
            self.signed = signed
        elif hasattr(self, 'signed'):
            if self.signed is None:
                if self.min() < self.null():
                    self.signed = True
                else:
                    self.signed = False
        else:
            if self.min() < self.null():
                self.signed = True
            else:
                self.signed = False

    @property
    def info(self):
        """Return Channel info dictionary."""
        info = collections.OrderedDict()
        info['name'] = self.name
        info['id'] = id(self)
        info['min'] = self.min()
        info['max'] = self.max()
        info['null'] = self.null()
        info['signed'] = self.signed
        return info

    def invert(self):
        """Invert channel values."""
        self.values = - self.values

    def mag(self):
        """Channel magnitude (maximum deviation from null)."""
        return self.major_extent

    @property
    def major_extent(self):
        """Maximum deviation from null."""
        return max((self.max() - self.null(), self.null() - self.min()))

    def max(self):
        """Maximum, ignorning nans."""
        return np.nanmax(self.values)

    def min(self):
        """Minimum, ignoring nans."""
        return np.nanmin(self.values)

    @property
    def minor_extent(self):
        """Minimum deviation from null."""
        return min((self.max() - self.null(), self.null() - self.min()))

    def normalize(self, axis=None):
        """Normalize a Channel, set `null` to 0 and the max to 1."""
        # process axis argument
        if axis is not None:
            if hasattr(axis, '__contains__'):  # list, tuple or similar
                axis = tuple((int(i) for i in axis))
            else:  # presumably a simple number
                axis = int(axis)
        # subtract off null
        self.values -= self.null()
        self._null = 0.
        # create dummy array
        dummy = self.values.copy()
        dummy[np.isnan(dummy)] = 0  # nans are propagated in np.amax
        if self.signed:
            dummy = np.absolute(dummy)
        # divide through by max
        self.values /= np.amax(dummy, axis=axis, keepdims=True)
        # finish

    def null(self):
        """Null value."""
        return self._null

    def trim(self, neighborhood, method='ztest', factor=3, replace='nan',
             verbose=True):
        """Remove outliers from the dataset.

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
        if verbose:
            print('%i outliers removed' % len(outliers))
        return outliers

    @ property
    def zmag(self):
        """Channel magnitude."""
        message = "use mag, not zmag"
        warnings.warn(message, wt_exceptions.VisibleDeprecationWarning)
        return self.mag

    @ property
    def zmax(self):
        """Channel maximum."""
        message = "use max, not zmax"
        warnings.warn(message, wt_exceptions.VisibleDeprecationWarning)
        return self.max

    @ property
    def zmin(self):
        """Channel minimum."""
        message = "use min, not zmin"
        warnings.warn(message, wt_exceptions.VisibleDeprecationWarning)
        return self.min

    @ property
    def znull(self):
        """Channel null."""
        message = "use null, not znull"
        warnings.warn(message, wt_exceptions.VisibleDeprecationWarning)
        return self.null()


class Data:
    """Central multidimensional data class."""

    def __init__(self, axes, channels, constants=[],
                 name='', source=None, **kwargs):
        """Create a ``Data`` object.

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
        **kwargs
            Additional keyword arguments are added to the attrs dictionary
            and to the natural namespace of the object (if possible).
        """
        # record version
        from .. import __version__
        self.__version__ = __version__
        # assign
        self.axes = axes
        self.constants = constants
        self.channels = channels
        self.name = name
        self.source = source
        self.attrs = kwargs
        # update
        self._update()

    def __repr__(self):
        """Return an unambiguous representation.

        Returns
        -------
        string
            Representation.
        """
        return 'WrightTools.data.Data object \'{0}\' {1} at {2}'.format(
            self.name, str(self.axis_names), str(id(self)))

    def _update(self):
        """Ensure that a Data Object is up to date.

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
            print('axis, constant, and channel names must all be unique')
            return
        for obj in self.axes + self.channels + self.constants:
            setattr(self, obj.name, obj)
        self.shape = self.channels[0].values.shape
        # attrs
        for key, value in self.attrs.items():
            identifier = wt_kit.string2identifier(key)
            if not hasattr(self, identifier):
                setattr(self, identifier, value)

    def bring_to_front(self, channel):
        """Bring a specific channel to the zero-indexed position in channels.

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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        # bring to front
        self.channels.insert(0, self.channels.pop(channel_index))
        self._update()

    def chop(self, *args, **kwargs):
        """Divide the dataset into its lower-dimensionality components.

        Parameters
        ----------
        axis : str or int (args)
            Axes of the returned data objects. Strings refer to the names of
            axes in this object, integers refer to their index. Provide multiple
            axes to return multidimensional data objects.
        at : dict (kwarg)
            Choice of position along an axis. Keys are axis names, values are lists
            ``[position, input units]``. If exact position does not exist,
            the closest valid position is used.
        verbose : bool, optional (kwarg)
            Toggle talkback. Default is True.

        Returns
        -------
        list of WrightTools.data.Data
            List of chopped data objects.

        Examples
        --------
        >>> data.axis_names
        ['d2', 'w1', 'w2']

        Get all w1 wigners.

        >>> datas = data.chop('d2', 'w1')
        >>> len(datas)
        51

        Get 2D frequency at d2=0 fs.

        >>> datas = data.chop('w1', 'w2', at={'d2': [0, 'fs']})
        >>> len(datas)
        0
        >>> datas[0].axis_names
        ['w1', 'w2']
        >>> datas[0].d2.points
        0.

        See Also
        --------
        collapse
            Collapse the dataset along one axis.
        split
            Split the dataset while maintaining its dimensionality.
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
        """Call ``Channel.clip``.

        Wrapper method for ``Channel.clip``.

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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
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
            is integrate. All methods but integrate disregard NANs.

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
            raise TypeError("axis: expected {int, str}, got %s" % type(axis))
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
                channel.values = np.nansum(channel.values, axis=axis_index)
            elif method in ['max', 'maximum']:
                channel.values = np.nanmax(channel.values, axis=axis_index)
            elif method in ['min', 'minimum']:
                channel.values = np.nanmin(channel.values, axis=axis_index)
            elif method in ['ave', 'average', 'mean']:
                channel.values = np.nanmean(channel.values, axis=axis_index)
            else:
                print('method not recognized in data.collapse')
        # cleanup ---------------------------------------------------------------------------------
        self.axes.pop(axis_index)
        self._update()

    def convert(self, destination_units, verbose=True):
        """Convert all compatable constants and axes to given units.

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
        """Get dimensionality of Data object."""
        return len(self.axes)

    def divide(self, divisor, channel=0, divisor_channel=0):
        """Divide a given channel by another data object.

        Divisor may be self.
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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        channel = self.channels[channel_index]
        # get divisor channel
        if isinstance(divisor_channel, int):
            divisor_channel_index = divisor_channel
        elif isinstance(divisor_channel, string_type):
            divisor_channel_index = divisor.channel_names.index(divisor_channel)
        else:
            raise TypeError("divisor_channel: expected {int, str}, got %s" % type(divisor_channel))
        divisor_channel = divisor.channels[divisor_channel_index]
        # do division
        channel.values /= divisor_channel.values
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
            raise TypeError("signal_channel: expected {int, str}, got %s" % type(signal_channel))
        # get reference channel
        if isinstance(reference_channel, int):
            reference_channel_index = reference_channel
        elif isinstance(reference_channel, string_type):
            reference_channel_index = self.channel_names.index(reference_channel)
        else:
            raise TypeError("reference_channel: expected {int, str}, got %s" %
                            type(reference_channel))
        # process
        intensity = self.channels[reference_channel_index].values.copy()
        d_intensity = self.channels[signal_channel_index].values.copy()
        if method == 'digital':
            out = -np.log10((intensity + d_intensity) / intensity)
        elif method == 'boxcar':
            # assume data collected with boxcar i.e.
            # sig = 1/2 dT
            # ref = T + 1/2 dT
            d_intensity *= 2
            out = -np.log10((intensity + d_intensity) / intensity)
        else:
            raise ValueError("Method '%s' not in {'digital', 'boxcar'}" % method)
        # finish
        self.channels[signal_channel_index].give_values(out)
        self.channels[signal_channel_index].signed = True
        self.channels[signal_channel_index]._null = 0

    def flip(self, axis):
        """Flip direction of arrays along an axis.

        Changes the index of elements without changing their correspondance to axis positions.

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
            raise TypeError("axis: expected {int, str}, got %s" % type(axis))
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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
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
            possible for 1D and 2D data. See `griddata`__ for more information.
            Default is linear.
        fill_value : number-like (optional)
            The value written to pixels that cannot be filled by interpolation.
            Default is nan.
        verbose : bool (optional)
            Toggle talkback. Default is True.


        __ http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html


        .. note:: Healing may take several minutes for large datasets.
           Interpolation time goes as nearest, linear, then cubic.


        """
        timer = wt_kit.Timer(verbose=False)
        with timer:
            # channel
            if isinstance(channel, int):
                channel_index = channel
            elif isinstance(channel, string_type):
                channel_index = self.channel_names.index(channel)
            else:
                raise TypeError("channel: expected {int, str}, got %s" % type(channel))
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
        # print
        if verbose:
            print('channel {0} healed in {1} seconds'.format(
                channel.name, np.around(timer.interval, decimals=3)))

    @property
    def info(self):
        """Retrieve info dictionary about a Data object."""
        info = collections.OrderedDict()
        info['name'] = self.name
        info['id'] = id(self)
        info['axes'] = self.axis_names
        info['channels'] = self.channel_names
        info['shape'] = self.shape
        info['version'] = self.__version__
        return info

    def level(self, channel, axis, npts, verbose=True):
        """For a channel, subtract the average value of several points at the edge of a given axis.

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
                raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        channel = self.channels[channel_index]
        # axis ------------------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            raise TypeError("axis: expected {int, str}, got %s" % type(axis))
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
        channel._null = 0.
        # print
        if verbose:
            axis = self.axes[axis_index]
            if npts > 0:
                points = axis.points[:npts]
            if npts < 0:
                points = axis.points[npts:]
            print('channel', channel.name, 'offset by', axis.name, 'between',
                  int(points.min()), 'and', int(points.max()), axis.units)

    def m(self, abs_data, channel=0, this_exp='TG', indices=None, m=None, bounds_error=True,
          verbose=True):
        """Perform m-factor corrections.

        Assumes all absorption functions are independent, so we can
        normalize each axis individually.

        Parameters
        ----------
        abs_data : wt.data.Data object
            Absorption data to normalize by
        channel : int or string (optional)
            Channel to correct (default is zero)
        this_exp : {'TG', 'TA'} (optional)
            Experimental configuration. Default is TG. Note that TG data
            should be processed on the intensity level.
        indices : list of integers (optional)
            axis indices. If None, indices are guessed from label_seed.
            Default is None.
        m : function (optional)
            m-factor function. Should take arguments a1 and a2.
        bounds_error : boolean (optional)
            Toggle bounds_error. Default is True.
        verbose : boolean (optional)
            Toggle talkback. Default is True.

        Notes
        -----
        m-factors originally derived by Carlson and Wright. [1]_

        References
        ----------
        .. [1] **Absorption and Coherent Interference Effects in Multiply Resonant
               Four-Wave Mixing Spectroscopy**
               Roger J. Carlson, and John C. Wright
               *Applied Spectroscopy* **1989** 43, 1195--1208
               `doi:10.1366/0003702894203408 <http://dx.doi.org/10.1366/0003702894203408>`_
        """
        # exp_name: [i], [m_i]
        exp_types = {
            'TG': [['1', '2'],
                   [lambda a1: 10**-a1,
                    lambda a2: ((1 - 10**-a2) / (a2 * np.log(10)))**2
                    ]
                   ],
            'TA': [['2'],
                   [lambda a2: (1 - 10**-a2) / (a2 * np.log(10))]
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
            raise KeyError('experiment {0} not recognized'.format(this_exp))
        # find which axes have m-factor dependence; move to the inside and operate
        m_axes = [axi for axi in self.axes if axi.units_kind == 'energy']
        # loop through 'indices' and find axis whole label_seeds contain indi
        for i, indi in enumerate(indices):
            t_order = list(range(len(self.axes)))
            # find axes indices that have the correct label seed
            # and also belong to the list of axes under consideration
            # ni = [j for j in range(len(m_axes)) if indi in
            ni = [j for j in range(len(self.axes)) if indi in self.axes[j].label_seed and
                  self.axes[j] in m_axes]
            if verbose:
                print(ni)
            # there should never be more than one axis that agrees
            if len(ni) > 1:
                raise RuntimeError('axes are not unique!')
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
                # evaluate ai
                abs_data.axes[0].convert(axi.units)
                Ei = abs_data.axes[0].points
                Ai = interp1d(Ei, abs_data.channels[0].values,
                              bounds_error=bounds_error)
                ai = Ai(axi.points)
                Mi = mi(ai)
                # apply Mi to channel
                self.channels[channel].values /= Mi
                # invert back out of the transpose
                t_inv = [t_order.index(j) for j in range(len(t_order))]
                if verbose:
                    print(t_inv)
                self.transpose(axes=t_inv, verbose=verbose)
            else:
                raise RuntimeError('{0} label_seed not found'.format(indi))

    def map_axis(self, axis, points, input_units='same', verbose=True):
        """Map points of an axis to new points using linear interpolation.

        Out-of-bounds points are written nan.

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
        points = np.array(points)
        # get axis index --------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            raise TypeError("axis: expected {int, str}, got %s" % type(axis))
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
        Normalize data in given channel so that null=0 and max=1.

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
                raise TypeError("channel: expected {int, str}, got %s" % type(channel))
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
        """Offset one axis based on another axis' values.

        Useful for correcting instrumental artifacts such as zerotune.

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
            possible for 1D and 2D data. See `griddata`__ for more information.
            Default is linear.
        verbose : bool (optional)
            Toggle talkback. Default is True.


        __ http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

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
            raise TypeError("along: expected {int, str}, got %s" % type(along))
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
            raise TypeError("offset_axis: expected {int, str}, got %s" % type(offset_axis))
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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        # remove
        self.channels.pop(channel_index)
        # finish
        self._update()

    def rename_attrs(self, **kwargs):
        """Rename a set of attributes.

        Keyword Arguments
        -----------------
        Each argument should have the key of a current axis or channel,
            and a value which is a string of its new name.

        The name will be set to str(val), and its natural naming identifier
        will be wt.kit.string2identifier(str(val))
        """
        changed = kwargs.keys()
        for k, v in kwargs.items():
            if getattr(self, k).__class__ not in (Channel, Axis):
                raise TypeError("Attribute for key %s: expected {Channel, Axis}, got %s" %
                                (k, getattr(self, k).__class__))
            if v not in changed and hasattr(self, v):
                raise wt_exceptions.NameNotUniqueError(v)
        for k, v in kwargs.items():
            axis = getattr(self, k)
            axis.name = str(v)
            delattr(self, k)
        self._update()

    def save(self, filepath=None, verbose=True):
        """Save using the `pickle`__ module.

        __ https://docs.python.org/3/library/pickle.html

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
        """Scale a channel.

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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        channel = self.channels[channel_index]
        # do scaling
        if kind in ['amp', 'amplitude']:
            channel.values = wt_kit.symmetric_sqrt(channel.values, out=channel.values)
        if kind in ['log']:
            channel.values = np.log10(channel.values)
        if kind in ['invert']:
            channel.values *= -1.

    def share_nans(self):
        """Share not-a-numbers between all channels.

        If any channel is nan at a given index, all channels will be nan
        at that index after this operation.

        Uses the share_nans method found in wt.kit.
        """
        arrs = [c.values for c in self.channels]
        outs = wt_kit.share_nans(arrs)
        for c, a, in zip(self.channels, outs):
            c.values = a

    def smooth(self, factors, channel=None, verbose=True):
        """Smooth a channel using an n-dimenional `kaiser window`__.

        __ https://en.wikipedia.org/wiki/Kaiser_window

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
                raise TypeError("channel: expected {int, str}, got %s" % type(channel))
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
            raise TypeError("axis: expected {int, str}, got %s" % type(axis))
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
                print('  {0} : {1} to {2} {3} (length {4})'.format(i, new_axis.points[0],
                                                                   new_axis.points[-1],
                                                                   new_axis.units,
                                                                   len(new_axis.points)))
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
        """Subtract a given channel by another data object.

        Subtrahend may be self.
        All axes in subtrahend must be contained in self.

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
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        channel = self.channels[channel_index]
        # get subtrahend channel
        if isinstance(subtrahend_channel, int):
            subtrahend_channel_index = subtrahend_channel
        elif isinstance(subtrahend_channel, string_type):
            subtrahend_channel_index = subtrahend.channel_names.index(subtrahend_channel)
        else:
            raise TypeError("subtrahend_channel: expected {int, str}, got %s" %
                            type(subtrahend_channel))
        subtrahend_channel = subtrahend.channels[subtrahend_channel_index]
        # do division
        channel.values -= subtrahend_channel.values
        # transpose out
        self.transpose(transpose_order, verbose=False)

    def trim(self, channel, **kwargs):
        """Call ``Channel.trim``.

        Wrapper method for ``Channel.trim``.

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
                raise KeyError('Keyword arguments to trim must be either an axis name or one of' +
                               '{method, factor, replace, verbose}.')
        # call trim
        return channel.trim(neighborhood=neighborhood, **inputs)

    def transform(self, transform=None):
        """Transform the dataset using arbitrary coordinates, then regrids the data.

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
        raise NotImplementedError

    def transpose(self, axes=None, verbose=True):
        """Transpose the dataset.

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
        """Zoom the data array using spline interpolation of the requested order.

        The number of points along each axis is increased by factor.
        See `scipy ndimage`__ for more info.

        __ http://docs.scipy.org/doc/scipy/reference/
                    generated/scipy.ndimage.interpolation.zoom.html

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
