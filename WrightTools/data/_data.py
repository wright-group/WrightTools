"""Central data class and associated."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import copy
import shutil
import weakref
import collections
import warnings
import tempfile

import posixpath

import numpy as np

import h5py

import scipy
from scipy.interpolate import griddata, interp1d

from .._base import Group, Dataset
from .. import collection as wt_collection
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


class Axis(Dataset):
    """Axis class."""

    def __init__(self, parent, id, units=None, symbol_type=None, label_seed=[''], **kwargs):
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
        self._parent = parent
        h5py.Dataset.__init__(self, id)
        # units
        if units is not None:
            self.attrs['units'] = units
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
        self.attrs.update(kwargs)
        self.attrs['name'] = h5py.h5i.get_name(self.id).decode().split('/')[-1]
        self.attrs['class'] = 'Axis'
        for key, value in self.attrs.items():
            identifier = wt_kit.string2identifier(key)
            if not hasattr(self, identifier):
                setattr(self, identifier, value)

    def __repr__(self):
        return '<WrightTools.Axis \'{0}\' at {1}>'.format(self.natural_name, self.fullpath)

    @property
    def fullpath(self):
        return self.parent.fullpath + posixpath.sep + self.natural_name

    @property
    def natural_name(self):
        return self.attrs['name']

    @property
    def info(self):
        """Axis info dictionary."""
        info = collections.OrderedDict()
        info['name'] = self.name
        info['id'] = id(self)
        if self.is_constant:
            info['point'] = self[0]
        else:
            info['range'] = '{0} - {1} ({2})'.format(self[:].min(),
                                                     self[:].max(), self.units)
            info['number'] = len(self[:])
        return info

    @property
    def label(self):  # noqa: D403
        """LaTeX formatted label."""
        return self.get_label()

    @property
    def parent(self):
        return self._parent

    @property
    def units(self):
        return self.attrs['units']

    def convert(self, destination_units):
        """Convert axis to destination units.

        Parameters
        ----------
        destination_units : string
            Destination units.
        """
        self[:] = wt_units.converter(self[:], self.units,
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
            if self[:] is not None:
                label += r'=\,' + str(np.round(self[:], decimals=decimals))
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

    def is_constant(self):
        """Axis constant flag."""
        try:
            len(self[:])
        except TypeError:
            return False
        finally:
            return True

    def max(self):
        """Axis max, ignoring nans."""
        return np.nanmax(self[:])

    def min(self):
        """Axis min, ignoring nans."""
        return np.nanmin(self[:])


class Channel(Dataset):
    """Channel."""

    def __init__(self, parent, id, units=None, null=None, signed=None,
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
        self._parent = parent
        h5py.Dataset.__init__(self, id)
        self.label = label
        self.label_seed = label_seed
        self.units = units
        self.dimensionality = len(self.shape)
        # attrs
        self.attrs.update(kwargs)
        self.attrs['name'] = h5py.h5i.get_name(self.id).decode().split('/')[-1]
        self.attrs['class'] = 'Channel'
        if signed is not None:
            self.attrs['signed'] = signed
        if null is not None:
            self.attrs['null'] = null
        for key, value in self.attrs.items():
            identifier = wt_kit.string2identifier(key)
            if not hasattr(self, identifier):
                setattr(self, identifier, value)

    def __getitem__(self, i):
        if hasattr(i, '__iter__'):
            indices = list(i)
        else:
            indices = [i]
        while len(indices) <= self.dimensionality:
            print(indices)
            indices.append(slice(None))
        indices = [indices[i] for i in self.transposition]
        return h5py.Dataset.__getitem__(self, tuple(indices)).transpose(self.transposition)

    def __repr__(self):
        return '<WrightTools.Channel \'{0}\' at {1}>'.format(self.natural_name,
                                                                  self.fullpath)

    @property
    def fullpath(self):
        return self.parent.fullpath + posixpath.sep + self.natural_name

    @property
    def minor_extent(self):
        """Minimum deviation from null."""
        return min((self.max() - self.null, self.null - self.min()))

    @property
    def natural_name(self):
        return self.attrs['name']

    @property
    def null(self):
        if not 'null' in self.attrs.keys():
            self.attrs['null'] = 0
        return self.attrs['null']

    @property
    def info(self):
        """Return Channel info dictionary."""
        info = collections.OrderedDict()
        info['name'] = self.name
        info['min'] = self.min()
        info['max'] = self.max()
        info['null'] = self.null
        info['signed'] = self.signed
        return info

    @property
    def major_extent(self):
        """Maximum deviation from null."""
        return max((self.max() - self.null, self.null - self.min()))

    @property
    def parent(self):
        return self._parent

    @property
    def signed(self):
        if not 'signed' in self.attrs.keys():
            self.attrs['signed'] = False
        return self.attrs['signed']

    @property
    def transposition(self):
        if not 'transposition' in self.attrs.keys():
            self.attrs['transposition'] = np.arange(len(self.shape))
        return self.attrs['transposition']

    def clip(self, min=None, max=None, replace='nan'):
        """Clip (limit) the values in a channel.

        Parameters
        ----------
        min : number (optional)
            New channel minimum. Default is None.
        max : number (optional)
            New channel maximum. Default is None.
        replace : {'val', 'nan'} (optional)
           Replace behavior. Default is nan.
        """
        # decide what min and max will actually be
        if max is None:
            max = self.max()
        if min is None:
            min = self.min()
        # replace values
        if replace == 'val':
            self[:].clip(min, max, out=self[:])
        elif replace == 'nan':
            self[self[:] < min] = np.nan
            self[self[:] > max] = np.nan
        else:
            print('replace not recognized in channel.clip')

    def invert(self):
        """Invert channel values."""
        self[:] *= -1

    def mag(self):
        """Channel magnitude (maximum deviation from null)."""
        return self.major_extent

    def max(self):
        """Maximum, ignorning nans."""
        return np.nanmax(self[:])

    def min(self):
        """Minimum, ignoring nans."""
        return np.nanmin(self[:])

    def normalize(self, axis=None):
        """Normalize a Channel, set `null` to 0 and the max to 1."""
        # process axis argument
        if axis is not None:
            if hasattr(axis, '__contains__'):  # list, tuple or similar
                axis = tuple((int(i) for i in axis))
            else:  # presumably a simple number
                axis = int(axis)
        # subtract off null
        self[:] -= self.null
        self._null = 0.
        # create dummy array
        dummy = self[:].copy()
        dummy[np.isnan(dummy)] = 0  # nans are propagated in np.amax
        if self.signed:
            dummy = np.absolute(dummy)
        # divide through by max
        self[:] /= np.amax(dummy, axis=axis, keepdims=True)
        # finish

    def transpose(self, axes):
        if axes is None:
            new = self.transposition[::-1]
        else:
            new = self.transposition
            for i, axis in enumerate(axes):
                new[i] = self.transposition[axis]
        self.attrs['transposition'] = new

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
        for idx in np.ndindex(self[:].shape):
            slices = []
            for i, di, size in zip(idx, neighborhood, self[:].shape):
                start = max(0, i - di)
                stop = min(size, i + di + 1)
                slices.append(slice(start, stop, 1))
            neighbors = self[:][slices]
            mean = np.nanmean(neighbors)
            limit = np.nanstd(neighbors) * factor
            if np.abs(self[:][idx] - mean) > limit:
                outliers.append(idx)
                means.append(mean)
        # replace outliers
        i = tuple(zip(*outliers))
        if replace == 'nan':
            self[:][i] = np.nan
        elif replace == 'mean':
            self[:][i] = means
        elif replace == 'mask':
            self[:] = np.ma.array(self[:])
            self[:][i] = np.ma.masked
        elif type(replace) in [int, float]:
            self[:][i] = replace
        else:
            raise KeyError('replace must be one of {nan, mean, mask} or some number')
        # finish
        if verbose:
            print('%i outliers removed' % len(outliers))
        return outliers


class Data(Group):
    """Central multidimensional data class."""
    class_name = 'Data'

    def __init__(self, *args, **kwargs):
        Group.__init__(self, *args, **kwargs)
        # the following are populated if not already recorded
        self.axis_names
        self.channel_names
        self.constant_names
        self.kind
        self.source

    def __getitem__(self, key):
        out = h5py.Group.__getitem__(self, key)
        if 'class' in out.attrs.keys():
            if out.attrs['class'] == 'Channel':
                return Channel(parent=self, id=out.id)
            elif out.attrs['class'] == 'Axis':
                return Axis(parent=self, id=out.id)

    def __repr__(self):
        return '<WrightTools.Data \'{0}\' {1} at {2}>'.format(
            self.natural_name, str(self.axis_names), '::'.join([self.filepath, self.name]))

    @property
    def axes(self):
        return [self[n] for n in self.axis_names]

    @property
    def axis_names(self):
        if 'axis_names' not in self.attrs.keys():
            self.attrs['axis_names'] = np.array([], dtype='S')
        return [s.decode() for s in self.attrs['axis_names']]

    @property
    def channel_names(self):
        if 'channel_names' not in self.attrs.keys():
            self.attrs['channel_names'] = np.array([], dtype='S')
        return [s.decode() for s in self.attrs['channel_names']]

    @property
    def channels(self):
        return [self[n] for n in self.axis_names]

    @property
    def constant_names(self):
        if 'constant_names' not in self.attrs.keys():
            self.attrs['constant_names'] = np.array([], dtype='S')
        return [s.decode() for s in self.attrs['constant_names']]

    @property
    def constants(self):
        return [self[n] for n in self.constant_names]

    @property
    def datasets(self):
        return [v for _, v in self.items() if isinstance(v, h5py.Dataset)]

    @property
    def dimensionality(self):
        """Get dimensionality of Data object."""
        return len(self.axes)

    @property
    def info(self):
        """Retrieve info dictionary about a Data object."""
        info = collections.OrderedDict()
        info['name'] = self.name
        info['axes'] = self.axis_names
        info['channels'] = self.channel_names
        info['shape'] = self.shape
        return info

    @property
    def kind(self):
        if 'kind' not in self.attrs.keys():
            self.attrs['kind'] = 'None'
        value = self.attrs['kind']
        return value if not value == 'None' else None

    @property
    def parent(self):
        group = super().parent
        parent = group.parent.name
        if parent == posixpath.sep:
            parent = None
        return wt_collection.Collection(self.filepath, parent=parent, name=group.attrs['name'])

    @property
    def shape(self):
        if len(self.axis_names) == 0:
            return tuple()
        return tuple([a.size for a in self.axes])

    @property
    def size(self):
        if len(self.channels) == 0:
            return 0
        return self.channels[0].size

    @property
    def source(self):
        if 'source' not in self.attrs.keys():
            self.attrs['source'] = 'None'
        value = self.attrs['source']
        return value if not value == 'None' else None

    def _update(self):
        """Ensure that a Data Object is up to date.

        Ensure that the ``axis_names``, ``constant_names``, ``channel_names``,
        and ``shape`` attributes are correct.
        """
        all_names = self.axis_names + self.channel_names + self.constant_names
        if len(all_names) == len(set(all_names)):
            pass
        else:
            raise wt_exceptions.NameNotUniqueError()
        for obj in self.axes + self.channels + self.constants:
            identifier = obj.name.split('/')[-1]
            setattr(self, identifier, obj)
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
        >>> datas[0].d2[:]
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
                length = len(getattr(self, name)[:])
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
                position = getattr(self, name)[:][idx]
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
                c_idx = np.argmin(abs(self.axes[idx][:] - val))
                constant_indicies.append(c_idx)
                obj = copy.copy(self.axes[idx])
                obj[:] = self.axes[idx][:][c_idx]
                constants.append(obj)
            # now handle axes
            axes_chopped = []
            for dim in axes_args:
                idx = [idx for idx in range(len(self.axes)) if self.axes[idx].name == dim][0]
                transpose_order.append(idx)
                axes_chopped.append(self.axes[idx])
            # ensure that everything is kosher
            if len(transpose_order) == len(self.channels[0][:].shape):
                pass
            else:
                print('chop failed: not enough dimensions specified')
                print(len(transpose_order))
                print(len(self.channels[0][:].shape))
                return
            if len(transpose_order) == len(set(transpose_order)):
                pass
            else:
                print('chop failed: same dimension used twice')
                return
            # chop
            for i in range(len(self.channels)):
                values = self.channels[i][:]
                values = values.transpose(transpose_order)
                for idx in constant_indicies:
                    values = values[idx]
                channels_chopped[i][:] = values
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
                channel[:] = np.trapz(
                    y=channel[:], x=self.axes[axis_index][:], axis=axis_index)
            elif method == 'sum':
                channel[:] = np.nansum(channel[:], axis=axis_index)
            elif method in ['max', 'maximum']:
                channel[:] = np.nanmax(channel[:], axis=axis_index)
            elif method in ['min', 'minimum']:
                channel[:] = np.nanmin(channel[:], axis=axis_index)
            elif method in ['ave', 'average', 'mean']:
                channel[:] = np.nanmean(channel[:], axis=axis_index)
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


    def create_axis(self, name, points, units, **kwargs):
        """Append a new axis, changing shape.

        Extant channels are broadcast into the new shape.

        Parameters
        ----------
        name : string
            Unique name for this axis.
        points : 1D array
            Axis points.
        units : string
            Axis units.

        Returns
        -------
        Axis
            Created axis.
        """
        # TODO: reshape extant channels
        id = self.require_dataset(name=name, data=points, shape=points.shape,
                                  dtype=points.dtype).id
        axis = Axis(self, id, units=units, **kwargs)
        self.axes.append(axis)
        self.attrs['axis_names'] = np.append(self.attrs['axis_names'], name.encode())
        self._update()
        return axis

    def create_constant(self, *args, **kwargs):
        raise NotImplementedError

    def create_channel(self, name, values=None, units=None, **kwargs):
        """Append a new channel.

        Parameters
        ----------
        name : string
            Unique name for this channel.
        values : array (optional)
            Array. If None, an empty array equaling the data shape is
            created. Default is None.
        units : string (optional)
            Channel units. Default is None.

        Returns
        -------
        Channel
            Created channel.
        """
        if values is None:
            shape = self.shape
            dtype = np.float64
        else:
            shape = values.shape
            dtype = values.dtype
            if not shape == self.shape:
                raise Exception  # TODO: better exception
        # create dataset
        id = self.require_dataset(name=name, data=values, shape=shape, dtype=dtype).id
        channel = Channel(self, id, units=units, **kwargs)
        self.channels.append(channel)
        self.attrs['channel_names'] = np.append(self.attrs['channel_names'], name.encode())
        self._update()
        return channel

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
                divisor.map_axis(name, axis[:])
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
        channel[:] /= divisor_channel[:]
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
        intensity = self.channels[reference_channel_index][:].copy()
        d_intensity = self.channels[signal_channel_index][:].copy()
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
        axis[:] = axis[:][::-1]
        # data
        for channel in self.channels:
            values = channel[:]
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
            channel[:] = values

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
        arr = channel[:]
        idxs = np.unravel_index(arr.argmin(), arr.shape)
        # finish
        return [a[:][i] for a, i in zip(self.axes, idxs)]

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
        arr = channel[:]
        idxs = np.unravel_index(arr.argmax(), arr.shape)
        # finish
        return [a[:][i] for a, i in zip(self.axes, idxs)]

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
            values = self.channels[channel_index][:]
            points = [axis[:] for axis in self.axes]
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
            self.channels[channel_index][:] = out
        # print
        if verbose:
            print('channel {0} healed in {1} seconds'.format(
                channel.name, np.around(timer.interval, decimals=3)))

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
        values = channel[:]
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
        channel[:] = values
        channel._null = 0.
        # print
        if verbose:
            axis = self.axes[axis_index]
            if npts > 0:
                points = axis[:][:npts]
            if npts < 0:
                points = axis[:][npts:]
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
                Ei = abs_data.axes[0][:]
                Ai = interp1d(Ei, abs_data.channels[0][:],
                              bounds_error=bounds_error)
                ai = Ai(axi[:])
                Mi = mi(ai)
                # apply Mi to channel
                self.channels[channel][:] /= Mi
                # invert back out of the transpose
                t_inv = [t_order.index(j) for j in range(len(t_order))]
                if verbose:
                    print(t_inv)
                self.transpose(axes=t_inv, verbose=verbose)
            else:
                raise RuntimeError('{0} label_seed not found'.format(indi))

    def map_axis(self, axis, points, input_units='same', edge_tolerance=0., verbose=True):
        """Map points of an axis to new points using linear interpolation.

        Out-of-bounds points are written nan.

        Parameters
        ----------
        axis : int or str
            The axis to map onto.
        points : 1D array-like or int
            If array, the new points. If int, new points will have the same
            limits, with int defining the number of evenly spaced points
            between.
        input_units : str (optional)
            The units of the new points. Default is same, which assumes
            the new points have the same units as the axis.
        edge_tolerance : float (optional)
            Axis edge points that are within this amount of the new edge
            points are coerced to the new edge points before interpolation.
            Default is 0.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        # get axis index --------------------------------------------------------------------------
        if isinstance(axis, int):
            axis_index = axis
        elif isinstance(axis, string_type):
            axis_index = self.axis_names.index(axis)
        else:
            raise TypeError("axis: expected {int, str}, got %s" % type(axis))
        axis = self.axes[axis_index]
        # get points ------------------------------------------------------------------------------
        if isinstance(points, int):
            points = np.linspace(axis[:][0], axis[:][-1], points)
            input_units = 'same'
        else:
            points = np.array(points)
        # transform points to axis units ----------------------------------------------------------
        if input_units == 'same':
            pass
        else:
            points = wt_units.converter(points, input_units, axis.units)
        # points must be ascending ----------------------------------------------------------------
        flipped = np.zeros(len(self.axes), dtype=np.bool)
        for i in range(len(self.axes)):
            if self.axes[i][:][0] > self.axes[i][:][-1]:
                self.flip(i)
                flipped[i] = True
        # handle edge tolerance -------------------------------------------------------------------
        for index in [0, -1]:
            old = axis[:][index]
            new = points[index]
            if new - edge_tolerance < old < new + edge_tolerance:
                axis[:][index] = new
        # interpn data ----------------------------------------------------------------------------
        old_points = [a[:] for a in self.axes]
        new_points = [a[:] if a is not axis else points for a in self.axes]
        if len(self.axes) == 1:
            for channel in self.channels:
                function = scipy.interpolate.interp1d(self.axes[0][:], channel[:])
                channel[:] = function(new_points[0])
        else:
            xi = tuple(np.meshgrid(*new_points, indexing='ij'))
            for channel in self.channels:
                values = channel[:]
                channel[:] = scipy.interpolate.interpn(old_points, values, xi,
                                                           method='linear',
                                                           bounds_error=False,
                                                           fill_value=np.nan)
        # cleanup ---------------------------------------------------------------------------------
        for i in range(len(self.axes)):
            if not i == axis_index:
                if flipped[i]:
                    self.flip(i)
        axis[:] = points
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
        corrections = function(axis[:])
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
        new_points = [a[:] for a in self.axes]
        old_offset_axis_points = self.axes[offset_axis_index][:]
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
        xi = tuple(np.meshgrid(*[a[:] for a in self.axes], indexing='ij'))
        for channel in self.channels:
            # 'undo' gridding
            arr = np.zeros((len(self.axes) + 1, channel[:].size))
            for i in range(len(self.axes)):
                arr[i] = xi[i].flatten()
            arr[-1] = channel[:].flatten()
            # do corrections
            corrections = list(corrections)
            corrections = corrections * int((len(arr[0]) / len(corrections)))
            arr[offset_axis_index] += corrections
            # grid data
            tup = tuple([arr[i] for i in range(len(arr) - 1)])
            # note that rescale is crucial in this operation
            out = griddata(tup, arr[-1], new_xi, method=method,
                           fill_value=np.nan, rescale=True)
            channel[:] = out
        self.axes[offset_axis_index][:] = new_offset_axis_points
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
        # TODO: documentation
        self.flush()  # ensure all changes are written to file
        if filepath is None:
            filepath = os.path.join(os.getcwd(), self.natural_name + '.wt5')
        elif len(os.path.basename(filepath).split('.')) == 1:
            filepath += '.wt5'
        filepath = os.path.expanduser(filepath)
        shutil.copyfile(src=self.filepath, dst=filepath)
        if verbose:
            print(filepath)
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
            channel[:] = wt_kit.symmetric_sqrt(channel[:], out=channel[:])
        if kind in ['log']:
            channel[:] = np.log10(channel[:])
        if kind in ['invert']:
            channel[:] *= -1.

    def share_nans(self):
        """Share not-a-numbers between all channels.

        If any channel is nan at a given index, all channels will be nan
        at that index after this operation.

        Uses the share_nans method found in wt.kit.
        """
        arrs = [c[:] for c in self.channels]
        outs = wt_kit.share_nans(arrs)
        for c, a, in zip(self.channels, outs):
            c[:] = a

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
            values = channel[:]
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
            channel[:] = values
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
            This decision is based on the value, not the index.
            Consider points [0, 1, 2, 3, 4, 5] and split value [3]. If direction
            is above the returned objects are [0, 1, 2] and [3, 4, 5]. If
            direction is below the returned objects are [0, 1, 2, 3] and
            [4, 5]. Default is below.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        list
            A list of data objects.
            Zero-length data objects are replaced with `None`.
            The order of the objects is such that the axis points retain their original order.

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
            idx = np.argmin(abs(axis[:] - position))
            indicies.append(idx)
        indicies.sort()

        # set direction according to units
        flip = direction == 'above'
        if axis[:][-1] < axis[:][0]:
            flip = not flip
        if flip:
            indicies = [i - 1 for i in indicies]
        # process ---------------------------------------------------------------------------------
        outs = []
        start = 0
        stop = -1
        for i in range(len(indicies) + 1):
            # get start and stop
            start = stop + 1  # previous value
            if i == len(indicies):
                stop = len(axis[:])
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
            new_data.axes[0][:] = new_data.axes[0][:][start:stop + 1]
            # channels
            for channel in new_data.channels:
                channel[:] = channel[:][start:stop + 1]
            # transpose out
            new_data.transpose(transpose_order, verbose=False)
            outs.append(new_data)
        # post process ----------------------------------------------------------------------------
        if verbose:
            print('split data into {0} pieces along {1}:'.format(len(indicies) + 1, axis.name))
            for i in range(len(outs)):
                new_data = outs[i]
                new_axis = new_data.axes[axis_index]
                if len(new_axis[:]) == 0:
                    print('  {0} : None'.format(i))
                else:
                    print('  {0} : {1} to {2} {3} (length {4})'.format(i, new_axis[:][0],
                                                                       new_axis[:][-1],
                                                                       new_axis.units,
                                                                       len(new_axis[:])))
        # deal with cases where only one element is left
        for i, new_data in enumerate(outs):
            if len(new_data.axes[axis_index][:]) == 1:
                # remove axis
                new_data.axis_names.pop(axis_index)
                axis = new_data.axes.pop(axis_index)
                new_data.constants.append(axis)
                # reshape channels
                shape = [i for i in new_data.channels[0][:].shape if not i == 1]
                shape = tuple(shape)
                for channel in new_data.channels:
                    channel[:].shape = shape
                new_data.shape = shape
            elif len(new_data.axes[axis_index][:]) == 0:
                outs[i] = None
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
                subtrahend.map_axis(name, axis[:])
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
        channel[:] -= subtrahend_channel[:]
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
            axes = range(len(self.channels[0][:].shape))[::-1]
        self.axes = [self.axes[i] for i in axes]
        self.attrs['axis_names'] = [axis.natural_name.encode() for axis in self.axes]
        for channel in self.channels:
            channel.transpose(axes=axes)
        if verbose:
            print('data transposed to', self.axis_names)

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
            axis[:] = scipy.ndimage.interpolation.zoom(axis[:],
                                                           factor,
                                                           order=order)
        # channels
        for channel in self.channels:
            channel[:] = scipy.ndimage.interpolation.zoom(channel[:],
                                                              factor,
                                                              order=order)
        # return
        if verbose:
            print('data zoomed to new shape:', self.channels[0][:].shape)
