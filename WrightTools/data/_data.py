"""Central data class and associated."""


# --- import --------------------------------------------------------------------------------------


import collections
import operator
import functools
import warnings

import numpy as np

import h5py

import scipy
from scipy.interpolate import griddata, interp1d
from skimage.transform import downscale_local_mean

from .._group import Group
from .. import collection as wt_collection
from .. import exceptions as wt_exceptions
from .. import kit as wt_kit
from .. import units as wt_units
from ._axis import Axis, identifier_to_operator
from ._channel import Channel
from ._constant import Constant
from ._variable import Variable


# --- define --------------------------------------------------------------------------------------


__all__ = ["Data"]


# --- class ---------------------------------------------------------------------------------------


class Data(Group):
    """Multidimensional dataset."""

    class_name = "Data"

    def __init__(self, *args, **kwargs):
        self._axes = []
        self._constants = []
        Group.__init__(self, *args, **kwargs)
        # populate axes from attrs string
        for identifier in self.attrs.get("axes", []):
            identifier = identifier.decode()
            expression, units = identifier.split("{")
            units = units.replace("}", "")
            for i in identifier_to_operator.keys():
                expression = expression.replace(i, identifier_to_operator[i])
            expression = expression.replace(" ", "")  # remove all whitespace
            axis = Axis(self, expression, units.strip())
            self._axes.append(axis)
        for identifier in self.attrs.get("constants", []):
            identifier = identifier.decode()
            expression, units = identifier.split("{")
            units = units.replace("}", "")
            for i in identifier_to_operator.keys():
                expression = expression.replace(i, identifier_to_operator[i])
            expression = expression.replace(" ", "")  # remove all whitespace
            axis = Axis(self, expression, units.strip())
            self._constants.append(axis)
        self._current_axis_identities_in_natural_namespace = []
        self._on_constants_updated()
        self._on_axes_updated()
        # the following are populated if not already recorded
        self.channel_names
        self.source
        self.variable_names

    def __repr__(self) -> str:
        return "<WrightTools.Data '{0}' {1} at {2}>".format(
            self.natural_name, str(self.axis_names), "::".join([self.filepath, self.name])
        )

    @property
    def axes(self) -> tuple:
        return tuple(self._axes)

    @property
    def axis_expressions(self) -> tuple:
        """Axis expressions."""
        return tuple(a.expression for a in self._axes)

    @property
    def axis_names(self) -> tuple:
        """Axis names."""
        return tuple(a.natural_name for a in self._axes)

    @property
    def constants(self) -> tuple:
        return tuple(self._constants)

    @property
    def constant_expressions(self) -> tuple:
        """Axis expressions."""
        return tuple(a.expression for a in self._constants)

    @property
    def constant_names(self) -> tuple:
        """Axis names."""
        return tuple(a.natural_name for a in self._constants)

    @property
    def channel_names(self) -> tuple:
        """Channel names."""
        if "channel_names" not in self.attrs.keys():
            self.attrs["channel_names"] = np.array([], dtype="S")
        return tuple(s.decode() for s in self.attrs["channel_names"])

    @channel_names.setter
    def channel_names(self, value):
        """Set channel names."""
        self.attrs["channel_names"] = np.array(value, dtype="S")

    @property
    def channels(self) -> tuple:
        """Channels."""
        return tuple(self[n] for n in self.channel_names)

    @property
    def datasets(self) -> tuple:
        """Datasets."""
        return tuple(v for _, v in self.items() if isinstance(v, h5py.Dataset))

    @property
    def kind(self):
        """Kind."""
        if "kind" not in self.attrs.keys():
            self.attrs["kind"] = "None"
        value = self.attrs["kind"]
        return value if not value == "None" else None

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        try:
            assert self._ndim is not None
        except (AssertionError, AttributeError):
            if len(self.variables) == 0:
                self._ndim = 0
            else:
                self._ndim = self.variables[0].ndim
        finally:
            return self._ndim

    @property
    def shape(self) -> tuple:
        """Shape."""
        try:
            assert self._shape is not None
        except (AssertionError, AttributeError):
            self._shape = wt_kit.joint_shape(*self.variables)
        finally:
            return self._shape

    @property
    def size(self) -> int:
        """Size."""
        return functools.reduce(operator.mul, self.shape)

    @property
    def source(self):
        """Source."""
        if "source" not in self.attrs.keys():
            self.attrs["source"] = "None"
        value = self.attrs["source"]
        return value if not value == "None" else None

    @property
    def units(self) -> tuple:
        """All axis units."""
        return tuple(a.units for a in self._axes)

    @property
    def constant_units(self) -> tuple:
        """All constant units."""
        return tuple(a.units for a in self._constants)

    @property
    def variable_names(self) -> tuple:
        """Variable names."""
        if "variable_names" not in self.attrs.keys():
            self.attrs["variable_names"] = np.array([], dtype="S")
        return tuple(s.decode() for s in self.attrs["variable_names"])

    @variable_names.setter
    def variable_names(self, value):
        """Set variable names."""
        self.attrs["variable_names"] = np.array(value, dtype="S")

    @property
    def variables(self) -> tuple:
        """Variables."""
        try:
            assert self._variables is not None
        except (AssertionError, AttributeError):
            self._variables = [self[n] for n in self.variable_names]
        finally:
            return tuple(self._variables)

    @property
    def _leaf(self):
        return "{0} {1}".format(self.natural_name, self.shape)

    def _on_axes_updated(self):
        """Method to run when axes are changed in any way.

        Propagates updated axes properly.
        """
        # update attrs
        self.attrs["axes"] = [a.identity.encode() for a in self._axes]
        # remove old attributes
        while len(self._current_axis_identities_in_natural_namespace) > 0:
            key = self._current_axis_identities_in_natural_namespace.pop(0)
            self.__dict__.pop(key)
        # populate new attributes
        for a in self._axes:
            key = a.natural_name
            setattr(self, key, a)
            self._current_axis_identities_in_natural_namespace.append(key)

    def _on_constants_updated(self):
        """Method to run when constants are changed in any way.

        Propagates updated constants properly.
        """
        # update attrs
        self.attrs["constants"] = [a.identity.encode() for a in self._constants]

    def _print_branch(self, prefix, depth, verbose):
        def print_leaves(prefix, lis, vline=True):
            for i, item in enumerate(lis):
                if vline:
                    a = "│   "
                else:
                    a = "    "
                if i + 1 == len(lis):
                    b = "└── "
                else:
                    b = "├── "
                s = prefix + a + b + "{0}: {1}".format(i, item._leaf)
                print(s)

        if verbose:
            # axes
            print(prefix + "├── axes")
            print_leaves(prefix, self.axes)
            # constants
            print(prefix + "├── constants")
            print_leaves(prefix, self.constants)
            # variables
            print(prefix + "├── variables")
            print_leaves(prefix, self.variables)
            # channels
            print(prefix + "└── channels")
            print_leaves(prefix, self.channels, vline=False)
        else:
            # axes
            s = "axes: "
            s += ", ".join(["{0} ({1})".format(a.expression, a.units) for a in self.axes])
            print(prefix + "├── " + s)
            # constants
            s = "constants: "
            s += ", ".join(
                ["{0} ({1} {2})".format(a.expression, a.value, a.units) for a in self.constants]
            )
            print(prefix + "├── " + s)
            # channels
            s = "channels: "
            s += ", ".join(self.channel_names)
            print(prefix + "└── " + s)

    def bring_to_front(self, channel):
        """Bring a specific channel to the zero-indexed position in channels.

        All other channels get pushed back but remain in order.

        Parameters
        ----------
        channel : int or str
            Channel index or name.
        """
        channel_index = wt_kit.get_index(self.channel_names, channel)
        new = list(self.channel_names)
        new.insert(0, new.pop(channel_index))
        self.channel_names = new

    def chop(self, *args, at={}, parent=None, verbose=True) -> wt_collection.Collection:
        """Divide the dataset into its lower-dimensionality components.

        Parameters
        ----------
        axis : str or int (args)
            Axes of the returned data objects. Strings refer to the names of
            axes in this object, integers refer to their index. Provide multiple
            axes to return multidimensional data objects.
        at : dict (optional)
            Choice of position along an axis. Keys are axis names, values are lists
            ``[position, input units]``. If exact position does not exist,
            the closest valid position is used.
        parent : WrightTools Collection instance (optional)
            Collection to place the new "chop" collection within. Default is
            None (new parent).
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        WrightTools Collection
            Collection of chopped data objects.

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
        from ._axis import operators, operator_to_identifier

        # parse args
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, int):
                args[i] = self._axes[arg].natural_name
            elif isinstance(arg, str):
                # same normalization that occurs in the natural_name @property
                arg = arg.strip()
                for op in operators:
                    arg = arg.replace(op, operator_to_identifier[op])
                args[i] = wt_kit.string2identifier(arg)

        # normalize the at keys to the natural name
        for k in list(at.keys()):
            for op in operators:
                if op in k:
                    nk = k.replace(op, operator_to_identifier[op])
                    at[nk] = at[k]
                    at.pop(k)
                    k = nk

        # get output collection
        out = wt_collection.Collection(name="chop", parent=parent)
        # get output shape
        kept = args + list(at.keys())
        kept_axes = [self._axes[self.axis_names.index(a)] for a in kept]
        removed_axes = [a for a in self._axes if a not in kept_axes]
        removed_shape = wt_kit.joint_shape(*removed_axes)
        if removed_shape == ():
            removed_shape = (1,) * self.ndim
        # iterate
        i = 0
        for idx in np.ndindex(removed_shape):
            idx = np.array(idx, dtype=object)
            idx[np.array(removed_shape) == 1] = slice(None)
            for axis, point in at.items():
                point, units = point
                destination_units = self._axes[self.axis_names.index(axis)].units
                point = wt_units.converter(point, units, destination_units)
                axis_index = self.axis_names.index(axis)
                axis = self._axes[axis_index]
                idx_index = np.array(axis.shape) > 1
                if np.sum(idx_index) > 1:
                    raise wt_exceptions.MultidimensionalAxisError("chop", axis.natural_name)
                idx_index = list(idx_index).index(True)
                idx[idx_index] = np.argmin(np.abs(axis[tuple(idx)] - point))
            data = out.create_data(name="chop%03i" % i)
            for v in self.variables:
                kwargs = {}
                kwargs["name"] = v.natural_name
                kwargs["values"] = v[idx]
                kwargs["units"] = v.units
                kwargs["label"] = v.label
                kwargs.update(v.attrs)
                data.create_variable(**kwargs)
            for c in self.channels:
                kwargs = {}
                kwargs["name"] = c.natural_name
                kwargs["values"] = c[idx]
                kwargs["units"] = c.units
                kwargs["label"] = c.label
                kwargs["signed"] = c.signed
                kwargs.update(c.attrs)
                data.create_channel(**kwargs)
            new_axes = [a.expression for a in kept_axes if a.expression not in at.keys()]
            new_axis_units = [a.units for a in kept_axes if a.expression not in at.keys()]
            data.transform(*new_axes)
            for ax in self.axis_expressions:
                if ax not in new_axes:
                    data.create_constant(ax, verbose=False)
            for j, units in enumerate(new_axis_units):
                data.axes[j].convert(units)
            i += 1
        out.flush()
        # return
        if verbose:
            print("chopped data into %d piece(s)" % len(out), "in", new_axes)
        return out

    def collapse(self, axis, method="integrate"):
        """
        Collapse the dataset along one axis, adding lower rank channels.

        New channels have names <channel name>_<axis name>_<method>.

        Parameters
        ----------
        axis : int or str
            The axis to collapse along.
            If given as an integer, the axis in the underlying array is used.
            If given as a string, the axis must exist, and be a 1D array-aligned axis.
            (i.e. have a shape with a single value which is not ``1``)
            The axis to collapse along is inferred from the shape of the axis.
        method : {'integrate', 'average', 'sum', 'max', 'min'} (optional)
            The method of collapsing the given axis. Method may also be list
            of methods corresponding to the channels of the object. Default
            is integrate. All methods but integrate disregard NANs.
            Can also be a list, allowing for different treatment for varied channels.
            In this case, None indicates that no change to that channel should occur.

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
        elif isinstance(axis, str):
            index = self.axis_names.index(axis)
            axes = [i for i in range(self.ndim) if self.axes[index].shape[i] > 1]
            if len(axes) > 1:
                raise wt_exceptions.MultidimensionalAxisError(axis, "collapse")
            elif len(axes) == 0:
                raise wt_exceptions.ValueError(
                    "Axis {} is a single point, cannot collapse".format(axis)
                )
            axis_index = axes[0]
        else:
            raise wt_exceptions.TypeError("axis: expected {int, str}, got %s" % type(axis))

        new_shape = list(self.shape)
        new_shape[axis_index] = 1
        # methods ---------------------------------------------------------------------------------
        if isinstance(method, list):
            if len(method) == len(self.channels):
                methods = method
            else:
                raise wt_exceptions.ValueError(
                    "method argument must have same number of elements as there are channels"
                )
            for m in methods:
                if m not in [
                    "sum",
                    "max",
                    "maximum",
                    "min",
                    "minimum",
                    "ave",
                    "average",
                    "mean",
                    "int",
                    "integrate",
                ]:
                    raise wt_exceptions.ValueError("method '{}' not recognized".format(m))
        elif isinstance(method, str):
            methods = [method for _ in self.channels]

        warnings.warn("collapse", category=wt_exceptions.EntireDatasetInMemoryWarning)

        # collapse --------------------------------------------------------------------------------
        for method, channel in zip(methods, self.channel_names):
            if method is None:
                continue

            if self[channel].shape[axis_index] == 1:
                continue  # Cannot collapse any further, don't clutter data object

            new_shape = list(self[channel].shape)
            new_shape[axis_index] = 1
            rtype = self[channel].dtype
            if method in ["ave", "average", "mean", "int", "integrate"]:
                rtype = np.result_type(self[channel].dtype, float)

            new = self.create_channel(
                "{}_{}_{}".format(channel, axis, method),
                values=np.empty(new_shape, dtype=rtype),
                units=self[channel].units,
            )

            channel = self[channel]

            if method == "sum":
                res = np.nansum(channel[:], axis=axis_index, keepdims=True)
                new[:] = res
            elif method in ["max", "maximum"]:
                res = np.nanmax(channel[:], axis=axis_index, keepdims=True)
                new[:] = res
            elif method in ["min", "minimum"]:
                res = np.nanmin(channel[:], axis=axis_index, keepdims=True)
                new[:] = res
            elif method in ["ave", "average", "mean"]:
                res = np.nanmean(channel[:], axis=axis_index, keepdims=True)
                new[:] = res
            elif method in ["int", "integrate"]:
                res = np.trapz(y=channel[:], x=self._axes[axis_index][:], axis=axis_index)
                res.shape = new_shape
                new[:] = res
            else:
                raise wt_exceptions.ValueError("method '{}' not recognized".format(m))

    def convert(self, destination_units, *, convert_variables=False, verbose=True):
        """Convert all compatable axes and constants to given units.

        Parameters
        ----------
        destination_units : str
            Destination units.
        convert_variables : boolean (optional)
            Toggle conversion of stored arrays. Default is False
        verbose : bool (optional)
            Toggle talkback. Default is True.

        See Also
        --------
        Axis.convert
            Convert a single axis object to compatable units. Call on an
            axis object in data.axes.
        """
        # get kind of units
        units_kind = wt_units.kind(destination_units)
        # apply to all compatible axes
        for axis in self.axes:
            if axis.units_kind == units_kind:
                orig = axis.units
                axis.convert(destination_units, convert_variables=convert_variables)
                if verbose:
                    print(
                        "axis {} converted from {} to {}".format(
                            axis.expression, orig, destination_units
                        )
                    )
        # apply to all compatible constants
        for constant in self.constants:
            if constant.units_kind == units_kind:
                orig = constant.units
                constant.convert(destination_units, convert_variables=convert_variables)
                if verbose:
                    print(
                        "constant {} converted from {} to {}".format(
                            constant.expression, orig, destination_units
                        )
                    )
        if convert_variables:
            for var in self.variables:
                if wt_units.kind(var.units) == units_kind:
                    orig = var.units
                    var.convert(destination_units)
                    if verbose:
                        print(
                            "variable {} converted from {} to {}".format(
                                var.natural_name, orig, destination_units
                            )
                        )
        self._on_axes_updated()
        self._on_constants_updated()

    def create_channel(
        self, name, values=None, *, shape=None, units=None, dtype=None, **kwargs
    ) -> Channel:
        """Append a new channel.

        Parameters
        ----------
        name : string
            Unique name for this channel.
        values : array (optional)
            Array. If None, an empty array equaling the data shape is
            created. Default is None.
        shape : tuple of int
            Shape to use. Must broadcast with the full shape.
            Only used if `values` is None.
            Default is the full shape of self.
        units : string (optional)
            Channel units. Default is None.
        dtype : numpy.dtype (optional)
            dtype to use for dataset, default is np.float64.
            Only used if `values` is None.
        kwargs : dict
            Additional keyword arguments passed to Channel instantiation.

        Returns
        -------
        Channel
            Created channel.
        """
        if name in self.channel_names:
            warnings.warn(name, wt_exceptions.ObjectExistsWarning)
            return self[name]

        require_kwargs = {}
        if values is None:
            if shape is None:
                require_kwargs["shape"] = self.shape
            else:
                require_kwargs["shape"] = shape
            if dtype is None:
                require_kwargs["dtype"] = np.dtype(np.float64)
            else:
                require_kwargs["dtype"] = dtype
            if require_kwargs["dtype"].kind in "fcmM":
                require_kwargs["fillvalue"] = np.nan
            else:
                require_kwargs["fillvalue"] = 0
        else:
            require_kwargs["data"] = values
            require_kwargs["shape"] = values.shape
            require_kwargs["dtype"] = values.dtype
        # create dataset
        dataset_id = self.require_dataset(name=name, chunks=True, **require_kwargs).id
        channel = Channel(self, dataset_id, units=units, **kwargs)
        # finish
        self.attrs["channel_names"] = np.append(self.attrs["channel_names"], name.encode())
        return channel

    def create_variable(
        self, name, values=None, *, shape=None, units=None, dtype=None, **kwargs
    ) -> Variable:
        """Add new child variable.

        Parameters
        ----------
        name : string
            Unique identifier.
        values : array-like (optional)
            Array to populate variable with. If None, an variable will be filled with NaN.
            Default is None.
        shape : tuple of int
            Shape to use. must broadcast with the full shape.
            Only used if `values` is None.
            Default is the full shape of self.
        units : string (optional)
            Variable units. Default is None.
        dtype : numpy.dtype (optional)
            dtype to use for dataset, default is np.float64.
            Only used if `values` is None.
        kwargs
            Additional kwargs to variable instantiation.

        Returns
        -------
        WrightTools Variable
            New child variable.
        """
        if name in self.variable_names:
            warnings.warn(name, wt_exceptions.ObjectExistsWarning)
            return self[name]
        if values is None:
            if shape is None:
                shape = self.shape
            if dtype is None:
                dtype = np.dtype(np.float64)
            if dtype.kind in "fcmM":
                fillvalue = np.nan
            else:
                fillvalue = 0
        else:
            shape = values.shape
            dtype = values.dtype
            fillvalue = None
        # create dataset
        id = self.require_dataset(
            name=name, data=values, shape=shape, dtype=dtype, fillvalue=fillvalue
        ).id
        variable = Variable(self, id, units=units, **kwargs)
        # finish
        self._variables = None
        self.attrs["variable_names"] = np.append(self.attrs["variable_names"], name.encode())
        return variable

    def downscale(self, tup, name=None, parent=None) -> "Data":
        """Down sample the data array using local averaging.

        See `skimage.transform.downscale_local_mean`__ for more info.

        __ http://scikit-image.org/docs/0.12.x/api/
            skimage.transform.html#skimage.transform.downscale_local_mean

        Parameters
        ----------
        tup : tuple of ints
            The collection of step sizes by which each axis is binned.
            Each axis is sliced with step size determined by the tuple.
            To keep an axis sampling unchanged, use 1 or None
        name : string (optional)
            The name of the string. Default is None.
        parent : WrightTools Collection instance (optional)
            Collection to place the downscaled data object. Default is
            None (new parent).

        Returns
        -------
        WrightTools Data instance
            New data object with the downscaled channels and axes

        See Also
        --------
        zoom
            Zoom the data array using spline interpolation of the requested order.
        """
        if name is None:
            name = self.natural_name + "_downscaled"
        if parent is None:
            newdata = Data(name=name)
        else:
            parent.create_data(name=name)

        for channel in self.channels:
            name = channel.natural_name
            newdata.create_channel(
                name=name, values=downscale_local_mean(channel[:], tup), units=channel.units
            )
        args = []
        for i, axis in enumerate(self.axes):
            if len(axis.variables) > 1:
                raise NotImplementedError("downscale only works with simple axes currently")
            variable = axis.variables[0]
            name = variable.natural_name
            args.append(name)
            slices = [slice(None, None, step) for step in tup]
            newdata.create_variable(name=name, values=variable[slices], units=variable.units)
        newdata.transform(*args)
        return newdata

    def get_nadir(self, channel=0) -> tuple:
        """Get the coordinates, in units, of the minimum in a channel.

        Parameters
        ----------
        channel : int or str (optional)
            Channel. Default is 0.

        Returns
        -------
        generator of numbers
            Coordinates in units for each axis.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, str):
            channel_index = self.channel_names.index(channel)
        else:
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        channel = self.channels[channel_index]
        # get indicies
        idx = channel.argmin()
        # finish
        return tuple(a[idx] for a in self._axes)

    def get_zenith(self, channel=0) -> tuple:
        """Get the coordinates, in units, of the maximum in a channel.

        Parameters
        ----------
        channel : int or str (optional)
            Channel. Default is 0.

        Returns
        -------
        generator of numbers
            Coordinates in units for each axis.
        """
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, str):
            channel_index = self.channel_names.index(channel)
        else:
            raise TypeError("channel: expected {int, str}, got %s" % type(channel))
        channel = self.channels[channel_index]
        # get indicies
        idx = channel.argmax()
        # finish
        return tuple(a[idx] for a in self._axes)

    def heal(self, channel=0, method="linear", fill_value=np.nan, verbose=True):
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
        warnings.warn("heal", category=wt_exceptions.EntireDatasetInMemoryWarning)
        timer = wt_kit.Timer(verbose=False)
        with timer:
            # channel
            if isinstance(channel, int):
                channel_index = channel
            elif isinstance(channel, str):
                channel_index = self.channel_names.index(channel)
            else:
                raise TypeError("channel: expected {int, str}, got %s" % type(channel))
            channel = self.channels[channel_index]
            values = self.channels[channel_index][:]
            points = [axis[:] for axis in self._axes]
            xi = tuple(np.meshgrid(*points, indexing="ij"))
            # 'undo' gridding
            arr = np.zeros((len(self._axes) + 1, values.size))
            for i in range(len(self._axes)):
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
            print(
                "channel {0} healed in {1} seconds".format(
                    channel.name, np.around(timer.interval, decimals=3)
                )
            )

    def level(self, channel, axis, npts, *, verbose=True):
        """Subtract the average value of npts at the edge of a given axis.

        Parameters
        ----------
        channel : int or str
            Channel to level.
        axis : int
            Axis to level along.
        npts : int
            Number of points to average for each slice. Positive numbers
            take points at leading indicies and negative numbers take points
            at trailing indicies.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        warnings.warn("level", category=wt_exceptions.EntireDatasetInMemoryWarning)
        channel_index = wt_kit.get_index(self.channel_names, channel)
        channel = self.channels[channel_index]
        # verify npts not zero
        npts = int(npts)
        if npts == 0:
            raise wt_exceptions.ValueError("npts must not be zero")
        # get subtrahend
        ss = [slice(None)] * self.ndim
        if npts > 0:
            ss[axis] = slice(0, npts, None)
        else:
            ss[axis] = slice(npts, None, None)
        subtrahend = np.nanmean(channel[ss], axis=axis)
        if self.ndim > 1:
            subtrahend = np.expand_dims(subtrahend, axis=axis)
        # level
        channel -= subtrahend
        # finish
        channel._null = 0
        if verbose:
            print("channel {0} leveled along axis {1}".format(channel.natural_name, axis))

    def map_variable(
        self, variable, points, input_units="same", *, name=None, parent=None, verbose=True
    ) -> "Data":
        """Map points of an axis to new points using linear interpolation.

        Out-of-bounds points are written nan.

        Parameters
        ----------
        variable : string
            The variable to map onto.
        points : array-like or int
            If array, the new points. If int, new points will have the same
            limits, with int defining the number of evenly spaced points
            between.
        input_units : str (optional)
            The units of the new points. Default is same, which assumes
            the new points have the same units as the axis.
        name : string (optional)
            The name of the new data object. If None, generated from
            natural_name. Default is None.
        parent : WrightTools.Collection (optional)
            Parent of new data object. If None, data is made at root of a
            new temporary file.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        WrightTools.Data
            New data object.
        """
        # get variable index
        variable_index = wt_kit.get_index(self.variable_names, variable)
        variable = self.variables[variable_index]
        # get points
        if isinstance(points, int):
            points = np.linspace(variable.min(), variable.max(), points)
        points = np.array(points)
        # points dimensionality
        if points.ndim < variable.ndim:
            for i, d in enumerate(variable.shape):
                if d == 1:
                    points = np.expand_dims(points, axis=i)
        # convert points
        if input_units == "same":
            pass
        else:
            points = wt_units.converter(points, input_units, variable.units)
        # construct new data object
        special = ["name", "axes", "constants", "channel_names", "variable_names"]
        kwargs = {k: v for k, v in self.attrs.items() if k not in special}
        if name is None:
            name = "{0}_{1}_mapped".format(self.natural_name, variable.natural_name)
        kwargs["name"] = name
        kwargs["parent"] = parent
        out = Data(**kwargs)
        # mapped variable
        values = points
        out.create_variable(values=values, **variable.attrs)
        # orthogonal variables
        for v in self.variables:
            if wt_kit.orthogonal(v.shape, variable.shape):
                out.create_variable(values=v[:], **v.attrs)
        out.transform(*self.axis_expressions)
        # interpolate
        if self.ndim == 1:

            def interpolate(dataset, points):
                function = scipy.interpolate.interp1d(variable[:], dataset[:], bounds_error=False)
                return function(points)

        else:
            pts = np.array([a.full.flatten() for a in self.axes]).T
            out_pts = np.array([a.full.flatten() for a in out.axes]).T

            def interpolate(dataset, points):
                values = dataset.full.flatten()
                function = scipy.interpolate.LinearNDInterpolator(pts, values, rescale=True)
                new = function(out_pts)
                new.shape = out.shape
                return new

        for v in self.variables:
            if v.natural_name not in out.variable_names:
                out.create_variable(values=interpolate(v, points), **v.attrs)
        out.variable_names = self.variable_names  # enforce old order
        out._variables = None  # force regeneration of variables @property
        for channel in self.channels:
            out.create_channel(values=interpolate(channel, points), **channel.attrs)
        # finish
        if verbose:
            print("data mapped from {0} to {1}".format(self.shape, out.shape))
        return out

    def offset(
        self,
        points,
        offsets,
        along,
        offset_axis,
        units="same",
        offset_units="same",
        mode="valid",
        method="linear",
        verbose=True,
    ):
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
        raise NotImplementedError
        # axis ------------------------------------------------------------------------------------
        if isinstance(along, int):
            axis_index = along
        elif isinstance(along, str):
            axis_index = self.axis_names.index(along)
        else:
            raise TypeError("along: expected {int, str}, got %s" % type(along))
        axis = self._axes[axis_index]
        # values & points -------------------------------------------------------------------------
        # get values, points, units
        if units == "same":
            input_units = axis.units
        else:
            input_units = units
        # check offsets is 1D or 0D
        if len(offsets.shape) == 1:
            pass
        else:
            raise RuntimeError("values must be 1D or 0D in offset!")
        # check if units is compatible, convert
        dictionary = getattr(wt_units, axis.units_kind)
        if input_units in dictionary.keys():
            pass
        else:
            raise RuntimeError("units incompatible in offset!")
        points = wt_units.converter(points, input_units, axis.units)
        # create correction array
        function = interp1d(points, offsets, bounds_error=False)
        corrections = function(axis[:])
        # remove nans
        finite_indicies = np.where(np.isfinite(corrections))[0]
        left_pad_width = finite_indicies[0]
        right_pad_width = len(corrections) - finite_indicies[-1] - 1
        corrections = np.pad(
            corrections[np.isfinite(corrections)],
            (int(left_pad_width), int(right_pad_width)),
            mode="edge",
        )
        # do correction ---------------------------------------------------------------------------
        # transpose so axis is last
        transpose_order = np.arange(len(self._axes))
        transpose_order[axis_index] = len(self._axes) - 1
        transpose_order[-1] = axis_index
        self.transpose(transpose_order, verbose=False)
        # get offset axis index
        if isinstance(offset_axis, int):
            offset_axis_index = offset_axis
        elif isinstance(offset_axis, str):
            offset_axis_index = self.axis_names.index(offset_axis)
        else:
            raise TypeError("offset_axis: expected {int, str}, got %s" % type(offset_axis))
        # new points
        new_points = [a[:] for a in self._axes]
        old_offset_axis_points = self._axes[offset_axis_index][:]
        spacing = abs(
            (old_offset_axis_points.max() - old_offset_axis_points.min())
            / float(len(old_offset_axis_points))
        )
        if mode == "old":
            new_offset_axis_points = old_offset_axis_points
        elif mode == "valid":
            _max = old_offset_axis_points.max() + corrections.min()
            _min = old_offset_axis_points.min() + corrections.max()
            n = int(abs(np.ceil((_max - _min) / spacing)))
            new_offset_axis_points = np.linspace(_min, _max, n)
        elif mode == "full":
            _max = old_offset_axis_points.max() + corrections.max()
            _min = old_offset_axis_points.min() + corrections.min()
            n = np.ceil((_max - _min) / spacing)
            new_offset_axis_points = np.linspace(_min, _max, n)
        new_points[offset_axis_index] = new_offset_axis_points
        new_xi = tuple(np.meshgrid(*new_points, indexing="ij"))
        xi = tuple(np.meshgrid(*[a[:] for a in self._axes], indexing="ij"))
        for channel in self.channels:
            # 'undo' gridding
            arr = np.zeros((len(self._axes) + 1, channel[:].size))
            for i in range(len(self._axes)):
                arr[i] = xi[i].flatten()
            arr[-1] = channel[:].flatten()
            # do corrections
            corrections = list(corrections)
            corrections = corrections * int((len(arr[0]) / len(corrections)))
            arr[offset_axis_index] += corrections
            # grid data
            tup = tuple([arr[i] for i in range(len(arr) - 1)])
            # note that rescale is crucial in this operation
            out = griddata(tup, arr[-1], new_xi, method=method, fill_value=np.nan, rescale=True)
            channel[:] = out
        self._axes[offset_axis_index][:] = new_offset_axis_points
        # transpose out
        self.transpose(transpose_order, verbose=False)

    def print_tree(self, *, verbose=True):
        """Print a ascii-formatted tree representation of the data contents."""
        print("{0} ({1})".format(self.natural_name, self.filepath))
        self._print_branch("", depth=0, verbose=verbose)

    def remove_channel(self, channel, *, verbose=True):
        """Remove channel from data.

        Parameters
        ----------
        channel : int or str
            Channel index or name to remove.
        verbose : boolean (optional)
            Toggle talkback. Default is True.
        """
        channel_index = wt_kit.get_index(self.channel_names, channel)
        new = list(self.channel_names)
        name = new.pop(channel_index)
        del self[name]
        self.channel_names = new
        if verbose:
            print("channel {0} removed".format(name))

    def remove_variable(self, variable, *, implied=True, verbose=True):
        """Remove variable from data.

        Parameters
        ----------
        variable : int or str
            Variable index or name to remove.
        implied : boolean (optional)
            Toggle deletion of other variables that start with the same
            name. Default is True.
        verbose : boolean (optional)
            Toggle talkback. Default is True.
        """
        if isinstance(variable, int):
            variable = self.variable_names[variable]
        # find all of the implied variables
        removed = []
        if implied:
            for n in self.variable_names:
                if n.startswith(variable):
                    removed.append(n)
        else:
            removed = [variable]
        # check that axes will not be ruined
        for n in removed:
            for a in self._axes:
                if n in [v.natural_name for v in a.variables]:
                    message = "{0} is contained in axis {1}".format(n, a.expression)
                    raise RuntimeError(message)
            for c in self._constants:
                if n in [v.natural_name for v in c.variables]:
                    warnings.warn(
                        "Variable being removed used in a constant",
                        wt_exceptions.WrightToolsWarning,
                    )

        # do removal
        for n in removed:
            variable_index = wt_kit.get_index(self.variable_names, n)
            new = list(self.variable_names)
            name = new.pop(variable_index)
            del self[name]
            self.variable_names = new
            self._variables = None
        # finish
        if verbose:
            print("{0} variable(s) removed:".format(len(removed)))
            for n in removed:
                print("  {0}".format(n))

    def rename_channels(self, *, verbose=True, **kwargs):
        """Rename a set of channels.

        Parameters
        ----------
        kwargs
            Keyword arguments of the form current:'new'.
        verbose : boolean (optional)
            Toggle talkback. Default is True
        """
        # ensure that items will remain unique
        changed = kwargs.keys()
        for k, v in kwargs.items():
            if v not in changed and v in self.keys():
                raise wt_exceptions.NameNotUniqueError(v)
        # compile references to items that are changing
        new = {}
        for k, v in kwargs.items():
            obj = self[k]
            index = self.channel_names.index(k)
            # rename
            new[v] = obj, index
            obj._instances.pop(obj.fullpath, None)
            obj.natural_name = str(v)
            # remove old references
            del self[k]
        # apply new references
        names = list(self.channel_names)
        for v, value in new.items():
            obj, index = value
            self[v] = obj
            names[index] = v
        self.channel_names = names
        # finish
        if verbose:
            print("{0} channel(s) renamed:".format(len(kwargs)))
            for k, v in kwargs.items():
                print("  {0} --> {1}".format(k, v))

    def rename_variables(self, *, implied=True, verbose=True, **kwargs):
        """Rename a set of variables.

        Parameters
        ----------
        kwargs
            Keyword arguments of the form current:'new'.
        implied : boolean (optional)
            Toggle inclusion of other variables that start with the same
            name. Default is True.
        verbose : boolean (optional)
            Toggle talkback. Default is True
        """
        # find all of the implied variables
        kwargs = collections.OrderedDict(kwargs)
        if implied:
            new = collections.OrderedDict()
            for k, v in kwargs.items():
                for n in self.variable_names:
                    if n.startswith(k):
                        new[n] = n.replace(k, v, 1)
            kwargs = new
        # ensure that items will remain unique
        changed = kwargs.keys()
        for k, v in kwargs.items():
            if v not in changed and v in self.keys():
                raise wt_exceptions.NameNotUniqueError(v)
        # compile references to items that are changing
        new = {}
        for k, v in kwargs.items():
            obj = self[k]
            index = self.variable_names.index(k)
            # rename
            new[v] = obj, index
            obj._instances.pop(obj.fullpath, None)
            obj.natural_name = str(v)
            # remove old references
            del self[k]
        # apply new references
        names = list(self.variable_names)
        for v, value in new.items():
            obj, index = value
            self[v] = obj
            names[index] = v
        self.variable_names = names
        units = self.units
        new = list(self.axis_expressions)
        for i, v in enumerate(kwargs.keys()):
            for j, n in enumerate(new):
                new[j] = n.replace(v, "{%i}" % i)
        for i, n in enumerate(new):
            new[i] = n.format(*kwargs.values())
        self.transform(*new)
        for a, u in zip(self._axes, units):
            a.convert(u)
        units = self.constant_units
        new = list(self.constant_expressions)
        for i, v in enumerate(kwargs.keys()):
            for j, n in enumerate(new):
                new[j] = n.replace(v, "{%i}" % i)
        for i, n in enumerate(new):
            new[i] = n.format(*kwargs.values())
        self.set_constants(*new)
        for c, u in zip(self._constants, units):
            c.convert(u)
        # finish
        if verbose:
            print("{0} variable(s) renamed:".format(len(kwargs)))
            for k, v in kwargs.items():
                print("  {0} --> {1}".format(k, v))

    def share_nans(self):
        """Share not-a-numbers between all channels.

        If any channel is nan at a given index, all channels will be nan
        at that index after this operation.

        Uses the share_nans method found in wt.kit.
        """

        def f(_, s, channels):
            outs = wt_kit.share_nans(*[c[s] for c in channels])
            for c, o in zip(channels, outs):
                c[s] = o

        self.channels[0].chunkwise(f, self.channels)

    def smooth(self, factors, channel=None, verbose=True) -> "Data":
        """Smooth a channel using an n-dimenional `kaiser window`__.

        Note, all arrays are loaded into memory.

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
        warnings.warn("smooth", category=wt_exceptions.EntireDatasetInMemoryWarning)
        # get factors -----------------------------------------------------------------------------

        if isinstance(factors, list):
            pass
        else:
            dummy = np.zeros(len(self._axes))
            dummy[::] = factors
            factors = list(dummy)
        # get channels ----------------------------------------------------------------------------
        if channel is None:
            channels = self.channels
        else:
            if isinstance(channel, int):
                channel_index = channel
            elif isinstance(channel, str):
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
                transpose_order = [
                    len(values.shape) - 1 if i == axis_index else i for i in transpose_order
                ]
                transpose_order[len(values.shape) - 1] = axis_index
                values = values.transpose(transpose_order)
                # get kaiser window
                beta = 5.0
                w = np.kaiser(2 * factor + 1, beta)
                # for all slices...
                for index in np.ndindex(values[..., 0].shape):
                    current_slice = values[index]
                    temp_slice = np.pad(current_slice, int(factor), mode=str("edge"))
                    values[index] = np.convolve(temp_slice, w / w.sum(), mode=str("valid"))
                # transpose out
                values = values.transpose(transpose_order)
            # return array to channel object
            channel[:] = values
        if verbose:
            print("smoothed data")

    def split(
        self, expression, positions, *, units=None, parent=None, verbose=True
    ) -> wt_collection.Collection:
        """
        Split the data object along a given expression, in units.

        Parameters
        ----------
        expression : int or str
            The expression to split along. If given as an integer, the axis at that index
            is used.
        positions : number-type or 1D array-type
            The position(s) to split at, in units.
        units : str (optional)
            The units of the given positions. Default is same, which assumes
            input units are identical to first variable units.
        parent : WrightTools.Collection (optional)
            The parent collection in which to place the 'split' collection.
            Default is a new Collection.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        WrightTools.collection.Collection
            A Collection of data objects.
            The order of the objects is such that the axis points retain their original order.

        See Also
        --------
        chop
            Divide the dataset into its lower-dimensionality components.
        collapse
            Collapse the dataset along one axis.
        """
        # axis ------------------------------------------------------------------------------------
        old_expr = self.axis_expressions
        old_units = self.units
        out = wt_collection.Collection(name="split", parent=parent)
        if isinstance(expression, int):
            if units is None:
                units = self._axes[expression].units
            expression = self._axes[expression].expression
        elif isinstance(expression, str):
            pass
        else:
            raise TypeError("expression: expected {int, str}, got %s" % type(expression))

        self.transform(expression)
        if units:
            self.convert(units)

        try:
            positions = [-np.inf] + sorted(list(positions)) + [np.inf]
        except TypeError:
            positions = [-np.inf, positions, np.inf]

        values = self._axes[0].full
        masks = [(values >= lo) & (values < hi) for lo, hi in wt_kit.pairwise(positions)]
        omasks = []
        cuts = []
        for mask in masks:
            try:
                omasks.append(wt_kit.mask_reduce(mask))
                cuts.append([i == 1 for i in omasks[-1].shape])
                # Ensure at least one axis is kept
                if np.all(cuts[-1]):
                    cuts[-1][0] = False
            except ValueError:
                omasks.append(None)
                cuts.append(None)
        for i in range(len(positions) - 1):
            out.create_data("split%03i" % i)

        for var in self.variables:
            for i, (imask, omask, cut) in enumerate(zip(masks, omasks, cuts)):
                if omask is None:
                    # Zero length split
                    continue
                omask = wt_kit.enforce_mask_shape(omask, var.shape)
                omask.shape = tuple([s for s, c in zip(omask.shape, cut) if not c])
                out_arr = np.full(omask.shape, np.nan)
                imask = wt_kit.enforce_mask_shape(imask, var.shape)
                out_arr[omask] = var[:][imask]
                out[i].create_variable(values=out_arr, **var.attrs)

        for ch in self.channels:
            for i, (imask, omask, cut) in enumerate(zip(masks, omasks, cuts)):
                if omask is None:
                    # Zero length split
                    continue
                omask = wt_kit.enforce_mask_shape(omask, ch.shape)
                omask.shape = tuple([s for s, c in zip(omask.shape, cut) if not c])
                out_arr = np.full(omask.shape, np.nan)
                imask = wt_kit.enforce_mask_shape(imask, ch.shape)
                out_arr[omask] = ch[:][imask]
                out[i].create_channel(values=out_arr, **ch.attrs)

        if verbose:
            for d in out.values():
                try:
                    d.transform(expression)
                except IndexError:
                    continue

            print("split data into {0} pieces along <{1}>:".format(len(positions) - 1, expression))
            for i, (lo, hi) in enumerate(wt_kit.pairwise(positions)):
                new_data = out[i]
                if new_data.shape == ():
                    print("  {0} : None".format(i))
                else:
                    new_axis = new_data.axes[0]
                    print(
                        "  {0} : {1:0.2f} to {2:0.2f} {3} {4}".format(
                            i, lo, hi, new_axis.units, new_axis.shape
                        )
                    )

        for d in out.values():
            try:
                d.transform(*old_expr)
                keep = []
                for ax in d.axes:
                    if ax.size > 1:
                        keep.append(ax.expression)
                    else:
                        d.create_constant(ax.expression, verbose=False)
                d.transform(*keep)
                for ax, u in zip(d.axes, old_units):
                    ax.convert(u)
            except IndexError:
                continue
            tempax = Axis(d, expression)
            if all(
                np.all(
                    np.sum(~np.isnan(tempax.masked), axis=tuple(set(range(tempax.ndim)) - {j}))
                    <= 1
                )
                for j in range(tempax.ndim)
            ):
                d.create_constant(expression, verbose=False)
        self.transform(*old_expr)
        for ax, u in zip(self.axes, old_units):
            ax.convert(u)

        return out

    def transform(self, *axes, verbose=True):
        """Transform the data.

        Parameters
        ----------
        axes : strings
            Expressions for the new set of axes.
        verbose : boolean (optional)
            Toggle talkback. Default is True

        See Also
        --------
        set_constants
            Similar method except for constants
        """
        # TODO: ensure that transform does not break data
        # create
        new = []
        newt = "newt" in self.axis_expressions
        current = {a.expression: a for a in self._axes}
        for expression in axes:
            axis = current.get(expression, Axis(self, expression))
            new.append(axis)
        self._axes = new
        # units
        for a in self._axes:
            if a.units is None:
                a.convert(a.variables[0].units)
        # finish
        self.flush()
        self._on_axes_updated()
        nownewt = "newt" in self.axis_expressions
        if verbose and nownewt and not newt:
            print("Look she turned me into a newt")
        elif verbose and newt and not nownewt:
            print("I got better")

    def set_constants(self, *constants, verbose=True):
        """Set the constants associated with the data.

        Parameters
        ----------
        constants : str
            Expressions for the new set of constants.
        verbose : boolean (optional)
            Toggle talkback. Default is True

        See Also
        --------
        transform
            Similar method except for axes.
        create_constant
            Add an individual constant.
        remove_constant
            Remove an individual constant.
        """
        # create
        new = []
        current = {c.expression: c for c in self._constants}
        for expression in constants:
            constant = current.get(expression, Constant(self, expression))
            new.append(constant)
        self._constants = new
        # units
        for c in self._constants:
            if c.units is None:
                c.convert(c.variables[0].units)
        # finish
        self.flush()
        self._on_constants_updated()

    def create_constant(self, expression, *, verbose=True):
        """Append a constant to the stored list.

        Parameters
        ----------
        expression : str
            Expression for the new constant.
        verbose : boolean (optional)
            Toggle talkback. Default is True
            
        See Also
        --------
        set_constants
            Remove and replace all constants.
        remove_constant
            Remove an individual constant.
        """
        if expression in self.constant_expressions:
            wt_exceptions.ObjectExistsWarning.warn(expression)
            return self.constants[self.constant_expressions.index(expression)]
        constant = Constant(self, expression)
        if constant.units is None:
            constant.convert(constant.variables[0].units)
        self._constants.append(constant)
        self.flush()
        self._on_constants_updated()
        if verbose:
            print("Constant '{}' added".format(constant.expression))
        return constant

    def remove_constant(self, constant, *, verbose=True):
        """Remove a constant from the stored list.

        Parameters
        ----------
        constant : str or Constant or int
            Expression for the new constant.
        verbose : boolean (optional)
            Toggle talkback. Default is True
            
        See Also
        --------
        set_constants
            Remove and replace all constants.
        create_constant
            Add an individual constant.
        """
        if isinstance(constant, (str, int)):
            constant_index = wt_kit.get_index(self.constant_expressions, constant)
        elif isinstance(constant, Constant):
            constant_index = wt_kit.get_index(self.constants, constant)
        constant = self._constants[constant_index]
        self._constants.pop(constant_index)
        self.flush()
        self._on_constants_updated()
        if verbose:
            print("Constant '{}' removed".format(constant.expression))

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

        See Also
        --------
        downscale
            Down-sample the data array using local averaging.
        """
        raise NotImplementedError
        import scipy.ndimage

        # axes
        for axis in self._axes:
            axis[:] = scipy.ndimage.interpolation.zoom(axis[:], factor, order=order)
        # channels
        for channel in self.channels:
            channel[:] = scipy.ndimage.interpolation.zoom(channel[:], factor, order=order)
        # return
        if verbose:
            print("data zoomed to new shape:", self.shape)
