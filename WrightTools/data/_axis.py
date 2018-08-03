"""Axis class and associated."""


# --- import --------------------------------------------------------------------------------------


import re
import numexpr
import operator
import functools

import numpy as np

from .. import exceptions as wt_exceptions
from .. import kit as wt_kit
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ["Axis"]

operator_to_identifier = {}
operator_to_identifier["/"] = "__d__"
operator_to_identifier["="] = "__e__"
operator_to_identifier["-"] = "__m__"
operator_to_identifier["+"] = "__p__"
operator_to_identifier["*"] = "__t__"
identifier_to_operator = {value: key for key, value in operator_to_identifier.items()}
operators = "".join(operator_to_identifier.keys())


# --- class ---------------------------------------------------------------------------------------


class Axis(object):
    """Axis class."""

    def __init__(self, parent, expression, units=None):
        """Data axis.

        Parameters
        ----------
        parent : WrightTools.Data
            Parent data object.
        expression : string
            Axis expression.
        units : string (optional)
            Axis units. Default is None.
        """
        self.parent = parent
        self.expression = expression
        if units is None:
            self.units = self.variables[0].units
        else:
            self.units = units

    def __getitem__(self, index):
        vs = {}
        for variable in self.variables:
            arr = variable[index]
            vs[variable.natural_name] = wt_units.converter(arr, variable.units, self.units)
        return numexpr.evaluate(self.expression.split("=")[0], local_dict=vs)

    def __repr__(self):
        return "<WrightTools.Axis {0} ({1}) at {2}>".format(
            self.expression, str(self.units), id(self)
        )

    @property
    def _leaf(self):
        out = self.expression
        if self.units is not None:
            out += " ({0}) {1}".format(self.units, self.shape)
        return out

    @property
    def full(self):
        arr = self[:]
        for i in range(arr.ndim):
            if arr.shape[i] == 1:
                arr = np.repeat(arr, self.parent.shape[i], axis=i)
        return arr

    @property
    def identity(self):
        """Complete identifier written to disk in data.attrs['axes']."""
        return self.natural_name + " {%s}" % self.units

    @property
    def label(self):
        symbol = wt_units.get_symbol(self.units)
        label = self.expression
        for v in self.variables:
            vl = "%s_{%s}" % (symbol, v.label)
            vl = vl.replace("_{}", "")  # label can be empty, no empty subscripts
            label = label.replace(v.natural_name, vl)
        if self.units_kind:
            units_dictionary = getattr(wt_units, self.units_kind)
            label += r"\,"
            label += r"\left("
            label += units_dictionary[self.units][2]
            label += r"\right)"
        else:
            pass
        label = r"$\mathsf{%s}$" % label
        return label

    @property
    def natural_name(self):
        name = self.expression.strip()
        for op in operators:
            name = name.replace(op, operator_to_identifier[op])
        return wt_kit.string2identifier(name)

    @property
    def ndim(self):
        """Get number of dimensions."""
        try:
            assert self._ndim is not None
        except (AssertionError, AttributeError):
            self._ndim = self.variables[0].ndim
        finally:
            return self._ndim

    @property
    def points(self):
        """Squeezed array."""
        return np.squeeze(self[:])

    @property
    def shape(self):
        """Shape."""
        return wt_kit.joint_shape(*self.variables)

    @property
    def size(self):
        """Size."""
        return functools.reduce(operator.mul, self.shape)

    @property
    def units_kind(self):
        """Units kind."""
        return wt_units.kind(self.units)

    @property
    def variables(self):
        """Variables."""
        try:
            assert self._variables is not None
        except (AssertionError, AttributeError):
            pattern = "|".join(map(re.escape, operators))
            keys = re.split(pattern, self.expression)
            indices = []
            for key in keys:
                if key in self.parent.variable_names:
                    indices.append(self.parent.variable_names.index(key))
            self._variables = [self.parent.variables[i] for i in indices]
        finally:
            return self._variables

    def convert(self, destination_units, *, convert_variables=False):
        """Convert axis to destination_units.

        Parameters
        ----------
        destination_units : string
            Destination units.
        convert_variables : boolean (optional)
            Toggle conversion of stored arrays. Default is False.
        """
        if not wt_units.is_valid_conversion(self.units, destination_units):
            kind = wt_units.kind(self.units)
            valid = list(wt_units.dicts[kind].keys())
            raise wt_exceptions.UnitsError(valid, destination_units)
        if convert_variables:
            for v in self.variables:
                v.convert(destination_units)
        self.units = destination_units

    def max(self):
        """Axis max."""
        return np.max(self[:])

    def min(self):
        """Axis min."""
        return np.min(self[:])
