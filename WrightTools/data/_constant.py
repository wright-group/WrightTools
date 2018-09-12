"""Constant class and associated."""


# --- import --------------------------------------------------------------------------------------


import re
import numexpr
import operator
import functools

import numpy as np

from ._variable import Variable
from ._axis import Axis
from .. import exceptions as wt_exceptions
from .. import kit as wt_kit
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ["Constant"]


# --- class ---------------------------------------------------------------------------------------


class Constant(Axis):
    """Constant class."""

    def __init__(self, parent, expression, units=None):
        """Data axis.

        Parameters
        ----------
        parent : WrightTools.Data
            Parent data object.
        expression : string
            Constant expression.
        units : string (optional)
            Constant units. Default is None.
        """
        super().__init__(parent, expression, units)

    def __repr__(self) -> str:
        return "<WrightTools.Constant {0} = {1} {2} at {2}>".format(
            self.expression, self.value, str(self.units), id(self)
        )

    @property
    def _leaf(self):
        out = self.expression
        out += " = {0}".format(self.value)
        if self.units is not None:
            out += " {0}".format(self.units)
        return out

    @property
    def label(self) -> str:
        label = self.expression.replace("_", "\\;")
        if self.units_kind:
            symbol = wt_units.get_symbol(self.units)
            for v in self.variables:
                vl = "%s_{%s}" % (symbol, v.label)
                vl = vl.replace("_{}", "")  # label can be empty, no empty subscripts
                label = label.replace(v.natural_name, vl)
        label += r"\,=\,{}".format(self.value)
        if self.units_kind:
            units_dictionary = getattr(wt_units, self.units_kind)
            label += r"\,"
            label += units_dictionary[self.units][2]
        label = r"$\mathsf{%s}$" % label
        return label

    @property
    def value(self) -> complex:
        """The value of the constant."""
        return np.nanmean(self.masked)

    @property
    def std(self) -> complex:
        """The value of the constant."""
        return np.nanstd(self.masked)
