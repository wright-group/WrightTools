"""Constant class and associated."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

from ._axis import Axis
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ["Constant"]


# --- class ---------------------------------------------------------------------------------------


class Constant(Axis):
    """Constant class."""

    def __init__(self, parent, expression, units=None, format_spec="0.3g", round_spec=None):
        """Data constant.

        Parameters
        ----------
        parent : WrightTools.Data
            Parent data object.
        expression : string
            Constant expression.
        units : string (optional)
            Constant units. Default is None.
        format_spec : string (optional)
            Format string specification, as passed to :meth:`format`
            Default is "0.3g"
        round_spec : int or None (optional)
            Decimal digits to round to before formatting, as passed to :meth:`round`.
            Default is None (no rounding).
        """
        super().__init__(parent, expression, units)
        self.format_spec = format_spec
        self.round_spec = round_spec

    def __repr__(self) -> str:
        return "<WrightTools.Constant {0} = {1} {2} at {3}>".format(
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
        """A latex formatted label representing constant expression and united value."""
        label = self.expression.replace("_", "\\;")
        if self.units_kind:
            symbol = wt_units.get_symbol(self.units)
            for v in self.variables:
                vl = "%s_{%s}" % (symbol, v.label)
                vl = vl.replace("_{}", "")  # label can be empty, no empty subscripts
                label = label.replace(v.natural_name, vl)
                val = (
                    round(self.value, self.round_spec)
                    if self.round_spec is not None
                    else self.value
                )
        label += r"\,=\,{}".format(format(val, self.format_spec))
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
        """The standard deviation of the constant."""
        return np.nanstd(self.masked)
