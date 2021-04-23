.. _units:

Units
=====

WrightTools provides its own units system.
You can use it directly, if you wish.

.. code-block:: python

   >>> import WrightTools as wt
   >>> wt.units.convert(2., 'eV', 'nm')
   619.9209921660013

Under the hood, the WrightTools units system is backed by a `Pint <https://pint.readthedocs.io` unit registry.
This registry has all of the units in the default Pint registry, plus a few added by WrightTools (including some extra aliases of pint units for backwards compatibility with older WrightTools).
Additionally this unit registry automatically enables the spectroscopy context as well as a custom context for handling delay behavior (which inserts a factor of 2 as delays typically increase the path of light by two times their physical movement).

The pint unit registry can be directly accessed at ``wt.units.ureg``.

To query whether a specific conversion is applicable, you can use :meth:`wt.units.is_valid_conversion`.
To get a list of commonly used units that are valid conversions, you can use :meth:`wt.units.get_valid_conversions`.

This same units system enables the units-aware properties throughout WrightTools.

The units system also provides a symbol for many units, enabling easy plotting.
You can get the symbol using :meth:`wt.units.get_symbol`.

The following table contains units commonly used in WrightTools, though many others (including compound units) may be used if desired.

=========  ====================  ====================
name       description           symbol
---------  --------------------  --------------------
rad        radian                :math:`\omega`
deg        degrees               :math:`\omega`
fs         femtoseconds          :math:`\tau`
ps         picoseconds           :math:`\tau`
ns         nanoseconds           :math:`\tau`
nm         nanometers            :math:`\lambda`
wn         wavenumbers           :math:`\bar{\nu}`
eV         electronvolts         :math:`\hslash\omega`
meV        millielectronvolts    :math:`\hslash\omega`
Hz         hertz                 :math:`f`
THz        terahertz             :math:`f`
GHz        gigahertz             :math:`f`
K          kelvin                :math:`T`
deg_C      celsius               :math:`T`
deg_F      fahrenheit            :math:`T`
mOD        mOD                   None
OD         OD                    None
nm_p       nanometers            None
um         microns               :math:`\lambda`
mm         millimeters           :math:`\lambda`
cm         centimeters           :math:`\lambda`
in         inches                :math:`\lambda`
fs_t       femtoseconds          :math:`\tau`
ps_t       picoseconds           :math:`\tau`
ns_t       nanoseconds           :math:`\tau`
us_t       microseconds          :math:`\tau`
ms_t       milliseconds          :math:`\tau`
s_t        seconds               :math:`\tau`
m_t        minutes               :math:`\tau`
h_t        hours                 :math:`\tau`
d_t        days                  :math:`\tau`
=========  ====================  ====================
