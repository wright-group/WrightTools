.. _units:

Units
=====

WrightTools provides its own units system.
You can use it directly, if you wish.

.. code-block:: python

   >>> import WrightTools as wt
   >>> wt.units.converter(2., 'eV', 'nm')
   620.0

This same units system enables the units-aware properties throughout WrightTools, as in ``Axis`` and ``Curve``.

In WrightTools, units are organized into kinds.
It is always possible to convert between units of the same kind, and never possible to convert between kinds.

The units system also provides a symbol for each unit, enabling easy plotting.

The following table contains every unit in WrightTools.

=========  ====================  ====================  ====================
name       description           kind                  symbol
---------  --------------------  --------------------  --------------------
rad        radian                angle                 None
deg        degrees               angle                 None
fs         femtoseconds          delay                 :math:`\tau`
ps         picoseconds           delay                 :math:`\tau`
ns         nanoseconds           delay                 :math:`\tau`
mm_delay   mm                    delay                 :math:`\tau`
nm         nanometers            energy                :math:`\lambda`
wn         wavenumbers           energy                :math:`\bar{\nu}`
eV         electron volts        energy                :math:`\hslash\omega`
meV        milla electron volts  energy                :math:`E`
Hz         hertz                 energy                :math:`f`
THz        terahertz             energy                :math:`f`
GHz        gigahertz             energy                :math:`f`
mOD        mOD                   optical density       None
OD         OD                    optical density       None
nm_p       nanometers            position              None
um         microns               position              None
mm         millimeters           position              None
cm         centimeters           position              None
in         inches                position              None
FWHM       full width half max   pulse width           :math:`\sigma`
fs_t       femtoseconds          time                  None
ps_t       picoseconds           time                  None
ns_t       nanoseconds           time                  None
us_t       microseconds          time                  None
ms_t       milliseconds          time                  None
s_t        seconds               time                  None
m_t        minutes               time                  None
h_t        hours                 time                  None
d_t        days                  time                  None
=========  ====================  ====================  ====================
