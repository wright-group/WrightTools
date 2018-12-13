.. _data:

Data
====

A data object contains your entire n-dimensional dataset, including axes, units, channels, and relevant metadata.
Once you have a data object, all of the other capabilities of WrightTools are immediately open to you, including processing, fitting, and plotting tools.

Instantiation
-------------

WrightTools aims to provide user-friendly ways of creating data directly from common spectroscopy file formats.
Here are the formats currently supported.

=============  ================================================================  =========================================
name           description                                                       API
-------------  ----------------------------------------------------------------  -----------------------------------------
BrunoldrRaman  Files from Brunold_ lab resonance raman measurements              :meth:`~WrightTools.data.from_BrunoldrRaman`
Cary           Files from Varian's Cary® Spectrometers                           :meth:`~WrightTools.collection.from_Cary`
COLORS         Files from Control Lots Of Research in Spectroscopy               :meth:`~WrightTools.data.from_COLORS`
JASCO          Files from JASCO_ optical spectrometers.                          :meth:`~WrightTools.data.from_JASCO`
KENT           Files from "ps control" by Kent Meyer                             :meth:`~WrightTools.data.from_KENT`
PyCMDS         Files from PyCMDS_.                                               :meth:`~WrightTools.data.from_PyCMDS`
Ocean Optics   .scope files from ocean optics spectrometers                      :meth:`~WrightTools.data.from_ocean_optics`
Shimadzu       Files from Shimadzu_ UV-VIS spectrophotometers.                   :meth:`~WrightTools.data.from_shimadzu`
SPCM           Files from Becker & Hickl spcm_ software                          :meth:`~WrightTools.data.from_spcm`
Tensor 27      Files from Bruker Tensor 27 FT-IR                                 :meth:`~WrightTools.data.from_Tensor27`
=============  ================================================================  =========================================

Is your favorite format missing?
It's easy to add---promise! Check out :ref:`contributing`.

Got bare numpy arrays and dreaming of data?
It is possible to create data objects directly in special circumstances, as shown below.

.. code-block:: python

   # import
   import numpy as np
   import WrightTools as wt
   # generate arrays for example
   def my_resonance(xi, yi, intensity=1, FWHM=500, x0=7000):
       def single(arr, intensity=intensity, FWHM=FWHM, x0=x0):
           return intensity*(0.5*FWHM)**2/((xi-x0)**2+(0.5*FWHM)**2)
       return single(xi) * single(yi)
   xi = np.linspace(6000, 8000, 75)[:, None]
   yi = np.linspace(6000, 8000, 75)[None, :]
   zi = my_resonance(xi, yi)
   # package into data object
   data = wt.Data(name='example')
   data.create_variable(name='w1', units='wn', values=xi)
   data.create_variable(name='w2', units='wn', values=yi)
   data.create_channel(name='signal', values=zi)
   data.transform('w1', 'w2')

Structure & properties
----------------------

So what is a data object anyway?
To put it simply, ``Data`` is a collection of ``Axis`` and ``Channel`` objects.
``Axis`` objects are composed of ``Variable`` objects.

===============  ============================
attribute        tuple of...
---------------  ----------------------------
data.axes        wt.data.Axis objects
data.constants   wt.data.Constant objects
data.channels    wt.data.Channel objects
data.variables   wt.data.Variable objects
===============  ============================

See also `Data.axis_expressions`, `Data.constant_expressions`, `Data.channel_names` and `Data.variable_names`.

Axis
````

Axes are the coordinates of the dataset. They have the following key attributes:

=================  ==========================================================
attribute          description
-----------------  ----------------------------------------------------------
axis.label         LaTeX-formatted label, appropriate for plotting
axis.min()         coordinates minimum, in current units
axis.max()         coordinates maximum, in current units
axis.natural_name  axis name
axis.units         current axis units (change with ``axis.convert``)
axis.variables     component variables
axis.expression    expression
=================  ==========================================================

Constant
````````

Constants are a special subclass of Axis objects, which is expected to be a single value.
Constant adds the value to to the label attribute, suitable for titles of plots to identify
static values associated with the plot.
Note that there is nothing enforcing that the value is actually static: constants still have
shapes and can be indexed to get the underlying numpy array.
You can control how this label is generated using the attributes ``format_spec`` an ``round_spec``.
``label`` uses the python builtin ``format``, an thus format_spec is a specification as in the 
`Format Specification Mini-Language`_.
Common examples would be "0.2f" or "0.3e" for decimal representation with two digits past the decimal
and engineers notation with 3 digits past the decimal, respectively.
``round_spec`` allows you to control the rounding of your number via the `builtin`_ ``round()``.
For instance, if you want a number rounded to the hundreds position, but represented as an integer, you may use ``round_spec=2; format_spec="0.0f"``.
For example, if you have a constant with value ``123.4567 nm``, a ``format_spec`` of ``0.3f``, and a ``round_spec`` of ``-2``, you will get a label something like ``\lambda_{1}=123.460 nm``, which will render as :math:`\lambda_{1}=123.460 nm`.
In addition to the above attributes, constants add/modify:

_`Format Specification Mini-Language`: https://docs.python.org/3/library/string.html#formatspec
_`builtin`: https://docs.python.org/3/library/functions.html#round

====================  ========================================================================
attribute             description\lambda_{1}=123.460 nm
--------------------  ------------------------------------------------------------------------
constant.format_spec  Format specification for how to represent the value, as in ``format()``.
constant.round_spec   Specify which digit to round to, as in `round()`
constant.label        LaTeX formatted label which includes a symbol and the constant value.
constant.value        The mean (ignoring NaNs) of the evaluated expression.
constant.std          The standard deviation of the points used to compute the value.
====================  ========================================================================

Channel
```````

Channels contain the n-dimensional data itself. They have the following key attributes:

===============  ==========================================================
attribute        description
---------------  ----------------------------------------------------------
channel.label    LaTeX-formatted label, appropriate for plotting
channel.mag()    channel magnitude (furthest deviation from null)
channel.max()    channel maximum
channel.min()    channel minimum
channel.name     channel name
channel.null     channel null (value of zero signal)
channel.signed   flag to indicate if channel is signed
===============  ==========================================================

Data
````

As mentioned above, the axes and channels within data can be accessed within the ``data.axes`` and ``data.channels`` lists.
Data also supports natural naming, so axis and channel objects can be accessed directly according to their name.
The natural syntax is recommended, as it tends to result in more readable code.

.. code-block:: python

   >>> data.axis_expressions
   ('w1', 'w2')
   >>> data.w2 == data.axes[1]
   True
   >>> data.channel_names
   ('signal', 'pyro1', 'pyro2', 'pyro3')
   >>> data.pyro2 == data.channels[2]
   True

The order of axes and channels is arbitrary.
However many methods within WrightTools operate on the zero-indexed channel by default.
For this reason, you can bring your favorite channel to zero-index using :meth:`~WrightTools.data.Data.bring_to_front`.

Units aware & interpolation ready
---------------------------------

Experiments are taken over all kinds of dynamic range, with all kinds of units.
You might wish to take the difference between a UV-VIS scan taken from 400 to 800 nm, 1 nm steps and a different scan taken from 1.75 to 2.00 eV, 1 meV steps.
This can be a huge pain!
Even if you converted them to the same unit system, you would still have to deal with the different absolute positions of the two coordinate arrays.
:meth:`~WrightTools.data.Data.map_variable` allows you to easily obtain a data object mapped onto a different set of coordinates.

WrightTools data objects know all about units, and they are able to use interpolation to map between different absolute coordinates.
Here we list some of the capabilities that are enabled by this behavior.

==================================================  ================================================================================
method                                              description
--------------------------------------------------  --------------------------------------------------------------------------------
:meth:`~WrightTools.data.Data.heal`                 use interpolation to guess the value of NaNs within a channel
:meth:`~WrightTools.data.join`                      join together multiple data objects, accounting for dimensionality and overlap
:meth:`~WrightTools.data.Data.map_variable`         re-map data coordinates
:meth:`~WrightTools.data.Data.offset`               offset one axis based on another
==================================================  ================================================================================

Dimensionality without the cursing
----------------------------------

Working with multidimensional data can be intimidating.
What axis am I looking at again?
Where am I in the other axis?
Is this slice unusual, or do they all look like that?

WrightTools tries to make multi-dimensional data easy to work with.
The following methods deal directly with dimensionality manipulation.

==================================================  ================================================================================
method                                              description
--------------------------------------------------  --------------------------------------------------------------------------------
:meth:`~WrightTools.data.Data.chop`                 chop data into a list of lower dimensional data
:meth:`~WrightTools.data.Data.collapse`             destroy one dimension of data using a mathematical strategy
:meth:`~WrightTools.data.Data.split`                split data at a series of coordinates, without reducing dimensionality
==================================================  ================================================================================

WrightTools seamlessly handles dimensionality throughout.
:ref:`Artists` is one such place where dimensionality is addressed explicitly.

Processing without the pain
---------------------------

There are many common data processing operations in spectroscopy.
WrightTools endeavors to make these operations easy.
A selection of important methods follows.

==================================================  ================================================================================
method                                              description
--------------------------------------------------  --------------------------------------------------------------------------------
:meth:`~WrightTools.data.Data.clip`                 clip values outside of a given range
:meth:`~WrightTools.data.gradient`                  take the derivative along an axis
:meth:`~WrightTools.data.join`                      join multiple data objects into one
:meth:`~WrightTools.data.Data.level`                level the edge of data along a certain axis
:meth:`~WrightTools.data.Data.smooth`               smooth a channel via convolution with a n-dimensional Kaiser window
:meth:`~WrightTools.data.Data.zoom`                 zoom a channel using spline interpolation
==================================================  ================================================================================

.. _Brunold: http://brunold.chem.wisc.edu/
.. _JASCO: https://jascoinc.com/products/spectroscopy/
.. _NISE: https://github.com/wright-group/NISE
.. _PyCMDS: https://github.com/wright-group/PyCMDS
.. _Shimadzu: http://www.ssi.shimadzu.com/products/productgroup.cfm?subcatlink=uvvisspectro
.. _spcm: http://www.becker-hickl.com/software/spcm.htm
