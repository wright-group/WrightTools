.. _data:

Data
====

A data object contains your entire n-dimensional dataset, including axes, units, channels, and relevant metadata.
Once you have a data object, all of the other capabilities of WrightTools are immediately open to you, including processing, fitting, and plotting tools.

Instantiation
-------------

WrightTools aims to provide user-friendly ways of creating data directly from common spectroscopy file formats.
Here are the formats currently supported.

.. TODO: link syntax fields directly into API documentation
=========  ================================================================  =========================================
name       description                                                       API
---------  ----------------------------------------------------------------  -----------------------------------------
Cary 50    Files from Varian's Cary® 50 UV-Vis                               :meth:`~WrightTools.data.from_Cary50`
COLORS     Files from Control Lots Of Research in Spectroscopy               :meth:`~WrightTools.data.from_COLORS`
JASCO      Files from JASCO_ optical spectrometers.                          :meth:`~WrightTools.data.from_JASCO`
KENT       Files from "ps control" by Kent Meyer                             :meth:`~WrightTools.data.from_KENT`
NISE       Measure objects from NISE_.                                       :meth:`~WrightTools.data.from_NISE`
PyCMDS     Files from PyCMDS_.                                               :meth:`~WrightTools.data.from_PyCMDS`
scope      .scope files from ocean optics spectrometers                      :meth:`~WrightTools.data.from_scope`
Shimadzu   Files from Shimadzu_ UV-VIS spectrophotometers.                   :meth:`~WrightTools.data.from_shimadzu`
Tensor 27  Files from Bruker Tensor 27 FT-IR                                 :meth:`~WrightTools.data.from_Tensor27`
=========  ================================================================  =========================================

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
       return single(xi)[:, None] * single(yi)[None, :]
   xi = np.linspace(6000, 8000, 75)
   yi = np.linspace(6000, 8000, 75)
   zi = my_resonance(xi, yi)
   # package into data object
   axes = []
   axes.append(wt.data.Axis(xi, units='wn', name='w1'))
   axes.append(wt.data.Axis(yi, units='wn', name='w2'))
   channels = []
   channels.append(wt.data.Channel(zi, name='resonance'))
   data = wt.data.Data(axes, channels, name='example')

Note that channel objects are matrix (ij) indexed.
Cartesian (xy) indexed packages like matplotlib will expect the transform.

Structure & properties
----------------------

So what is a data object anyway?
To put it simply, ``Data`` is a collection of ``Axis`` and ``Channel`` objects.

===============  ============================
attribute        contains
---------------  ----------------------------
data.axes        wt.data.Axis objects
data.channels    wt.data.Channel objects
===============  ============================

Axis
````

Axes are the coordinates of the dataset. They have the following key attributes:

===============  ==========================================================
attribute        description
---------------  ----------------------------------------------------------
axis.label       LaTeX-formatted label, appropriate for plotting
axis.min         coordinates minimum, in current units
axis.max         coordinates maximum, in current units
axis.name        axis name
axis.points      coordinates array, in current units
axis.units       current axis units (change with ``axis.convert``)
===============  ==========================================================

Axes can also be constants (data.constants), in which case they contain a single value in points.
This is crucial for keeping track of low dimensional data within a high dimensional experimental space.

Channel
```````

Channels contain the n-dimensional data itself. They have the following key attributes:

===============  ==========================================================
attribute        description
---------------  ----------------------------------------------------------
channel.label    LaTeX-formatted label, appropriate for plotting
channel.mag      channel magnitude (furthest deviation from null)
channel.max      channel maximum
channel.min      channel minimum
channel.name     channel name
channel.null     channel null (value of zero signal)
channel.signed   flag to indicate if channel is signed
channel.values   n-dimensional array
===============  ==========================================================

Data
````

As mentioned above, the axes and channels within data can be accessed within the ``data.axes`` and ``data.channels`` lists.
Data also supports natural naming, so axis and channel objects can be accessed directly according to their name.
The natural syntax is recommended, as it tends to result in more readable code.

.. code-block:: python

   >>> data.axis_names
   ['w1', 'w2']
   >>> data.w2 == data.axes[1]
   True
   >>> data.channel_names
   ['signal', 'pyro1', 'pyro2', 'pyro3']
   >>> data.pyro2 == data.channels[2]
   True

The order of the ``data.axes`` list is crucial, as the coordinate arrays must be kept aligned with the shape of the corresponding n-dimensional data arrays.

In contrast, the order of ``data.channels`` is arbitrary.
However many methods within WrightTools operate on the zero-indexed channel by default.
For this reason, you can bring your favorite channel to zero-index using :meth:`~WrightTools.data.Data.bring_to_front`.

At many points throughout WrightTools you will need to refer to a particular axis or channel.
In such a case, you can always refer by name (string) or index (integer).

Units aware & interpolation ready
---------------------------------

Experiments are taken over all kinds of dynamic range, with all kinds of units.
You might wish to take the difference between a UV-VIS scan taken from 400 to 800 nm, 1 nm steps and a different scan taken from 1.75 to 2.00 eV, 1 meV steps.
This can be a huge pain!
Even if you converted them to the same unit system, you would still have to deal with the different absolute positions of the two coordinate arrays.

WrightTools data objects know all about units, and they implicitly use interpolation to map between different absolute coordinates.
Here we list some of the capabilities that are enabled by this behavior.

==================================================  ================================================================================
method                                              description
--------------------------------------------------  --------------------------------------------------------------------------------
:meth:`~WrightTools.data.Data.divide`               divide one channel by another, interpolating the divisor
:meth:`~WrightTools.data.Data.heal`                 use interpolation to guess the value of NaNs within a channel
:meth:`~WrightTools.data.join`                      join together multiple data objects, accounting for dimensionality and overlap
:meth:`~WrightTools.data.Data.map_axis`             re-map axis coordinates
:meth:`~WrightTools.data.Data.offset`               offset one axis based on another
:meth:`~WrightTools.data.Data.subtract`             subtract one channel from another, interpolating the subtrahend
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
:meth:`~WrightTools.data.Data.transpose`            change the order of data axes
==================================================  ================================================================================

WrightTools seamlessly handles dimensionality throughout.
:ref:`Artists` and :ref:`Fit` are places where dimensionality is addressed explicitly.

Processing without the pain
---------------------------

There are many common data processing operations in spectroscopy.
WrightTools endeavors to make these operations easy.
A selection of important methods follows.

==================================================  ================================================================================
method                                              description
--------------------------------------------------  --------------------------------------------------------------------------------
:meth:`~WrightTools.data.Data.clip`                 clip values outside of a given range
:meth:`~WrightTools.data.Data.level`                level the edge of data along a certain axis
:meth:`~WrightTools.data.Data.m`                    apply m-factor corrections
:meth:`~WrightTools.data.Data.normalize`            normalize a channel such that mag --> 1 and null --> 0
:meth:`~WrightTools.data.Data.revert`               revert the data object to an earlier state
:meth:`~WrightTools.data.Data.scale`                apply a scaling to a channel, such as square root or log
:meth:`~WrightTools.data.Data.smooth`               smooth a channel via convolution with a n-dimensional Kaiser window
:meth:`~WrightTools.data.Data.trim`                 remove outliers via a statistical test
:meth:`~WrightTools.data.Data.zoom`                 zoom a channel using spline interpolation
==================================================  ================================================================================

.. _JASCO: https://jascoinc.com/products/spectroscopy/
.. _NISE: https://github.com/wright-group/NISE
.. _PyCMDS: https://github.com/wright-group/PyCMDS
.. _Shimadzu: http://www.ssi.shimadzu.com/products/productgroup.cfm?subcatlink=uvvisspectro
