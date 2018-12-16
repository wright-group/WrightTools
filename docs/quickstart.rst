.. quickstart_

Quick Start
===========

This "quick start" page is designed to introduce a few commonly-used features that you should know immediately as a user of WrightTools.
We assume that you have installed WrightTools and that you are somewhat comfortable using Python.
If you are brand new to Python, it's typically useful to run Python within an integrated development environment---our favorite is `Spyder <https://www.spyder-ide.org/>`_.

Each of the following code blocks builds on top of the previous code.
Read this document like a series of commands typed into a Python shell.
We recommend following along on your own machine.

Create a Data Object
--------------------

There are many ways to create a WrightTools data object.
One strategy is to open an existing wt5 file.
When you downloaded WrightTools you also downloaded a few example files.
The :mod:`WrightTools.datasets` package allows you to easily access the path to these files.
Let's create a data object now:

.. code-block:: python

   import WrightTools as wt
   # get the path to an example wt5 file
   from WrightTools import datasets
   p = datasets.wt5.v1p0p1_MoS2_TrEE_movie  # just a filepath
   # open data object
   data = wt.open(p)

The data contains some helpful attributes.
We can "inspect" these attributes by simply entering them into a Python shell.
Let's do that now:

.. code-block:: python

   >>> data.channel_names
   ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'mc']
   >>> data.axis_expressions
   ['w2', 'w1=wm', 'd2']
   >>> data.shape
   (41, 41, 23)

Alternatively, we can use the :meth:`~WrightTools.data.Data.print_tree` method to print out a whole bunch of information at once.

.. code-block:: python

   >>> data.print_tree()
   _001_dat (/tmp/811qwfvb.wt5)
   ├── axes
   │   ├── 0: w2 (nm) (41, 1, 1)
   │   ├── 1: w1=wm (nm) (1, 41, 1)
   │   └── 2: d2 (fs) (1, 1, 23)
   ├── constants
   ├── variables
   │   ├── 0: w2 (nm) (41, 1, 1)
   │   ├── 1: w1 (nm) (1, 41, 1)
   │   ├── 2: wm (nm) (1, 41, 1)
   │   ├── 3: d2 (fs) (1, 1, 23)
   │   ├── 4: w3 (nm) (1, 1, 1)
   │   ├── 5: d0 (fs) (1, 1, 1)
   │   └── 6: d1 (fs) (1, 1, 1)
   └── channels
       ├── 0: ai0 (41, 41, 23)
       ├── 1: ai1 (41, 41, 23)
       ├── 2: ai2 (41, 41, 23)
       ├── 3: ai3 (41, 41, 23)
       ├── 4: ai4 (41, 41, 23)
       └── 5: mc (41, 41, 23)

Notice that the data object is made out of ``axes``, ``constants``, ``variables``, and ``channels``.
All of these are arrays, and they have different shapes and units associated with them.
For now, this is all you need to understand about the contents of data objects---read :ref:`data` when you're ready to learn more.
Next we'll visualize our data.

Visualize Data
--------------

WrightTools strives to make data visualization as quick and painless as possible.

Axes, labels, and units are brought along implicitly.

WrightTools offers a few handy ways to quickly visualize a data object, shown below.
For more information, see :ref:`artists`, or check out our `Gallery`_.

quick1D
^^^^^^^

:meth:`~WrightTools.artists.quick1D` makes it as easy as possible to visualize a simple 1D slice of our data object.
We have to specify an axis to plot along---for this example let's choose ``w1=wm``.
By default, :meth:`~WrightTools.artists.quick1D` will plot all possible slices along our chosen axis.
Optionally, we can narrow down the number of generated plots by specifying what particular coordinate we are interested in.
In this example, we have fully specified all other axes using the ``at`` keyword argument, so only one plot will be generated.

.. code-block:: python

   wt.artists.quick1D(data, 'w1=wm', at={'w2': [2, 'eV'], 'd2': [-100, 'fs']})

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   ps = datasets.wt5.v1p0p1_MoS2_TrEE_movie
   data = wt.open(ps)
   wt.artists.quick1D(data, 'w1=wm', at={'w2': [2, 'eV'], 'd2': [-100, 'fs']})
   plt.show()

quick2D
^^^^^^^

:meth:`~WrightTools.artists.quick2D` is built with the same goals as :meth:`~WrightTools.artists.quick1D`, but for two dimensional representations.
This time, we have to specify two axes to plot along---``w1=wm`` and ``d2``, in this example.
Again, we use the ``at`` keyword argument so only one plot will be generated.

.. code-block:: python

   wt.artists.quick2D(data, 'w1=wm', 'd2', at={'w2': [2, 'eV']})

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
   data = wt.open(p)
   wt.artists.quick2D(data, 'w1=wm', 'd2', at={'w2': [2, 'eV']})
   plt.show()

interact2D
^^^^^^^^^^

:meth:`WrightTools.artists.interact2D` uses Matplotlib's interactive widgets framework to present an interactive graphical interface to a multidimensional data object.
You must choose two axes to plot against in the central two-dimensional plot.
All other axes are automatically represented as "sliders", and you can easily manipulate these two explore the dataset in its full dimensionality.
See :ref:`artists` for an example.

Process Data
------------

Now let's actually modify the arrays that make up our data object. Note that the raw data which we imported is not being modified, rather we are modifying the data as copied into our data object.

Convert
^^^^^^^

WrightTools has built in units support.
This enables us to easily convert our data object from one unit system to another:

.. code-block:: python

   >>> data.units
   ('nm', 'nm', 'fs')
   >>> data.convert('eV')
   axis w2 converted from nm to eV
   axis w1=wm converted from nm to eV
   >>> data.units
   ('eV', 'eV', 'fs')

Note that only compatable axes were converted---the trailing axis with units ``'fs'`` was ignored.
Want fine control?
You can always convert individual axes, *e.g.* ``data.w2.convert('wn')``.
For more information see :ref:`units`.

Split
^^^^^

Use :meth:`~WrightTools.data.Data.split` to break your dataset into smaller pieces.

.. code-block:: python

   >>> col = data.split('d2', -100.)
   split data into 2 pieces along <d2>:
     0 : -inf to 0.00 fs (1, 1, 15)
     1 : 0.00 to inf fs (1, 1, 8)

Note that :meth:`~WrightTools.data.Data.split` accepts axis expressions and unit-aware coordinates, not axis indices.

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
   data = wt.open(p)
   col = data.split('d2', -100.)
   fig, gs = wt.artists.create_figure(cols=[1,1])
   for i, d in enumerate(col.values()):
       d = d.chop("w1=wm", "d2", at={"w2": (2, "eV")})[0]
       ax = plt.subplot(gs[i])
       ax.pcolor(d)
       ax.set_xlim(data.w1__e__wm.min(), data.w1__e__wm.max())
       ax.set_ylim(data.d2.min(), data.d2.max())
   wt.artists.set_fig_labels(xlabel=data.w1__e__wm.label, ylabel=data.d2.label)

Clip
^^^^

Use :meth:`~WrightTools.data.Channel.clip` to ignore/remove points of a channel outside of a specific range.

.. code-block:: python

   data.ai0.clip(min=0.0, max=0.1)

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
   data = wt.open(p)
   data.ai0.clip(min=0.0, max=0.1)
   wt.artists.quick2D(data, 'w1=wm', 'd2', at={'w2': [2, 'eV']})

Transform
^^^^^^^^^

Use :meth:`~WrightTools.data.Data.transform` to choose a different set of axes for your data object.

.. code-block:: python

   data.ai0.transform('w1=wm', 'w2-wm', 'd2')

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
   data = wt.open(p)
   data.transform('w1=wm', 'w2-wm', 'd2')
   data.convert('eV')
   wt.artists.quick2D(data, 'w1=wm', 'w2-wm', at={'d2': (-100, 'fs')})

Save Data
---------

It's easy to save your data objects using WrightTools.

Save, Open
^^^^^^^^^^

Most simply, you can simply save...

.. code-block:: python

   data.save('my-path.wt5')

and then open...

.. code-block:: python

   data = wt.open('my-path.wt5')

You will pick right up at the state where you saved the object (even on different operating systems or machines)!

Collections
^^^^^^^^^^^

Collections are containers that can hold multiple data objects.
Collections can nest within each-other, much like folders in your computers file system.
Collections can help you store all associated data within a single wt5 file, keeping everything internally organized.
Creating collections is easy:

.. code-block:: python

   >>> collection = wt.Collection(name='test')

Filling collections with data objects is easy as well.
Again, let's use the :mod:`WrightTools.datasets` package:

.. code-block:: python

   >>> from WrightTools import datasets
   >>> p = datasets.COLORS.v0p2_d1_d2_diagonal
   >>> wt.data.from_COLORS(p, parent=collection)
   cols recognized as v0 (19)
   data created at /tmp/w1ijzsmv.wt5::/d1_d2_diagonal_dat
     axes: ('d1', 'd2')
     shape: (21, 21)
   >>> p = datasets.ocean_optics.tsunami
   >>> wt.data.from_ocean_optics(p, parent=collection)
   data created at /tmp/w1ijzsmv.wt5::/tsunami
     range: 339.95 to 1013.55 (nm)
     size: 2048
   >>> p = datasets.PyCMDS.wm_w2_w1_000
   >>> wt.data.from_PyCMDS(p, parent=collection)
   data created at /tmp/w1ijzsmv.wt5::/3d1580hi
     axes: ('wm', 'w2', 'w1')
     shape: (35, 11, 11)

Note that we are using from functions instead of :meth:`~WrightTools.open`.
That's because these aren't wt5 files---they're raw data files output by various instruments.
We use the ``parent`` keyword argument to create these data objects directly inside of our collection.
See :ref:`Data` for a complete list of supported file formats.

Much like data objects, collection objects have a method :meth:`~WrightTools.collection.Collection.print_tree` that prints out a bunch of information:

.. code-block:: python

   >>> collection.print_tree()
   test (/tmp/w1ijzsmv.wt5)
   ├── 0: d1_d2_diagonal_dat (21, 21)
   │   ├── axes: d1 (fs), d2 (fs)
   │   ├── constants:
   │   └── channels: ai0, ai1, ai2, ai3
   ├── 1: tsunami (2048,)
   │   ├── axes: energy (nm)
   │   ├── constants:
   │   └── channels: signal
   └── 2: 3d1580hi (35, 11, 11)
       ├── axes: wm (wn), w2 (wn), w1 (wn)
       ├── constants:
       └── channels: signal_diff, signal_mean, pyro1, pyro2, pyro3, PMT voltage

Collections can be saved inside of wt5 files, so be aware that :meth:`~WrightTools.open` may return a collection or a data object based on the contents of your wt5 file.

Learning More
-------------

We hope that this quick start page has been a useful introduction to you.
Now it's time to go forth and process data!
If you want to read further, consider the following links:

* more about data objects: :ref:`data`
* more about collection objects: :ref:`collection`
* more about WrightTools artists: :ref:`artists`
* a gallery of figures made using WrightTools (click for source code): `Gallery`_
* a complete list of WrightTools units: :ref:`units`
* a complete list of attributes and methods of the ``Data`` class: :class:`~WrightTools.data.Data`

.. _Gallery: auto_examples/index.html
