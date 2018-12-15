.. quickstart_

Quick Start
===========

This "quick start" page is designed to introduce a few commonly-used features that you should know right off the bat as a user of WrightTools.
We assume that you have installed WrightTools and that you are somewhat comfortable using Python.
If you are brand new to Python, it's typically useful to run Python within an integrated development environment---our favorite is (TODO: LINK) Spyder.

Each of the following code blocks builds on top of the previous code.
Read this document like a series of commands typed into a Python shell.
We recommend following along on your own machine.

Create a Data Object
--------------------

There are many ways to create a WrightTools data object.
The most straight-forward strategy is to open an existing wt5 file.
When you downloaded WrightTools you also downloaded a few example files.
The :mod:`WrightTools.datasets` package allows you to easily access the path to these files.
Let's create a data object now:

.. code-block:: python

   import WrightTools as wt
   # get an example wt5 file
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

Alternatively, we can use the :meth:`WrightTools.data.Data.print_tree` method to print out a whole bunch of information at once.

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
For more information, check see :ref:`artists`, or check out our `gallery`_.

quick1D
^^^^^^^

:meth:`WrightTools.artists.quick1D` makes it as easy as possible to visualize a simple 1D slice of our data object.
We have to specify an axis to plot along---for this example let's choose ``w1=wm``.
By default, :meth:`WrightTools.artists.quick1D` will plot all possible slices along our chosen axis.
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

:meth:`WrightTools.artists.quick2D` is built with the same goals as :meth:`WrightTools.artists.quick1D`, but for two dimensional representations.
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

TODO

Process Data
------------

Now let's actually modify the arrays that make up our data object.

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

Use ``split`` to break your dataset into smaller pieces.

.. code-block:: python

   >>> col = data.split('d2', 0.)
   split data into 2 pieces along <d2>:
     0 : -inf to 0.00 fs (1, 1, 15)
     1 : 0.00 to inf fs (1, 1, 8)

TODO: split at -100 instead.
TODO: note that coordinates, with units, were used---not indices.

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
   data = wt.open(p)
   col = data.split('d2', 0.)
   fig, gs = wt.artists.create_figure(cols=[1,1])
   for i, d in enumerate(col.values()):
       d = d.chop("w1=wm", "d2", at={"w2": (2, "eV")})[0]
       ax = plt.subplot(gs[i])
       ax.pcolor(d)
       ax.set_xlim(data.w1__e__wm.min(), data.w1__e__wm.max())
       ax.set_ylim(data.d2.min(), data.d2.max())
   wt.artists.set_fig_labels(xlabel=data.w1__e__wm.label, ylabel=data.d2.label)
   plt.show()

Clip
^^^^

Use ``clip`` to ignore points outside of a specific range.

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
   plt.show()

.. _gallery: auto_examples/index.html

Transform
^^^^^^^^^

TODO

Save Data
---------

TODO

Save, Open
^^^^^^^^^^

TODO

Collections
^^^^^^^^^^^

TODO
