.. quickstart_

Quick Start
===========

Create a Data Object
--------------------

Create a data object from one of the built-in datasets.

.. code-block:: python

   import WrightTools as wt
   # get some example files
   from WrightTools import datasets
   p = datasets.wt5.v1p0p1_MoS2_TrEE_movie  # list of filepaths
   # open data object
   data = wt.open(p)

The data contains some helpful attributes that we can inspect now:

.. code-block:: python

   >>> data.channel_names
   ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'mc']
   >>> data.axis_expressions
   ['w2', 'w1=wm', 'd2']
   >>> data.shape
   (41, 41, 23)

Make a Plot
-----------

WrightTools strives to make data visualization as quick and painless as possible.

Axes, labels, and units are brought along implicitly.

WrightTools offers a few handy ways to quickly visualize a data object, shown below.
For more information, check see :ref:`artists`, or check out our `gallery`_.

1D
^^

.. code-block:: python

   wt.artists.quick1D(data, 'w1', at={'w2': [2, 'eV'], 'd2': [-100, 'fs']})

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   ps = datasets.wt5.v1p0p1_MoS2_TrEE_movie
   data = wt.open(ps)
   wt.artists.quick1D(data, 'w1=wm', at={'w2': [2, 'eV'], 'd2': [-100, 'fs']})
   plt.show()

2D
^^

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

Interact with the Data
----------------------

WrightTools has built in units support. For more information see :ref:`units`.

Convert
^^^^^^^

.. code-block:: python

   >>> data.units
   ('nm', 'nm', 'fs')
   >>> data.convert('eV')
   axis w2 converted from nm to eV
   axis w1=wm converted from nm to eV
   >>> data.units
   ('eV', 'eV', 'fs')

Want fine control? You can always convert individual axes, *e.g.* ``data.w2.convert('wn')``.

Split (Coming soon)
^^^^^^^^^^^^^^^^^^^

Use ``split`` to break your dataset into smaller pieces.

.. code-block:: python

   >>> data.split('d2', 0.)
   split data into 2 pieces along d2:
     0 : -599.79 to -40.06 fs (length 15)
     1 : 39.91 to 279.70 fs (length 7)

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
