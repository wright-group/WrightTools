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
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie  # list of filepaths
   # create data object
   data = wt.data.from_COLORS(ps)

The data contains some helpful attributes that we can inspect now:

.. code-block:: python

   >>> data.channel_names
   ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'array']
   >>> data.axis_names
   ['w2', 'w1', 'd2']
   >>> data.shape
   (41, 41, 23)

Make a Plot
-----------

WrightTools strives to make data visualization as quick and painless as possible.

Axes, labels, and units are brought along implicitly.

WrightTools offers a few handy ways to quickly visualize a data object, shown below.
For more information, check see :ref:`artists`, or check out our :ref:`gallery`.

1D
^^

.. code-block:: python

   artist = wt.artists.mpl_1D(data, 'w1', at={'w2': [2, 'eV'], 'd2': [-100, 'fs']})
   artist.plot()

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   data = wt.data.from_COLORS(ps)
   artist = wt.artists.mpl_1D(data, 'w1', at={'w2': [2, 'eV'], 'd2': [-100, 'fs']})
   artist.plot()
   plt.show()

2D
^^

.. code-block:: python

   artist = wt.artists.mpl_2D(data, 'w1', 'd2', at={'w2': [2, 'eV']})
   artist.plot()

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   data = wt.data.from_COLORS(ps)
   artist = wt.artists.mpl_2D(data, 'w1', 'd2', at={'w2': [2, 'eV']})
   artist.plot()
   plt.show()

Interact with the Data
----------------------

WrightTools has built in units support. For more information see :ref:`units`.

Convert
^^^^^^^

.. code-block:: python

   >>> [a.units for a in data.axes]
   ['wn', 'wn', 'fs']
   >>> data.convert('eV')
   axis w2 converted
   axis w1 converted
   >>> [a.units for a in data.axes]
   ['eV', 'eV', 'fs']

Want fine control? You can always convert individual axes, *e.g.* ``data.w2.convert('nm')``.

Split
^^^^^

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

   data.clip('ai0', zmin=0.0, zmax=0.1)

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   data = wt.data.from_COLORS(ps)
   data.clip('ai0', zmin=0.0, zmax=0.1)
   artist = wt.artists.mpl_2D(data, 'w1', 'd2', at={'w2': [2, 'eV']})
   artist.plot()
   plt.show()
