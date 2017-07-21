.. _quickstart

Quick Start
===========

Create a Data Object
--------------------

.. code-block:: python

   # get some example files
   from WrightTools import dataset
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   # create data object
   data = wt.data.from_COLORS(ps)

Make a Plot
-----------


.. code-block:: python

   artist = wt.artists.mpl_2D(data, 'w1', 'd2', at={'w2': [1.5, 'eV']})
	   artist.plot()

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   data = wt.data.from_COLORS(ps)
   artist = wt.artists.mpl_2D(data, 'w1', 'd2', at={'w2': [2, 'eV']})
   artist.plot()
   plt.show()

Interact with The Data
----------------------

(TESTING)

:ref:`sphx_glr_auto_examples_myexample.py`

