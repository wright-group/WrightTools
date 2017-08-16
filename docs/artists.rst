.. _artists:

Artists
=======

The artists module contains a variety of data visualizaton tools.

Artist objects
--------------

==================================================  ==================================================  ====================================
artist                                              description                                         gallery links
==================================================  ==================================================  ====================================
:meth:`~WrightTools.artists.mpl_1D`                 generic 1D slice(s)                                 :ref:`1 <sphx_glr_auto_examples_simple_1D.py>`
:meth:`~WrightTools.artists.mpl_2D`                 generic 2D slice(s)                                 :ref:`1 <sphx_glr_auto_examples_simple_2D.py>`
:meth:`~WrightTools.artist.absorbance`              absorbance spectra                                  :ref:`1 <sphx_glr_auto_examples_absorbance.py>`, :ref:`2 <sphx_glr_auto_examples_batch_4_comparison.py>` 
:meth:`~WrightTools.artists.difference_2D`          2D difference slice(s)                              :ref:`1 <sphx_glr_auto_examples_diff2D.py>`
==================================================  ==================================================  ====================================

Colors
------

Two-dimensional data is often represented using "heatmaps".
Your choice of colormap is a crucial part of how your data is perceived.
WrightTools has a few choice colormaps built-in.

.. plot::
   :include-source: False

   import numpy as np
   import matplotlib.pyplot as plt
   import WrightTools as wt
   
   num = len(wt.artists.colormaps)
   fig, axes = plt.subplots(nrows=num*3, figsize=(6, num/2.5))
   fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
   gradient = np.linspace(0, 1, 256)
   gradient = np.vstack((gradient, gradient))
   axis_index = 0
   
   for name, cmap in wt.artists.colormaps.items():
       # color
       ax = axes[axis_index]
       ax.imshow(gradient, aspect='auto', cmap=wt.artists.grayify_cmap(cmap))
       axis_index += 1
       # color
       ax = axes[axis_index]
       ax.imshow(gradient, aspect='auto', cmap=cmap)
       pos = list(ax.get_position().bounds)
       x_text = pos[0] - 0.01
       y_text = pos[1] + pos[3]
       fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
       axis_index += 2
   
   for ax in axes:
           ax.set_axis_off()

All of these are held in the `colormaps` dictionary.

.. code-block:: python

   >>> wt.artists.colormaps['default']
   <matplotlib.colors.LinearSegmentedColormap at 0x7f6d8b658d30>

Throughout WrightTools you can refer to colormaps by their name.
By default, WrightTools will use the default (signed) colormap when plotting un(signed) channels.

There are many great resources on how to choose the best colormap.
`Choosing Colormaps`_ is a great place to start reading.
WrightTools tries to use perceptual colormaps wherever possible.
When a large dynamic range is needed, the data can always be scaled to accommodate. 

The default colormap is based on the wonderful cubehelix color scheme. [#green2006]_
The cubehelix parameters have been fine-tuned to roughly mimic the colors of the historically popular "jet" colormap.

The isoluminant series are instances of the color scheme proposed by Kindlmann *et al.* [#kindlmann2002]_

The skyebar series were designed by Schuyler (Skye) Kain for use in his instrumental software package COLORS.

wright and signed_old are kept for legacy purposes.

Custom figures
--------------

WrightTools offers specialized tools for custom figure generation.
It is often difficult to 

Layout
^^^^^^

Layout documentation coming soon.

Plot
^^^^

Plot documentation coming soon.

Beautify
^^^^^^^^

Beautify documentation coming soon.

Save
^^^^

Save documentation coming soon.

.. _Choosing Colormaps: https://matplotlib.org/users/colormaps.html#choosing-colormaps  

.. [#green2006] **A colour scheme for the display of astronomical intensity images**
                Dave Green
                *Bulletin of the Astronomical Society of India* **2011**
                `arXiv:1108.5083 <https://arxiv.org/abs/1108.5083>`_

.. [#kindlmann2002] **Face-based luminace matching for perceptual colormap generation**
                    G. Kindlmann, E. Reinhard, and S Creem
                    *IEEE Visualization* **2002**
                    `doi:10.1109/visual.2002.1183788 <http://dx.doi.org/10.1109/visual.2002.1183788>`_
