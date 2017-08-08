.. _artists:

Artists
=======

The artists module contains a variety of data visualizaton tools.

Artist objects
--------------

==================================================  ==================================================  ====================================
artist                                              description                                         gallery links
--------------------------------------------------  --------------------------------------------------  ------------------------------------
:meth:`~WrightTools.artists.mpl_1D`                 generic 1D slice(s)                                 :ref:`1 <sphx_glr_auto_examples_simple_1D.py>`
:meth:`~WrightTools.artists.mpl_2D`                 generic 2D slice(s)                                 :ref:`1 <sphx_glr_auto_examples_simple_2D.py>`
:meth:`~WrightTools.artist.absorbance`              absorbance spectra                                  :ref:`1 <sphx_glr_auto_examples_absorbance.py>`
:meth:`~WrightTools.artists.difference_2D`          2D difference slice(s)                              :ref:`1 <sphx_glr_auto_examples_diff2D.py>`
==================================================  ==================================================  ====================================

Colors
------

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

Custom figures
--------------

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
