.. _artists:

Artists
=======

The artists module contains a variety of data visualizaton tools.

Quick artists
-------------

To facilitate rapid and easy visualization of data, WrightTools offers
“quick” artist functions which quickly generate 1D or 2D
representations.
These functions are made to make good representations by default, but
they do have certain keyword arguments to make popular customization
easy.
These are particular useful functions within the context of
auto-generated plots in acquisition software.

:meth:`WrightTools.artists.quick1D` is a function that generates 1D representations.

.. plot::

    import WrightTools as wt
    from WrightTools import datasets
    import matplotlib.pyplot as plt
    wt.artists.apply_rcparams('default')
    # import data
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    data = wt.open(p)
    data.transform("w1", "w2", "d2")
    # probe freqency trace
    wt.artists.quick1D(data, axis=0, at={"w2": [1.7, "eV"], "d2": [0, "fs"]})
    # delay trace
    wt.artists.quick1D(data, axis="d2", at={"w2": [1.7, "eV"], "w1": [1.65, "eV"]})
    plt.show()

:meth:`WrightTools.artists.quick2D` is a function that generates 2D representations.

.. plot::

    import WrightTools as wt
    from WrightTools import datasets
    import matplotlib.pyplot as plt
    wt.artists.apply_rcparams('default')
    # import data
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    data = wt.open(p)
    data.transform("w1", "w2", "d2")
    # probe wigner
    wt.artists.quick2D(data, xaxis=0, yaxis=2, at={"w2": [1.7, "eV"]})
    # 2D-frequency
    wt.artists.quick2D(data, xaxis="w1", yaxis="w2", at={"d2": [0, "fs"]})
    plt.show()

Note that the actual quick functions are each one-liners. Keyword
arguments such as ``autosave`` and ``save_directory`` may be supplied if
the user desires to save images (not typical for users in interactive
mode). The ``channel`` kwarg allows users to specify what channel they
would like to plot.

Perhaps the most powerful feature of :meth:`WrightTools.artists.quick1D`
and :meth:`WrightTools.artists.quick2D` are
their ability to treat higher-dimensional datasets by automatically
generating multiple figures. When handing a dataset of higher
dimensionality to these artists, the user may choose which axes will
be plotted against using keyword arguments.
Any axis not plotted against will be iterated over such that an image
will be generated at each coordinate in that axis. Users may also
provide a dictionary with entries of the form
``{axis_name: [position, units]}`` to choose a specific coordinates
along non-plotted axes. Positions along non-plotted axes are reported
in the title of each plot and overlines are shown when applicable.
These functionalities are derived from :meth:`WrightTools.data.Data.chop`.

Interactive artists
-------------------

:meth:`WrightTools.artists.interact2D` facilitates interaction with multidimensional
datasets.

.. plot::

    import WrightTools as wt
    from WrightTools import datasets
    import matplotlib.pyplot as plt
    # import data
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    data = wt.open(p)
    interact = wt.artists.interact2D(data, xaxis=0, yaxis=2, local=True, verbose=False)
    # show-off functionality. The following lines are not needed when in an interactive mode.
    interact[1]['w2'].set_val(40) # hack w2 slider
    fig = plt.gcf()
    # simulate mouse event to get crosshairs
    fig.canvas.button_release_event(160, 375, 1)
    plt.show()

Side plots show x and y projections of the slice (shaded gray). Left
clicks on the main axes draw 1D slices on side plots at the coordinates
selected. Right clicks remove the 1D slices. For 3+ dimensional data,
sliders below the main axes are used to change which slice is viewed.
:meth:`WrightTools.artists.interact2D` allows users to easily vizualize 2D slices of arbitrarly
high dimension data.

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
