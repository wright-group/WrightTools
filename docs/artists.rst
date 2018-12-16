.. _artists:

Artists
=======

The artists module contains a variety of data visualization tools.

.. toctree:: artists
   :maxdepth: 3

Quick artists
-------------

To facilitate rapid and easy visualization of data, ``WrightTools`` offers
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
:meth:`WrightTools.artists.interact2D` allows users to easily vizualize 2D slices of arbitrarily
high dimension data.

Colors
------

Two-dimensional data is often represented using "heatmaps".
Your choice of colormap is a crucial part of how your data is perceived.
``WrightTools`` has a few choice colormaps built-in.

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
By default, WrightTools will use the "default" colormap when plotting unsigned channels and the "signed" colormap when plotting signed channels.

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

``WrightTools`` offers specialized tools for custom publication quality figures.
As an example, we will break down the figure in :ref:`sphx_glr_auto_examples_custom_fig.py`,
exploring the relationships between ``WrightTools`` and the underlying ``matplotlib``.

The preprocessing of data is handled in tools covered in :ref:`data`.

First, the full code and the image it creates:

.. literalinclude:: auto_examples/custom_fig.py
   :lines: 10-

.. image:: _images/sphx_glr_custom_fig_001.png

Layout
^^^^^^

``WrightTools`` defines a handy function, :meth:`~WrightTools.artists.create_figure`, for easily and flexibly making complicated figures.
When made with this function, :class:`~WrightTools.artists.Axes` created have additional functionality built in to work with :class:`~WrightTools.data.Data` objects directly.

:meth:`~WrightTools.artists.create_figure` makes it easy to create figures the perfect size for ``"single"`` or ``double"`` column figures for journal articles (though they are convenient in other contexts as well).

:meth:`~WrightTools.artists.create_figure` also creates a :class:`~matplotlib.GridSpec` to help layout subplots.
Columns are created with a weighted list with the number of columns, passed as ``cols``.
A special weight, ``"cbar"``, provides a fixed width column intended for color bars.
All other columns are proportionally distributed according to their weights.
The number of rows in the grid are specified with the ``nrows`` kwarg.
You can modify the aspect ratio of particular rows independently using the ``aspects`` and ``default_aspect`` kwargs.

Spacing between figures can be adjusted with the ``wspace`` and ``hspace`` kwargs for the width and height, respectively.

Axes can be accessed with :meth:`matplotlib.pyplot.subplot`.
Importantly, axes may span multiple rows/columns by using slice syntax into the gridspec.
This is demonstrated with the color bar axes here, which takes up two rows in the last column.


.. code-block:: python

   # prepare figure gridspec
   cols = [1, 1, "cbar"]
   aspects = [[[0, 0], .3]]
   fig, gs = wt.artists.create_figure(
       width="double", cols=cols, nrows=3, aspects=aspects, wspace=1.35, hspace=.35
   )   
   # plot wigners
   indxs = [(row, col) for row in range(1, 3) for col in range(2)]
   for indx, wigner, color in zip(indxs, wigners, wigner_colors):
       ax = plt.subplot(gs[indx])
   ...
   indxs = [(0, col) for col in range(2)]
   for indx, color, traces in zip(indxs, trace_colors, tracess):
       ax = plt.subplot(gs[indx])
   ...
   cax = plt.subplot(gs[1:3, -1])

Plot
^^^^

Once you have axes with the :meth:`~matplotlib.pyplot.subplot` call, it can be used as you are used to using :class:`matplotlib.axes.Axes` objects (though some defaults, such as colormap, differ from bare matplotlib).
However, you can also pass :class:`WrightTools.data.Data` objects in directly (and there are some kwargs available when you do).
These :class:`WrightTools.artists.Axes` will extract out the proper arrays and plot the data.

.. code-block:: python

   for indx, wigner, color in zip(indxs, wigners, wigner_colors):
       ax = plt.subplot(gs[indx])
       ax.pcolor(wigner, vmin=0, vmax=1)  # global colormpa
       ax.contour(wigner)  # local contours
   ...
   for indx, color, traces in zip(indxs, trace_colors, tracess):
       ax = plt.subplot(gs[indx])
       for trace, w_color in zip(traces, wigner_colors):
           ax.plot(trace, color=w_color, linewidth=1.5)

Beautify
^^^^^^^^

Once the main data is plotted, additional information can be overlaid on the axes.
Of course, standard matplotlib methods like :meth:`~matplotlib.axes.Axes.axhline` or :meth:`~matplotlib.axes.Axes.set_xlim` are all available.
In addition, ``WrightTools`` defines some small helper functions for common tasks.

- :meth:`~WrightTools.artists.set_ax_spines` Easily set color/width of the outline (spines) of an axis

  - Great for using color to connect different parts of a figure (or figures throughout a larger work)

- :meth:`~WrightTools.artists.corner_text` Quick and easy plot labeling within a dense grid

  - Pairs well with :attr:`WrightTools.data.Constant.label`

- :meth:`~WrightTools.artists.plot_colorbar` Add a colorbar in a single function call
- :meth:`~WrightTools.artists.set_fig_labels` Label axes in a whole row/column of a figure

  - Allows using slice objects to limit range affected
  - Removes axis labels from other axes in the rectangle
  - Pairs well with :attr:`WrightTools.data.Axes.label`


.. code-block:: python

   wigner_colors = ["C0", "C1", "C2", "C3"]
   trace_colors = ["#FE4EDA", "#00B7EB"]
   ...
   for indx, wigner, color in zip(indxs, wigners, wigner_colors):
       ...
       ax.grid()
       wt.artists.set_ax_spines(ax=ax, c=color)
       # set title as value of w2
       wigner.constants[0].format_spec = ".2f"
       wigner.round_spec = -1
       wt.artists.corner_text(wigner.constants[0].label, ax=ax)
       # plot overlines
       for d2, t_color in zip(d2_vals, trace_colors):
           ax.axhline(d2, color=t_color, alpha=.5, linewidth=6)
       # plot w2 placement
       ax.axvline(wigner.w2.points, color="grey", alpha=.75, linewidth=6)
   ...
   for indx, color, traces in zip(indxs, trace_colors, tracess):
       ...
       ax.set_xlim(trace.axes[0].min(), trace.axes[0].max())
       wt.artists.set_ax_spines(ax=ax, c=color)
   # plot colormap
   cax = plt.subplot(gs[1:3, -1])
   ticks = np.linspace(data.ai0.min(), data.ai0.max(), 11)
   wt.artists.plot_colorbar(cax=cax, label="amplitude", cmap="default", ticks=ticks)
   # set axis labels
   wt.artists.set_fig_labels(xlabel=data.w1__e__wm.label, ylabel=data.d2.label, col=slice(0, 1))

Save
^^^^

Saving figures is as easy as calling :meth:`~WrightTools.artists.savefig`.
This is a simple wrapper for :meth:`matplotlib.pyplot.savefig` which allows us to override defaults so that figures created with :meth:`~WrightTools.artists.create_figure` have proper margins and resolution.
If you wish to change margin padding or transparancy settings, the matplotlib function will work just as well.

.. code-block:: python

   # saving the figure as a png
   wt.artists.savefig("custom_fig.png", fig=fig, close=False)


.. _Choosing Colormaps: https://matplotlib.org/users/colormaps.html#choosing-colormaps  

.. [#green2006] **A colour scheme for the display of astronomical intensity images**
                Dave Green
                *Bulletin of the Astronomical Society of India* **2011**
                `arXiv:1108.5083 <https://arxiv.org/abs/1108.5083>`_

.. [#kindlmann2002] **Face-based luminace matching for perceptual colormap generation**
                    G. Kindlmann, E. Reinhard, and S Creem
                    *IEEE Visualization* **2002**
                    `doi:10.1109/visual.2002.1183788 <http://dx.doi.org/10.1109/visual.2002.1183788>`_
