.. _wt5:

The wt5 File Format
===================

WrightTools stores data in binary wt5 files.

wt5 is a sub-format of `HDF5 <https://support.hdfgroup.org/HDF5/>`_.

wt5
---

``wt5`` files are hdf5 files with particular structure and attributes defined.
``wt5`` objects may appear embedded within a larger hdf5 file or vise-versa, however this is untested.
At the root of a ``wt5`` file, a :class:`~WrightTools.collection.Collection` or :class:`~WrightTools.data.Data` object is found.
:class:`~WrightTools.collection.Collection` and :class:`~WrightTools.data.Data` are hdf5 groups.
A :class:`~WrightTools.collection.Collection` may have children consisting of :class:`~WrightTools.collection.Collection` and/or :class:`~WrightTools.data.Data`.
A :class:`~WrightTools.data.Data` may have children consisting of :class:`~WrightTools.data.Variable` and/or :class:`~WrightTools.data.Channel`.
:class:`~WrightTools.data.Variable` and :class:`~WrightTools.data.Channel` are hdf5 datasets.

Metadata
^^^^^^^^

The following metadata is handled within ``WrightTools`` and define the necessary attributes to be a ``wt5`` file.
It is recommended not to write over these attributes manually except at import time (e.g. ``from_<x>`` function).

===================  ===========  ==========  ==========  ==========  ============================================
name                 Collection   Data        Variable    Channel     description/notes
===================  ===========  ==========  ==========  ==========  ============================================
``name``             yes          yes         yes         yes         Usually matches the last component of the path,
                                                                      except for root, ``/``, which does not have a path with it's name
``class``            yes          yes         yes         yes         Identifies which kind of WrightTools object it is.
``created``          yes          yes                                 Timestamp of when the object was made,
                                                                      can be overwritten with source file creation time by ``from_<x>`` functions.
``__version__``      yes          yes                                 ``wt5`` version identifier
``item_names``       yes          yes                                 Ordered list of the children
``variable_names``                yes                                 Ordered list of all Variables
``channel_names``                 yes                                 Ordered list of all Channels
``axes``                          yes                                 Ordered list of axes expressions which define how a Data object is represented
``constants``                     yes                                 Ordered list of expressions for values which are constant
``kind``                          yes                                 Short description of what type of file it originated
                                                                      from, usually the instrument
``source``                        yes                                 File path/url to the original file as read in
``label``                                     yes         yes         Identifier used to create more complex labels in
                                                                      Axes or Constants, which are used to plot
``units``                                     yes         yes         Units assigned to the dataset
``min``                                       yes         yes         Cached minimum value
``max``                                       yes         yes         Cached maximum value
``argmin``                                    yes         yes         Cached index of minimum value
``argmax``                                    yes         yes         Cached index of maximum value
``signed``                                                yes         Boolean for treating channel as signed/unsigned
===================  ===========  ==========  ==========  ==========  ============================================


HDF5
----

The HDF5 data model contains two primary objects: the group and the dataset.
Groups are used to hierarchically organize content within the file.
Each group is a container for datasets and other groups.
Think of groups like folders in your computers file system.
Every HDF5 file contains a top-level root group, signified by ``/``.

Datasets are specialty containers for raw data values.
Think of datasets like multidimensional arrays, similar to the numpy `ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_.
Each dataset has a specific data type, such as integer, float, or character.

Groups and datasets can contain additional `metadata <https://en.wikipedia.org/wiki/Metadata>`_.
This metadata is stored in a key: value pair system called ``attrs``, similar to a python dictionary.

Much more information can be found on the `HDF5 tutorial <https://support.hdfgroup.org/HDF5/Tutor/>`_.

WrightTools relies upon the `h5py package <http://www.h5py.org/>`_, a Pythonic interface to HDF5.

Access
------

wt5 is a binary format, so it cannot be interpreted with traditional text editors.
Since wt5 is a sub-format of HDF5, WrightTools benefits from the ecosystem of HDF5 tools that already exists.
This means that it is possible to import and interact with wt5 files without WrightTools, or even without python.

ASCII
^^^^^

Export an HDF5 file to a human-readable ASCII file using `h5dump <https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Dump>`_.

See also `HDF to Excel <https://support.hdfgroup.org/HDF5/HDF5-FAQ.html#toexcel>`_.

Fortran
^^^^^^^

Use the official `HDF5 Fortran Library <https://support.hdfgroup.org/HDF5/doc/fortran/index.html>`_.

Graphical
^^^^^^^^^

`HDF COMPASS <https://support.hdfgroup.org/projects/compass/index.html>`_, a simple tool for navigating and viewing data within HDF5 files (no editing functionality).

`HDF VIEW <https://support.hdfgroup.org/products/java/hdfview/index.html>`_, a visual tool for browsing and editing HDF5 files.

MATLAB
^^^^^^

MATLAB offers built-in `high-level HDF5 functions <https://www.mathworks.com/help/matlab/high-level-functions.html>`_ including ``h5disp``, ``h5read``, and ``h5readatt``.

Python (without WrightTools)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We reccomend the amazing `h5py package <http://www.h5py.org/>`_.

Shell
^^^^^

`h5cli <https://gitlab.com/h5cli/h5cli>`_: bash-like interface to interacting with HDF5 files.

`h5diff <https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Diff>`_: compare two HDF5 files, reporting the differences.

`h5ls <https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Ls>`_: print information about one or more HDF5 files.

`Complete list of official HDF5 tools <https://support.hdfgroup.org/HDF5/doc/RM/Tools.html>`_


Changes
-------

Version 1.0.0
^^^^^^^^^^^^^

Initial release of the format.

Version 1.0.1
^^^^^^^^^^^^^

Changes internal handling of strings. Bare strings are no longer required to call ``encode()`` before storing.

Version 1.0.2
^^^^^^^^^^^^^

Adds "constants" as a stored attribute in the attrs dictionary, a list of strings just like axes.

Version 1.0.3
^^^^^^^^^^^^^

Changed identity as stored in attrs dictionary (``axis`` and ``constant``) to use the ``expression`` including operators.
Previous versions exhibited a bug where decimal points would be ignored when the expression was generated from the attrs (thus "2.0" would be stored as "2_0" and read in as "20").

