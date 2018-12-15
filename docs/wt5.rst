.. _wt5:

The wt5 File Format
===================

WrightTools stores data in binary wt5 files.

wt5 is a sub-format of `HDF5 <https://support.hdfgroup.org/HDF5/>`_.

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

Changes internal handling of strings. Bare strings are no longer call ``encode()`` before storing.

Version 1.0.2
^^^^^^^^^^^^^

Adds "constants" as a stored attribute in the attrs dictionary, a list of strings just like axes.
