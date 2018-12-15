.. _wt5:

wt5
===

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


wt5 sub-format
--------------

The wt5 sub-format defines a few rules on top of the HDF5 data model.
Within wt5, there are two container objects derived from group: collection and data.
There are also two array objects derived from dataset: variable and channel.

Collection
^^^^^^^^^^

Collection objects are containers like folders in a file system.
They can contain any mixture of collections and data objects.
The contents of a collection can be accessed in a variety of convinient ways with WrightTools.
As an example, let's create a simple wt5 file now.

.. code-block:: python

   import WrightTools as wt
   results = wt.Collection(name='results')

We have created a new file with a root-level collection named results.
Let's add some data to our collection.

.. code-block:: python

   results.create_data(name='neat')
   results.create_data(name='messy')
   results.create_data(name='confusing')

We can access treat our collection like a dictionary with methods ``keys``, ``values``, and ``items``.

.. code-block:: python

   >>> list(results.values())
   [<WrightTools.Data 'neat'>, <WrightTools.Data 'messy'>, <WrightTools.Data 'confusing'>]

We can also access by key, or by index. 
We can even use natural naming!

.. code-block:: python

   >>> results[1]
   <WrightTools.Data 'messy'>
   >>> results['neat']
   <WrightTools.Data 'neat'>
   >>> results.confusing
   <WrightTools.Data 'confusing'>

Jeez, it would be nice to also keep track of the calibration data from our experiment.
Let's add a child collection called calibration within our root results collection.
We'll fill this collection with our calibration data.

.. code-block:: python

   calibration = results.create_collection(name='calibration')
   calibration.create_data(name='OPA1_tune_test')
   calibration.create_data(name='OPA2_tune_test')

This child collection can be accessed in all of the ways mentioned above (dictionary, index, natural naming).
The child collections and data objects hold a reference to the parent.

.. code-block:: python

   >>> calibration.parent
   <WrightTools.Collection 'results'>

In sumarry, we have created a wt5 file with the following structure:

.. code-block:: bash

   collection results
   ├─ data neat
   ├─ data messy
   ├─ data confusing
   └─ collection calibration
      ├─ data OPA1_tune_test
      └─ data OPA2_tune_test 

Collections can be nested and added to arbitrarily to optimally organize and share results.

Note that the collections do not directly contain datasets.
Datsets are children of the data objects.
We discuss data objects in the next section.

Data
^^^^

Data is, in some sense, the central object of WrightTools.
Within the HDF5 file, a data object is merely a group containing severeal datasets.
According to the rules of wt5, these datasets have very specific relationships that bind them together into a single cohesive data object.

Each data object has a specific multidimensional shape.

Datasets are divided into two categories: variables and channels.
Variables correspond to independent axes of the experiment: things like OPA color, delay, and monochromator position.
Channels correspond to dependent measurements: output intensities measured by detectors.
Every variable and channel must have the same number of dimensions as its parent data object.
However, the length of one or more of those dimensions may be one.
This means that these arrays need not contain a unique point for every location in data.

Axes are a thin wrapper that describe a specific algebraic expression made up of one or more component variables.
Axes are key for plottinng and interacting with data objects, but they do not directly contain the arrays within the HDF5 file.
Rather, axes are simply stored as strings within the ``attrs`` metadata dictionary of the data object.


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
