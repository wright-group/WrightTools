.. _write_from_function:

Writing a New From Function
===========================

From functions are the entry point into the ``WrightTools`` ecosystem.
In order to use all of the data manipulations and plotting tools to their fullest, you
must have a data object to work with.
These functions come in two flavors: ``data`` from functions and ``collection`` from functions.

Data from functions create a single data object.
If multiple data objects would be generated, they should be wrapped in a collection, and be placed
in the :py:mod:`WrightTools.collection` package instead.
The process is much the same, other than the wrapper object.
Here, we will focus on the more common ``data`` flavor of from function.

Additionally, if there is extra processing that *needs* to be done at import time, it should be
questioned whether there is a raw form that is a ``data`` from function, and the processing can
then be placed in a ``collection`` from function which returns both the raw form and processed form.

Ideally any processing steps can be performed with functions of ``data``, not in the import stage.
Additional processing is more tolerated in ``collection`` from functions.

We will walk through by way of example, using :meth:`~WrightTools.data.from_JASCO`:

.. code-block:: python

    # --- import --------------------------------------------------------------
    import os
    import pathlib
    import numpy as np
    from ._data import Axis, Channel, Data
    from .. import exceptions as wt_exceptions
    # --- define ---------------------------------------------------------------
    __all__ = ["from_JASCO"]
    # --- from function --------------------------------------------------------
    def from_JASCO(filepath, name=None, parent=None, verbose=True) -> Data:
        """Create a data object from JASCO UV-Vis spectrometers.

        Parameters
        ----------
        filepath : path-like
            Path to .txt file.
            Can be either a local or remote file (http/ftp).
            Can be compressed with gz/bz2, decompression based on file name.
        name : string (optional)
            Name to give to the created data object. If None, filename is used.
            Default is None.
        parent : WrightTools.Collection (optional)
            Collection to place new data object within. Default is None.
        verbose : boolean (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        data
            New data object(s).
        """
        # parse filepath
        filestr = os.fspath(filepath)
        filepath = pathlib.Path(filepath)

        if not ".txt" in filepath.suffixes:
            wt_exceptions.WrongFileTypeWarning.warn(filepath, ".txt")
        # parse name
        if not name:
            name = filepath.name.split(".")[0]
        # create data
        kwargs = {"name": name, "kind": "JASCO", "source": filestr}
        if parent is None:
            data = Data(**kwargs)
        else:
            data = parent.create_data(**kwargs)
        # array
        ds = np.DataSource(None)
        f = ds.open(filestr, "rt")
        arr = np.genfromtxt(f, skip_header=18).T
        f.close()

        # chew through all scans
        data.create_variable(name="energy", values=arr[0], units="nm")
        data.create_channel(name="signal", values=arr[1])
        data.transform("energy")
        # finish
        if verbose:
            print("data created at {0}".format(data.fullpath))
            print("  range: {0} to {1} (nm)".format(data.energy[0], data.energy[-1]))
            print("  size: {0}".format(data.size))
        return data



Function Signature and Docstring
--------------------------------

By convention, the function name should be ``from_<kind>``.
The first argument should be a file path to the data file being read in.
If possible, this should be the only required argument to the function.
Ideally, ``from_`` functions are free of additional processing, except what is needed to
faithfully represent the data object in it's raw form.
Options which toggle or adjust processing are discouraged, as they should be performed by
users after instantiation of the object.
If there are specialized functions, consider adding them as separate functions elsewhere,
such as the :class:`WrightTools.data.Data` class.

The other standard, optional arguments are ``name``, ``parent``, and ``verbose``.
Where possible, the default ``name`` should be derived from metadata in the file itself.
If that is not possible, it should derive from the ``filename`` itself.
Consider using :meth:`~WrightTools.kit.string2identifier` to ensure that the name is a valid
python identifier.

By default, a brand new data object should be created at root of a new ``wt5`` file.
This can be overwritten by passing a :class:`~WrightTools.collection.Collection` object as ``parent``.

Finally, ``verbose`` is a boolean toggle for printing to standard out.
By convention, this is ``True`` by default.
Additionally, ``verbose`` and any custom keyword arguments should be keyword-only arguments.

The function should have a docstring that documents all parameters.
The summary line should tell about the source of the data.
Feel free to add additional information in the body of the docstring, where appropriate.
Check out the existing examples for formatting, such as the example from :meth:`~WrightTools.data.from_JASCO`.

.. code-block:: python

    def from_JASCO(filepath, name=None, parent=None, verbose=True) -> Data:
        """Create a data object from JASCO UV-Vis spectrometers.

        Parameters
        ----------
        filepath : path-like
            Path to .txt file.
            Can be either a local or remote file (http/ftp).
            Can be compressed with gz/bz2, decompression based on file name.
        name : string (optional)
            Name to give to the created data object. If None, filename is used.
            Default is None.
        parent : WrightTools.Collection (optional)
            Collection to place new data object within. Default is None.
        verbose : boolean (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        data
            New data object(s).
        """


Validation
----------

A few simple validation checks can be performed.
If it is not possible to read a data object, it should raise a ``WrightTools`` exception. See :mod:`~WrightTools.exceptions`.
If it is simply an unexpected feature, such as unusual file extension, it should raise a warning.
``WrightTools`` includes a specific warning for unexpected file type: :class:`~WrightTools.exceptions.WrongFileTypeWarning`.
We use :data:`pathlib.PurePath.suffixes` to allow for compound file extensions like ``.txt.gz``.
You should also validate the name, and extract the default in this step.

The reason to have both ``filestr`` and ``filepath`` is that :class:`pathlib.Path` objects
do not work well for urls (particularly on Windows), but pathlib is nice for performing validation.


.. code-block:: python

        # parse filepath
        filestr = os.fspath(filepath)
        filepath = pathlib.Path(filepath)
        if not ".txt" in filepath.suffixes:
            wt_exceptions.WrongFileTypeWarning.warn(filepath, ".txt")
        # parse name
        if not name:
             name = filepath.name.split(".")[0]


Create the Data object
----------------------

Instantiating the new data object involves inspecting the ``parent`` argument.
By convention, arguments to the instantiation are passed in as a keyword argument dictionary.
This should include, minimally, the ``name`` (described above), ``kind``
(specific to the particular function), and ``source`` (typically the local file path)
If the time of creation for the data is in the metadata, it should be added here, in RFC3339_ format.
The :class:`~WrightTools.kit.TimeStamp` class has a handy way of getting timestamps in this format.
Additional keyword arguments not expected by either :class:`~WrightTools.data.Data` or
:class:`~WrightTools.Group` initialization are added directly to the ``attrs`` dictionary.

.. _RFC3339: https://www.ietf.org/rfc/rfc3339.txt

.. code-block:: python

        kwargs = {"name": name, "kind": "JASCO", "source": filestr}
        if parent is None:
            data = Data(**kwargs)
        else:
            data = parent.create_data(**kwargs)

Add Metadata
------------

Additional pieces of metadata can be added into the ``attrs`` dictionary of the data object.
This can include text, numbers or even arrays.
These are arbitrary, and can be accessed like a dictionary.
Avoid using the "privileged" attributes for tasks other than their pre-defined purpose,
as overwriting may cause unexpected behavior or for them to be overwritten internally:

- ``name``
- ``class``
- ``created``
- ``kind``
- ``__version__``
- ``item_names``
- ``axes``
- ``constants``
- ``source``
- ``variable_names``
- ``channel_names``
- ``label``
- ``units``
- ``signed``
- ``null``
- ``filepath``

One way to add them is to add to the ``kwargs`` dictionary in the previous section.
Alternatively, they can be added directly:

.. code-block:: python

        data.attrs["key"] = "value"
        data.attrs.update(dictionary)



Create Variables and Channels
-----------------------------

Creating variables (things you set) and channels (things you measure) is painless.
Once you have a ``numpy`` array, (see tools such as :func:`numpy.genfromtxt`), all you have to
do is add a name, and (optionally) units.

Units are supported for both variables and channels, though tend to be more common on variables.
Supported units can be found in :mod:`~WrightTools.units`.
If there are units important to you that are not yet supported, please file an issue_.

.. _issue: https://github.com/wright-group/WrightTools/issues

For one-dimensional data formats, this is particularly easy:

.. code-block:: python

        # array
        ds = np.DataSource(None)
        f = ds.open(filestr, "rt")
        arr = np.genfromtxt(f, skip_header=18).T
        f.close()
        # add variable and channels
        data.create_variable(name="energy", values=arr[0], units="nm")
        data.create_channel(name="signal", values=arr[1])

:class:`numpy.DataSource` is a class which provides transparent decompression and remote file retrieval.
:func:`numpy.genfromtxt` will handle this itself, however it will leave the downloaded files in the
working directory, and opening explicitly allows you to use the file more directly as well.
Using ``np.DataSource(None)`` causes it to use temporary files which are removed automatically.
Opening in ``"rt"`` mode ensures that you are reading as text.

Parsing multidimensional datasets (and in particular formats which allow arbitrary dimensionality)
provides real benefit, but becomes a much more arduous task to generalize.
This is where it becomes important to consider the ``shape`` and ``units`` of the Data object.
All variables and channels must be the same rank (``ndim``) and broadcast together to get the full shape.
If variables in particular can be collapsed to a lower dimension, they should be; this is accomplished by placing a ``1`` in the shape.

For particularly complex parsing, see :meth:`~WrightTools.data.from_PyCMDS`,
:meth:`~WrightTools.data.from_KENT`, and :meth:`~WrightTools.data.from_COLORS`.
These are existing multidimensional formats used by the Wright Group, and can provide some insights.
:meth:`~WrightTools.data.from_Aramis` is an example of a multidimensional binary data format.
Feel free to reach out to the maintainers (via our `issue tracker`_) if you have any questions.

.. _issue tracker: https://github.com/wright-group/WrightTools/issues


Transform to Create Axes
------------------------

To get ``Data`` objects to behave as expected, they should be transformed to the natural axes of the
data itself.
Axes are algebraic combinations of variables (linear combinations are guaranteed to be supported).

.. code-block:: python

        data.transform("energy")

You may also add constants to your data object in your from function.
These are expressions of variables which have a constant value
(potentially with noise) in the whole of the data.

.. code-block:: python

        data.set_constants("x", "y-z")

Verbose Output
--------------

It is expected that From functions print out information at the end.
This should include the file path where the data is made, and a few lines which help users confirm
that they imported the correct data object.
Printing should be no more than about 5 lines.

For one-dimensional data, the print output tends to be the range of the axis and the size:

.. code-block:: python

        # finish
        if verbose:
            print("data created at {0}".format(data.fullpath))
            print("  range: {0} to {1} (nm)".format(data.energy[0], data.energy[-1]))
            print("  size: {0}".format(data.size))
        return data

For multidimensional formats, it tends to be the axes and shape:

.. code-block:: python

        # return
        if verbose:
            print("data created at {0}".format(data.fullpath))
            print("  axes: {0}".format(data.axis_names))
            print("  shape: {0}".format(data.shape))
        return data

Also remember to return the data object, otherwise it will not be usable immediately.

Contributing for Others to Use
------------------------------

Once you have the function, it is useful to share your code for others to use.
If you wish for your function to be included in the upstream code, take the following steps:

- Read our :ref:`contributing` page to learn how to submit a Pull Request.
- Place your function in the ``WrightTools/data`` folder with the filename ``_<lowercase kind>.py``
- Add ``__all__ = ["from_<kind>"]`` to the file.
- Import your file and add a line to the ``__all__`` defined in ``WrightTools/data/__init__.py``
- Add an example dataset in an appropriately labeled folder in ``WrightTools/datasets``
- Add your dataset to ``WrightTools/datasets/__init__.py``, e.g.:

    .. code-block:: python

        JASCO = DatasetContainer()
        JASCO._from_files("JASCO")

- Add your data kind to ``__all__`` in ``datasets/__init__.py``
- Add your dataset (with citation, if appropriate) to the table in ``docs/datasets.rst``
- Write a test which calls your ``from_<kind>`` function at ``tests/data/from_<kind>.py`` (See examples in that directory)
- Submit your Pull Request

If you have any questions, feel free to contact us via our `issue tracker`_.
