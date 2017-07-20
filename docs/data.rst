.. _data:

Data
====

The ``Data`` class is an incredibly general class capable of representing any rectangular dataset.
It fills two important roles within WrightTools:

#. ``Data`` defines a single format to represent the results of almost any measurement or simulation.
#. ``Data`` provides a suite of methods for data manipulation.

Almost every capability of WrightTools relies upon ``Data`` to some exent.
In this way, ``Data`` can be thought of as the glue that holds the entire Wright group software stack together.

Structure & properties
----------------------

The heart of ``Data`` are the ``np.ndarrays`` that contain the numbers themselves.
Everything else is, in some sense, decoration.
The arrays are not attributes of ``Data`` itself, rather they are attributes of the closely related ``wt.data.Axis`` and ``wt.data.Channel`` classes.
``Data`` contains lists of ``Axis`` es and ``Channel`` s.
To understand the basic structure of ``Data``, then, one must first understand the structure of ``Axis`` and ``Channel``.

Axes are the coordinates of points in a dataset.
Since all datasets are rectangular, axes are always one-dimensional.
``wt.data.Axis`` is the class that contains these coordinates and their properties.
Axes may be regular or differential.

Channels are the values in a dataset.
A ``Data`` instance may contain many channels, since each coordinate may have multiple recorded values e.g. signal, pyro1, and pyro2.
``wt.data.Channel`` is the class that contains these values and their properties.
The ``signed`` flag of ``Channel`` changes how other parts of WrightTools interact with the dataset.
For example, an artist may choose to use a different colorbar if the dataset is signed.

Allowing the null value of a datset to be non-zero is an important feature in WrightTools.
When possible, developers should anticipate non-zero ``znull``.
Also note that ``Channel.values`` may contain ``NaN`` s.
When manipulating the ``values`` array directly, call ``Channel._update()`` to force the flags and attributes of the object to update.

The ``Data`` object has two primary attributes: ``axes``, a list of ``Axis`` objects and ``channels``, a list of ``Channel`` objects.
``axes`` contain exactly one ``Axis`` object for each dimension of the dataset.
The order of ``axes`` maters, so that the nth member of ``axes`` contains the coordinates along the nth dimension of the dataset.
The order of ``channels`` is arbitrary.
It is typical for code in WrightTools to assume that the most important channel is first.
For this reason, ``Data`` offers a ``bring_to_front`` method.

WrightTools data objects use natural naming.

Instantiation
-------------

Manipulaton
-----------

