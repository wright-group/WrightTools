.. _collection:

Collection
==========

Collection
----------

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
