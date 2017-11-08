.. _fit:

Fit
===

WrightTools provides a suite of fitting tools.

Function
--------

Function objects are the real workforce of fitting in WrightTools.
Let's look at one now.

.. code-block:: python

   >>> import WrightTools as wt
   >>> g = wt.fit.Gaussian()
   >>> g.params
   ['mean', 'width', 'amplitude', 'baseline']
   >>> g.limits
   {'width': [0, inf]}

At their simplest, function objects can be directly evaluated with a given set of parameters.

.. plot::

   #import
   import numpy as np
   import matplotlib.pyplot as plt
   import WrightTools as wt
   # parameters
   xi = np.linspace(6000, 8000, 101)
   mean = 7000.
   width = 100.
   amplitude = 1.
   baseline = 0.
   # create, evaluate
   g = wt.fit.Gaussian()
   yi = g.evaluate([mean, width, amplitude, baseline], xi)
   # plot
   plt.plot(xi, yi)

They can also be used to fit bare arrays.

.. plot::

   #import
   import numpy as np
   import matplotlib.pyplot as plt
   import WrightTools as wt
   # noisey gaussian
   xi = np.linspace(-100, 100, 25)
   yi = 5*np.exp(-0.5*((xi-5)/20.)**2)
   yi = np.random.poisson(yi)
   plt.scatter(xi, yi)
   # fitted
   g = wt.fit.Gaussian()
   ps = g.fit(yi, xi)
   xi = np.linspace(-100, 100, 101)
   model = g.evaluate(ps, xi)
   plt.plot(xi, model)

Finally, of course, the can be handed to Fitter to iteratively fit entire data objects.

Here is a list of the functions currently supplied.

==================================================  ============================================================
function                                            parameters
--------------------------------------------------  ------------------------------------------------------------
:meth:`~WrightTools.fit.ExpectationValue`           ['value']
:meth:`~WrightTools.fit.Exponential`                ['amplitude', 'tau', 'offset']
:meth:`~WrightTools.fit.Gaussian`                   ['mean', 'width', 'amplitude', 'baseline']
:meth:`~WrightTools.fit.Moments`                    ['integral', 'one', 'two', 'three', 'four', 'baseline']
==================================================  ============================================================

Fitter
------

The Fitter class is specially made to work seamlessly with data objects.
   
WrightTools is especially good at dimensionality reduction through fitting.
This concept is best demonstrated through an example.

Let's load in some test data.

.. code-block:: python

   #import
   import WrightTools as wt
   from WrightTools import datasets
   # create
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   data = wt.data.from_COLORS(ps)
   # cleanup
   data.level('ai0', 'd2', -3)
   data.scale()
   data.convert('eV')
   data.name = 'MoS2'

This is a three dimensional dataset.

.. code-block:: python

   >>> data.axis_names
   ['w2', 'w1', 'd2']
   >>> data.shape
   (41, 41, 23)

We could plot it as an animation to see each and every pixel:

.. image:: _static/v2p1_MoS2_TrEE_movie.gif

This is great, but we might still have questions about the data.

Instead we could imagine fitting every decay (:math:`\tau_{21}` trace) to an exponential.
Then we could plot the amplitude and time constant of that exponential decay.
This helps us get at subtle questions about the data.
Do the lineshapes narrow with time?
Does the redder feature decay slower than the bluer feature? Faster?

WrightTools makes it easy to do all of these exponential fits at once, through the fit module.

.. code-block:: python

   # isolate only relevant data
   data = data.split('w1', 1.75)[1].split('d2', 0)[0]
   # prepare a function
   function = wt.fit.Exponential()
   function.limits['amplitude'] = [0, 1]
   function.limits['offset'] = [0, 0]
   function.limits['tau'] = [0, 2000]
   # do the fit
   fitter = wt.fit.Fitter(function, data, 'd2')
   outs = fitter.run()

When we call ``fitter.run()``, every slice of the data object will be fit according to the given function object.
Fitter automatically creates two new data objects when this happens.
``outs`` contains the fit parameters, in this case amplitude, tau, and offset.
Accordingly, ``outs`` is lower-dimensional than the original data object.
``model`` contains the fit evaluated at each coordinate of the original dataset---it's really useful for inspecting the quality of your fit procedure.

Let's look at one of the fits now:

.. code-block:: python

   import matplotlib.pyplot as plt
   # plot
   fig, gs = wt.artists.create_figure()
   ax = plt.subplot(gs[0, 0])
   at = {'w1': [2.0, 'eV'], 'w2': [2.0, 'eV']}
   ax.plot_data(fitter.model.chop('d2', at=at)[0])
   ax.plot_data(data.chop('d2', at=at)[0])

.. plot::
   :include-source: False

   #import
   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   # create
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   #data = wt.data.from_COLORS(ps)
   # cleanup
   #data.level('ai0', 'd2', -3)
   #data.scale()
   #data.convert('eV')
   #data.name = 'MoS2'
   # isolate only relevant data
   #data = data.split('w1', 1.75)[1].split('d2', 0)[0]
   # prepare a function
   #function = wt.fit.Exponential()
   #function.limits['amplitude'] = [0, 1]
   #function.limits['offset'] = [0, 0]
   #function.limits['tau'] = [0, 2000]
   # do the fit
   #fitter = wt.fit.Fitter(function, data, 'd2')
   #outs = fitter.run()
   # plot
   #fig, gs = wt.artists.create_figure()
   #ax = plt.subplot(gs[0, 0])
   #at = {'w1': [2.0, 'eV'], 'w2': [2.0, 'eV']}
   #ax.plot_data(fitter.model.chop('d2', at=at)[0])
   #ax.plot_data(data.chop('d2', at=at)[0])

Looks reasonable.

Since outs is just another data object, we can plot it directly using :meth:`~WrightTools.artists.mpl_2D`.

.. plot::

   #import
   import matplotlib.pyplot as plt
   import WrightTools as wt
   from WrightTools import datasets
   # create
   ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
   #data = wt.data.from_COLORS(ps)
   # cleanup
   #data.level('ai0', 'd2', -3)
   #data.scale()
   #data.convert('eV')
   #data.name = 'MoS2'
   # isolate only relevant data
   #data = data.split('w1', 1.75)[1].split('d2', 0)[0]
   # prepare a function
   #function = wt.fit.Exponential()
   #function.limits['amplitude'] = [0, 0.75]
   #function.limits['offset'] = [0, 0]
   #function.limits['tau'] = [0, 2000]
   # do the fit
   #fitter = wt.fit.Fitter(function, data, 'd2')
   #outs = fitter.run()
   # plot
   #a = wt.artists.mpl_2D(outs, 'w1', 'w2')
   #a.plot('amplitude')
   #a.plot('tau')

We can easily see that the two large peaks decay slower than the rest of the spectra.
