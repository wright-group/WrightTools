.. _kit:

Kit
===

WrightTools provides a toolbox of methods and classes in the kit module.

kit contains classes and methods which are of general utility but do not fit in the context of the other modules in WrightTools. Some of the methods are simple reparameterizations of an external module’s functionality. kit is a convenient place for keeping methods and classes which are broadly applicable to the handling and representation of multidimensional spectroscopy data. 

Some of the useful classes that kit contains include:

============  =====================================================================
Class         Use
============  =====================================================================
TimeStamp     Represents a moment in time in many different ways.
INI           Interacts with .ini files.
Spline        Less picky paramaterization of scipy.UnivariateSpline
Timer         Allows controlled execution of code while offering timing abilities.
============  =====================================================================

Some of the useful array manipulation methods kit offers include:

- :meth:`~WrightTools.kit.closest_pair` for finding the pair of indices corresponding to the closest elements in an array 
- :meth:`~WrightTools.kit.diff` for taking numerical derivatives while keeping original array shapes
- :meth:`~WrightTools.kit.fft` for taking the fourier transform of an array and returning _sensible_ arrays 
