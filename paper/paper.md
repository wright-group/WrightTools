---
title: 'WrightTools: a Python package for multidimensional spectroscopy'
tags:
  - example
  - tags
  - for the paper
authors:
 - name: Blaise J. Thompson
   orcid: 0000-0002-3845-824X
   affiliation: 1
 - name: Kyle F. Sunden
   orcid: 0000-0001-6167-8059
   affiliation: 1
 - name: Darien J. Morrow
   orcid: 0000-0002-8922-8049
   affiliation: 1
 - name: Daniel D. Kohler
   orcid: 0000-0003-4602-2961
   affiliation: 1
affiliations:
 - name: University of Wisconsin--Madison
   index: 1
date: 26 November 2018
bibliography: paper.bib
---

# Introduction

"Multidimensional spectroscopy" (MDS) is an umbrella term for a family of analytical techniques that interrogate the response of a material to multiple stimuli, typically multiple electric fields.
This approach has several unique capabilities:

- resolving congested states [@ZhaoWei1999b; @DonaldsonPaulMurray2008a]
- extracting spectra that would otherwise be selection-rule disallowed [@BoyleErinSelene2013b; @BoyleErinSelene2014a],
- resolving fully coherent dynamics [@PakoulevAndreiV2009a],
- measuring coupling [@WrightJohnCurtis2011a]
- resolving ultrafast dynamics

In our view, the most exciting thing about these techinques is the great number of different approaches that scientists can take to learn about material quantum states.
Often, a great number of these experiments can be accomplished with a single instrument.

Modern innovations in optics and laser science are bringing multidimensional spectroscopy to more and more laboratories around the world.
At the same time, increasing automation and computer control are allowing traditional "one-dimensional" spectroscopies to be recorded against other dimensions.
These multidimensional datasets are increasingly challenging to manipulate, visualize, store, and share using traditional data processing toolkits.

``WrightTools`` is a new toolkit that is made specifically for multidimensional spectroscopy.
To our knowledge, ``WrightTools`` is the first MDS toolkit to be freely avaliable and openly licensed.

# Challenges and Implementation

There are several recurring challenges in MDS data processing and representation

- Dimensionality of datasets can typically be greater than two, complicating representation.
- Shape and dimensionality change, and relevant axes can be different from the scanned dimensions.
- Data can be awkwardly large-ish (several million pixels), to legitimately large---it is not always possible to store entire arrays in memory.
- There are no agreed-upon file formats.

The excelent Scientific Python ecosystem is well suited to adress all of these challenges.
Numpy supports interaction and manipulation of multidmensional datasets. [cite]
Matplotlib supports one, two, and even three-dimensional plotting. [cite]
h5py interfaces with hdf5, allowing for storage and access to large multidmensional arrays in a binary format that can be accessed from a variety of different popular languages, including MATLAB and Fortran. [cite]
``WrightTools`` does not intend to replace or reimplement these core libraries.
Instead, ``WrightTools`` offers an interface that impedence-matches multidimensional spectroscopy and Scientific Python.

``WrightTools`` defines a universal MDS data format: the wt5 file.
These are simply hdf5 files with certain internal conventions that are designed for MDS.
These internal conventions enable the flexability and ease-of-use that we discuss in the rest of this section.
The multdimensional spectroscopic data within these files is dynamically interacted with through ``WrightTools``'s various classes, which are children of ``h5py`` classes.
``WrightTools`` offers a variety of functions that try hard to convert data stored in various other formats to wt5.

``WrightTools`` defines a unique and flexable strategy of storing and manipulating axes and variables.
MDS data is typically quite structured, so completely flattened solutions are bulky and not performant because they don't take advantage of the natural internal structure of the dataset.
On the other hand, there are important reasons to manipuate and display MDS data in less structured ways. [cite]
``WrightTools`` allows users to ``transform`` their data, relaxing the strict structure requirements when necessary.

``WrightTools`` offers a suite of data manipulation tools with MDS in mind.
Users can access portions of their data using high-level methods like ``chop``, ``split``, and ``clip``.
They can process their data using simple mathematical operations or more specific tools like ``level``, ``gradient``, and ``collapse``.
Users can even join multiple datasets together, creating higher-dimensional datasets when appropriate.
All of these operations refer to the self-describing internal structure of the wt5 file wherever possible.
Users are not asked to refer to the specific shape and indicies of their data arrays.
Instead, they deal with simple axis names and unit-aware coordinates.

``WrightTools`` offers a set of "artists" to quickly draw typical representations.
These make it trivial to make beutiful Matplotlib representations of MDS datasets.
Again, the self-describing internal structure is capitalized upon, auto-filling labels and auto-scaling axes.
For higher-than-two dimensional datasets, ``WrightTools`` makes it easy to plot many separate figures that can be looped through using an image viewer or stitched into a looping animated gif.
There is also an "interactive" artist which takes advantage of Matplotlib's interactive widgets.
There are even specialty artists for drawing common MDS diagrams, such as WMELs [cite].

# Availability 

``WrightTools`` is archived with Zenodo [cite].
``WrightTools`` is installable with pip and conda package managers with documentation available at [wright.tools](http://wright.tools).


# Impact

To our knowledge, ``WrightTools`` has directly enabled no fewer than ten publications.
A partial list of these can be found under "Citing WrightTools" at our documentation website.
Many of these publications have associated open datasets and ``WrightTools``-based processing scripts which enhance the communities ability to audit and reproduce the published work.
These practices are not yet common in the MDS community.

Though still relatively uncommon, MDS is an increasingly important family of analytical techniques used by Chemists and Physists to interrogate especially complex systems and to answer especially challenging questions.
We hope that ``WrightTools``, and the universal wt5 file format, will become a useful open source core technology for this growing community.
We are particularly excited about ongoing projects that build on top of ``WrightTools``, including packages for data acquisition and simulation.

# Acknowledgements

The development of ``WrightTools`` has been supported by the Department of Energy, Office of Basic Energy Sciences, Division of Materials Sciences and Engineering, under award DE--FG02--09ER46664 and by the National Science Foundation Division of Chemistry under Grant No. CHE-1709060. 
D.J.M acknowledges support from the Link Foundation.

# References
