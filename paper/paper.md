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
 - name: John C. Wright
   orcid: 0000-0002-6926-1837
   affiliation: 1
affiliations:
 - name: University of Wisconsin--Madison
   index: 1
date: 26 November 2018
bibliography: paper.bib
---

# Introduction

"Multidimensional spectroscopy" (MDS) is a family of analytical techniques that record the response of a material to multiple stimuli---typically multiple ultrafast pulses of light.
This approach has several unique capabilities;

- resolving congested states [@ZhaoWei1999b; @DonaldsonPaulMurray2008a],
- extracting spectra that would otherwise be selection-rule disallowed [@BoyleErinSelene2013b; @BoyleErinSelene2014a],
- resolving fully coherent dynamics [@PakoulevAndreiV2009a],
- measuring coupling [@WrightJohnCurtis2011a],
- and resolving ultrafast dynamics [@SmallwoodChristopherL2018a; @CzechKyleJonathan2015a].

In our view, the most exciting aspect of these techniques is the vast number of different approaches that scientists can take to learn about material quantum states.
Often, a great number of these experiments can be accomplished with a single instrument.
The diversity of related-but-unique approaches to interrogating quantum systems is a large strength of MDS.

Modern innovations in optics and laser science are bringing multidimensional spectroscopy to more and more laboratories around the world.
At the same time, increasing automation and computer control are allowing traditionally "one-dimensional" spectroscopies to be recorded against other dimensions.

Due to its diversity and dimensionality, MDS data is challenging to process and represent.
The data processing tools that a scientist develops to process one experiment may not work when she attempts to process an experiment where different experimental variables are explored.
Historically, this processing strategy has resulted in MR-CMDS practitioners have using custom, one-off data processing workflows that need to be changed for each particular  experiment.
These changes take time to implement, and can become annoyances or opportunities for error.
Even worse, the challenge of designing a new processing workflow may dissuade a scientist from creatively modifying their experimental strategy, or comparing their data with data taken from another group.
This limit to creativity and flexibility defeats one of the main advantages of the MDS "family approach".

``WrightTools`` is a new Python package that is made specifically for multidimensional spectroscopy.
It aims to be a core toolkit that is general enough to handle all MDS datasets and processing workloads.
Being built for and by MDS practitioners, ``WrightTools`` has an intuitive high-level interface for the experimental spectroscopist.
To our knowledge, ``WrightTools`` is the first MDS-focused toolkit to be freely avaliable and openly licensed.

# Challenges and Implementation

There are several recurring challenges in MDS data processing and representation

- Dimensionality of datasets can typically be greater than two, complicating representation.
- Shape and dimensionality change, and relevant axes can be different from the scanned dimensions.
- Data can be awkwardly large-ish (several million pixels), to legitimately large (it is not always possible to store entire arrays in memory).
- There are no agreed-upon file formats.

The excellent Scientific Python ecosystem is well suited to adress all of these challenges. [@OliphantTravisE2007a]
Numpy supports interaction and manipulation of multidmensional datasets. [@OliphantTravisE2006a]
Matplotlib supports one, two, and even three-dimensional plotting. [@HunterJohnD2007a]
h5py interfaces with hdf5, allowing for storage and access to large multidmensional arrays in a binary format that can be accessed from a variety of different popular languages, including MATLAB and Fortran. [@hdf5]
``WrightTools`` does not intend to replace or reimplement these core libraries.
Instead, ``WrightTools`` offers an interface that impedence-matches multidimensional spectroscopy and Scientific Python.

``WrightTools`` defines a universal MDS data format: the ``wt5`` file.
These are simply hdf5 files with certain internal conventions that are designed for MDS.
These internal conventions enable the flexability and ease-of-use that we discuss in the rest of this section.
The multdimensional spectroscopic data within these files is dynamically interacted with through instances of ``WrightTools``'s various classes, which are children of h5py classes.
``WrightTools`` offers a variety of functions that try hard to convert data stored in various other formats to ``wt5``.

``WrightTools`` defines a unique and flexable strategy of storing and manipulating MDS datasets.
A single dataset is implemented as a group containing many separate arrays.
There are two principle multidimensional array classes: ``Channel`` and ``Variable``.
Conceptually, these correspond to independent (scanned) dimensions---"variables" and dependent (measured) signals---"channels".
Channels typically contain measured signals from all of the different sensors that are employed simultaniously during a MDS experiment.
Variables contain coordinates of different light manipulation hardware that are scanned against eachother to make up an MDS experiment.
All variables are recorded, including coordinates for hardware that is not actually moved during that experiment (an array with one unique value) or other independent variables, such as lab time.

There can be many variables that change in the context of a single MDS experiment.
The typical spectroscopist only really cares about a small subset of these variables, but exactly what subset matters may change as different strategies are used to explain the measurement.
Furthermore, it is often useful to "combine" mutiple variables using simple algebraic relationships to exploit the natural symmetry of many MDS experiments and to draw comparisons between different members of the MDS family. [@NeffMallonNathanA2017a]
In light of these details, ``WrightTools`` provides a high-level ``Axis`` class that allows users to transparently define which variables, or variable relationships, are important to them.
Each ``Axis`` contains an ``expression``, which dictates its relationship with one or more variables.
Given 5 variables with names [``'w1'``, ``'w2'``, ``'wm'``, ``'d1'``, ``'d2'``] , example valid expressions include ``'w1'``, ``'w1=wm'``, ``'w1+w2'``, ``'2*w1'``, ``'d1-d2'``, and ``'wm-w1+w2'``.
Users may treat axes like multidimensional arrays, using ``__getitem__`` syntax and slicing, but axes do not themselves contain arrays.
Instead, the appropriate axis value at each dataset coordinate is computed on-the-fly using the given expression.
Users may at any time change their axes by simply ``transform``ing their dataset using new expressions.

``WrightTools`` offers a suite of data manipulation tools with MDS in mind.
Users can access portions of their data using high-level methods like ``chop``, ``split``, and ``clip``.
They can process their data using simple mathematical operations or more specific tools like ``level``, ``gradient``, ``collapse``, and ``smooth``.
Users can even join multiple datasets together, creating higher-dimensional datasets when appropriate.
All of these operations refer to the self-describing internal structure of the ``wt5`` file wherever possible.
Users are not asked to refer to the specific shape and indicies of their data arrays.
Instead, they deal with simple axis expressions and unit-aware coordinates.

``WrightTools`` offers a set of "artists" to quickly draw typical representations.
These make it trivial to make beautiful Matplotlib representations of MDS datasets.
Again, the self-describing internal structure is capitalized upon, auto-filling labels and auto-scaling axes.
For higher-than-two dimensional datasets, ``WrightTools`` makes it easy to plot many separate figures that can be looped through using an image viewer or stitched into a looping animated gif.

# Availability

``WrightTools`` hosted on GitHub [cite] and archived on Zenodo [@ThompsonBlaiseJonathan2018WrightTools].
``WrightTools`` is distributed using [pip](http://pypi.org/project/WrightTools/) and [conda](http://anaconda.org/conda-forge/wrighttools) (through conda-forge).
Documentation is available at [wright.tools](http://wright.tools).

# Impact

``WrightTools`` has directly enabled no fewer than eleven publications. [@CzechKyleJonathan2015a; @KohlerDanielDavid2017a; @NeffMallonNathanA2017a; @ChenJie2017a; @MorrowDarienJames2017a; @HorakErikH2018a; @SundenKyleFoster2018a; @KohlerDanielDavid2018a; @HandaliJonathanDaniel2018a; @MorrowDarienJames2018a; @HandaliJonathanDaniel2018b]
Many of these publications have associated open datasets and ``WrightTools``-based processing scripts which enhance the scientific community's ability to audit and reproduce the published work.
These practices are not yet common in the MDS community.

Though still relatively uncommon, MDS is an increasingly important family of analytical techniques used by Chemists and Physists to interrogate especially complex systems and to answer especially challenging questions.
We hope that ``WrightTools``, and the universal ``wt5`` file format, will become a useful open source core technology for this growing community.
We are particularly excited about ongoing projects that build on top of ``WrightTools``, including packages for data acquisition and simulation. [@ThompsonBlaiseJonathan2018PyCMDS; @SundenKyleFoster2018a]

# Acknowledgements

The development of ``WrightTools`` has been supported by the Department of Energy, Office of Basic Energy Sciences, Division of Materials Sciences and Engineering, under award DE--FG02--09ER46664 and by the National Science Foundation Division of Chemistry under Grant No. CHE--1709060.
D.J.M acknowledges support from the Link Foundation.

# References
