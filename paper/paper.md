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
 - name: Darien Morrow
   orcid: 0000-0002-8922-8049
   affiliation: 1
affiliations:
 - name: University of Wisconsin--Madison
   index: 1
date: 06 June 2018
bibliography: paper.bib
---

# Summary

"Multidimensional spectroscopy" (MDS) is an umbrella term for a family of analytical techniques that interrogate the response of a material to multiple stimuli, typically multiple electric fields.
This approach has several unique capabilities:

- resolving congested states [@ZhaoWei1999b, @DonaldsonPaulMurray2008a]
- extracting spectra that would otherwise be selection-rule disallowed [@BoyleErinSelene2013b, @BoyleErinSelene2014a],
- resolving fully coherent dynamics [@PakoulevAndreiV2009a],
- measuring coupling [@WrightJohnCurtis2011a]
- resolving ultrafast dynamics

In our view, the most exciting thing about these techinques is the great number of different approaches that scientists can take to learn about material quantum states.
Often, a great number of these experiments can be accomplished with a single instrument.

Modern innovations in optics and laser science are bringing multidimensional spectroscopy to more and more laboratories around the world.
At the same time, increasing automation and computer control are allowing traditional experiments to increase their dimensionality.
The increasing magnitude and complexity of these multidimensional datasets are increasingly challenging to manipulate, visualize, and share using traditional data processing toolkits.
``WrightTools`` is a new toolkit that is made specifically with multidimensional spectroscopy in mind.
To our knowledge, ``WrightTools`` is the first MDS toolkit to be freely avaliable and openly licensed.

There are several recurring challenges in MDS data processing and representation
- Dimensionality of datasets can typically be greater than two, complicating representation.
- Shape and dimensionality change, and relevant axes can be different from the scanned dimensions.
- Data can be awkwardly large-ish (several million pixels), to legitimately large---it is not always possible to store entire arrays in memory.
- There are no agreed-upon file formats.

The excelent scientific python ecosystem is well suited to adress all of these challenges.
Numpy---multidimensional data
Matplotlib---1, 2 & 3D plotting
H5Py for storage, access to large datasets
etc...
``WrightTools`` builds on top of the scientific python ecosystem, focusing on solving several impedence-matching issues unique to MDS.

``WrightTools`` defines a universal MDS data format: the wt5 file.
These are simply hdf5 files with certain internal conventions.
The multdimensional spectroscopic data within these files is dynamically interacted with through ``WrightTools``'s various classes, which are children of ``h5py`` classes.
``WrightTools`` offers a variety of functions that try hard to convert data stored in various other formats to wt5.

``WrightTools`` defines a unique and flexable strategy of storing and manipulating axes and variables.

``WrightTools`` offers a set of "artists" to quickly draw typical representations.
Also diagrams.
Also interact.

WrightTools is written to be used in scripts and in the command line.
It does not have any graphical components built in, except for the ability to generate plots using matplotlib [@HunterJohnD2007a].
Being built in this way gives WrightTools users maximum flexibility, and allows for rapid collaborative development.
It also allows other software packages to use WrightTools as a ``back-end'' foundational software, as has already been done in simulation and acquisition software created in the Wright Group.  %

TODO: list scientific publications that WrightTools has enabled

TODO: close by clarifying that we wish to extend WrightTools to as many applications as possible in MDS.

# References
