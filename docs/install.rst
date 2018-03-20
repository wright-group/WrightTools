.. _install:

Installation
============

WrightTools requires Python 3.5 or newer.

conda-forge
-----------

Conda_ is a multilingual package/environment manager.
It seamlessly handles non-Python library dependencies which many scientific Python tools rely upon.
Conda is reccomended, especially for Windows users.
If you don't have Python yet, start by `installing Anaconda`_ or `miniconda`_.

`conda-forge`_ is a community-driven conda channel. `conda-forge contains a WrightTools feedstock`_.

.. code-block:: bash

    conda config --add channels conda-forge
    conda install wrighttools

To upgrade:

.. code-block:: bash

    conda update wrighttools

pip
---

pip_ is Python's official package manager. `WrightTools is hosted on PyPI`_.


.. code-block:: bash

    pip install wrighttools

To upgrade:

.. code-block:: bash

    pip install wrighttools --upgrade

.. _Conda: https://conda.io/docs/intro.html
.. _installing Anaconda: https://www.continuum.io/downloads
.. _conda-forge: https://conda-forge.org/
.. _conda-forge contains a WrightTools feedstock: https://github.com/conda-forge/wrighttools-feedstock
.. _miniconda: https://conda.io/miniconda.html
.. _pip: https://pypi.python.org/pypi/pip
.. _WrightTools is hosted on PyPI: https://pypi.org/project/WrightTools/
