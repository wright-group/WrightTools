.. _contributing:

Contributing
============

Thank you so much for contributing to WrightTools!
We really appreciate your help.

If you have any questions at all, please either `open an issue on GitHub <https://github.com/wright-group/WrightTools/issues>`_ or email a WrightTools maintainer. The current maintainers can always be found in `CONTRIBUTORS <https://github.com/wright-group/WrightTools/blob/master/CONTRIBUTORS>`_.

Are you interested in adding support for yet another data format?
Please see :ref:`write_from_function`.

Preparing
---------

#. fork the `WrightTools repository <https://github.com/wright-group/WrightTools>`_ (if you have push access to the main repository you can skip this step)
#. clone WrightTools to your machine:

    .. code-block:: bash

        $ git clone <your fork>


#. in the cloned directory (note, to install to system python, you may need to use ``sudo`` for this command):

    .. code-block:: bash

        $ pip install -e .[dev]

#. run tests

    .. code-block:: bash

        $ pytest

   Note: On ``*nix`` machines (unfortunately this does not work on Windows),
   the tests may be multiprocessed using `pytest-mp <https://github.com/ansible/pytest-mp>`_:

   .. code-block:: bash

       $ pip install pytest-mp
       $ pytest --mp


Contributing
------------

#. ensure that the changes you intend to make have corresponding `issues on GitHub <https://github.com/wright-group/WrightTools/issues>`_
    a) if you aren't sure how to break your ideas into atomic issues, feel free to open a discussion issue
    b) looking for low-hanging fruit? check out the `help wanted label <https://github.com/wright-group/WrightTools/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_ for beginner-friendly issues

    .. code-block:: bash

        $ # Create the branch, including remote
        $ git branch <your branch> --set-upstream-to origin origin/<your branch>
        $ git checkout <your branch> # Switch to the newly created branch

#. run all tests to ensure that nothing is broken right off the start

    .. code-block:: bash

        $ pytest

#. make your changes, commiting often

    .. code-block:: bash

        $ git status # See which files you have changed/added
        $ git diff # See changes since your last commit
        $ git add <files you wish to commit>
        $ git commit -m "Description of changes" -m "More detail if needed"

#. mark your issues as resolved (within your commit message):

    .. code-block:: bash

        $ git commit -m "added crazy colormap (resolves #99)"

    a. If your commit is related to an issue, but does not resolve it, use ``addresses #99`` in the commit message
#. if appropriate, add tests that address your changes (if you just fixed a bug, it is strongly reccomended that you add a test so that the bug cannot come back unanounced)
#. once you are done with your changes, run your code through flake8 and pydocstyle

    .. code-block:: bash

        $ flake8 file.py
        $ pydocstyle file.py

#. rerun tests
#. add yourself to `CONTRIBUTORS <https://github.com/wright-group/WrightTools/blob/master/CONTRIBUTORS>`_
#. push your changes to the remote branch (github)

    .. code-block:: bash

        $ git pull # make sure your branch is up to date
        $ git push

#. make a pull request to the master branch
#. communicate with the maintainers in your pull request, assuming any further work needs to be done
#. celebrate! 🎉

Style
-----

Internally we use the following abbreviations:
    WrightTools
        ``import WrightTools as wt``
    Matplotlib
        ``import matplotlib as mpl``
    Pyplot
        ``from matplotlib import pyplot as plt``
    NumPy
        ``import numpy as np``

WrightTools follows `pep8 <https://www.python.org/dev/peps/pep-0008/>`_, with the following modifications:

#. Maximum line length from 79 characters to 99 characters.

WrightTools also folows `numpy Docstring Convention`_, which is a set of adjustments to `pep257`_.
WrightTools additionally ignores one guideline:

#. WrightTools does not require all magic methods (e.g. ``__add__``) to have a docstring.

    a) It remains encourged to add a docstring if there is any ambiguity of the meaning.

.. _numpy docstring convention: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _pep257: https://www.python.org/dev/peps/pep-0257/

We use `flake8 <http://flake8.pycqa.org/en/latest/>`_ for automated code style enforcement, and `pydocstyle <http://www.pydocstyle.org>`_ for automated docstring style checking.


.. code-block:: bash

     $ # These will check the whole directory (recursively)
     $ flake8
     $ pydocstyle

Consider using `black <https://pypi.org/project/black/>`_ for automated code corrections.
Black is an opinionated code formatter for unambiguous standardization.

.. code-block:: bash

     $ git commit -m "Describe changes"
     $ black file.py
     $ git diff # review changes
     $ git add file.py
     $ git commit -m "black style fixes"

We also provide a configuration to use git hooks to automatically apply ``black`` style to edited files.
This hook can be installed using ``pre-commit``:

.. code-block:: bash

     $ pre-commit install

When committing, it will automatically apply the style, and prevent the commit from completing if changes are made.
If that is the case, simply re-add the changed files and then commit again.
This prevents noisy commit logs with changes that are purely style conformity.
