Contributing to WrightTools
===========================

Thank you so much for contributing to WrightTools!
We really appreciate your help.

If you have any questions at all, please either `open an issue on GitHub <https://github.com/wright-group/WrightTools/issues>`_ or email a WrightTools maintainer. The current maintainers can always be found in `CONTRIBUTORS <https://github.com/wright-group/WrightTools/blob/master/CONTRIBUTORS>`_.

Preparing to edit WrightTools
-----------------------------

#. fork the `WrightTools repository <https://github.com/wright-group/WrightTools>`_ (if you have push access to the main repository you can skip this step)
#. clone WrightTools to your machine: ``$ git clone <your fork>``
#. in the cloned directory, ``$ python setup.py develop``
#. run tests

Contributing
------------

#. ensure that the changes you intend to make have corresponding `issues on GitHub <https://github.com/wright-group/WrightTools/issues>`_
    a) if you aren't sure how to break your ideas into atomic issues, feel free to open a discussion issue
    b) looking for low-hanging fruit? check out the `help wanted label <https://github.com/wright-group/WrightTools/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_ for beginner-friendly issues
#. if you are working on documentation or tests, please work within the dedicated branches---otherwise create your own feature branch
    
    a. If you wish to track changes over multiple days prior to submitting a pull request, also create a dedicated branch, and merge into the apropriate branch when ready to submit
#. run all tests to ensure that nothing is broken right off the start
#. make your changes, commiting often
#. mark your issues as resolved (within your commit message): ``added crazy colormap (resolves #99)``
    a. If your commit is related to an issue, but does not resolve it, use ``addresses #99`` in the commit message
#. if appropriate, add tests that address your changes (if you just fixed a bug, it is strongly reccomended that you add a test so that the bug cannot come back unanounced)
#. once you are done with your changes, run your code through flake8
#. rerun tests
#. add yourself to `CONTRIBUTORS <https://github.com/wright-group/WrightTools/blob/master/CONTRIBUTORS>`_
#. make a pull request to the development branch
#. communicate with the maintainers in your pull request, assuming any further work needs to be done
#. celebrate!

Style
-----

Internally WrightTools is imported as ``import WrightTools as wt`` 

WrightTools follows `pep8 <https://www.python.org/dev/peps/pep-0008/>`_, with the following modifications:

#. maximum line length from 79 characters to 99 charachters
#. `NumPy style docstrings <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_

We use `flake8 <http://flake8.pycqa.org/en/latest/>`_ for automated code style enforcement.


.. code-block:: bash

     $ flake8 --max-line-length=99

Consider using `autopep8 <https://pypi.python.org/pypi/autopep8>`_ for automated code corrections --- Make sure to confirm that the output is expected

.. code-block:: bash

     $ git commit
     $ autopep8 --max-line-lenghth=99 file.py 
     $ git diff # review changes
     $ git commit
