"""Test building the documentation."""
import sphinx
import os
import shutil


def test_build_docs():
    docsdir = os.path.abspath(os.path.dirname(__file__)) + '/../../docs'
    exitCode = sphinx.build_main(['_sphinx.py', docsdir, docsdir + '/__testbuild'])
    # The following code works in sphinx >= 1.7.0
    # exitCode = sphinx.build([docsdir, docsdir + '/__testbuild'])
    assert exitCode == 0
    shutil.rmtree(docsdir + '/__testbuild')


if __name__ == '__main__':
    test_build_docs()
