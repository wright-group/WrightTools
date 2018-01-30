"""Test building the documentation."""
import sphinx
import os
import shutil


def test_build_docs():
    print(dir(sphinx))
    docsdir = os.path.abspath(os.path.dirname(__file__) + '/../../docs')
    exitCode = sphinx.build_main(['sphinx', docsdir, docsdir + '/__testbuild'])
    assert exitCode == 0
    shutil.rmtree(docsdir + '/__testbuild')
