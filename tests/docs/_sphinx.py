"""Test building the documentation."""

import pytest
import sys
from sphinx.cmd import build
import os
import shutil


@pytest.mark.skipif(sys.version_info > (3, 11), reason="Only building with python 3.11")
def test_build_docs():
    docsdir = os.path.abspath(os.path.dirname(__file__)) + "/../../docs"
    exitCode = build.build_main([docsdir, docsdir + "/__testbuild"])
    # The following code works in sphinx >= 1.7.0
    # exitCode = sphinx.build([docsdir, docsdir + '/__testbuild'])
    assert exitCode == 0
    shutil.rmtree(docsdir + "/__testbuild")


if __name__ == "__main__":
    test_build_docs()
