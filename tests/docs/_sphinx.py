"""Test building the documentation."""

import pytest
import sys
from sphinx.cmd import build
import pathlib
import shutil


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11) or sys.platform == "win32",
    reason="Only need to build with ubuntu, python 3.11",
)
def test_build_docs():
    docsdir = pathlib.Path(__file__).resolve().parent.parent.parent / "docs"
    exitCode = build.build_main([str(docsdir), str(docsdir / "__testbuild")])
    # The following code works in sphinx >= 1.7.0
    # exitCode = sphinx.build([docsdir, docsdir + '/__testbuild'])
    assert exitCode == 0
    shutil.rmtree(docsdir / "__testbuild")


if __name__ == "__main__":
    test_build_docs()
