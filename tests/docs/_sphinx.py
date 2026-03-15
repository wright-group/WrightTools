"""Test building the documentation."""

import pytest
import sys
from sphinx.cmd import build
import pathlib
import shutil
import subprocess


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12) or sys.platform == "win32",
    reason="Only need to build with ubuntu, python 3.12",
)
def test_build_docs():
    docsdir = pathlib.Path(__file__).resolve().parent.parent.parent / "docs"
    (docsdir / "__testbuild").mkdir(exist_ok=True)
    result = subprocess.run(["sphinx-build", str(docsdir), str(docsdir / "__testbuild")])
    try:
        assert result.returncode == 0
    finally:
        shutil.rmtree(docsdir / "__testbuild")


if __name__ == "__main__":
    test_build_docs()
