#! /usr/bin/env python3

# --- import --------------------------------------------------------------------------------------

import subprocess

# --- define --------------------------------------------------------------------------------------

from WrightTools import datasets

def test_tree():
    subprocess.check_call(["wt-tree ", datasets.wt5.v1p0p0_perovskite_TA])

# --- run -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    test_tree()