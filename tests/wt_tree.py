#! /usr/bin/env python3

# --- import --------------------------------------------------------------------------------------


import subprocess


# --- define --------------------------------------------------------------------------------------


from WrightTools import datasets


def test_tree():
    subprocess.run(
        ["wt-tree ", datasets.wt5.v1p0p0_perovskite_TA], encoding="utf-8", check=True
    )


# --- run -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    test_tree()
