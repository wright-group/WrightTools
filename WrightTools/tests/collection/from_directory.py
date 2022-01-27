#! /usr/bin/env python3

import os
import WrightTools as wt
from WrightTools import datasets


def test_from_directory():
    from_dict = {
        "*.data": wt.data.from_PyCMDS,
        "*.csv": wt.collection.from_Cary,
        "KENT": None,
        "COLORS": None,
    }
    p = datasets.PyCMDS.w1_000
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    col = wt.collection.from_directory(p, from_dict)
    # Print the tree in case tests fail, also will catch errors in printing
    col.print_tree()
    assert col.natural_name == "datasets"
    assert col.PyCMDS["w1 000"]
    assert col.Cary["CuPCtS_H2O_vis"]
    assert "KENT" not in col.item_names
    col.close()


if __name__ == "__main__":
    test_from_directory()
