#! /usr/bin/env python3

import pathlib
import WrightTools as wt
from WrightTools import datasets


def test_from_directory():
    from_dict = {
        "*.csv": wt.collection.from_Cary,
        "*.txt": wt.data.from_JASCO,
        "KENT": None,
        "Shimadzu": None,
    }
    p = datasets.PyCMDS.w1_000
    print(p)
    col = wt.collection.from_directory(pathlib.Path(*p.parts[:-2]), from_dict)
    # Print the tree in case tests fail, also will catch errors in printing
    col.print_tree()
    assert col.natural_name == "datasets"
    assert col.JASCO["PbSe batch 1"]
    assert col.Cary["CuPCtS_H2O_vis"]
    assert "KENT" not in col.item_names
    col.close()


if __name__ == "__main__":
    test_from_directory()
