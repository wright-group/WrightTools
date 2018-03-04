#! /usr/bin/env python3


import WrightTools as wt
from WrightTools import datasets


def test_created():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)

    assert int(data.created.unix) == 1487860880

    copy = data.copy()
    assert int(copy.created.unix) == 1487860880


if __name__ == '__main__':
    test_created()
