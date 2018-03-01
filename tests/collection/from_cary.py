#! /usr/bin/env python3

import os
import WrightTools as wt
from WrightTools import datasets


def test_CuPCtS_H2O_vis():
    p = wt.datasets.Cary.CuPCtS_H2O_vis
    col = wt.collection.from_Cary(p)
    data = col.sample1
    assert col.natural_name == 'cary'
    assert data.axis_names == ('wavelength',)
    assert data.units == ('nm',)
    assert data.shape == (141,)
    col.close()


if __name__ == '__main__':
    test_CuPCtS_H2O_vis()
