#! /usr/bin/env python3


import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt
import numpy as np


def test_imshow_transform():
    p = datasets.PyCMDS.d1_d2_000
    data = wt.data.from_PyCMDS(p)

    fig, gs = wt.artists.create_figure(cols=[1, 1])
    ax0 = plt.subplot(gs[0])
    data.transform("d1", "d2")
    im1 = ax0.imshow(data)

    ax1 = plt.subplot(gs[1])
    data.transform("d2", "d1")
    im2 = ax1.imshow(data)

    assert np.all(im1.get_array() == im2.get_array().T)

    data.close()


def test_imshow_approx_pcolormesh():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)

    fig, gs = wt.artists.create_figure(cols=[1, 1])
    ax0 = plt.subplot(gs[0])
    mesh = ax0.pcolormesh(data)
    ax1 = plt.subplot(gs[1])
    image = ax1.imshow(data)

    lim0 = ax0.get_xlim() + ax0.get_ylim()
    lim1 = ax1.get_xlim() + ax1.get_ylim()
    assert np.allclose(lim0, lim1, atol=1e-3, rtol=1), f"unequal axis limits: {lim0}, {lim1}"

    bbox = mesh.get_datalim(ax0.transData)
    meshbox = [bbox.x0, bbox.x1, bbox.y0, bbox.y1]
    imagebox = image.get_extent()
    imagebox = [*sorted(imagebox[:2]), *sorted(imagebox[2:])]
    assert np.allclose(
        meshbox, imagebox, atol=1e-3, rtol=1e-3
    ), f"unequal limits: mesh {meshbox} image {imagebox}"

    assert np.isclose(mesh.norm.vmin, image.norm.vmin), "unequal norm.vmin"
    assert np.isclose(mesh.norm.vmax, image.norm.vmax), "unequal norm.vmax"

    data.close()


if __name__ == "__main__":
    test_imshow_transform()
    test_imshow_approx_pcolormesh()
