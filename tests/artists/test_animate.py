import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt


def test_animate2D():
    """
    smokescreen test: just ensure the ani object initializes

    if you are running this locally, you can view the animation
    by running this script
    """
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    d.channels[0].signed = True
    from matplotlib.colors import CenteredNorm
    from functools import partial

    ani = wt.artists.animate2D(d, interval=100, norm=partial(CenteredNorm, vcenter=0))
    return ani


if __name__ == "__main__":
    ani = test_animate2D()
    plt.show()
