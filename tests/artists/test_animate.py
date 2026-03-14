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

    ani = wt.artists.animate2D(
        d, interval=50, back_and_forth=True, norm=partial(CenteredNorm, vcenter=0)
    )
    return ani


def test_animate_interact2D():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    d.channels[0].signed = True

    out = wt.artists.interact2D(d, local=True)
    ani = wt.artists.animate_interact2D(out, back_and_forth=True, interval=500)
    return ani


def test_animate_quick2D():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    d.channels[0].signed = True

    ani = wt.artists.animate_quick2D(d, fa_kwargs=dict(interval=100))
    return ani
    

if __name__ == "__main__":
    ani1 = test_animate2D()
    ani2 = test_animate_interact2D()
    ani3 = test_animate_quick2D()
    plt.show()
