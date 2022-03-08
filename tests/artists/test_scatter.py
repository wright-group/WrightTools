import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt


def test_scatter_broadcast_vars():
    data = wt.data.Data(name="data")
    rng1 = np.random.rand(50)[:, None]
    rng2 = np.random.rand(50)[None, :]
    data.create_variable("x", values=2 * rng1 - 1)
    data.create_variable("y", values=2 * rng2 - 1)
    data.create_channel("z", values=np.exp(-(data.x[:] ** 2 + data.y[:] ** 2)))
    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.scatter(data, "x", "y", channel="z", s=20, alpha=0.5)
    data.close()


def test_scatter_signed_channel():
    data = wt.data.Data(name="data")
    rng1 = np.random.rand(10**3)
    rng2 = np.random.rand(10**3)
    data.create_variable("x", values=4 * np.pi * (rng1 - 0.5))
    data.create_variable("y", values=4 * np.pi * (rng2 - 0.5))
    data.create_channel("z", values=np.sin((data.x[:] ** 2 + data.y[:] ** 2) ** 0.5), signed=True)
    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.scatter(data, "x", "y", channel="z", s=20, alpha=0.5)
    data.close()


if __name__ == "__main__":
    t0 = test_scatter_broadcast_vars()
    t1 = test_scatter_signed_channel()
    plt.show()
