import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt


def test_scatter_broadcast_vars():
    rng = np.random.default_rng(seed=42)
    data = wt.data.Data(name="data")
    a = rng.random((50, 1))
    b = rng.random((1, 50))
    data.create_variable("x", values=2 * a - 1)
    data.create_variable("y", values=2 * b - 1)
    data.create_channel("z", values=np.exp(-(data.x[:] ** 2 + data.y[:] ** 2)))
    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.scatter(data, x="x", y="y", channel="z", s=20, alpha=0.5)
    data.close()


def test_scatter_signed_channel():
    rng = np.random.default_rng(seed=42)
    data = wt.data.Data(name="data")
    a = rng.random((10**3))
    b = rng.random((10**3))
    data.create_variable("x", values=4 * np.pi * (a - 0.5))
    data.create_variable("y", values=4 * np.pi * (b - 0.5))
    data.create_channel("z", values=np.sin((data.x[:] ** 2 + data.y[:] ** 2) ** 0.5), signed=True)
    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.scatter(data, x="x", y="y", channel="z", s=20, alpha=0.5)
    data.close()


if __name__ == "__main__":
    test_scatter_broadcast_vars()
    test_scatter_signed_channel()
