import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt


def test_scatter_heatmap_unstructured():
    data = wt.data.Data(name="data")
    rng1 = np.random.rand(10**3)
    rng2 = np.random.rand(10**3)
    data.create_variable("x", values=2 * rng1 - 1)
    data.create_variable("y", values=2 * rng2 - 1)
    data.create_channel("z", values=np.exp(-(data.x[:] ** 2 + data.y[:] ** 2)))
    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.scatter_heatmap(data, "x", "y", channel="z")
    plt.show()


if __name__ == "__main__":
    test_scatter_heatmap_unstructured()
