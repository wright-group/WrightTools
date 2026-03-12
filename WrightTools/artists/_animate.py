"""library for making animations"""

"""TODO: push these functions into WrightTools"""

import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from matplotlib.animation import FuncAnimation

# typing
from ._interact import interact2D_fig

__all__ = ["animate2D", "animate_interact2D"]


def animate2D(
    d,
    cmap,
    norm,
    snake=False,
    back_and_forth=False,
    **ani_kwargs,
):
    """
    animate pcolormesh of a nd dataset (ndim >=2),
    mesh plots last two axes of the dataset (use `Data.transform` if needed)
    uses first channel in dataset (use `bring_to_front` if needed)

    Note: snake, back_and_forth are not yet implemented are no-ops
    """

    # initialize canvas
    # need fig, ax, art, and norm to get this working
    fig, ax = plt.subplots(subplot_kw=dict(projection="wright"), dpi=140, layout="constrained")
    art = ax.pcolormesh(d[tuple([0 for i in d.shape[:-2]])], cmap=cmap, norm=norm)
    fig.colorbar(art, ax=ax)

    # funcs for updating
    def title(ind):
        parts = [
            f"{var.natural_name} = {var[:].squeeze()[ind]:.2f} {var.units}"
            for var in map(lambda a: a.variables[0], d.axes[:-2])
        ]
        return "\n".join(parts)

    ax.set_title(title(0))

    def update2D(frame, data, fig, ax, mesh, norm):
        print(frame)
        # for ind, axis in zip(frame, data.axes[:-2]):
        mesh.set_array(data.channels[0][frame])
        ax.set_title(title(frame))
        mesh.set_norm(norm)
        fig.canvas.draw_idle()
        return mesh

    frames = list(np.ndindex(d.shape[:-2]))

    return FuncAnimation(
        fig=fig,
        func=partial(update2D, data=d, mesh=art, fig=fig, ax=ax, norm=norm),
        frames=frames,
        **ani_kwargs,
    )


def animate_interact2D(interact2D: interact2D_fig, snake=False, back_and_forth=False, **kwargs):
    """
    Take an interact2D figure and create an animation by moving the sliders.

    Note: snake, back_and_forth are not yet implemented are no-ops
    """

    def update(frame):
        print(frame)
        for ind, slider in zip(frame, interact2D.sliders.values()):
            slider.set_val(ind)

    frames = list(np.ndindex(tuple([s.valmax + 1 for s in interact2D.sliders.values()])))
    print(f"beginning animation: {len(frames)} frames to write")
    return FuncAnimation(
        fig=interact2D.fig,
        func=update,
        frames=frames,
    )
