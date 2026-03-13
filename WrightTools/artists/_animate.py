"""helpers for making animations"""


import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from matplotlib.animation import FuncAnimation

from ._interact import interact2D_fig


__all__ = ["animate2D", "animate_interact2D"]


def animate2D(
    data,
    norm,
    channel=0,
    cmap=None,
    snake:bool=False,
    back_and_forth:bool=False,
    **ani_kwargs,
):
    """
    animate pcolormesh of a nd dataset (ndim >=2),
    mesh plots last two axes of the dataset (use `Data.transform` if needed)
    uses first channel in dataset (use `bring_to_front` if needed)

    Parameters
    ----------
    
    data: WrightTools.data
        dataset to animate.  take the last two axes as the ones that are plotted;
        other axes compose the frames of the animation
    
    norm: Normalize instance or callable
        determines the normalization rules to follow.  
        If channel is signed, defaults to CenteredNorm with null center.
        If channel is unsigned, defaults to Normalize from null to max.

    channel: string, index, or Channel
        Select which channel to plot
        
    cmap: str or Colormap (optional)
        colormap used.  Defaults to WrightTools default

    **kwargs: dict items
        all extra kwargs are passed to matplotlib.FuncAnimation
        
    Example
    -------
    General usage (create an animation procedure and then write to file):
    ```
    norm=CenteredNorm(vcenter=0, halfrange=np.abs(d.channels[0][:]).max())
    ani = wt.artists.animate2D(
        d, cmap="bwr", norm=norm, interval=100
    )
    with open(f"{d.natural_name}_animation.html", "w") as f:
        f.write(ani.to_html5_video())
    ani.pause()  # if you use interactive viewer after, animation will loop unless you pause
    ```
    For colorbar normalized at each frame, you can use functools.partial:
    ```
    norm = partial(CenteredNorm, vcenter=0)  # halfrange evaluated for each frame
    ```

    Notes
    -----
    
    snake, back_and_forth are not yet implemented
    """

    channel = data.get_channel(channel)

    # initialize canvas
    fig, ax = plt.subplots(subplot_kw=dict(projection="wright"), dpi=140, layout="constrained")
    art = ax.pcolormesh(data[tuple([0 for i in data.shape[:-2]])], cmap=cmap, norm=norm() if callable(norm) else norm)
    colorbar = fig.colorbar(art, ax=ax)
    colorbar.set_label(channel.label)

    # funcs for updating
    def title(ind):
        parts = [
            f"{var.natural_name} = {var[:].squeeze()[ind]:.2f} {var.units}"
            for var in map(lambda a: a.variables[0], data.axes[:-2])
        ]
        return "\n".join(parts)

    ax.set_title(title(0))
    # with layout well set, turn off the engine (avoids jitter)
    fig.set_layout_engine("none")

    def update2D(frame, data, fig, ax, mesh, norm):
        print(f"{frame=}")
        # for ind, axis in zip(frame, data.axes[:-2]):
        mesh.set_array(data.channels[0][frame])
        ax.set_title(title(frame))
        mesh.set_norm(norm() if callable(norm) else norm)
        fig.canvas.draw_idle()
        return mesh

    frames = list(np.ndindex(data.shape[:-2]))
    return FuncAnimation(
        fig=fig,
        func=partial(update2D, data=data, mesh=art, fig=fig, ax=ax, norm=norm),
        frames=frames,
        **ani_kwargs,
    )


def animate_quick2D(quick2D: plt.Figure, snake=False, back_and_forth=False, **kwargs):
    """animate a quick2D series"""
    raise NotImplementedError


def animate_interact2D(interact2D: interact2D_fig, snake=False, back_and_forth=False, **kwargs):
    """
    Take an interact2D figure and create an animation by moving the sliders.

    Note: snake, back_and_forth are not yet implemented
    """

    def update(frame):
        print(f"{frame=}")
        for ind, slider in zip(frame, interact2D.sliders.values()):
            slider.set_val(ind)

    frames = list(np.ndindex(tuple([s.valmax + 1 for s in interact2D.sliders.values()])))
    return FuncAnimation(
        fig=interact2D.fig,
        func=update,
        frames=frames,
    )
