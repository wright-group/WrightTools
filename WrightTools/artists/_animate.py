"""helpers for making animations"""

import matplotlib.pyplot as plt
import numpy as np
import logging

from functools import partial
from matplotlib.animation import FuncAnimation
from inspect import isclass

from ._helpers import norm_from_channel
from ._interact import interact2D_fig
from ._quick import _quick2D

__all__ = ["animate2D", "animate_interact2D", "animate_quick2D"]
logger = logging.getLogger("animation")


def animate2D(
    data,
    norm=None,
    channel=0,
    cmap=None,
    back_and_forth: bool = False,
    **ani_kwargs,
):
    """
    animate pcolormesh of a nd dataset (ndim >=2)
    mesh plots last two axes of the dataset (use `Data.transform` if needed)

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

    back_and_forth: bool = False
        when True, the animation will go in reverse after going forward,
        creating a continuous loop when repeat is no

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
    ```
    The animation has write to file utilities like `to_html5_video`:
    ```
    with open(f"{d.natural_name}_animation.html", "w") as f:
        f.write(ani.to_html5_video())
    ```
    Alternatively, you can show in the interactive viewer and watch the animation:
    ```
    plt.show()
    ```
    For colorbar normalized at each frame, you can use `functools.partial`:
    ```
    from functools import partial
    norm = partial(CenteredNorm, vcenter=0)  # halfrange evaluated for each frame
    ```
    """

    channel = data.get_channel(channel)
    if norm is None:
        norm = norm_from_channel(channel)
    if cmap is None:
        cmap = "signed" if channel.signed else "default"
    # detect whether to call norm each frame
    # probably not an optimal implementation, but working for now
    call_norm = isclass(norm) or isinstance(norm, partial)

    def gen_title(ind):
        parts = [
            f"{var.natural_name} = {var[:].squeeze()[ind]:.2f} {var.units}"
            for var in map(lambda a: a.variables[0], data.axes[:-2])
        ]
        return "\n".join(parts)

    # initialize canvas
    fig, ax = plt.subplots(subplot_kw=dict(projection="wright"), dpi=140, layout="constrained")
    art = ax.pcolormesh(
        data[tuple([0 for i in data.shape[:-2]])],
        cmap=cmap,
        norm=norm() if call_norm else norm,
    )
    colorbar = fig.colorbar(art, ax=ax)
    colorbar.set_label(channel.label)

    ax.set_title(gen_title(tuple([0 for _ in data.shape[:-2]])))
    # with layout well set, turn off the engine (avoids jittering frames)
    fig.set_layout_engine("none")

    def updater(frame):
        logger.info(f"{frame=}")
        art.set_array(data.channels[0][frame])
        ax.set_title(gen_title(frame))
        art.set_norm(norm() if call_norm else norm)
        fig.canvas.draw_idle()
        return art

    # generate frame sequence
    frames = list(np.ndindex(data.shape[:-2]))
    if back_and_forth:
        frames += reversed(frames)

    return FuncAnimation(
        fig=fig,
        func=updater,
        frames=frames,
        **ani_kwargs,
    )


def animate_quick2D(data, *args, **kwargs):
    """
    animate a quick2D series

    function accepts same arguments as artists.Quick2D

    animation kwargs can be passed through a dictionary `fa_kwargs`

    unlike other animation functions, this enforces repeat=False
    """

    # trying for minimal code here, and just drawing from quick2D's constructs
    # quick2D is constructed as a generator, so looping and reversing is 
    # not simple to implement
    fa_kwargs = kwargs.pop("fa_kwargs", dict())
    q2d = _quick2D(data, *args, **kwargs)
    generator = q2d.__iter__()
    fig = generator.__next__()

    def updater(frame):
        logger.info(f"{frame=}")
        generator.__next__()
        fig.canvas.draw_idle()
        return

    return FuncAnimation(fig=fig, func=updater, frames=q2d.nfigs, **fa_kwargs, repeat=False)


def animate_interact2D(interact2D: interact2D_fig, back_and_forth=False, **kwargs):
    """
    Take an interact2D figure and create an animation by moving the sliders.

    Parameters
    ----------
    interact2D: interact2D_fig
        the output of an interact2D call

    back_and_forth: bool = False
        when True, the animation will go in reverse after going forward,
        creating a continuous loop when repeat is no

    **kwargs: dict items
        all extra kwargs are passed to matplotlib.FuncAnimation
    """

    def update(frame):
        logger.info(f"{frame=}")
        for ind, slider in zip(frame, interact2D.sliders.values()):
            slider.set_val(ind)

    frames = list(np.ndindex(tuple([s.valmax + 1 for s in interact2D.sliders.values()])))
    if back_and_forth:
        frames += reversed(frames)

    return FuncAnimation(
        fig=interact2D.fig,
        func=update,
        frames=frames,
    )
