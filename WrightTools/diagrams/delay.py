"""Delay space."""


# --- import --------------------------------------------------------------------------------------


import matplotlib.pyplot as plt


# --- label sectors -------------------------------------------------------------------------------


def label_sectors(
    *,
    labels=["I", "II", "IV", "VI", "V", "III"],
    ax=None,
    lw=2,
    lc="k",
    cs=None,
    c_zlevel=2,
    c_alpha=0.5,
    fontsize=40
):
    """Label the six time-orderings in a three-pulse experiment.

    Parameters
    ----------
    labels : list of strings
        Labels to place within sectors, starting in the upper left and
        proceeding clockwise. Default is ['I', 'II', 'IV', 'VI', 'V', 'III'].
    ax : matplotlib axis object (optional)
        Axis to label. If None, uses current axis. Default is None.
    cs : list of matplotlib colors (optional)
        Color to label sectors. If None, sectors are not colored. Default is
        None.
    c_zlevel : number (optional)
        Matplotlib zlevel of color. Default is 2.
    c_alpha : number between 0 and 1.
        Transparency of color. Default is 0.5
    """
    if ax is None:
        ax = plt.gca()
    # label
    factors = [
        [0.25, 0.75],
        [2 / 3, 5 / 6],
        [5 / 6, 2 / 3],
        [0.75, 0.25],
        [1 / 3, 1 / 6],
        [1 / 6, 1 / 3],
    ]
    transform = ax.transAxes
    for label, factor in zip(labels, factors):
        ax.text(
            *factor + [label], fontsize=fontsize, va="center", ha="center", transform=transform
        )
    # lines
    if lw > 0:
        ax.axhline(0, c=lc, lw=lw)
        ax.axvline(0, c=lc, lw=lw)
        ax.plot([0, 1], [0, 1], c=lc, lw=lw, transform=transform)
    # colors
    if cs is None:
        cs = ["none"] * 6
    xbound = ax.get_xbound()
    ybound = ax.get_ybound()
    factors = []
    factors.append([[xbound[0], 0], [0, 0], [ybound[1], ybound[1]]])
    factors.append([[0, xbound[1]], [0, ybound[1]], [ybound[1], ybound[1]]])
    factors.append([[0, xbound[1]], [0, 0], [0, ybound[1]]])
    factors.append([[0, xbound[1]], [ybound[0], ybound[0]], [0, 0]])
    factors.append([[xbound[0], 0], [ybound[0], ybound[0]], [ybound[0], 0]])
    factors.append([[xbound[0], 0], [ybound[0], 0], [0, 0]])
    for color, factor in zip(cs, factors):
        poly = ax.fill_between(*factor, facecolor=color, edgecolor="none", alpha=c_alpha)
        poly.set_zorder(c_zlevel)


# --- testing -------------------------------------------------------------------------------------


if __name__ == "__main__":

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([-1, 1], [-1, 1], color="k", linewidth=2)
    ax.axhline(0, color="k", linewidth=2)
    ax.axvline(0, color="k", linewidth=2)

    ax.text(-0.5, 0.5, "I", fontsize=30, verticalalignment="center", horizontalalignment="center")
    ax.text(0.25, 0.6, "II", fontsize=30, verticalalignment="center", horizontalalignment="center")
    ax.text(
        -0.6, -0.25, "III", fontsize=30, verticalalignment="center", horizontalalignment="center"
    )
    ax.text(0.6, 0.25, "IV", fontsize=30, verticalalignment="center", horizontalalignment="center")
    ax.text(
        -0.25, -0.6, "V", fontsize=30, verticalalignment="center", horizontalalignment="center"
    )
    ax.text(0.5, -0.5, "VI", fontsize=30, verticalalignment="center", horizontalalignment="center")

    ax.set_xlabel(r"d1 $\mathrm{(\tau_{22^{\prime}})}$", fontsize=15)
    ax.set_ylabel(r"d2 $\mathrm{(\tau_{21})}$", fontsize=15)
    ax.set_title("ultimate representation")

    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        left="off",
        right="off",
        labelleft="off",
        labelbottom="off",
    )

    ax.set_aspect(1.)

    plt.savefig("TRIEE_delay_space.png", transparent=True)

    plt.close()

    # as collected --------------------------------------------------------------------------------

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([-1, 1], [1, -1], color="k", linewidth=2)
    ax.axhline(0, color="k", linewidth=2)
    ax.axvline(0, color="k", linewidth=2)

    ax.text(0.5, 0.5, "I", fontsize=30, verticalalignment="center", horizontalalignment="center")
    ax.text(
        -0.25, 0.6, "II", fontsize=30, verticalalignment="center", horizontalalignment="center"
    )
    ax.text(
        0.6, -0.25, "III", fontsize=30, verticalalignment="center", horizontalalignment="center"
    )
    ax.text(
        -0.6, 0.25, "IV", fontsize=30, verticalalignment="center", horizontalalignment="center"
    )
    ax.text(0.25, -0.6, "V", fontsize=30, verticalalignment="center", horizontalalignment="center")
    ax.text(
        -0.5, -0.5, "VI", fontsize=30, verticalalignment="center", horizontalalignment="center"
    )

    ax.set_xlabel(r"d1 $\mathrm{(\tau_{2^{\prime}2})}$", fontsize=15)
    ax.set_ylabel(r"d2 $\mathrm{(\tau_{21})}$", fontsize=15)
    ax.set_title("as collected")

    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        left="off",
        right="off",
        labelleft="off",
        labelbottom="off",
    )

    ax.set_aspect(1.)

    plt.savefig("TRIEE_delay_space_as_collected.png", transparent=True)
