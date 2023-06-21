"""Colormaps."""


# --- import --------------------------------------------------------------------------------------

import copy

import numpy as np
from numpy import r_

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.gridspec as grd

from ._turbo import turbo


# --- define -------------------------------------------------------------------------------------


__all__ = [
    "colormaps",
    "get_color_cycle",
    "grayify_cmap",
    "overline_colors",
    "plot_colormap_components",
]


# --- functions ----------------------------------------------------------------------------------


def make_cubehelix(name="WrightTools", gamma=0.5, s=0.25, r=-1, h=1.3, reverse=False, darkest=0.7):
    """Define cubehelix type colorbars.

    Look `here`__ for more information.

    __ http://arxiv.org/abs/1108.5083


    Parameters
    ----------
    name : string (optional)
        Name of new cmap. Default is WrightTools.
    gamma : number (optional)
        Intensity factor. Default is 0.5
    s : number (optional)
        Start color factor. Default is 0.25
    r : number (optional)
        Number and direction of rotations. Default is -1
    h : number (option)
        Hue factor. Default is 1.3
    reverse : boolean (optional)
        Toggle reversal of output colormap. By default (Reverse = False),
        colormap goes from light to dark.
    darkest : number (optional)
        Default is 0.7

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap

    See Also
    --------
    plot_colormap_components
        Displays RGB components of colormaps.
    """
    rr = 0.213 / 0.30
    rg = 0.715 / 0.99
    rb = 0.072 / 0.11

    def get_color_function(p0, p1):
        def color(x):
            # Calculate amplitude and angle of deviation from the black to
            # white diagonal in the plane of constant perceived intensity.
            xg = darkest * x**gamma
            lum = 1 - xg  # starts at 1
            if reverse:
                lum = lum[::-1]
            a = lum.copy()
            a[lum < 0.5] = h * lum[lum < 0.5] / 2.0
            a[lum >= 0.5] = h * (1 - lum[lum >= 0.5]) / 2.0
            phi = 2 * np.pi * (s / 3 + r * x)
            out = lum + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
            return out

        return color

    rgb_dict = {
        "red": get_color_function(-0.14861 * rr, 1.78277 * rr),
        "green": get_color_function(-0.29227 * rg, -0.90649 * rg),
        "blue": get_color_function(1.97294 * rb, 0.0),
    }
    cmap = matplotlib.colors.LinearSegmentedColormap(name, rgb_dict)
    return cmap


def make_colormap(seq, name="CustomMap", plot=False):
    """Generate a LinearSegmentedColormap.

    Parameters
    ----------
    seq : list of tuples
        A sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
    name : string (optional)
        A name for the colormap
    plot : boolean (optional)
        Use to generate a plot of the colormap (Default is False).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap


    `Source`__

    __ http://nbviewer.ipython.org/gist/anonymous/a4fa0adb08f9e9ea4f94
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    cmap = mplcolors.LinearSegmentedColormap(name, cdict)
    if plot:
        plot_colormap_components(cmap)
    return cmap


def nm_to_rgb(nm):
    """Convert a wavelength to corresponding RGB values [0.0-1.0].

    Parameters
    ----------
    nm : int or float
        The wavelength of light.

    Returns
    -------
    List of [R,G,B] values between 0 and 1


    `original code`__

    __ http://www.physics.sfasu.edu/astro/color/spectra.html
    """
    w = int(nm)
    # color ---------------------------------------------------------------------------------------
    if w >= 380 and w < 440:
        R = -(w - 440.0) / (440.0 - 350.0)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.0) / (490.0 - 440.0)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.0) / (510.0 - 490.0)
    elif w >= 510 and w < 580:
        R = (w - 510.0) / (580.0 - 510.0)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.0) / (645.0 - 580.0)
        B = 0.0
    elif w >= 645 and w <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    # intensity correction ------------------------------------------------------------------------
    if w >= 380 and w < 420:
        SSS = 0.3 + 0.7 * (w - 350) / (420 - 350)
    elif w >= 420 and w <= 700:
        SSS = 1.0
    elif w > 700 and w <= 780:
        SSS = 0.3 + 0.7 * (780 - w) / (780 - 700)
    else:
        SSS = 0.0
    SSS *= 255
    return [
        float(int(SSS * R) / 256.0),
        float(int(SSS * G) / 256.0),
        float(int(SSS * B) / 256.0),
    ]


def plot_colormap_components(cmap):
    """Plot the components of a given colormap."""
    from ._helpers import set_ax_labels  # recursive import protection

    plt.figure(figsize=[8, 4])
    gs = grd.GridSpec(3, 1, height_ratios=[1, 10, 1], hspace=0.05)
    # colorbar
    ax = plt.subplot(gs[0])
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(cmap.name, fontsize=20)
    ax.set_axis_off()
    # components
    ax = plt.subplot(gs[1])
    x = np.arange(cmap.N)
    colors = cmap(x)
    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]
    RGB_weight = [0.299, 0.587, 0.114]
    k = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    r.clip(0, 1, out=r)
    g.clip(0, 1, out=g)
    b.clip(0, 1, out=b)
    xi = np.linspace(0, 1, x.size)
    plt.plot(xi, r, "r", linewidth=5, alpha=0.6)
    plt.plot(xi, g, "g", linewidth=5, alpha=0.6)
    plt.plot(xi, b, "b", linewidth=5, alpha=0.6)
    plt.plot(xi, k, "k", linewidth=5, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    set_ax_labels(ax=ax, xlabel=None, xticks=False, ylabel="intensity")
    # grayified colorbar
    cmap = grayify_cmap(cmap)
    ax = plt.subplot(gs[2])
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_axis_off()


def grayify_cmap(cmap):
    """Return a grayscale version of the colormap.

    `Source`__

    __ https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
    """
    if not isinstance(cmap, matplotlib.colors.Colormap):
        cmap = matplotlib.colormaps[cmap]
    colors = cmap(np.arange(cmap.N))
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return mplcolors.LinearSegmentedColormap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def get_color_cycle(n, cmap="rainbow", rotations=3):
    """Get a list of RGBA colors following a colormap.

    Useful for plotting lots of elements, keeping the color of each unique.

    Parameters
    ----------
    n : integer
        The number of colors to return.
    cmap : string (optional)
        The colormap to use in the cycle. Default is rainbow.
    rotations : integer (optional)
        The number of times to repeat the colormap over the cycle. Default is 3.

    Returns
    -------
    list
        List of RGBA lists.
    """
    cmap = colormaps[cmap]
    if np.mod(n, rotations) == 0:
        per = np.floor_divide(n, rotations)
    else:
        per = np.floor_divide(n, rotations) + 1
    vals = list(np.linspace(0, 1, per))
    vals = vals * rotations
    vals = vals[:n]
    out = cmap(vals)
    return out


# --- color maps ----------------------------------------------------------------------------------


cubehelix = make_cubehelix()

greenscale = ["#000000", "#00FF00"]  # black  # green

greyscale = ["#FFFFFF", "#000000"]  # white  # black

invisible = ["#FFFFFF", "#FFFFFF"]  # white  # white

# isoluminant colorbar based on the research of Kindlmann et al.
# http://dx.doi.org/10.1109/VISUAL.2002.1183788
c = mplcolors.ColorConverter().to_rgb
isoluminant1 = make_colormap(
    [
        c(r_[1.000, 1.000, 1.000]),
        c(r_[0.847, 0.057, 0.057]),
        1 / 6.0,
        c(r_[0.847, 0.057, 0.057]),
        c(r_[0.527, 0.527, 0.000]),
        2 / 6.0,
        c(r_[0.527, 0.527, 0.000]),
        c(r_[0.000, 0.592, 0.000]),
        3 / 6.0,
        c(r_[0.000, 0.592, 0.000]),
        c(r_[0.000, 0.559, 0.559]),
        4 / 6.0,
        c(r_[0.000, 0.559, 0.559]),
        c(r_[0.316, 0.316, 0.991]),
        5 / 6.0,
        c(r_[0.316, 0.316, 0.991]),
        c(r_[0.718, 0.000, 0.718]),
    ],
    name="isoluminant`",
)

isoluminant2 = make_colormap(
    [
        c(r_[1.000, 1.000, 1.000]),
        c(r_[0.718, 0.000, 0.718]),
        1 / 6.0,
        c(r_[0.718, 0.000, 0.718]),
        c(r_[0.316, 0.316, 0.991]),
        2 / 6.0,
        c(r_[0.316, 0.316, 0.991]),
        c(r_[0.000, 0.559, 0.559]),
        3 / 6.0,
        c(r_[0.000, 0.559, 0.559]),
        c(r_[0.000, 0.592, 0.000]),
        4 / 6.0,
        c(r_[0.000, 0.592, 0.000]),
        c(r_[0.527, 0.527, 0.000]),
        5 / 6.0,
        c(r_[0.527, 0.527, 0.000]),
        c(r_[0.847, 0.057, 0.057]),
    ],
    name="isoluminant2",
)

isoluminant3 = make_colormap(
    [
        c(r_[1.000, 1.000, 1.000]),
        c(r_[0.316, 0.316, 0.991]),
        1 / 5.0,
        c(r_[0.316, 0.316, 0.991]),
        c(r_[0.000, 0.559, 0.559]),
        2 / 5.0,
        c(r_[0.000, 0.559, 0.559]),
        c(r_[0.000, 0.592, 0.000]),
        3 / 5.0,
        c(r_[0.000, 0.592, 0.000]),
        c(r_[0.527, 0.527, 0.000]),
        4 / 5.0,
        c(r_[0.527, 0.527, 0.000]),
        c(r_[0.847, 0.057, 0.057]),
    ],
    name="isoluminant3",
)

signed_old = [
    "#0000FF",  # blue
    "#00BBFF",  # blue-aqua
    "#00FFFF",  # aqua
    "#FFFFFF",  # white
    "#FFFF00",  # yellow
    "#FFBB00",  # orange
    "#FF0000",  # red
]

skyebar = [
    "#FFFFFF",  # white
    "#000000",  # black
    "#0000FF",  # blue
    "#00FFFF",  # cyan
    "#64FF00",  # light green
    "#FFFF00",  # yellow
    "#FF8000",  # orange
    "#FF0000",  # red
    "#800000",  # dark red
]

skyebar_d = [
    "#000000",  # black
    "#0000FF",  # blue
    "#00FFFF",  # cyan
    "#64FF00",  # light green
    "#FFFF00",  # yellow
    "#FF8000",  # orange
    "#FF0000",  # red
    "#800000",  # dark red
]

skyebar_i = [
    "#000000",  # black
    "#FFFFFF",  # white
    "#0000FF",  # blue
    "#00FFFF",  # cyan
    "#64FF00",  # light green
    "#FFFF00",  # yellow
    "#FF8000",  # orange
    "#FF0000",  # red
    "#800000",  # dark red
]

wright = ["#FFFFFF", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000", "#881111"]


class cmapdict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        self[key] = plt.get_cmap(key)
        return self.get(key)


colormaps = cmapdict()

colormaps["cubehelix"] = copy.copy(plt.get_cmap("cubehelix_r"))
colormaps["default"] = cubehelix
colormaps["signed"] = copy.copy(plt.get_cmap("bwr"))
colormaps["greenscale"] = mplcolors.LinearSegmentedColormap.from_list("greenscale", greenscale)
colormaps["greyscale"] = mplcolors.LinearSegmentedColormap.from_list("greyscale", greyscale)
colormaps["invisible"] = mplcolors.LinearSegmentedColormap.from_list("invisible", invisible)
colormaps["isoluminant1"] = isoluminant1
colormaps["isoluminant2"] = isoluminant2
colormaps["isoluminant3"] = isoluminant3
colormaps["signed_old"] = mplcolors.LinearSegmentedColormap.from_list("signed", signed_old)
colormaps["skyebar1"] = mplcolors.LinearSegmentedColormap.from_list("skyebar", skyebar)
colormaps["skyebar2"] = mplcolors.LinearSegmentedColormap.from_list("skyebar dark", skyebar_d)
colormaps["skyebar3"] = mplcolors.LinearSegmentedColormap.from_list("skyebar inverted", skyebar_i)
colormaps["turbo"] = turbo
colormaps["wright"] = mplcolors.LinearSegmentedColormap.from_list("wright", wright)


# enforce grey as 'bad' value for colormaps
for cmap in colormaps.values():
    cmap.set_bad([0.75] * 3, 1)
# enforce under and over for default colormap
colormaps["default"].set_under([0.50] * 3, 1)
colormaps["default"].set_over("m")
# enforce under and over for signed colormap
colormaps["signed"].set_under("c")
colormaps["signed"].set_over("m")


# a nice set of line colors
overline_colors = ["#CCFF00", "#FE4EDA", "#FF6600", "#00FFBF", "#00B7EB"]
