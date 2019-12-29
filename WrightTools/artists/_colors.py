"""Colormaps."""


# --- import --------------------------------------------------------------------------------------

import collections

import numpy as np
from numpy import r_

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.gridspec as grd


# --- define -------------------------------------------------------------------------------------


__all__ = [
    "colormaps",
    "get_color_cycle",
    "grayify_cmap",
    "overline_colors",
    "plot_colormap_components",
]


# --- functions ----------------------------------------------------------------------------------


def make_cubehelix(
    name="WrightTools", gamma=0.5, s=0.25, r=-1, h=1.3, reverse=False, darkest=0.7
):
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
            xg = darkest * x ** gamma
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
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return mplcolors.LinearSegmentedColormap.from_list(
        cmap.name + "_grayscale", colors, cmap.N
    )


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

experimental = [
    "#FFFFFF",
    "#0000FF",
    "#0080FF",
    "#00FFFF",
    "#00FF00",
    "#FFFF00",
    "#FF8000",
    "#FF0000",
    "#881111",
]

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

signed = [
    "#0000FF",  # blue
    "#002AFF",
    "#0055FF",
    "#007FFF",
    "#00AAFF",
    "#00D4FF",
    "#00FFFF",
    "#FFFFFF",  # white
    "#FFFF00",
    "#FFD400",
    "#FFAA00",
    "#FF7F00",
    "#FF5500",
    "#FF2A00",
    "#FF0000",
]  # red

signed_old = [
    "#0000FF",  # blue
    "#00BBFF",  # blue-aqua
    "#00FFFF",  # aqua
    "#FFFFFF",  # white
    "#FFFF00",  # yellow
    "#FFBB00",  # orange
    "#FF0000",
]  # red

skyebar = [
    "#FFFFFF",  # white
    "#000000",  # black
    "#0000FF",  # blue
    "#00FFFF",  # cyan
    "#64FF00",  # light green
    "#FFFF00",  # yellow
    "#FF8000",  # orange
    "#FF0000",  # red
    "#800000",
]  # dark red

skyebar_d = [
    "#000000",  # black
    "#0000FF",  # blue
    "#00FFFF",  # cyan
    "#64FF00",  # light green
    "#FFFF00",  # yellow
    "#FF8000",  # orange
    "#FF0000",  # red
    "#800000",
]  # dark red

skyebar_i = [
    "#000000",  # black
    "#FFFFFF",  # white
    "#0000FF",  # blue
    "#00FFFF",  # cyan
    "#64FF00",  # light green
    "#FFFF00",  # yellow
    "#FF8000",  # orange
    "#FF0000",  # red
    "#800000",
]  # dark red

# see https://github.com/matplotlib/matplotlib/issues/15091#issuecomment-565870497
# https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
# this colormap will be removed when turbo is added to Matplotlib's distributed version
turbo_seq = [
    [0.18995, 0.07176, 0.23217],
    [0.19483, 0.08339, 0.26149],
    [0.19956, 0.09498, 0.29024],
    [0.20415, 0.10652, 0.31844],
    [0.20860, 0.11802, 0.34607],
    [0.21291, 0.12947, 0.37314],
    [0.21708, 0.14087, 0.39964],
    [0.22111, 0.15223, 0.42558],
    [0.22500, 0.16354, 0.45096],
    [0.22875, 0.17481, 0.47578],
    [0.23236, 0.18603, 0.50004],
    [0.23582, 0.19720, 0.52373],
    [0.23915, 0.20833, 0.54686],
    [0.24234, 0.21941, 0.56942],
    [0.24539, 0.23044, 0.59142],
    [0.24830, 0.24143, 0.61286],
    [0.25107, 0.25237, 0.63374],
    [0.25369, 0.26327, 0.65406],
    [0.25618, 0.27412, 0.67381],
    [0.25853, 0.28492, 0.69300],
    [0.26074, 0.29568, 0.71162],
    [0.26280, 0.30639, 0.72968],
    [0.26473, 0.31706, 0.74718],
    [0.26652, 0.32768, 0.76412],
    [0.26816, 0.33825, 0.78050],
    [0.26967, 0.34878, 0.79631],
    [0.27103, 0.35926, 0.81156],
    [0.27226, 0.36970, 0.82624],
    [0.27334, 0.38008, 0.84037],
    [0.27429, 0.39043, 0.85393],
    [0.27509, 0.40072, 0.86692],
    [0.27576, 0.41097, 0.87936],
    [0.27628, 0.42118, 0.89123],
    [0.27667, 0.43134, 0.90254],
    [0.27691, 0.44145, 0.91328],
    [0.27701, 0.45152, 0.92347],
    [0.27698, 0.46153, 0.93309],
    [0.27680, 0.47151, 0.94214],
    [0.27648, 0.48144, 0.95064],
    [0.27603, 0.49132, 0.95857],
    [0.27543, 0.50115, 0.96594],
    [0.27469, 0.51094, 0.97275],
    [0.27381, 0.52069, 0.97899],
    [0.27273, 0.53040, 0.98461],
    [0.27106, 0.54015, 0.98930],
    [0.26878, 0.54995, 0.99303],
    [0.26592, 0.55979, 0.99583],
    [0.26252, 0.56967, 0.99773],
    [0.25862, 0.57958, 0.99876],
    [0.25425, 0.58950, 0.99896],
    [0.24946, 0.59943, 0.99835],
    [0.24427, 0.60937, 0.99697],
    [0.23874, 0.61931, 0.99485],
    [0.23288, 0.62923, 0.99202],
    [0.22676, 0.63913, 0.98851],
    [0.22039, 0.64901, 0.98436],
    [0.21382, 0.65886, 0.97959],
    [0.20708, 0.66866, 0.97423],
    [0.20021, 0.67842, 0.96833],
    [0.19326, 0.68812, 0.96190],
    [0.18625, 0.69775, 0.95498],
    [0.17923, 0.70732, 0.94761],
    [0.17223, 0.71680, 0.93981],
    [0.16529, 0.72620, 0.93161],
    [0.15844, 0.73551, 0.92305],
    [0.15173, 0.74472, 0.91416],
    [0.14519, 0.75381, 0.90496],
    [0.13886, 0.76279, 0.89550],
    [0.13278, 0.77165, 0.88580],
    [0.12698, 0.78037, 0.87590],
    [0.12151, 0.78896, 0.86581],
    [0.11639, 0.79740, 0.85559],
    [0.11167, 0.80569, 0.84525],
    [0.10738, 0.81381, 0.83484],
    [0.10357, 0.82177, 0.82437],
    [0.10026, 0.82955, 0.81389],
    [0.09750, 0.83714, 0.80342],
    [0.09532, 0.84455, 0.79299],
    [0.09377, 0.85175, 0.78264],
    [0.09287, 0.85875, 0.77240],
    [0.09267, 0.86554, 0.76230],
    [0.09320, 0.87211, 0.75237],
    [0.09451, 0.87844, 0.74265],
    [0.09662, 0.88454, 0.73316],
    [0.09958, 0.89040, 0.72393],
    [0.10342, 0.89600, 0.71500],
    [0.10815, 0.90142, 0.70599],
    [0.11374, 0.90673, 0.69651],
    [0.12014, 0.91193, 0.68660],
    [0.12733, 0.91701, 0.67627],
    [0.13526, 0.92197, 0.66556],
    [0.14391, 0.92680, 0.65448],
    [0.15323, 0.93151, 0.64308],
    [0.16319, 0.93609, 0.63137],
    [0.17377, 0.94053, 0.61938],
    [0.18491, 0.94484, 0.60713],
    [0.19659, 0.94901, 0.59466],
    [0.20877, 0.95304, 0.58199],
    [0.22142, 0.95692, 0.56914],
    [0.23449, 0.96065, 0.55614],
    [0.24797, 0.96423, 0.54303],
    [0.26180, 0.96765, 0.52981],
    [0.27597, 0.97092, 0.51653],
    [0.29042, 0.97403, 0.50321],
    [0.30513, 0.97697, 0.48987],
    [0.32006, 0.97974, 0.47654],
    [0.33517, 0.98234, 0.46325],
    [0.35043, 0.98477, 0.45002],
    [0.36581, 0.98702, 0.43688],
    [0.38127, 0.98909, 0.42386],
    [0.39678, 0.99098, 0.41098],
    [0.41229, 0.99268, 0.39826],
    [0.42778, 0.99419, 0.38575],
    [0.44321, 0.99551, 0.37345],
    [0.45854, 0.99663, 0.36140],
    [0.47375, 0.99755, 0.34963],
    [0.48879, 0.99828, 0.33816],
    [0.50362, 0.99879, 0.32701],
    [0.51822, 0.99910, 0.31622],
    [0.53255, 0.99919, 0.30581],
    [0.54658, 0.99907, 0.29581],
    [0.56026, 0.99873, 0.28623],
    [0.57357, 0.99817, 0.27712],
    [0.58646, 0.99739, 0.26849],
    [0.59891, 0.99638, 0.26038],
    [0.61088, 0.99514, 0.25280],
    [0.62233, 0.99366, 0.24579],
    [0.63323, 0.99195, 0.23937],
    [0.64362, 0.98999, 0.23356],
    [0.65394, 0.98775, 0.22835],
    [0.66428, 0.98524, 0.22370],
    [0.67462, 0.98246, 0.21960],
    [0.68494, 0.97941, 0.21602],
    [0.69525, 0.97610, 0.21294],
    [0.70553, 0.97255, 0.21032],
    [0.71577, 0.96875, 0.20815],
    [0.72596, 0.96470, 0.20640],
    [0.73610, 0.96043, 0.20504],
    [0.74617, 0.95593, 0.20406],
    [0.75617, 0.95121, 0.20343],
    [0.76608, 0.94627, 0.20311],
    [0.77591, 0.94113, 0.20310],
    [0.78563, 0.93579, 0.20336],
    [0.79524, 0.93025, 0.20386],
    [0.80473, 0.92452, 0.20459],
    [0.81410, 0.91861, 0.20552],
    [0.82333, 0.91253, 0.20663],
    [0.83241, 0.90627, 0.20788],
    [0.84133, 0.89986, 0.20926],
    [0.85010, 0.89328, 0.21074],
    [0.85868, 0.88655, 0.21230],
    [0.86709, 0.87968, 0.21391],
    [0.87530, 0.87267, 0.21555],
    [0.88331, 0.86553, 0.21719],
    [0.89112, 0.85826, 0.21880],
    [0.89870, 0.85087, 0.22038],
    [0.90605, 0.84337, 0.22188],
    [0.91317, 0.83576, 0.22328],
    [0.92004, 0.82806, 0.22456],
    [0.92666, 0.82025, 0.22570],
    [0.93301, 0.81236, 0.22667],
    [0.93909, 0.80439, 0.22744],
    [0.94489, 0.79634, 0.22800],
    [0.95039, 0.78823, 0.22831],
    [0.95560, 0.78005, 0.22836],
    [0.96049, 0.77181, 0.22811],
    [0.96507, 0.76352, 0.22754],
    [0.96931, 0.75519, 0.22663],
    [0.97323, 0.74682, 0.22536],
    [0.97679, 0.73842, 0.22369],
    [0.98000, 0.73000, 0.22161],
    [0.98289, 0.72140, 0.21918],
    [0.98549, 0.71250, 0.21650],
    [0.98781, 0.70330, 0.21358],
    [0.98986, 0.69382, 0.21043],
    [0.99163, 0.68408, 0.20706],
    [0.99314, 0.67408, 0.20348],
    [0.99438, 0.66386, 0.19971],
    [0.99535, 0.65341, 0.19577],
    [0.99607, 0.64277, 0.19165],
    [0.99654, 0.63193, 0.18738],
    [0.99675, 0.62093, 0.18297],
    [0.99672, 0.60977, 0.17842],
    [0.99644, 0.59846, 0.17376],
    [0.99593, 0.58703, 0.16899],
    [0.99517, 0.57549, 0.16412],
    [0.99419, 0.56386, 0.15918],
    [0.99297, 0.55214, 0.15417],
    [0.99153, 0.54036, 0.14910],
    [0.98987, 0.52854, 0.14398],
    [0.98799, 0.51667, 0.13883],
    [0.98590, 0.50479, 0.13367],
    [0.98360, 0.49291, 0.12849],
    [0.98108, 0.48104, 0.12332],
    [0.97837, 0.46920, 0.11817],
    [0.97545, 0.45740, 0.11305],
    [0.97234, 0.44565, 0.10797],
    [0.96904, 0.43399, 0.10294],
    [0.96555, 0.42241, 0.09798],
    [0.96187, 0.41093, 0.09310],
    [0.95801, 0.39958, 0.08831],
    [0.95398, 0.38836, 0.08362],
    [0.94977, 0.37729, 0.07905],
    [0.94538, 0.36638, 0.07461],
    [0.94084, 0.35566, 0.07031],
    [0.93612, 0.34513, 0.06616],
    [0.93125, 0.33482, 0.06218],
    [0.92623, 0.32473, 0.05837],
    [0.92105, 0.31489, 0.05475],
    [0.91572, 0.30530, 0.05134],
    [0.91024, 0.29599, 0.04814],
    [0.90463, 0.28696, 0.04516],
    [0.89888, 0.27824, 0.04243],
    [0.89298, 0.26981, 0.03993],
    [0.88691, 0.26152, 0.03753],
    [0.88066, 0.25334, 0.03521],
    [0.87422, 0.24526, 0.03297],
    [0.86760, 0.23730, 0.03082],
    [0.86079, 0.22945, 0.02875],
    [0.85380, 0.22170, 0.02677],
    [0.84662, 0.21407, 0.02487],
    [0.83926, 0.20654, 0.02305],
    [0.83172, 0.19912, 0.02131],
    [0.82399, 0.19182, 0.01966],
    [0.81608, 0.18462, 0.01809],
    [0.80799, 0.17753, 0.01660],
    [0.79971, 0.17055, 0.01520],
    [0.79125, 0.16368, 0.01387],
    [0.78260, 0.15693, 0.01264],
    [0.77377, 0.15028, 0.01148],
    [0.76476, 0.14374, 0.01041],
    [0.75556, 0.13731, 0.00942],
    [0.74617, 0.13098, 0.00851],
    [0.73661, 0.12477, 0.00769],
    [0.72686, 0.11867, 0.00695],
    [0.71692, 0.11268, 0.00629],
    [0.70680, 0.10680, 0.00571],
    [0.69650, 0.10102, 0.00522],
    [0.68602, 0.09536, 0.00481],
    [0.67535, 0.08980, 0.00449],
    [0.66449, 0.08436, 0.00424],
    [0.65345, 0.07902, 0.00408],
    [0.64223, 0.07380, 0.00401],
    [0.63082, 0.06868, 0.00401],
    [0.61923, 0.06367, 0.00410],
    [0.60746, 0.05878, 0.00427],
    [0.59550, 0.05399, 0.00453],
    [0.58336, 0.04931, 0.00486],
    [0.57103, 0.04474, 0.00529],
    [0.55852, 0.04028, 0.00579],
    [0.54583, 0.03593, 0.00638],
    [0.53295, 0.03169, 0.00705],
    [0.51989, 0.02756, 0.00780],
    [0.50664, 0.02354, 0.00863],
    [0.49321, 0.01963, 0.00955],
    [0.47960, 0.01583, 0.01055],
]

wright = ["#FFFFFF", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000", "#881111"]


class cmapdict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        self[key] = plt.get_cmap(key)
        return self.get(key)


colormaps = cmapdict()

colormaps["cubehelix"] = plt.get_cmap("cubehelix_r")
colormaps["default"] = cubehelix
colormaps["signed"] = plt.get_cmap("bwr")
colormaps["greenscale"] = mplcolors.LinearSegmentedColormap.from_list(
    "greenscale", greenscale
)
colormaps["greyscale"] = mplcolors.LinearSegmentedColormap.from_list(
    "greyscale", greyscale
)
colormaps["invisible"] = mplcolors.LinearSegmentedColormap.from_list(
    "invisible", invisible
)
colormaps["isoluminant1"] = isoluminant1
colormaps["isoluminant2"] = isoluminant2
colormaps["isoluminant3"] = isoluminant3
colormaps["signed_old"] = mplcolors.LinearSegmentedColormap.from_list(
    "signed", signed_old
)
colormaps["skyebar1"] = mplcolors.LinearSegmentedColormap.from_list("skyebar", skyebar)
colormaps["skyebar2"] = mplcolors.LinearSegmentedColormap.from_list(
    "skyebar dark", skyebar_d
)
colormaps["skyebar3"] = mplcolors.LinearSegmentedColormap.from_list(
    "skyebar inverted", skyebar_i
)
colormaps["turbo"] = mplcolors.ListedColormap(turbo_seq, "turbo")
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
