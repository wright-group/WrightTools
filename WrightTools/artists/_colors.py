"""Colormaps."""


# --- import --------------------------------------------------------------------------------------

import collections

import numpy as np
from numpy import r_

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors

# --- define -------------------------------------------------------------------------------------


__all__ = ['colormaps']


# --- functions ----------------------------------------------------------------------------------


def make_cubehelix(gamma=0.5, s=0.25, r=-1, h=1.3, reverse=False, darkest=0.7):
    """Define cubehelix type colorbars.

    Look `here`__ for more information.

    __ cubehelix_


    Parameters
    ----------
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
    rr = (.213 / .30)
    rg = (.715 / .99)
    rb = (.072 / .11)

    def get_color_function(p0, p1):
        def color(x):
            # Calculate amplitude and angle of deviation from the black to
            # white diagonal in the plane of constant perceived intensity.
            xg = darkest * x**gamma
            lum = 1 - xg  # starts at 1
            if reverse:
                lum = lum[::-1]
            a = lum.copy()
            a[lum < 0.5] = h * lum[lum < 0.5] / 2.
            a[lum >= 0.5] = h * (1 - lum[lum >= 0.5]) / 2.
            phi = 2 * np.pi * (s / 3 + r * x)
            out = lum + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
            return out
        return color
    rgb_dict = {'red': get_color_function(-0.14861 * rr, 1.78277 * rr),
                'green': get_color_function(-0.29227 * rg, -0.90649 * rg),
                'blue': get_color_function(1.97294 * rb, 0.0)}
    cmap = matplotlib.colors.LinearSegmentedColormap('cubehelix', rgb_dict)
    return cmap


def make_colormap(seq, name='CustomMap', plot=False):
    """Generate a LinearSegmentedColormap.

    Parameters
    ----------
    seq : list of tuples
        A sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
    name : string (optional)
        A name for the colormap
    plot : boolean (optional)
        Use to generate a plot of the colormap (Defalut is False).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap


    `Source`__

    __ colormap_
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
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

    __ nmtorgb_
    """
    w = int(nm)
    # color ---------------------------------------------------------------------------------------
    if w >= 380 and w < 440:
        R = -(w - 440.) / (440. - 350.)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.) / (490. - 440.)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.) / (510. - 490.)
    elif w >= 510 and w < 580:
        R = (w - 510.) / (580. - 510.)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.) / (645. - 580.)
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
    return [float(int(SSS * R) / 256.),
            float(int(SSS * G) / 256.),
            float(int(SSS * B) / 256.)]


def plot_colormap_components(cmap):
    """Plot the components of a given colormap."""
    plt.figure(figsize=[8, 4])
    gs = grd.GridSpec(2, 1, height_ratios=[1, 10], hspace=0.05)
    # colorbar
    ax = plt.subplot(gs[0])
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=cmap, vmin=0., vmax=1.)
    ax.set_axis_off()
    # components
    ax = plt.subplot(gs[1])
    x = gradient[0]
    r = cmap._segmentdata['red'](x)
    g = cmap._segmentdata['green'](x)
    b = cmap._segmentdata['blue'](x)
    k = .3 * r + .59 * g + .11 * b
    # truncate
    r.clip(0, 1, out=r)
    g.clip(0, 1, out=g)
    b.clip(0, 1, out=b)
    # plot
    plt.plot(x, r, 'r', linewidth=5, alpha=0.6)
    plt.plot(x, g, 'g', linewidth=5, alpha=0.6)
    plt.plot(x, b, 'b', linewidth=5, alpha=0.6)
    plt.plot(x, k, 'k:', linewidth=5, alpha=0.6)
    ax.set_ylim(-.1, 1.1)
    # finish
    plt.grid()
    plt.xlabel('value', fontsize=17)
    plt.ylabel('intensity', fontsize=17)


# --- color maps ----------------------------------------------------------------------------------


cubehelix = make_cubehelix()

experimental = ['#FFFFFF',
                '#0000FF',
                '#0080FF',
                '#00FFFF',
                '#00FF00',
                '#FFFF00',
                '#FF8000',
                '#FF0000',
                '#881111']

greenscale = ['#000000',  # black
              '#00FF00']  # green

greyscale = ['#FFFFFF',  # white
             '#000000']  # black

invisible = ['#FFFFFF',  # white
             '#FFFFFF']  # white

# isoluminant colorbar based on the research of Kindlmann et al.
# http://dx.doi.org/10.1109/VISUAL.2002.1183788
c = mplcolors.ColorConverter().to_rgb
isoluminant1 = make_colormap([
    c(r_[1.000, 1.000, 1.000]), c(r_[0.847, 0.057, 0.057]), 1 / 6.,
    c(r_[0.847, 0.057, 0.057]), c(r_[0.527, 0.527, 0.000]), 2 / 6.,
    c(r_[0.527, 0.527, 0.000]), c(r_[0.000, 0.592, 0.000]), 3 / 6.,
    c(r_[0.000, 0.592, 0.000]), c(r_[0.000, 0.559, 0.559]), 4 / 6.,
    c(r_[0.000, 0.559, 0.559]), c(r_[0.316, 0.316, 0.991]), 5 / 6.,
    c(r_[0.316, 0.316, 0.991]), c(r_[0.718, 0.000, 0.718])],
    name='isoluminant`')

isoluminant2 = make_colormap([
    c(r_[1.000, 1.000, 1.000]), c(r_[0.718, 0.000, 0.718]), 1 / 6.,
    c(r_[0.718, 0.000, 0.718]), c(r_[0.316, 0.316, 0.991]), 2 / 6.,
    c(r_[0.316, 0.316, 0.991]), c(r_[0.000, 0.559, 0.559]), 3 / 6.,
    c(r_[0.000, 0.559, 0.559]), c(r_[0.000, 0.592, 0.000]), 4 / 6.,
    c(r_[0.000, 0.592, 0.000]), c(r_[0.527, 0.527, 0.000]), 5 / 6.,
    c(r_[0.527, 0.527, 0.000]), c(r_[0.847, 0.057, 0.057])],
    name='isoluminant2')

isoluminant3 = make_colormap([
    c(r_[1.000, 1.000, 1.000]), c(r_[0.316, 0.316, 0.991]), 1 / 5.,
    c(r_[0.316, 0.316, 0.991]), c(r_[0.000, 0.559, 0.559]), 2 / 5.,
    c(r_[0.000, 0.559, 0.559]), c(r_[0.000, 0.592, 0.000]), 3 / 5.,
    c(r_[0.000, 0.592, 0.000]), c(r_[0.527, 0.527, 0.000]), 4 / 5.,
    c(r_[0.527, 0.527, 0.000]), c(r_[0.847, 0.057, 0.057])],
    name='isoluminant3')

signed = ['#0000FF',  # blue
          '#002AFF',
          '#0055FF',
          '#007FFF',
          '#00AAFF',
          '#00D4FF',
          '#00FFFF',
          '#FFFFFF',  # white
          '#FFFF00',
          '#FFD400',
          '#FFAA00',
          '#FF7F00',
          '#FF5500',
          '#FF2A00',
          '#FF0000']  # red

signed_old = ['#0000FF',  # blue
              '#00BBFF',  # blue-aqua
              '#00FFFF',  # aqua
              '#FFFFFF',  # white
              '#FFFF00',  # yellow
              '#FFBB00',  # orange
              '#FF0000']  # red

skyebar = ['#FFFFFF',  # white
           '#000000',  # black
           '#0000FF',  # blue
           '#00FFFF',  # cyan
           '#64FF00',  # light green
           '#FFFF00',  # yellow
           '#FF8000',  # orange
           '#FF0000',  # red
           '#800000']  # dark red

skyebar_d = ['#000000',  # black
             '#0000FF',  # blue
             '#00FFFF',  # cyan
             '#64FF00',  # light green
             '#FFFF00',  # yellow
             '#FF8000',  # orange
             '#FF0000',  # red
             '#800000']  # dark red

skyebar_i = ['#000000',  # black
             '#FFFFFF',  # white
             '#0000FF',  # blue
             '#00FFFF',  # cyan
             '#64FF00',  # light green
             '#FFFF00',  # yellow
             '#FF8000',  # orange
             '#FF0000',  # red
             '#800000']  # dark red

wright = ['#FFFFFF',
          '#0000FF',
          '#00FFFF',
          '#00FF00',
          '#FFFF00',
          '#FF0000',
          '#881111']

colormaps = collections.OrderedDict()
colormaps['coolwarm'] = plt.get_cmap('coolwarm')
colormaps['cubehelix'] = plt.get_cmap('cubehelix_r')
colormaps['default'] = cubehelix
colormaps['flag'] = plt.get_cmap('flag')
colormaps['greenscale'] = mplcolors.LinearSegmentedColormap.from_list('greenscale', greenscale)
colormaps['greyscale'] = mplcolors.LinearSegmentedColormap.from_list('greyscale', greyscale)
colormaps['invisible'] = mplcolors.LinearSegmentedColormap.from_list('invisible', invisible)
colormaps['isoluminant1'] = isoluminant1
colormaps['isoluminant2'] = isoluminant2
colormaps['isoluminant3'] = isoluminant3
colormaps['prism'] = plt.get_cmap('prism')
colormaps['rainbow'] = plt.get_cmap('rainbow')
colormaps['seismic'] = plt.get_cmap('seismic')
colormaps['signed'] = plt.get_cmap('bwr')
colormaps['signed_old'] = mplcolors.LinearSegmentedColormap.from_list('signed', signed_old)
colormaps['skyebar1'] = mplcolors.LinearSegmentedColormap.from_list('skyebar', skyebar)
colormaps['skyebar2'] = mplcolors.LinearSegmentedColormap.from_list('skyebar dark', skyebar_d)
colormaps['skyebar3'] = mplcolors.LinearSegmentedColormap.from_list('skyebar inverted', skyebar_i)
colormaps['wright'] = mplcolors.LinearSegmentedColormap.from_list('wright', wright)


# enforce grey as 'bad' value for colormaps
for cmap in colormaps.values():
    cmap.set_bad([0.75] * 3, 1)


# a nice set of line colors
overline_colors = ['#CCFF00', '#FE4EDA', '#FF6600', '#00FFBF', '#00B7EB']
