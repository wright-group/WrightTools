'''
tools used in plotting data
'''


### import ####################################################################


import os
import collections

import numpy as np
from numpy import r_

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as grd
import matplotlib.colors as mplcolors
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

import kit


### artist helpers ############################################################


def make_cubehelix(gamma=1.0, s=0.5, r=-1.5, h=0.5,
                   lum_rev=False, darkest=0.8, plot=False):
    '''
    Define cubehelix type colorbars. \n
    gamma intensity factor, s start color, 
    r rotations, h 'hue' saturation factor \n
    Returns white to black LinearSegmentedColormap. \n
    Written by Dan \n
    For more information see http://arxiv.org/abs/1108.5083 .
    '''
    # isoluminescent curve--helical color cycle
    def get_color_function(p0, p1):
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            #xg = x ** gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            xg = darkest * x**gamma
            lum = 1-xg # starts at 1
            if lum_rev:
                lum = lum[::-1]
            a = lum.copy()#h * lum*(1-lum)/2.
            a[lum<0.5] = h * lum[lum<0.5]/2.
            a[lum>=0.5] = h * (1-lum[lum>=0.5])/2.
            phi = 2 * np.pi * (s / 3 + r * x)
            out = lum + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
            return out
        return color
    rgb_dict = {'red':   get_color_function(-0.14861, 1.78277),
                'green': get_color_function(-0.29227, -0.90649),
                'blue':  get_color_function(1.97294, 0.0)}
    cbar = matplotlib.colors.LinearSegmentedColormap('cubehelix', rgb_dict)
    if plot:
        # maybe broken? - BJT 2015.08.11
        plt.figure()
        x = np.linspace(0, 1.)
        r = cbar._segmentdata['red'](x)
        g = cbar._segmentdata['green'](x)
        b = cbar._segmentdata['blue'](x)
        k = .3*r + .59*g + .11*b
        plt.plot(x, r, 'r', linewidth=5, alpha=0.6)
        plt.plot(x, g, 'g', linewidth=5, alpha=0.6)
        plt.plot(x, b, 'b', linewidth=5, alpha=0.6)
        plt.plot(x, k, 'k:', linewidth=5, alpha=0.6)
        plt.grid()
    return cbar
    

def make_colormap(seq, name='CustomMap'):
    '''
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1). \n
    from http://nbviewer.ipython.org/gist/anonymous/a4fa0adb08f9e9ea4f94#
    '''
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mplcolors.LinearSegmentedColormap(name, cdict)
    

def nm_to_rgb(nm):
    '''
    returns list [r, g, b] (zero to one scale) for given input in nm \n
    original code - http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    w = int(nm)

    # color -------------------------------------------------------------------

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

    # intensity correction ----------------------------------------------------

    if w >= 380 and w < 420:
        SSS = 0.3 + 0.7*(w - 350) / (420 - 350)
    elif w >= 420 and w <= 700:
        SSS = 1.0
    elif w > 700 and w <= 780:
        SSS = 0.3 + 0.7*(780 - w) / (780 - 700)
    else:
        SSS = 0.0
    SSS *= 255

    return [float(int(SSS*R)/256.),
            float(int(SSS*G)/256.),
            float(int(SSS*B)/256.)]


def pcolor_helper(xi, yi, zi):
    '''
    accepts xi, yi, zi as the normal rectangular arrays
    that would be given to contorf etc \n
    returns list [X, Y, Z] appropriate for feeding directly
    into matplotlib.pyplot.pcolor so that the pixels are centered correctly. \n
    '''

    x_points = np.zeros(len(xi)+1)
    y_points = np.zeros(len(yi)+1)

    for points, axis in [[x_points, xi], [y_points, yi]]:
        for j in range(len(points)):
            if j == 0:  # first point
                points[j] = axis[0] - (axis[1] - axis[0])
            elif j == len(points)-1:  # last point
                points[j] = axis[-1] + (axis[-1] - axis[-2])
            else:
                points[j] = np.average([axis[j], axis[j-1]])

    X, Y = np.meshgrid(x_points, y_points)

    return X, Y, zi


### color maps ################################################################


cubehelix = make_cubehelix(gamma=0.5, s=0.25, r=-6/6., h=1.25, 
                           lum_rev=False, darkest=0.7)

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
isoluminant = make_colormap([
    c(r_[1.000,1.000,1.000]), c(r_[0.847,0.057,0.057]), 1/6.,
    c(r_[0.847,0.057,0.057]), c(r_[0.527,0.527,0.000]), 2/6.,
    c(r_[0.527,0.527,0.000]), c(r_[0.000,0.592,0.000]), 3/6.,
    c(r_[0.000,0.592,0.000]), c(r_[0.000,0.559,0.559]), 4/6.,
    c(r_[0.000,0.559,0.559]), c(r_[0.316,0.316,0.991]), 5/6.,
    c(r_[0.316,0.316,0.991]), c(r_[0.718,0.000,0.718])],
    name='isoluminant')

isoluminant2 = make_colormap([
    c(r_[1.000,1.000,1.000]), c(r_[0.718,0.000,0.718]), 1/6.,
    c(r_[0.718,0.000,0.718]), c(r_[0.316,0.316,0.991]), 2/6.,
    c(r_[0.316,0.316,0.991]), c(r_[0.000,0.559,0.559]), 3/6.,
    c(r_[0.000,0.559,0.559]), c(r_[0.000,0.592,0.000]), 4/6.,
    c(r_[0.000,0.592,0.000]), c(r_[0.527,0.527,0.000]), 5/6.,
    c(r_[0.527,0.527,0.000]), c(r_[0.847,0.057,0.057])],
    name='isoluminant2')
    
isoluminant3 = make_colormap([
    c(r_[1.000,1.000,1.000]), c(r_[0.316,0.316,0.991]), 1/5.,
    c(r_[0.316,0.316,0.991]), c(r_[0.000,0.559,0.559]), 2/5.,
    c(r_[0.000,0.559,0.559]), c(r_[0.000,0.592,0.000]), 3/5.,
    c(r_[0.000,0.592,0.000]), c(r_[0.527,0.527,0.000]), 4/5.,
    c(r_[0.527,0.527,0.000]), c(r_[0.847,0.057,0.057])],
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

colormaps = {'CMRmap': plt.get_cmap('CMRmap_r'),
             'cubehelix': plt.get_cmap('cubehelix_r'),
             'default': cubehelix,
             'experimental': mplcolors.LinearSegmentedColormap.from_list('experimental', experimental),
             'flag': plt.get_cmap('flag'),
             'earth': plt.get_cmap('gist_earth'),
             'gnuplot2': plt.get_cmap('gnuplot2_r'),
             'greenscale': mplcolors.LinearSegmentedColormap.from_list('greenscale', greenscale),
             'greyscale': mplcolors.LinearSegmentedColormap.from_list('greyscale', greyscale),
             'invisible': mplcolors.LinearSegmentedColormap.from_list('invisible', invisible),
             'isoluminant': isoluminant,
             'isoluminant2': isoluminant2,
             'isoluminant3': isoluminant3,
             'ncar': plt.get_cmap('gist_ncar'),
             'paired': plt.get_cmap('Paired'),
             'prism': plt.get_cmap('prism'),
             'rainbow': plt.get_cmap('rainbow'),
             'seismic': plt.get_cmap('seismic'),
             'signed':  mplcolors.LinearSegmentedColormap.from_list('signed', signed),
             'signed_old':  mplcolors.LinearSegmentedColormap.from_list('signed', signed_old),
             'skyebar':  mplcolors.LinearSegmentedColormap.from_list('skyebar', skyebar),
             'skyebar_d': mplcolors.LinearSegmentedColormap.from_list('skyebar dark', skyebar_d), 
             'skyebar_i': mplcolors.LinearSegmentedColormap.from_list('skyebar inverted', skyebar_i),                          
             'spectral': plt.get_cmap('nipy_spectral'),
             'wright': mplcolors.LinearSegmentedColormap.from_list('wright', wright)}


### general purpose artists ###################################################


class mpl_1D:

    def __init__(self, data, xaxis = 0, at = {}, verbose = True):
        # import data
        self.data = data
        self.chopped = self.data.chop(xaxis, at, verbose = False)
        if verbose:
            print 'mpl_1D recieved data to make %d plots'%len(self.chopped)
        # defaults
        self.font_size = 15

    def plot(self, channel = 0, local = False,
             autosave = False, output_folder = None, fname = None,
             verbose = True):
        fig = None
        if len(self.chopped) > 10:
            if not autosave:
                print 'too many images will be generated ({}): forcing autosave'.format(len(self.chopped))
                autosave = True
                
        # prepare output folder
        if autosave:
            if output_folder:
                pass
            else:
                if len(self.chopped) == 1:
                    output_folder = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.data.name
                else:
                    folder_name = 'mpl_1D ' + kit.get_timestamp()
                    os.mkdir(folder_name)
                    output_folder = folder_name
        
        # chew through image generation
        for i in range(len(self.chopped)):

            if fig: plt.close(fig)

            fig = plt.figure(figsize=(8, 6))

            current_chop = self.chopped[i]
            axes = current_chop.axes
            channels = current_chop.channels
            constants = current_chop.constants

            xi = axes[0].points
            zi = channels[channel].values

            plt.plot(xi, zi)
            plt.grid()

            # limits ----------------------------------------------------------

            if local:
                pass
            else:
                plt.ylim(channels[channel].zmin, channels[channel].zmax)

            # label axes ------------------------------------------------------

            plt.xlabel(axes[0].get_label())
            plt.ylabel(channels[channel].name)

            # title -----------------------------------------------------------

            title_text = self.data.name

            constants_text = '\n'
            for constant in constants:
                constants_text += constant.get_label(show_units=True, points=True) + '    '

            plt.suptitle(title_text + constants_text, fontsize=self.font_size)
            
            # cleanup ---------------------------------------------------------

            # plt.tight_layout()
            factor = 0.12
            fig.subplots_adjust(left = factor, right = 1-factor, top = 1-factor, bottom = 0.15)

            # save figure -----------------------------------------------------

            if autosave:
                if fname:
                    file_name = fname + ' ' + str(i).zfill(3)
                else:
                    file_name = str(i).zfill(3)
                fpath = os.path.join(output_folder, file_name + '.png')
                plt.savefig(fpath, transparent = True)
                plt.close()

                if verbose:
                    print 'image saved at', fpath


class mpl_2D:

    def __init__(self, data, xaxis = 1, yaxis = 0, at = {}, verbose = True):
        # import data
        self.data = data
        self.chopped = self.data.chop(yaxis, xaxis, at, verbose = False)
        if verbose:
            print 'mpl_2D recieved data to make %d plots'%len(self.chopped)
        # defaults
        self.font_size = 15
        self._xsideplot = False
        self._ysideplot = False
        self._xsideplotdata = []
        self._ysideplotdata = []

    def sideplot(self, data, x = True, y = True):
        data = data.copy()
        if x:
            if self.chopped[0].axes[1].units_kind == data.axes[0].units_kind:
                data.convert(self.chopped[0].axes[1].units)
                self._xsideplot = True
                self._xsideplotdata.append([data.axes[0].points, data.channels[0].values])
            else:
                print 'given data ({0}), does not aggree with x ({1})'.format(data.axes[0].units_kind, self.chopped[0].axes[1].units_kind)
        if y: 
            if self.chopped[0].axes[0].units_kind == data.axes[0].units_kind:
                data.convert(self.chopped[0].axes[0].units)
                self._ysideplot = True
                self._ysideplotdata.append([data.axes[0].points, data.channels[0].values])
            else:
                print 'given data ({0}), does not aggree with y ({1})'.format(data.axes[0].units_kind, self.chopped[0].axes[0].units_kind)

    def plot(self, channel_index=0,
             contours=9, pixelated=True, lines=True, cmap='default', facecolor='w',
             dynamic_range = False, local = False, contours_local = True, normalize_slices = 'both',
             xbin= False, ybin=False, xlim=None, ylim=None,
             autosave=False, output_folder=None, fname=None,
             verbose=True):
        '''
        set contours to zero to turn off

        dynamic_range forces the colorbar to use all of its colors (only matters
        for signed data)
        '''
        fig = None
        if len(self.chopped) > 10:
            if not autosave:
                print 'too many images will be generated: forcing autosave'
                autosave = True
        
        # prepare output folder
        if autosave:
            if output_folder:
                pass
            else:
                if len(self.chopped) == 1:
                    output_folder = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.data.name
                else:
                    folder_name = 'mpl_2D ' + kit.get_timestamp()
                    os.mkdir(folder_name)
                    output_folder = folder_name

        # chew through image generation
        for i in range(len(self.chopped)):

            # get data to plot ------------------------------------------------

            current_chop = self.chopped[i]
            axes = current_chop.axes
            channels = current_chop.channels
            constants = current_chop.constants

            xaxis = axes[1]
            yaxis = axes[0]
            channel = channels[channel_index]
            zi = channel.values

            # normalize slices -------------------------------------------------

            if normalize_slices == 'both':
                pass
            elif normalize_slices == 'horizontal':
                nmin = channel.znull
                #normalize all x traces to a common value
                maxes = zi.max(axis=1)
                numerator = (zi - nmin)
                denominator = (maxes - nmin)
                for j in range(zi.shape[0]):
                    zi[j] = numerator[j]/denominator[j]
                channel.zmax = zi.max()
                channel.zmin = zi.min()
                channel.znull = 0
            elif normalize_slices == 'vertical':
                nmin = channel.znull
                maxes = zi.max(axis=0)
                numerator = (zi - nmin)
                denominator = (maxes - nmin)
                for j in range(zi.shape[1]):
                    zi[:,j] = numerator[:,j] / denominator[j]
                channel.zmax = zi.max()
                channel.zmin = zi.min()
                channel.znull = 0

            # create figure ---------------------------------------------------

            if fig:
                plt.close(fig)

            fig = plt.figure(figsize=(8, 7))

            gs = grd.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.1)

            subplot_main = plt.subplot(gs[0])
            subplot_main.patch.set_facecolor(facecolor)

            # levels ----------------------------------------------------------

            if channel.signed:

                if dynamic_range:
                    limit = min(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                else:
                    limit = max(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                levels = np.linspace(-limit + channel.znull, limit + channel.znull, 200)

            else:

                if local:
                    levels = np.linspace(channel.znull, zi.max(), 200)
                else:
                    levels = np.linspace(channel.znull, channel.zmax, 200)

            # main plot -------------------------------------------------------

            #get colormap
            mycm = colormaps[cmap]
            mycm.set_bad(facecolor)
            mycm.set_under(facecolor)

            #fill in main data environment
            if pixelated:
                xi, yi, zi = pcolor_helper(xaxis.points, yaxis.points, zi)
                cax = plt.pcolor(xi, yi, zi, cmap=mycm,
                                 vmin=levels.min(), vmax=levels.max())
                plt.xlim(xaxis.points.min(), xaxis.points.max())
                plt.ylim(yaxis.points.min(), yaxis.points.max())
            else:
                cax = subplot_main.contourf(xaxis.points, yaxis.points, zi,
                                            levels, cmap=mycm)

            plt.xticks(rotation = 45)
            plt.xlabel(xaxis.get_label(), fontsize = self.font_size)
            plt.ylabel(yaxis.get_label(), fontsize = self.font_size)

            # variable marker lines -------------------------------------------

            if lines:
                for constant in constants:
                        if constant.units_kind == 'energy':
                            #x axis
                            if xaxis.units == constant.units:
                                plt.axvline(constant.points, color = 'k', linewidth = 4, alpha = 0.25)
                            #y axis
                            if yaxis.units == constant.units:
                                plt.axhline(constant.points, color = 'k', linewidth = 4, alpha = 0.25)

            # grid ------------------------------------------------------------

            plt.grid(b = True)

            if xaxis.units == yaxis.units:
                # add diagonal line
                if xlim:
                    x = xlim
                else:
                    x = xaxis.points
                if ylim:
                    y = ylim
                else:
                    y = yaxis.points

                diag_min = max(min(x), min(y))
                diag_max = min(max(x), max(y))
                plt.plot([diag_min, diag_max],[diag_min, diag_max],'k:')

            # contour lines ---------------------------------------------------

            if contours:
                if contours_local:
                    # force top and bottom contour to be just outside of data range
                    # add two contours
                    contours_levels = np.linspace(channel.znull-1e-10, zi.max()+1e-10, contours+2)
                else:
                    contours_levels = contours
                subplot_main.contour(xaxis.points, yaxis.points, zi,
                                     contours_levels, colors = 'k')

            # finish main subplot ---------------------------------------------

            if xlim:
                subplot_main.set_xlim(xlim[0], xlim[1])
            else:
                subplot_main.set_xlim(xaxis.points[0], xaxis.points[-1])
            if ylim:
                subplot_main.set_ylim(ylim[0], ylim[1])
            else:
                subplot_main.set_ylim(yaxis.points[0], yaxis.points[-1])

            # sideplots -------------------------------------------------------

            divider = make_axes_locatable(subplot_main)

            if xbin or self._xsideplot:

                axCorrx = divider.append_axes('top', 0.75, pad=0.0, sharex=subplot_main)
                axCorrx.autoscale(False)
                axCorrx.set_adjustable('box-forced')
                plt.setp(axCorrx.get_xticklabels(), visible=False)
                plt.setp(axCorrx.get_yticklabels(), visible=False)
                plt.grid(b = True)
                if channel.signed:
                    axCorrx.set_ylim([-1.1,1.1])
                else:
                    axCorrx.set_ylim([0,1.1])

                # bin
                if xbin:
                    x_ax_int = zi.sum(axis=0) - channel.znull * len(yaxis.points)
                    # normalize (min is a pixel)
                    xmax = max(np.abs(x_ax_int))
                    x_ax_int = x_ax_int / xmax
                    axCorrx.plot(xaxis.points,x_ax_int, lw = 2)
                    axCorrx.set_xlim([xaxis.points.min(), xaxis.points.max()])

                # data
                if self._xsideplot:
                    for s_xi, s_zi in self._xsideplotdata:
                        xlim =  axCorrx.get_xlim()
                        min_index = np.argmin(abs(s_xi - min(xlim)))
                        max_index = np.argmin(abs(s_xi - max(xlim)))
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        s_zi = s_zi - min(s_zi_in_range)
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        s_zi = s_zi / max(s_zi_in_range)
                        axCorrx.plot(s_xi, s_zi, lw = 2)
                        
                # line
                if lines:
                    for constant in constants:
                        if constant.units_kind == 'energy':
                            if xaxis.units == constant.units:
                                axCorrx.axvline(constant.points, color = 'k', linewidth = 4, alpha = 0.25)

            if ybin or self._ysideplot:

                axCorry = divider.append_axes('right', 0.75, pad=0.0, sharey=subplot_main)
                axCorry.autoscale(False)
                axCorry.set_adjustable('box-forced')
                plt.setp(axCorry.get_xticklabels(), visible=False)
                plt.setp(axCorry.get_yticklabels(), visible=False)
                plt.grid(b = True)
                if channel.signed:
                    axCorry.set_xlim([-1.1,1.1])
                else:
                    axCorry.set_xlim([0,1.1])

                # bin
                if ybin:
                    y_ax_int = zi.sum(axis=1) - channel.znull * len(xaxis.points)
                    # normalize (min is a pixel)
                    ymax = max(np.abs(y_ax_int))
                    y_ax_int = y_ax_int / ymax
                    axCorry.plot(y_ax_int, yaxis.points, lw = 2)
                    axCorry.set_ylim([yaxis.points.min(), yaxis.points.max()])

                # data
                if self._ysideplot:
                    for s_xi, s_zi in self._ysideplotdata:
                        xlim =  axCorry.get_ylim()
                        min_index = np.argmin(abs(s_xi - min(xlim)))
                        max_index = np.argmin(abs(s_xi - max(xlim)))
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        s_zi = s_zi - min(s_zi_in_range)
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        s_zi = s_zi / max(s_zi_in_range)
                        axCorry.plot(s_zi, s_xi, lw = 2)

                # line
                if lines:
                    for constant in constants:
                        if constant.units_kind == 'energy':
                            if yaxis.units == constant.units:
                                axCorry.axvline(constant.points, color = 'k', linewidth = 4, alpha = 0.25)

            # colorbar --------------------------------------------------------

            if True:
                subplot_cb = plt.subplot(gs[1])
                cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
                cbar = plt.colorbar(cax, cax=subplot_cb, ticks=cbar_ticks)
                cbar.set_label(channel.name)
                

            # title -----------------------------------------------------------

            title_text = self.data.name

            constants_text = '\n'
            for constant in constants:
                constants_text += constant.get_label(show_units = True, points = True) + '    '

            plt.suptitle(title_text + constants_text, fontsize = self.font_size)

            # cleanup ---------------------------------------------------------

            # plt.tight_layout()
            factor = 0.12
            fig.subplots_adjust(left = factor, right = 1-factor, top = 1-factor, bottom = 0.15)

            # save figure -----------------------------------------------------

            if autosave:
                if fname:
                    file_name = fname + ' ' + str(i).zfill(3)
                else:
                    file_name = str(i).zfill(3)
                fpath = os.path.join(output_folder, file_name + '.png')
                plt.savefig(fpath, facecolor = 'none')
                plt.close()

                if verbose:
                    print 'image saved at', fpath


### specific artists ###########################################################


class absorbance:

    def __init__(self, data):

        if not type(data) == list:
            data = [data]

        self.data = data

    def plot(self, channel_index = 0, font_size = 12, xlim = None, ylim = None,
             yticks = True, derivative = True, n_smooth = 10,):

        # prepare plot environment --------------------------------------------

        matplotlib.rcParams.update({'font.size': font_size})
        self.font_size = font_size

        if derivative:
            gs = grd.GridSpec(2, 1, hspace = 0.05)
            self.ax1 = plt.subplot(gs[0])
            plt.ylabel('OD')
            plt.grid()
            plt.setp(self.ax1.get_xticklabels(), visible=False)
            self.ax2 = plt.subplot(gs[1], sharex = self.ax1)
            plt.grid()
            plt.ylabel('2nd derivative')
        else:
            self.ax1 = plt.subplot(111)
            plt.ylabel('OD')
            plt.grid()

        for data in self.data:

            # import data -----------------------------------------------------

            xi = data.axes[0].points
            zi = data.channels[channel_index].values

            # scale -----------------------------------------------------------

            if xlim:
                plt.xlim(xlim[0], xlim[1])

                min_index = np.argmin(abs(xi - min(xlim)))
                max_index = np.argmin(abs(xi - max(xlim)))

                zi_truncated = zi[min(min_index,max_index):max(min_index, max_index)]
                zi -= zi_truncated.min()

                zi_truncated = zi[min(min_index,max_index):max(min_index, max_index)]
                zi /= zi_truncated.max()

            # plot absorbance -------------------------------------------------

            self.ax1.plot(xi, zi, lw = 2)

            # now plot 2nd derivative -----------------------------------------

            if derivative:
                # compute second derivative
                xi2, zi2= self._smooth(np.array([xi,zi]), n_smooth)
                plotData = kit.diff(xi2, zi2, order = 2)
                # plot the data!
                self.ax2.plot(plotData[0], plotData[1], lw = 2)
                self.ax2.grid(b=True)
                plt.xlabel(data.axes[0].get_label())

        # legend --------------------------------------------------------------

        #self.ax1.legend([data.name for data in self.data])

        # ticks ---------------------------------------------------------------

        if not yticks: self.ax1.get_yaxis().set_ticks([])
        if derivative:
            self.ax2.get_yaxis().set_ticks([])
            self.ax2.axhline(0, color = 'k', ls = ':')

        # title ---------------------------------------------------------------

        if len(self.data) == 1: #only attempt this if we are plotting one data object
            title_text = self.data[0].name
            print title_text
            plt.suptitle(title_text, fontsize = self.font_size)

        # finish --------------------------------------------------------------

        if xlim:
            plt.xlim(xlim[0], xlim[1])
            for axis, xi, zi in [[self.ax1, xi, zi], [self.ax2, plotData[0], plotData[1]]]:
                min_index = np.argmin(abs(xi - min(xlim)))
                max_index = np.argmin(abs(xi - max(xlim)))
                zi_truncated = zi[min_index:max_index]
                extra = (zi_truncated.max() - zi_truncated.min())*0.1
                axis.set_ylim(zi_truncated.min() - extra, zi_truncated.max() + extra)

        if ylim:
            self.ax1.set_ylim(ylim)

    def _smooth(self, dat1, n=20, window_type='default'):
        '''
        data is an array of type [xlis,ylis] \n
        smooth to prevent 2nd derivative from being noisy
        '''
        for i in range(n, len(dat1[1])-n):
            # change the x value to the average
            window = dat1[1][i-n:i+n].copy()
            dat1[1][i] = window.mean()
        return dat1[:][:,n:-n]


class difference_2D():

    def __init__(self, minuend, subtrahend, xaxis=1, yaxis=0, at={}, 
                 verbose=True):
        '''
        plot the difference between exactly two datasets in 2D \n
        both data objects must have the same axes with the same name \n
        axes do not need to be in the same order or have the same points \n
        '''
        self.minuend = minuend.copy()
        self.subtrahend = subtrahend.copy()
        # check if axes are valid - same axis names in both data objects
        minuend_counter = collections.Counter(self.minuend.axis_names)
        subrahend_counter = collections.Counter(self.subtrahend.axis_names)
        if minuend_counter == subrahend_counter:
            pass
        else:
            print 'axes are not equivalent - difference_2D cannot initialize'
            print '  minuhend axes -', self.minuend.axis_names
            print '  subtrahend axes -', self.subtrahend.axis_names
            raise RuntimeError('axes incompataible')
        # transpose subrahend to agree with minuend
        transpose_order = [self.minuend.axis_names.index(name) for name in self.subtrahend.axis_names]
        self.subtrahend.transpose(transpose_order, verbose=False)
        # map subtrahend axes onto minuend axes
        for i in range(len(self.minuend.axes)):            
            self.subtrahend.axes[i].convert(self.minuend.axes[i].units)
            self.subtrahend.map_axis(i, self.minuend.axes[i].points)
        # chop
        self.minuend_chopped = self.minuend.chop(yaxis, xaxis, at, verbose = False)
        self.subtrahend_chopped = self.subtrahend.chop(yaxis, xaxis, at, verbose = False)
        if verbose:
            print 'difference_2D recieved data to make %d plots'%len(self.minuend_chopped)
        # defaults
        self.font_size = 18

    def plot(self, channel_index=0,
         contours=9, pixelated=True, cmap='default', facecolor='grey',
         dynamic_range = False, local = False, contours_local = True,
         xlim=None, ylim=None,
         autosave=False, output_folder=None, fname=None,
         verbose=True):
        '''
        set contours to zero to turn off

        dynamic_range forces the colorbar to use all of its colors (only matters
        for signed data)
        '''
        fig = None
        if len(self.minuend_chopped) > 10:
            if not autosave:
                print 'too many images will be generated: forcing autosave'
                autosave = True
        
        # prepare output folder
        if autosave:
            plt.ioff()
            if output_folder:
                pass
            else:
                if len(self.minuend_chopped) == 1:
                    output_folder = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.minuend.name
                else:
                    folder_name = 'difference_2D ' + kit.get_timestamp()
                    os.mkdir(folder_name)
                    output_folder = folder_name

        # chew through image generation
        for i in range(len(self.minuend_chopped)):
            
            # create figure ---------------------------------------------------

            if fig:
                plt.close(fig)

            fig = plt.figure(figsize=(22, 7))

            gs = grd.GridSpec(1, 6, width_ratios=[20, 20, 1, 1, 20, 1], wspace=0.1)

            subplot_main = plt.subplot(gs[0])
            subplot_main.patch.set_facecolor(facecolor)

            # levels ----------------------------------------------------------

            '''
            if channel.signed:

                if dynamic_range:
                    limit = min(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                else:
                    limit = max(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                levels = np.linspace(-limit + channel.znull, limit + channel.znull, 200)

            else:

                if local:
                    levels = np.linspace(channel.znull, zi.max(), 200)
                else:
                    levels = np.linspace(channel.znull, channel.zmax, 200)
            '''
            levels = np.linspace(0, 1, 200)

            # main plot -------------------------------------------------------

            #get colormap
            mycm = colormaps[cmap]
            mycm.set_bad(facecolor)
            mycm.set_under(facecolor)

            for j in range(2):
                
                if j == 0:
                    current_chop = chopped = self.minuend_chopped[i]
                elif j == 1:
                    current_chop = self.subtrahend_chopped[i]
                
                axes = current_chop.axes
                channels = current_chop.channels
                constants = current_chop.constants
    
                xaxis = axes[1]
                yaxis = axes[0]
                channel = channels[channel_index]
                zi = channel.values

                plt.subplot(gs[j])

                #fill in main data environment
                if pixelated:
                    xi, yi, zi = pcolor_helper(xaxis.points, yaxis.points, zi)
                    cax = plt.pcolor(xi, yi, zi, cmap=mycm,
                                     vmin=levels.min(), vmax=levels.max())
                    plt.xlim(xaxis.points.min(), xaxis.points.max())
                    plt.ylim(yaxis.points.min(), yaxis.points.max())
                else:
                    cax = subplot_main.contourf(xaxis.points, yaxis.points, zi,
                                                levels, cmap=mycm)
    
                plt.xticks(rotation = 45)
                #plt.xlabel(xaxis.get_label(), fontsize = self.font_size)
                #plt.ylabel(yaxis.get_label(), fontsize = self.font_size)

                # grid --------------------------------------------------------
    
                plt.grid(b = True)
    
                if xaxis.units == yaxis.units:
                    # add diagonal line
                    if xlim:
                        x = xlim
                    else:
                        x = xaxis.points
                    if ylim:
                        y = ylim
                    else:
                        y = yaxis.points
    
                    diag_min = max(min(x), min(y))
                    diag_max = min(max(x), max(y))
                    plt.plot([diag_min, diag_max],[diag_min, diag_max],'k:')
    
                # contour lines -----------------------------------------------
    
                if contours:
                    if contours_local:
                        # force top and bottom contour to be just outside of data range
                        # add two contours
                        contours_levels = np.linspace(channel.znull-1e-10, np.nanmax(zi)+1e-10, contours+2)
                    else:
                        contours_levels = contours
                    plt.contour(xaxis.points, yaxis.points, zi,
                                contours_levels, colors = 'k')
    
                # finish main subplot -----------------------------------------
    
                if xlim:
                    subplot_main.set_xlim(xlim[0], xlim[1])
                else:
                    subplot_main.set_xlim(xaxis.points[0], xaxis.points[-1])
                if ylim:
                    subplot_main.set_ylim(ylim[0], ylim[1])
                else:
                    subplot_main.set_ylim(yaxis.points[0], yaxis.points[-1])

            
            # colorbar --------------------------------------------------------

            subplot_cb = plt.subplot(gs[2])
            cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
            cbar = plt.colorbar(cax, cax=subplot_cb, ticks=cbar_ticks)

            # difference ------------------------------------------------------

            #get colormap
            mycm = colormaps['seismic']
            mycm.set_bad(facecolor)
            mycm.set_under(facecolor)
                
            dzi = self.minuend_chopped[i].channels[0].values - self.subtrahend_chopped[i].channels[0].values
                  
            dax = plt.subplot(gs[4])
            plt.subplot(dax)
            
            X, Y, Z = pcolor_helper(xaxis.points, yaxis.points, dzi)
            
            largest = np.nanmax(np.abs(dzi))
            
            dcax = dax.pcolor(X, Y, Z, vmin=-largest, vmax=largest, cmap=mycm)
            
            dax.set_xlim(xaxis.points.min(), xaxis.points.max())
            dax.set_ylim(yaxis.points.min(), yaxis.points.max())            
            
            differenc_cb = plt.subplot(gs[5])
            dcbar = plt.colorbar(dcax, cax=differenc_cb)
            dcbar.set_label(self.minuend.channels[channel_index].name +
                            ' - ' + self.subtrahend.channels[channel_index].name)
            
            # title -----------------------------------------------------------

            title_text = self.minuend.name + ' - ' + self.subtrahend.name

            constants_text = '\n'
            for constant in constants:
                constants_text += constant.get_label(show_units = True, points = True) + '    '

            plt.suptitle(title_text + constants_text, fontsize = self.font_size)

            fig.subplots_adjust(left=0.075, right=1-0.075, top=0.90, bottom=0.15)
            
            plt.figtext(0.03, 0.5, yaxis.get_label(), fontsize = self.font_size, rotation = 90)
            plt.figtext(0.5, 0.01, xaxis.get_label(), fontsize = self.font_size, horizontalalignment = 'center')            

            # cleanup ---------------------------------------------------------

            plt.setp(plt.subplot(gs[1]).get_yticklabels(), visible=False)
            plt.setp(plt.subplot(gs[4]).get_yticklabels(), visible=False)

            # save figure -----------------------------------------------------

            if autosave:
                if fname:
                    file_name = fname + ' ' + str(i).zfill(3)
                else:
                    file_name = str(i).zfill(3)
                fpath = os.path.join(output_folder, file_name + '.png')
                plt.savefig(fpath, facecolor = 'none')
                plt.close()

                if verbose:
                    print 'image saved at', fpath
        
        plt.ion()

