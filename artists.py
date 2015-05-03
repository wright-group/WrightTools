import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as grd
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.colors as mplcolors

from scipy.interpolate import griddata, interp1d

import kit

### artist helpers #############################################################

def nm_to_rgb(nm):
    '''
    returns list [r, g, b] (zero to one scale) for given input in nm \n
    original code - http://www.physics.sfasu.edu/astro/color/spectra.html
    '''    
    
    w = int(wavelength)

    #color----------------------------------------------------------------------

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

    #intensity correction-------------------------------------------------------

    if w >= 380 and w < 420:
        SSS = 0.3 + 0.7*(w - 350) / (420 - 350)
    elif w >= 420 and w <= 700:
        SSS = 1.0
    elif w > 700 and w <= 780:
        SSS = 0.3 + 0.7*(780 - w) / (780 - 700)
    else:
        SSS = 0.0
    SSS *= 255

    return [float(int(SSS*R)/256.), float(int(SSS*G)/256.), float(int(SSS*B)/256.)]

### color maps #################################################################

default = ['#FFFFFF',
           '#0000FF',
           '#00FFFF',
           '#00FF00',
           '#FFFF00',
           '#FF0000',
           '#881111']
           
experimental = ['#FFFFFF',
                '#0000FF',
                '#0080FF',
                '#00FFFF',
                '#00FF00',
                '#FFFF00',
                '#FF8000',
                '#FF0000',
                '#881111']

greenscale = ['#000000', #black
              '#00FF00'] #green
                              
greyscale = ['#FFFFFF', #white
             '#000000'] #black
             
invisible = ['#FFFFFF', #white
             '#FFFFFF'] #white

signed = ['#0000FF', #blue
          '#002AFF', 
          '#0055FF',
          '#007FFF', 
          '#00AAFF', 
          '#00D4FF', 
          '#00FFFF',
          '#FFFFFF', #white
          '#FFFF00', 
          '#FFD400', 
          '#FFAA00', 
          '#FF7F00', 
          '#FF5500', 
          '#FF2A00',
          '#FF0000'] #red

signed_old = ['#0000FF', #blue
              '#00BBFF', #blue-aqua
              '#00FFFF', #aqua
              '#FFFFFF', #white
              '#FFFF00', #yellow
              '#FFBB00', #orange
              '#FF0000'] #red
          
skyebar = ['#FFFFFF', #white
           '#000000', #black
           '#0000FF', #blue
           '#00FFFF', #cyan
           '#64FF00', #light green
           '#FFFF00', #yellow
           '#FF8000', #orange
           '#FF0000', #red
           '#800000'] #dark red          

colormaps = {'cubehelix': plt.get_cmap('cubehelix'),
             'default': mplcolors.LinearSegmentedColormap.from_list('wright', default),
             'experimental': mplcolors.LinearSegmentedColormap.from_list('experimental', experimental),
             'flag': plt.get_cmap('flag'),
             'earth': plt.get_cmap('gist_earth'),
             'greenscale': mplcolors.LinearSegmentedColormap.from_list('greenscale', greenscale), 
             'greyscale': mplcolors.LinearSegmentedColormap.from_list('greyscale', greyscale),
             'invisible': mplcolors.LinearSegmentedColormap.from_list('invisible', invisible),
             'ncar': plt.get_cmap('gist_ncar'),
             'paired': plt.get_cmap('Paired'),
             'prism': plt.get_cmap('prism'),
             'rainbow': plt.get_cmap('rainbow'),
             'signed':  mplcolors.LinearSegmentedColormap.from_list('signed', signed),
             'signed_old':  mplcolors.LinearSegmentedColormap.from_list('signed', signed_old),
             'skyebar':  mplcolors.LinearSegmentedColormap.from_list('skyebar', skyebar),
             'spectral': plt.get_cmap('nipy_spectral')} 

### general purpose artists ####################################################

class mpl_1D:
    
    def __init__(self, data, xaxis = 0, at = {}, verbose = True):

        #import data------------------------------------------------------------

        self.data = data
        self.chopped = self.data.chop(xaxis, at, verbose = False)
        
        if verbose:
            print 'mpl_1D recieved data to make %d plots'%len(self.chopped)
            
        #defaults---------------------------------------------------------------
        
        self.font_size = 15
        
    def plot(self, channel = 0, local = False,
             autosave = False, output_folder = None, fname = None,
             verbose = True):
        
        fig = None
        
        if len(self.chopped) > 10:
            if not autosave:
                print 'too many images will be generated ({}): forcing autosave'.format(len(self.chopped))
                autosave = True
                
        #prepare output folder--------------------------------------------------
        
        if autosave:
            if output_folder:
                pass
            else:
                timestamp = kit.get_timestamp()
                os.mkdir(timestamp)
                output_folder = timestamp
                       
        #chew through image generation
        for i in range(len(self.chopped)):
        
            if fig: plt.close(fig)
            
            fig = plt.figure(figsize=(8, 6))
        
            axes, channels, constants = self.chopped[i]
            
            xi = axes[0].points
            zi = channels[channel].values
            
            plt.plot(xi, zi)
            plt.grid()
            
            #limits-------------------------------------------------------------
            
            if local:
                pass
            else:
                plt.ylim(channels[channel].zmin, channels[channel].zmax)

            #label axes---------------------------------------------------------

            plt.xlabel(axes[0].get_label())
            plt.ylabel(channels[channel].name)
            
            #title--------------------------------------------------------------
            
            title_text = self.data.name
            
            constants_text = '\n'
            for constant in constants:
                constants_text += constant.get_label(units = True, points = True) + '    '       
                
            plt.suptitle(title_text + constants_text, fontsize = self.font_size)
            
            #save figure--------------------------------------------------------
            
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
        
        #import data------------------------------------------------------------

        self.data = data
        self.chopped = self.data.chop(yaxis, xaxis, at, verbose = False)
        
        if verbose:
            print 'mpl_2D recieved data to make %d plots'%len(self.chopped)
            
        #defaults---------------------------------------------------------------
        
        self.font_size = 15
        
        self._xsideplot = False
        self._ysideplot = False
        self._xsideplotdata = []
        self._ysideplotdata = []
    
    def sideplot(self, data, x = True, y = True):

        if x:
            self._xsideplot = True
            self._xsideplotdata.append([data.axes[0].points, data.channels[0].values])
        
        if y:
            self._ysideplot = True
            self._ysideplotdata.append([data.axes[0].points, data.channels[0].values])

    def plot(self, channel_index,
             contours = 9, pixelated = False, lines = True, cmap = 'default', 
             dynamic_range = False, local = False, contours_local = True, normalize_slices = 'both',
             xbin = False, ybin = False,
             autosave = False, output_folder = None, fname = None,
             verbose = True):
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
        
        #prepare output folder--------------------------------------------------
        
        if autosave:
            if output_folder:
                pass
            else:
                timestamp = kit.get_timestamp()
                os.mkdir(timestamp)
                output_folder = timestamp  
             
        #chew through image generation
        for i in range(len(self.chopped)):
            
            #get data to plot---------------------------------------------------
            
            axes, channels, constants = self.chopped[i]
            
            xaxis = axes[1]
            yaxis = axes[0]
            channel = channels[channel_index]
            zi = channel.values
            
            #normalize slices---------------------------------------------------
            
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
            
            #create figure------------------------------------------------------            
            
            if fig: plt.close(fig)            
            
            fig = plt.figure(figsize=(8, 7))
            
            gs = grd.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.1)            
            
            subplot_main = plt.subplot(gs[0])
                        
            #levels-------------------------------------------------------------
            
            if channel.signed:
                
                if dynamic_range:
                    limit = min(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                else:
                    limit = max(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                levels = np.linspace(-limit + channel.znull, limit + channel.znull, 200)
                
            else:
                
                if local: 
                    levels = np.linspace(zi.min(), zi.max(), 200)
                else:
                    levels = np.linspace(channel.zmin, channel.zmax, 200)
                    
            #main plot----------------------------------------------------------
            
            #get colormap
            mycm = colormaps[cmap]

            #fill in main data environment
            if pixelated:
                x_points = np.zeros(len(xaxis.points) + 1)
                y_points = np.zeros(len(yaxis.points) + 1)
                for points, axis in [[x_points, xaxis], [y_points, yaxis]]:
                    for j in range(len(points)):
                        if j == 0:               #first point
                            points[j] = axis.points[0] - (axis.points[1] - axis.points[0])
                        elif j == len(points)-1: #last point
                            points[j] = axis.points[-1] +  (axis.points[-1] - axis.points[-2])
                        else:
                            points[j] = np.average([axis.points[j], axis.points[j-1]])
                #plot
                xi, yi = np.meshgrid(x_points, y_points)
                cax = plt.pcolor(xi, yi, zi, cmap = mycm,
                                 vmin = levels.min(), vmax = levels.max())
                plt.xlim(x_points.min(), x_points.max())
                plt.ylim(y_points.min(), y_points.max())
            else:
                cax = subplot_main.contourf(xaxis.points, yaxis.points, zi,
                                            levels, cmap = mycm)
            
            plt.xticks(rotation = 45)
            plt.xlabel(xaxis.get_label(), fontsize = self.font_size)
            plt.ylabel(yaxis.get_label(), fontsize = self.font_size)
            
            #variable marker lines----------------------------------------------
            
            if lines:
                for constant in constants:
                        if constant.units_kind == 'energy':
                            #x axis
                            if xaxis.units == constant.units:
                                plt.axvline(constant.points, color = 'k', linewidth = 4, alpha = 0.25)
                            #y axis
                            if yaxis.units == constant.units:
                                plt.axhline(constant.points, color = 'k', linewidth = 4, alpha = 0.25)
    
            #grid---------------------------------------------------------------
            
            plt.grid(b = True)
            
            if xaxis.units == yaxis.units:
                #add diagonal line
                diag_min = max(min(xaxis.points),min(yaxis.points))
                diag_max = min(max(xaxis.points),max(yaxis.points))
                plt.plot([diag_min, diag_max],[diag_min, diag_max],'k:')
            
            #contour lines------------------------------------------------------

            if contours:
                if contours_local:
                    contours_levels = np.linspace(zi.min(), zi.max(), contours)
                else:
                    contours_levels = contours
                subplot_main.contour(xaxis.points, yaxis.points, zi, 
                                     contours_levels, colors = 'k')
                      
            #finish main subplot------------------------------------------------
                                     
            subplot_main.set_xlim(xaxis.points[0], xaxis.points[-1])                     
            subplot_main.set_ylim(yaxis.points[0], yaxis.points[-1])

            #sideplots----------------------------------------------------------

            divider = make_axes_locatable(subplot_main)
            
            if xbin or self._xsideplot:            
            
                axCorrx = divider.append_axes('top', 0.75, pad=0.3, sharex=subplot_main)
                axCorrx.autoscale(False)
                axCorrx.set_adjustable('box-forced')
                plt.setp(axCorrx.get_xticklabels(), visible=False)
                plt.setp(axCorrx.get_yticklabels(), visible=False)
                plt.grid(b = True)
                if channel.signed:
                    axCorrx.set_ylim([-1.1,1.1])
                else:
                    axCorrx.set_ylim([0,1.1])
                
                #bin
                if xbin:
                    x_ax_int = zi.sum(axis=0) - channel.znull * len(yaxis.points)
                    #normalize (min is a pixel)
                    xmax = max(np.abs(x_ax_int))
                    x_ax_int = x_ax_int / xmax
                    axCorrx.plot(xaxis.points,x_ax_int, lw = 2)
                    axCorrx.set_xlim([xaxis.points.min(), xaxis.points.max()])
                    
                #data
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
                                
            if ybin or self._ysideplot:
            
                axCorry = divider.append_axes('right', 0.75, pad=0.3, sharey=subplot_main)
                axCorry.autoscale(False)
                axCorry.set_adjustable('box-forced')
                plt.setp(axCorry.get_xticklabels(), visible=False)
                plt.setp(axCorry.get_yticklabels(), visible=False)   
                plt.grid(b = True)  
                if channel.signed:
                    axCorry.set_xlim([-1.1,1.1])
                else:
                    axCorry.set_xlim([0,1.1])
                
                #bin
                if ybin:
                    y_ax_int = zi.sum(axis=1) - channel.znull * len(xaxis.points)
                    #normalize (min is a pixel)
                    ymax = max(np.abs(y_ax_int))
                    y_ax_int = y_ax_int / ymax
                    axCorry.plot(y_ax_int, yaxis.points, lw = 2)
                    axCorry.set_ylim([yaxis.points.min(), yaxis.points.max()])
                
                #data
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
                                
            #colorbar-----------------------------------------------------------

            if True:
                subplot_cb = plt.subplot(gs[1])
                plt.colorbar(cax, cax = subplot_cb)
            
            #title--------------------------------------------------------------
            
            title_text = self.data.name
            
            constants_text = '\n'
            for constant in constants:
                constants_text += constant.get_label(units = True, points = True) + '    '         
                
            plt.suptitle(title_text + constants_text, fontsize = self.font_size)
            
            #cleanup------------------------------------------------------------
            
            #plt.tight_layout()
            factor = 0.12
            fig.subplots_adjust(left = factor, right = 1-factor, top = 1-factor, bottom = 0.15)
            
            #save figure--------------------------------------------------------

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
        
### specific artists ###########################################################
        
class absorbance:
    
    def __init__(self, data):

        self.data = data
        
    def plot(self, channel_index = 0, font_size = 12, xlim = None, ylim = None):

        '''
        data can be a single object or a list of objects
        '''
        
        if not type(data) == list:
            data = [data]

        self.data = data
        

        xi = self.data.axes[0].points
        zi = self.data.channels[channel_index].values
        name = self.data.name

    def plot(self, channel_index = 0, font_size = 12, xlim = None, ylim = None):
        
        #prepare plot environment-----------------------------------------------
    
        self.ax1 = plt.subplot(211)
        self.ax1.legend(loc=4)
        self.ax1.grid(b=True)
        self.ax1.set_ylim(0, 1.1)
        plt.ylabel('absorbance')
        
        self.ax2 = plt.subplot(212, sharex=self.ax1)
        self.ax2.set_ylim(-0.001, 0.001)
        
        plt.ylabel('2nd derivative')
        matplotlib.rcParams.update({'font.size': font_size})

        for data in self.data:        
        
            #import data--------------------------------------------------------
            
            xi = data.axes[0].points
            zi = data.channels[channel_index].values

            #scale--------------------------------------------------------------
            
            if xlim:
                plt.xlim(xlim[0], xlim[1])

                min_index = np.argmin(abs(xi - min(xlim)))
                max_index = np.argmin(abs(xi - max(xlim)))
                
                zi_truncated = zi[min(min_index,max_index):max(min_index, max_index)]
                zi -= zi_truncated.min()
                
                zi_truncated = zi[min(min_index,max_index):max(min_index, max_index)]
                zi /= zi_truncated.max()

            #plot absorbance----------------------------------------------------        
            
            self.ax1.plot(xi, zi, lw = 2)
            
            #now plot 2nd derivative--------------------------------------------
            
            #compute second derivative
            xi2, zi2= self._smooth(np.array([xi,zi]))
            plotData = np.array([np.delete(xi2, [0, len(xi2)-1]), np.diff(zi2, n=2)])
            
            #plot the data!
            self.ax2.plot(plotData[0], plotData[1], lw = 2)
            
            self.ax2.grid(b=True)
            plt.xlabel(r'$\bar\nu / cm^{-1}$')
            
        self.ax1.legend([data.name for data in self.data])
            
            
        self.ax1.get_yaxis().set_ticks([])
        
        self.ax2.get_yaxis().set_ticks([])
        self.ax2.axhline(0, color = 'k', ls = ':')
        
        
        
        
        #finish-----------------------------------------------------------------
        
        if xlim:
            plt.xlim(xlim[0], xlim[1])
            for axis, xi, zi in [[self.ax1, xi, zi], [self.ax2, plotData[0], plotData[1]]]:
                min_index = np.argmin(abs(xi - min(xlim)))
                max_index = np.argmin(abs(xi - max(xlim)))
                zi_truncated = zi[min_index:max_index]
                extra = (zi_truncated.max() - zi_truncated.min())*0.1
                axis.set_ylim(zi_truncated.min() - extra, zi_truncated.max() + extra)
                
            
            
            
        
    def _smooth(self, dat1, n=20, window_type='default'):
        #data is an array of type [xlis,ylis]        
        #smooth to prevent 2nd derivative from being noisy
        for i in range(n, len(dat1[1])-n):
            #change the x value to the average
            window = dat1[1][i-n:i+n].copy()
            dat1[1][i] = window.mean()
        return dat1[:][:,n:-n]