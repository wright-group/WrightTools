import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as grd
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

from scipy.interpolate import griddata, interp1d

import kit

### color maps #################################################################

# colormap
signed_cm = ['#0000FF', #blue
             '#00BBFF', #blue-aqua
             '#00FFFF', #aqua
             '#FFFFFF', #white
             '#FFFF00', #yellow
             '#FFBB00', #orange
             '#FF0000'] #red
            
wrightcm = ['#FFFFFF',
            '#0000FF',
            '#00FFFF',
            '#00FF00',
            '#FFFF00',
            '#FF0000',
            '#881111']

import matplotlib.colors as mplcolors
colormaps = {'default': mplcolors.LinearSegmentedColormap.from_list('wright', wrightcm),
             'signed':  mplcolors.LinearSegmentedColormap.from_list('signed', signed_cm)} 

### general purpose artists ####################################################

class mpl_1D:
    
    def plot(self, data, axis, channel = 0, alt_z='raw', 
                   aspect=None, floor=None, ceiling=None):
                       
        xi = data.axes[axis].points
        zi = data.zis[channel]
        plt.plot(xi, zi)
        plt.grid()


class mpl_2D:
    
    def plot(self, data, xaxis, yaxis, at = {}, channel = 0, 
             autosave = False, output_folder = None, verbose = True):
                 
        #ensure plot environment clear------------------------------------------
                 
        plt.close()                 
                 
        #import data------------------------------------------------------------

        self.data = data
        self.chopped = self.data.chop(xaxis, yaxis, at)
        
        #warn if many images will be generated
        if not autosave and len(self.chop_list) > 10:
            print 'warning, you are about to generate %d images.'%d
            #in future put exit case
            
        #prepare output folder
        if autosave:
            if output_folder:
                pass
            else:
                timestamp = kit.get_timestamp()
                os.mkdir(timestamp)
                output_folder = timestamp  
             
        for i in range(len(self.chopped)):
            
            axes, zis, constants = self.chopped[i]
            
            plt.figure()
            
            xi = axes[1].points
            yi = axes[0].points
            zi = zis[channel]
            
            plt.contourf(xi, yi, zi, 200)
            
            #save figure
            if autosave:
                fpath = os.path.join(output_folder, '%d.png'%i)
                plt.savefig(fpath, transparent = True)
                plt.close()
            
            
            
        
    
    
        
### specific artists ###########################################################
        
class absorbance:
        
    def plot(self, data, channel = 0, font_size = 12):
        
        #import data------------------------------------------------------------
        
        xi = data.axes[0].points
        zi = data.zis[0]
        name = data.name
        
        #prepare plot environment-----------------------------------------------
    
        self.ax1 = plt.subplot(211)
        self.ax2 = plt.subplot(212, sharex=self.ax1)
        matplotlib.rcParams.update({'font.size': font_size})

        #plot absorbance--------------------------------------------------------        
        
        self.ax1.plot(xi, zi, label=name)
        
        plt.ylabel('abs (a.u.)')
        self.ax1.legend(loc=4)
        self.ax1.grid(b=True)
        
        #now plot 2nd derivative------------------------------------------------
        
        #compute second derivative
        xi2, zi2= self._smooth(np.array([xi,zi]))
        plotData = np.array([np.delete(xi2, [0, len(xi2)-1]), np.diff(zi2, n=2)])
        
        #plot the data!
        self.ax2.plot(plotData[0], plotData[1], label=name)
        
        self.ax2.grid(b=True)
        plt.xlabel(r'$\bar\nu / cm^{-1}$')
        
    def _smooth(self, dat1, n=20, window_type='default'):
        #data is an array of type [xlis,ylis]        
        #smooth to prevent 2nd derivative from being noisy
        for i in range(n, len(dat1[1])-n):
            #change the x value to the average
            window = dat1[1][i-n:i+n].copy()
            dat1[1][i] = window.mean()
        return dat1[:][:,n:-n]