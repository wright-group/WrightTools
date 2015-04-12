import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata, interp1d
import matplotlib.gridspec as grd
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

class mpl_2D:
    """
        class for initializing plotting functions
        depends on dimensionality, presets, etc.
    """
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
    # define colormaps
    mycm=mplcolors.LinearSegmentedColormap.from_list('wright',wrightcm)
    altcm=mplcolors.LinearSegmentedColormap.from_list('signed',signed_cm)

    aspect=None
    # font style attributes
    font_size = 14
    font_family = 'sans-serif'
    # plot windowing--use to concatenate image
    limits = False
    xlim=[]
    ylim=[]

    contour_n = 11
    contour_kwargs={'colors':'k',
                    'linewidths':2}
    # attributes of sideplots
    side_plot_proj_kwargs={'linewidth':2}
    side_plot_proj_linetype = 'b'
    side_plot_else_kwargs={'linewidth':2}
    side_plot_else_linetype = 'r'

    def __init__(self, **kwargs):
        # you can pass it xvar and yvars so it sets up the axes properly
        for key in ['xaxis', 'yaxis']:
            val = kwargs.get(key)
            if val is not None:
                setattr(self, key, kwargs.get(key))
        '''
        if isinstance(self.xaxis, Axis):
            self.xlabel = self.xaxis.name
        if isinstance(self.yaxis, Axis):
            self.ylabel = self.yaxis.name
            if self.xaxis.pulse_var == self.yaxis.pulse_var:
                self.aspect = 'equal'
        '''
    
    def colorbar(self):
        """
            adds colorbar to the contour plot figure
            only after all contour plot embellishments have been performed
        """
        if self.s1:
            ax_cb = plt.subplot(self.gs[1])
        else:
            print 'must create plot before adding colorbar'
            return
        if self.alt_z == 'int':
            ticks = np.linspace(-1,1,21)
            # find the intersection of the range of data displayed and ticks
            ticks = [ticki for ticki in ticks if ticki >= 
                min(self.z_norm.min(), self.znull) and 
                ticki <= max(self.znull, self.z_norm.max())]
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_z == '+':
            ticks = np.linspace(self.znull,self.zmax,11)
            # find the intersection of the range of data displayed and ticks
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_z == '-':
            ticks = np.linspace(self.zmin,self.znull,11)
            # find the intersection of the range of data displayed and ticks
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_z == 'amp':
            ticks = np.linspace(0,1,11)
            ampmin = self.floor
            ampmax = self.ceiling
            ticks = np.linspace(ampmin,ampmax,num=11)
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        elif self.alt_z == 'log':
            logmin = self.floor
            logmax = self.ceiling
            ticks = np.linspace(logmin,logmax,num=11)
            # determine how much precision is necessary in the ticks:
            decimals = int(np.floor(-np.log10(np.abs(
                ticks[-1]-ticks[0])))) + 2
            ticklabels = np.around(ticks,decimals)
            self.p1.colorbar(self.cax, ticks=ticks).ax.set_yticklabels(ticklabels)
        elif self.alt_z in [None, 'raw']: # raw colorbar
            ticks = np.linspace(min([self.znull, self.zmin]),
                                max(self.znull, self.zmax),num=11)
            decimals = int(np.floor(-np.log10(np.abs(
                ticks[-1]-ticks[0])))) + 2
            ticklabels = np.around(ticks,decimals)
            self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb).ax.set_yticklabels(ticklabels)
            #self.p1.colorbar(self.cax, ticks=ticks, cax=ax_cb)
        else: #could not determine colorbar type
            print 'color scale used not recognized:  cannot produce colorbar'

    def plot(self, data, xaxis, yaxis = None, channel = 0, alt_z='raw', 
               scantype=None, contour=False, aspect=None, pixelated=False, 
               dynamic_range=False, floor=None, ceiling=None):
        """
            dynamic_range=True will force the colorbar to use all of it's colors
            floor used exclusively for divergent zero signal scales (amp and log)
            floor is the cutoff to be established in the scaled space
            
            plot lays out the contour figure, but also stores some info about xyz
            to make it easy to calculate colorbars and such
        """
        
        #unpack chop------------------------------------------------------------

        axes, zi = data.chop(xaxis, yaxis, channel)
        
        #FOR NOW WE FORCE THINGS TO WORK BY CREATING AN XYZ OBJECT DIRECTLY
        xyz = XYZ(axes[0].points, axes[1].points, zi,
                  znull=data.znull, zmin=data.zmin, zmax=data.zmax)

        #prepare matplotlib-----------------------------------------------------        
        
        matplotlib.rcParams.update({'font.size':self.font_size})
        
        # delete old plot data stored in the plt class
        plt.close()

        # update parameters
        p1 = plt.figure()
        gs = grd.GridSpec(1,2, width_ratios=[20,1], wspace=0.1)
        if self.aspect is None and aspect is None:
            s1 = p1.add_subplot(gs[0])
        else:
            if aspect is not None:
                self.aspect = aspect
            s1 = p1.add_subplot(gs[0], aspect=self.aspect)
            if self.aspect=='equal':
                diag_min = max(min(xyz.x),min(xyz.y))
                diag_max = min(max(xyz.x),max(xyz.y))
                plt.plot([diag_min, diag_max],[diag_min, diag_max],'k:')
                
        # attach to the plot objects so further manipulations can be done
        self.p1=p1
        self.gs=gs
        self.s1=s1

        #z scaling--------------------------------------------------------------

        if alt_z in ['int', None, 'raw', '+', '-']:
            znull = None
            if alt_z == '+':
                znull = xyz.znull
                z_norm = np.ma.masked_less(xyz.z, znull)
                z_norm = z_norm.filled(znull)
                lbound = znull
                ubound = xyz.zmax
            elif alt_z == '-':
                znull = xyz.znull
                z_norm = np.ma.masked_greater(xyz.z, znull)
                z_norm = z_norm.filled(znull)
                lbound = xyz.zmin
                ubound = znull
            else:
                if alt_z == 'int':
                    # for regular normalized (unscaled, normalized to znull-zmax range)
                    # first offset and normalize data
                    z_sign_mag = max(np.abs([xyz.zmax-xyz.znull, xyz.zmin-xyz.znull]))
                    z_norm = (xyz.z - xyz.znull) / z_sign_mag
                    znull = 0.
                else: # alt_z in [None, 'raw']
                    z_norm = xyz.z
                    znull = xyz.znull
                if xyz.zmax == xyz.z.max():
                    zmax = max(znull, z_norm.max())
                else:
                    zmax = xyz.zmax
                if xyz.zmin == xyz.z.min():
                    zmin = min(znull, z_norm.min())
                else:
                    zmin = xyz.zmin
                # now I have to whether or not the data is signed, if zmin and zmax
                # are on the same side of znull, then the data only has one sign!
                if znull >= max(zmin, zmax):
                    # data is negative sign
                    print 'data has only negative sign'
                    if dynamic_range:
                        ubound = zmax
                    else:
                        ubound = znull
                    lbound = zmin
                elif znull <= min(zmin, zmax):
                    # data is positive sign
                    print 'data has only positive sign'
                    if dynamic_range:
                        lbound = zmin
                    else:
                        lbound = znull
                    ubound = zmax
                else:
                    # data has positive and negative sign, so center the colorbar
                    print 'data has positive and negative sign'
                    if dynamic_range:
                        # check for whether positive or negative signals extend less
                        # using smaller range on both sides of znull ensures full 
                        # dynamic range of colorbar
                        if -zmin + znull < zmax - znull:
                            ubound = np.abs(zmin)
                        else:
                            ubound = np.abs(zmax)
                    else:
                        # using larger range on both sides of znull ensures full 
                        # range of data will be shown
                        if -zmin + znull < zmax - znull:
                            ubound = np.abs(zmax)
                        else:
                            ubound = np.abs(zmin)
                    lbound = -ubound
            print 'lower and upper bounds:', lbound, ubound
            levels = np.linspace(lbound, ubound, num=200)
            # artist need this information for colorbar
            self.zmin, self.zmax = zmin, zmax
            self.znull = znull
            
        elif alt_z in ['amp', 'log']:
            z_norm = np.ma.masked_less_equal(
                (xyz.z - xyz.znull) / (xyz.zmax - xyz.znull), 0.)
            if alt_z == 'amp':
                # for sqrt scale (amplitude)
                z_norm = np.sqrt(z_norm)
                if floor is not None:
                    self.floor = floor
                else:
                    self.floor = 0.
                z_norm = np.ma.masked_less_equal(z_norm, self.floor)
                z_norm = z_norm.filled(self.floor)
                if ceiling is not None:
                    self.ceiling = ceiling
                else:
                    self.ceiling = 1 #np.sqrt(self.zmax / (self.zmax-self.znull))
                z_norm = np.ma.masked_greater(z_norm, self.ceiling)
                z_norm = z_norm.filled(self.ceiling)
            elif alt_z == 'log':
                # for log scale
                z_norm = np.log10(z_norm)
                # cutoffs can be defined in terms of log scale as well
                if floor is not None:
                    self.floor = floor
                else:
                    self.floor = z_norm.min()
                z_norm = np.ma.masked_less_equal(z_norm, self.floor)
                z_norm = z_norm.filled(self.floor)
                if ceiling is not None:
                    self.ceiling = ceiling
                else:
                    self.ceiling = np.log10(self.zmax / (self.zmax-self.znull))
                z_norm = np.ma.masked_greater(z_norm, self.ceiling)
                z_norm = z_norm.filled(self.ceiling)
            levels = np.linspace(self.floor, self.ceiling, num=200)
        else:
            print 'alt_z type {0} not recognized; plotting on raw scale'.format(alt_z)
            z_norm = xyz.z
            levels = 200 
        xyz.alt_z=alt_z
        xyz.z_norm = z_norm
        # artist needs to know alt_z as well!
        self.alt_z = alt_z
                
        #plot the data----------------------------------------------------------
        
        if pixelated:
            # need to input step size to get centering to work
            x_step = np.abs(xyz.x[1] - xyz.x[0])
            y_step = np.abs(xyz.y[1] - xyz.y[0])
            if aspect:
                pixel_aspect=aspect
            else:
                # this weighting makes the plot itself square
                pixel_aspect = (xyz.x.max() - xyz.x.min()) / (xyz.y.max() - xyz.y.min())
                # this weighting gives square pixels...?
                #pixel_aspect = 1. / pixel_aspect
            cax = plt.imshow(z_norm, origin='lower', cmap=self.mycm, 
                             interpolation='nearest', 
                             vmin=levels.min(), vmax=levels.max(),
                             extent=[xyz.x.min() - x_step/2., 
                                     xyz.x.max() + x_step/2., 
                                     xyz.y.min() - y_step/2., 
                                     xyz.y.max() + y_step/2.])#,
                             #aspect=pixel_aspect)
            plt.gca().set_aspect(pixel_aspect, adjustable='box-forced')
        else:
            cax = plt.contourf(xyz.x, xyz.y, z_norm, levels, 
                               cmap=self.mycm)
        self.cax=cax
        if contour:
            plt.contour(xyz.x, xyz.y, z_norm, self.contour_n, 
                        **self.contour_kwargs)
                        
        #enforce plot consistency-----------------------------------------------                        
        
        #matplotlib.axes.rcParams.viewitems
        plt.xticks(rotation=45)
        plt.grid(b=True)
        if self.limits:
            v = np.array([self.xlim[0], self.xlim[1],
                          self.ylim[0], self.ylim[1]])
        else:
            v = np.array([xyz.x.min(), xyz.x.max(),
                          xyz.y.min(), xyz.y.max()])
            #x_decimal=max(0,int(np.ceil(np.log10(np.abs(v[1]-v[0]))+2)))
            #y_decimal=max(0,int(np.ceil(np.log10(np.abs(v[3]-v[2]))+2)))
            #v[0:2] = np.around(v[0:2], decimals=x_decimal)
            #v[2:] = np.around(v[2:], decimals=y_decimal)
        s1.axis(v)

        if aspect:
            s1.set_aspect(aspect)
        # window the plot; use either 2d plot dimensions or set window
        try:
            plt.ylabel(self.ylabel, fontsize=self.font_size)
            plt.xlabel(self.xlabel, fontsize=self.font_size)
        except:
            print 'labels failed to print'
        p1.subplots_adjust(bottom=0.18)
        #s1.set_adjustable('box-forced')
        s1.autoscale(False)
        print 'plotting finished!'

    def side_plots(self, subplot, 
                    # do we project (bin) either axis?
                    x_proj=False, y_proj=False, 
                    # provide a list of coordinates for sideplot
                    x_list=None, y_list=None,
                    # provide a NIRscan object to plot
                    x_obj=None, y_obj=None):
        """
            position complementary axis plot on x and/or y axes of subplot
        """
        #if there is no 1d_object, try to import one
        divider = make_axes_locatable(subplot)
        if x_proj or x_list or x_obj:
            axCorrx = divider.append_axes('top', 0.75, pad=0.3, sharex=subplot)
            axCorrx.autoscale(False)
            axCorrx.set_adjustable('box-forced')
            # make labels invisible
            plt.setp(axCorrx.get_xticklabels(), visible=False)
            axCorrx.get_yaxis().set_visible(False)
            axCorrx.grid(b=True)
        if y_proj or y_list or y_obj:
            axCorry = divider.append_axes('right', 0.75, pad=0.3, sharey=subplot)
            axCorry.autoscale(False)
            axCorry.set_adjustable('box-forced')
            # make labels invisible
            plt.setp(axCorry.get_yticklabels(), visible=False)
            axCorry.get_xaxis().set_visible(False)
            axCorry.grid(b=True)
        if x_proj:
            #integrate the axis
            x_ax_int = self.zi.sum(axis=0) - self.znull * len(self.yi)
            #normalize (min is a pixel)
            xmax = max(np.abs(x_ax_int))
            x_ax_int = x_ax_int / xmax
            axCorrx.plot(self.xi,x_ax_int,self.side_plot_proj_linetype,
                         **self.side_plot_proj_kwargs)
            if min(x_ax_int) < 0:
                axCorrx.set_ylim([-1.1,1.1])
            else:
                axCorrx.set_ylim([0,1.1])
            axCorrx.set_xlim([self.xi.min(), self.xi.max()])
        if y_proj:
            #integrate the axis
            y_ax_int = self.zi.sum(axis=1) - self.znull * len(self.xi)
            #normalize (min is a pixel)
            ymax = max(np.abs(y_ax_int))
            y_ax_int = y_ax_int / ymax
            axCorry.plot(y_ax_int,self.yi,self.side_plot_proj_linetype,
                         **self.side_plot_proj_kwargs)
            if min(y_ax_int) < 0:
                axCorry.set_xlim([-1.1,1.1])
            else:
                axCorry.set_xlim([0,1.1])
            axCorry.set_ylim([self.yi.min(), self.yi.max()])
        if isinstance(x_list, np.ndarray): 
            print x_list.shape
            axCorrx.plot(x_list[0],x_list[1], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
            axCorrx.set_ylim([0.,1.1])
        elif x_obj:
            try:
                x_list = x_obj.data[0][2].copy()
            except IndexError:
                print 'Import failed--data type was not recognized'
            # spectrometer has units of nm, so make sure these agree
            if self.xvar in ['w1','w2','wm']:
                x_list[0] = 10**7 / x_list[0]
            #normalize the data set
            x_list_max = x_list[1].max()
            x_list[1] = x_list[1] / x_list_max
            axCorrx.plot(x_list[0],x_list[1], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
            axCorrx.set_ylim([0.,1.1])
            axCorrx.set_xlim([self.xi.min(), self.xi.max()])
        if isinstance(y_list, np.ndarray):
            axCorry.plot(y_list[1],y_list[0], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
        elif y_obj:
            try:
                y_list = y_obj.data[0][2].copy()
            except IndexError:
                print 'Import failed--data type was not recognized'
            if self.yvar in ['w1','w2','wm']:
                y_list[0] = 10**7 / y_list[0]
            #normalize the data set
            y_list_max = y_list[1].max()
            y_list[1] = y_list[1] / y_list_max
            axCorry.plot(y_list[1],y_list[0], self.side_plot_else_linetype,
                         **self.side_plot_else_kwargs)
            #axCorry.set_xlim([0.,1.1])
            axCorry.set_ylim([self.yi.min(), self.yi.max()])

    def savefig(self, fname=None, **kwargs):
        """
            generates the image file by autonaming the file
            default image type is 'png'
        """
        try:
            self.p1
        except NameError:
            print 'no plot is associated with the data. cannot save'
            return
        if not fname:
            fname = self.filename
            filepath = self.filepath
            file_suffix = 'png'
        else:
            filepath, fname, file_suffix = filename_parse(fname)
            if not file_suffix:
                file_suffix = 'png' 
        if 'transparent' not in kwargs.keys():
            kwargs['transparent'] = True
        if filepath:
            fname = filepath + '\\' + fname
        fname = find_name(fname, file_suffix)
        fname = fname + '.' + file_suffix
        self.p1.savefig(fname, **kwargs)
        print 'image saved as {0}'.format(fname)
        
        
        
        
        
        
        
        
class XYZ:
    """
        a class for manipulating 2d data objects and their axes
            -plotting
            -decompositions and fitting
            -"slicing" data
            -normal array operations (adding and subtracting, etc.)
    """
    def __init__(self, x, y, z, 
                 znull=None, zmin=None, zmax=None):
        self.x = x
        self.y = y 
        self.z = z
        if znull is None:
            self.znull = 0.
        else: self.znull = znull
        if zmin is None:
            self.zmin = z.min()
        else: self.zmin = zmin
        if zmax is None:
            self.zmax = z.max()
        else: self.zmax = zmax

    def zoom(self, factor, order=1):
        import scipy.ndimage
        self.x = scipy.ndimage.interpolation.zoom(self.x, factor, order=order)
        self.y = scipy.ndimage.interpolation.zoom(self.y, factor, order=order)
        self.z = scipy.ndimage.interpolation.zoom(self.z, factor, order=order)
        
    def svd(self, verbose=False):
        """
            singular value decomposition of gridded data z
        """
        #give feedback on top (normalized) singular values
        U, s, V = np.linalg.svd(self.z)
        if verbose:
            # report significant stats on svd
            plt.figure()
            s_max = s.max()
            plt.scatter(s / s_max)
        return U, s, V

    def center(self, axis=None, center=None):
        if center == 'max':
            print 'listing center as the point of maximum value'
            if axis == 0 or axis in ['x', self.xvar]:
                index = self.zi.argmax(axis=0)
                set_var = self.xi
                max_var = self.yi
                out = np.zeros(self.xi.shape)
            elif axis == 1 or axis in ['y', self.yvar]:
                index = self.zi.argmax(axis=1)
                set_var = self.yi
                max_var = self.xi
                out = np.zeros(self.yi.shape)
            else:
                print 'Input error:  axis not identified'
                return
            for i in range(len(set_var)):
                out[i] = max_var[index[i]]
        else:
            # find center by average value
            out = self.exp_value(axis=axis, moment=1)
        return out
                
    def exp_value(self, axis=None, moment=1, norm=True, noise_filter=None):
        """
            returns the weighted average for fixed points along axis
            specify the axis you want to have exp values for (x or y)
            good for poor-man's 3peps, among other things
            moment argument can be any integer; meaningful ones are:
                0 (area, set norm False)
                1 (average, mu) or 
                2 (variance, or std**2)
            noise filter, a number between 0 and 1, specifies a cutoff for 
                values to consider in calculation.  zi values less than the 
                cutoff (on a normalized scale) will be ignored
            
        """
        if axis == 0:
            # an output for every x var
            z = self.z.copy()
            int_var = self.y
            out = np.zeros(self.x.shape)
        elif axis == 1:
            # an output for every y var
            z = self.z.T.copy()
            int_var = self.x
            out = np.zeros(self.y.shape)
        else:
            print 'Input error:  axis not identified'
            return
        if not isinstance(moment, int):
            print 'moment must be an integer.  recieved {0}'.format(moment)
            return
        for i in range(out.shape[0]):
            # ignoring znull for this calculation, and offseting my slice by min
            z_min = z[:,i].min()
            #zi_max = zi[:,i].max()
            temp_z = z[:,i] - z_min
            if noise_filter is not None:
                cutoff = noise_filter * (temp_z.max() - z_min)
                temp_z[temp_z < cutoff] = 0
            #calculate the normalized moment
            if norm == True:
                out[i] = np.dot(temp_z,int_var**moment) / temp_z.sum()#*np.abs(int_var[1]-int_var[0]) 
            else:
                out[i] = np.dot(temp_z,int_var**moment)
        return out

    def fit_gauss(self, axis=None):
        """
            least squares optimization of traces
            intial params p0 guessed by moments expansion
        """
        if axis == 0:
            # an output for every x var
            z = self.z.copy()
            var = self.y
            #out = np.zeros((len(self.xi), 3))
        elif axis == 1:
            # an output for every y var
            z = self.z.T.copy()
            var = self.x
            #out = np.zeros((len(self.yi), 3))

        # organize the list of initial params by calculating moments
        m0 = self.exp_value(axis=axis, moment=0, norm=False)
        m1 = self.exp_value(axis=axis, moment=1, noise_filter=0.1)
        m2 = self.exp_value(axis=axis, moment=2, noise_filter=0.1)        

        mu_0 = m1
        s0 = np.sqrt(np.abs(m2 - mu_0**2))
        A0 = m0 / (s0 * np.sqrt(2*np.pi))
        offset = np.zeros(m0.shape)
        
        p0 = np.array([A0, mu_0, s0, offset])
        out = p0.copy()
        from scipy.optimize import leastsq
        for i in range(out.shape[1]):
            #print leastsq(gauss_residuals, p0[:,i], args=(zi[:,i], var))
            try:
                out[:,i] = leastsq(gauss_residuals, p0[:,i], args=(z[:,i]-self.znull, var))[0]
            except:
                print 'least squares failed on {0}:  initial guesses will be used instead'.format(i)
                out[:,i] = p0[:,i]
        out[2] = np.abs(out[2])
        return out
        
    def smooth(self, 
               x=0,y=0, 
               window='kaiser',
               debug = False): #smoothes via adjacent averaging            
        """
            convolves the signal with a 2D window function
            currently only equipped for kaiser window
            'x' and 'y', both integers, are the nth nearest neighbor that get 
                included in the window
            Decide whether to perform xaxis smoothing or yaxis by setting the 
                boolean true
        """
        # n is the seed of the odd numbers:  n is how many nearest neighbors 
        # in each direction
        # make sure n is integer and n < grid dimension
        # account for interpolation using grid factor
        nx = x
        ny = y
        # create the window function
        if window == 'kaiser':
            # beta, a real number, is a form parameter of the kaiser window
            # beta = 5 makes this look approximately gaussian in weighting 
            # beta = 5 similar to Hamming window, according to numpy
            # over window (about 0 at end of window)
            beta=5.0
            wx = np.kaiser(2*nx+1, beta)
            wy = np.kaiser(2*ny+1, beta)
        # for a 2D array, y is the first index listed
        w = np.zeros((len(wy),len(wx)))
        for i in range(len(wy)):
            for j in range(len(wx)):
                w[i,j] = wy[i]*wx[j]
        # create a padded array of zi
        # numpy 1.7.x required for this to work
        temp_z = np.pad(self.zi, ((ny,ny), 
                                   (nx,nx)), 
                                    mode='edge')
        from scipy.signal import convolve
        out = convolve(temp_z, w/w.sum(), mode='valid')
        if debug:
            plt.figure()
            sp1 = plt.subplot(131)
            plt.contourf(self.zi, 100)
            plt.subplot(132, sharex=sp1, sharey=sp1)
            plt.contourf(w,100)
            plt.subplot(133)
            plt.contourf(out,100)
        self.z=out
        # reset zmax
        self.zmax = self.z.max()
        self.zmin = self.z.min()