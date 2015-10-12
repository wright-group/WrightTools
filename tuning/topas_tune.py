#import~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#os interaction packages
import os
import re
import sys
import imp
import time
import copy
import inspect
import subprocess
import ConfigParser
import glob #used to search through folders for filesd

#qt is used for gui handling, see Qtforum.org for info
from PyQt4.QtCore import * #* means import all
from PyQt4.QtGui import *

#matplotlib is used for plot generation and manipulation
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as grd
from mpl_toolkits.axes_grid1 import make_axes_locatable

#numpy is used for general math in python
import numpy as np
from numpy import sin, cos
                
#scipy is used for some nice array manipulation
import scipy
from scipy.interpolate import griddata, interp1d, interp2d, UnivariateSpline
import scipy.integrate as integrate
from scipy.optimize import leastsq

#pylab
from pylab import *

#filepath for relative navigation purposes
filepath_of_folder = os.path.abspath( __file__ )
filepath_of_folder = filepath_of_folder.replace(r'\topas_tune.pyc', '')
filepath_of_folder = filepath_of_folder.replace(r'\topas_tune.py', '')
topas_tune_ini_filepath = filepath_of_folder + r'\topas_tune.ini'

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through). 
    
    from http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
   
    with wt.kit.suppress_stdout_stderr():
        rogue_function()
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))
    
    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)
    
    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


#constants~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

spitfire_output = 800. #nm
             
wright_colormap_array = ['#FFFFFF','#0000FF','#00FFFF','#00FF00','#FFFF00','#FF0000','#881111']
wright_colormap = mplcolors.LinearSegmentedColormap.from_list('wright', wright_colormap_array)
wright_colormap.set_bad('grey',1.)

#internal format of tune point arrays
#  0 - setpoint      - nm
#  1 - m0(c1)        - us
#  2 - m1(d1)        - us
#  3 - m2(c2)        - us
#  4 - m3(d2)        - us
#  5 - m4(M1)        - us
#  6 - m5(M2)        - us
#  7 - m6(M3)        - us
#  8 - fit center    - nm
#  9 - fit amplitude - a.u.
# 10 - fit FWHM      - nm
# 11 - fit GoF       - percent
# 12 - fit mismatch  - nm
# 13 - source color  - nm
# 14 - reserved
# 15 - reserved
# 16 - reserved
# 17 - reserved
# 18 - reserved
# 19 - reserved
             
#functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
             
def create_clean_starting_curve(OPA, 
                                interaction_string):
    
    #import old curve-----------------------------------------------------------
    
    old_tunepoints = _crv('read', OPA, interaction_string = interaction_string)    
    
    #choose new curve points----------------------------------------------------

    if interaction_string in ['NON-NON-NON-Sig', 'NON-NON-NON-Idl', 'preamp']: 
    
        num_tunepoints = 30
        
        max_c1 = max(old_tunepoints[:, 1]) + 300
        min_c1 = min(old_tunepoints[:, 1]) - 300
        c1_positions = np.linspace(min_c1, max_c1, num_tunepoints)
        
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 1], old_tunepoints[:, 0], k=2, s=100000)
        d1_spline = UnivariateSpline(old_tunepoints[:, 0], old_tunepoints[:, 2], k=2, s=100000)
        c2_spline = UnivariateSpline(old_tunepoints[:, 0], old_tunepoints[:, 3], k=2, s=100000)
        d2_spline = UnivariateSpline(old_tunepoints[:, 0], old_tunepoints[:, 4], k=2, s=100000)

    
        new_tunepoints = np.zeros([num_tunepoints, 20])
        for i in range(num_tunepoints):
            c1_position = int(c1_positions[i])
            setpoint = setpoint_spline(c1_position)
            if np.isnan(setpoint):
                if i < num_tunepoints/2:
                    setpoint = 1100 + 0.25*i
                else:
                    setpoint = 1640 + 0.25*i
            new_tunepoints[i, 0] = setpoint            
            new_tunepoints[i, 1] = c1_position
            new_tunepoints[i, 2] = d1_spline(setpoint)
            new_tunepoints[i, 3] = c2_spline(setpoint)
            new_tunepoints[i, 4] = d2_spline(setpoint)
            new_tunepoints[i, 13] = old_tunepoints[0, 13]
            
        #check if setpoints are reasonable
        for i in range(len(new_tunepoints)):
            if new_tunepoints[i, 0] > 1625:
                new_tunepoints[i, 0] = 1630 - 0.25*(len(new_tunepoints)-i)
            if new_tunepoints[i, 0] < 1115:
                new_tunepoints[i, 0] = 1110 + 0.25*(i+1)
                
    elif interaction_string == 'NON-NON-SH-Idl': # - - - - - - - - - - - - - - -
    
        num_tunepoints = 15
        
        #import source curve, decide on new source points
        source_tunepoints = _crv('read', OPA, interaction_string = 'NON-NON-NON-Idl')[:, 0]
        sourcepoints = np.linspace(source_tunepoints.min(), source_tunepoints.max(), num_tunepoints)        
        
        #create splines of mixer positions, setpoints vs source color
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 0], k=2, s=1000)
        m1_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 5], k=2, s=1000)
        
        #create new tunepoints array
        new_tunepoints = np.zeros([len(sourcepoints), 20])
        for i in range(len(sourcepoints)):
            sourcepoint = sourcepoints[i]
            new_tunepoints[i, 13] = sourcepoint
            new_tunepoints[i, 0] = setpoint_spline(sourcepoint)
            new_tunepoints[i, 5] = m1_spline(sourcepoint)
        
    elif interaction_string == 'NON-SH-NON-Sig': # - - - - - - - - - - - - - - -
    
        num_tunepoints = 15
    
        #import source curve, decide on new source points
        source_tunepoints = _crv('read', OPA, interaction_string = 'NON-NON-NON-Sig')[:, 0]
        sourcepoints = np.linspace(source_tunepoints.min(), source_tunepoints.max(), num_tunepoints)        
        
        #create splines of mixer positions, setpoints vs source color
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 0], k=2, s=1000)
        m2_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 6], k=2, s=1000)
        
        #create new tunepoints array
        new_tunepoints = np.zeros([len(sourcepoints), 20])
        for i in range(len(sourcepoints)):
            sourcepoint = sourcepoints[i]
            new_tunepoints[i, 13] = sourcepoint
            new_tunepoints[i, 0] = setpoint_spline(sourcepoint)
            new_tunepoints[i, 6] = m2_spline(sourcepoint)
            
    elif interaction_string == 'NON-NON-SF-Idl': # - - - - - - - - - - - - - - -
    
        num_tunepoints = 10
    
        #import source curve, decide on new source points
        source_tunepoints = _crv('read', OPA, interaction_string = 'NON-NON-NON-Idl')[:, 0]
        sourcepoints = np.linspace(source_tunepoints.min(), 2400., num_tunepoints)        
        
        #create splines of mixer positions, setpoints vs source color
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 0], k=2, s=1000)
        m1_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 5], k=2, s=1000)
        
        #create new tunepoints array
        new_tunepoints = np.zeros([len(sourcepoints), 20])
        for i in range(len(sourcepoints)):
            sourcepoint = sourcepoints[i]
            new_tunepoints[i, 13] = sourcepoint
            new_tunepoints[i, 0] = setpoint_spline(sourcepoint)
            new_tunepoints[i, 5] = m1_spline(sourcepoint)
    
    elif interaction_string == 'NON-NON-SF-Sig': # - - - - - - - - - - - - - - -
    
        num_tunepoints = 10
    
        #import source curve, decide on new source points
        source_tunepoints = _crv('read', OPA, interaction_string = 'NON-NON-NON-Sig')[:, 0]
        sourcepoints = np.linspace(source_tunepoints.min(), source_tunepoints.max(), num_tunepoints)        
        
        #create splines of mixer positions, setpoints vs source color
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 0], k=2, s=1000)
        m1_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 5], k=2, s=1000)
        
        #create new tunepoints array
        new_tunepoints = np.zeros([len(sourcepoints), 20])
        for i in range(len(sourcepoints)):
            sourcepoint = sourcepoints[i]
            new_tunepoints[i, 13] = sourcepoint
            new_tunepoints[i, 0] = setpoint_spline(sourcepoint)
            new_tunepoints[i, 5] = m1_spline(sourcepoint)
    
    elif interaction_string == 'NON-SH-SH-Idl': #- - - - - - - - - - - - - - - -
    
        num_tunepoints = 20
    
        #import source curve, decide on new source points
        source_tunepoints = _crv('read', OPA, interaction_string = 'NON-NON-SH-Idl')[:, 0]
        sourcepoints = np.linspace(source_tunepoints.min(), source_tunepoints.max(), num_tunepoints)        
        
        #create splines of mixer positions, setpoints vs source color
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 0], k=2, s=1000)
        m2_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 6], k=2, s=1000)
        
        #create new tunepoints array
        new_tunepoints = np.zeros([len(sourcepoints), 20])
        for i in range(len(sourcepoints)):
            sourcepoint = sourcepoints[i]
            new_tunepoints[i, 13] = sourcepoint
            new_tunepoints[i, 0] = setpoint_spline(sourcepoint)
            new_tunepoints[i, 6] = m2_spline(sourcepoint)
    
    elif interaction_string == 'SH-SH-NON-Sig': #- - - - - - - - - - - - - - - -
    
        num_tunepoints = 20
    
        #import source curve, decide on new source points
        source_tunepoints = _crv('read', OPA, interaction_string = 'NON-SH-NON-Sig')[:, 0]
        sourcepoints = np.linspace(source_tunepoints.min(), source_tunepoints.max(), num_tunepoints)        
        
        #create splines of mixer positions, setpoints vs source color
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 0], k=2, s=1000)
        m3_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 7], k=2, s=1000)
        
        #create new tunepoints array
        new_tunepoints = np.zeros([len(sourcepoints), 20])
        for i in range(len(sourcepoints)):
            sourcepoint = sourcepoints[i]
            new_tunepoints[i, 13] = sourcepoint
            new_tunepoints[i, 0] = setpoint_spline(sourcepoint)
            new_tunepoints[i, 7] = m3_spline(sourcepoint)
    
    elif interaction_string == 'DF1-NON-NON-Sig': #- - - - - - - - - - - - - - -
    
        num_tunepoints = 20
    
        #import source curve, decide on new source points
        source_tunepoints = _crv('read', OPA, interaction_string = 'NON-NON-NON-Sig')[:, 0]
        sourcepoints = np.linspace(source_tunepoints.min(), source_tunepoints.max(), num_tunepoints)        
        
        #create splines of mixer positions, setpoints vs source color
        setpoint_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 0], k=2, s=1000)
        m3_spline = UnivariateSpline(old_tunepoints[:, 13], old_tunepoints[:, 7], k=2, s=1000)
        
        #create new tunepoints array
        new_tunepoints = np.zeros([len(sourcepoints), 20])
        for i in range(len(sourcepoints)):
            sourcepoint = sourcepoints[i]
            new_tunepoints[i, 13] = sourcepoint
            new_tunepoints[i, 0] = setpoint_spline(sourcepoint)
            new_tunepoints[i, 7] = m3_spline(sourcepoint)
        
        pass

    else: #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
        print 'interaction string {} not recognized in create_clean_starting_curve'.format(interaction_string)
        
    #write new curve------------------------------------------------------------

    new_curve_filepath = _crv('write', 
                              OPA, 
                              interaction_string = interaction_string, 
                              curve = new_tunepoints,
                              source_colors = 'new',
                              dummy = True)
                              
    print new_curve_filepath


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def parse_motortune(OPA, 
                    interaction_string, 
                    fit_filepath = None,
                    write_crv = True,
                    output_filepath_seed = None):
                        
    #hack for poor old scipy
    import warnings
    warnings.filterwarnings('ignore')
    
    source_colors = 'old'    
    
    #constants------------------------------------------------------------------

    continue_tuning = False
    next_range = 0.0
    next_step = 0.0

    #import fit data------------------------------------------------------------

    if fit_filepath:
        pass
    else:
        os.chdir(DATA_folderpath)
        try: 
            fit_filepath = max(glob.iglob('*Motor.fit'), key=os.path.getctime)
        except  ValueError:
            fit_filepath = max(glob.iglob('*Topas.fit'), key=os.path.getctime)
    
    fits = _load_fit(fit_filepath)
    
    #parse fits array based on setpoint-----------------------------------------
    
    num_set_points = len(set(fits[:, 0]))
    num_motor_points = len(fits)/num_set_points
    
    fits_grouped = np.zeros([num_set_points, num_motor_points, 20])
    for i in range(num_set_points):
        for j in range(num_motor_points):
            fits_grouped[i][j] = fits[(i*num_motor_points)+j]
            
    #choose best motor positions------------------------------------------------

    if interaction_string == 'preamp': # - - - - - - - - - - - - - - - - - - - -
        
        #store values to preamp_old
        preamp_old = np.zeros([len(fits_grouped), 3])
        scan_range = max(fits_grouped[-1, :, 2]) - min(fits_grouped[-1, :, 2])
        for i in range(len(fits_grouped)):
            preamp_old[i, 0] = fits_grouped[i, 0, 0]
            preamp_old[i, 1] = fits_grouped[i, 0, 1]
            preamp_old[i, 2] = np.average(fits_grouped[i, :, 2])
            
        #create, clean arrays of tune points
        center = fits[:, 8].tolist()
        amplitude = fits[:, 9].tolist()
        c1 = fits[:, 1].tolist()
        d1 = fits[:, 2].tolist()
        to_pop = []
        for i in range(len(center)):
            if (center[i] > 1650) or (center[i] < 1100):
                to_pop.append(i)
            elif np.isnan(amplitude[i]):
                to_pop.append(i)
            elif amplitude[i] < (0.01*np.nanmax(amplitude)):
                to_pop.append(i)
            else:
                pass
        to_pop.reverse()
        for i in range(len(to_pop)):
            center.pop(to_pop[i])
            amplitude.pop(to_pop[i])
            c1.pop(to_pop[i])
            d1.pop(to_pop[i])
        center = np.array(center)
        amplitude = np.array(amplitude)
        c1 = np.array(c1)
        d1 = np.array(d1)  

        #decide on setpoints
        setpoints = np.linspace(1140, 1620, 25)
            
        #create plot environment 
        plt.close()
        background_color = 'grey'
        fig = plt.figure(figsize = [7, 7])
        fig_b = fig.add_subplot(111, axisbg = background_color)
        
        #add tunepoints to plots
        c1_list = fits_grouped[:, :, 1] 
        d1_list = fits_grouped[:, :, 2]
        _get_color(1)
        fig_b.contourf(c1_list, d1_list, fits_grouped[:, :, 8], 200, cmap = rainbow_cmap)
        fig_b.grid()
                
        #get contours of each color setpoint, add contours to plots
        CS = fig_b.contour(c1_list, d1_list, fits_grouped[:, :, 8], 20, colors = 'grey', levels = setpoints)
        fig_b.contourf(c1_list, d1_list, fits_grouped[:, :, 9], 200, cmap = wright_colormap)
        
        #plot old tunepoints, range        
        fig_b.plot(preamp_old[:, 1], preamp_old[:, 2], color = 'k', linewidth = 1)
        fig_b.plot(preamp_old[:, 1], preamp_old[:, 2] + scan_range/2, color = 'k', linewidth = 1, linestyle = 'dotted')
        fig_b.plot(preamp_old[:, 1], preamp_old[:, 2] - scan_range/2, color = 'k', linewidth = 1, linestyle = 'dotted')
        fig_b.plot((preamp_old[0, 1], preamp_old[0, 1]), (preamp_old[0, 2] - scan_range/2,  preamp_old[0, 2] + scan_range/2), color = 'k', linewidth = 1, linestyle = 'dotted')
        fig_b.plot((preamp_old[-1, 1], preamp_old[-1, 1]), (preamp_old[-1, 2] - scan_range/2,  preamp_old[-1, 2] + scan_range/2), color = 'k', linewidth = 1, linestyle = 'dotted')
        fig_b.grid()
        
        #grid data
        c1_grid = np.arange(c1_list.min() - 10., c1_list.max() + 10., 5.)
        d1_grid = np.arange(d1_list.min() - 10., d1_list.max() + 10., 5.)
        grid = np.zeros([len(c1_grid), len(d1_grid), 2])
        for i in range(len(grid[0])): 
            grid[:, i, 0] = c1_grid
        for i in range(len(grid)): 
            grid[i, :, 1] = d1_grid
        amp_grid = scipy.interpolate.griddata((c1, d1), amplitude, grid)
        cen_grid = scipy.interpolate.griddata((c1, d1), center, grid)
        
        #get indicies with centers near setpoints (bin data)
        within = 2
        fits_by_setpoint = []
        for i in range(len(setpoints)):
            fits_temp = []
            max_value = 0.
            for j in range(len(c1_grid)):
                for k in range(len(d1_grid)):
                    if np.isnan(cen_grid[j, k]):
                        pass
                    else:
                        if np.abs(cen_grid[j, k]-setpoints[i]) < within:
                            motor_positions = [c1_grid[j], d1_grid[k]]
                            amplitude = amp_grid[j, k]
                            if amplitude > max_value:
                                max_value = amplitude
                            fits_temp.append([motor_positions, amplitude])
                        else:
                            pass
            to_pop = []
            for j in range(len(fits_temp)):
                if fits_temp[j][1] < 0.25 * max_value:
                    to_pop.append(j)
                else:
                    pass
            to_pop.reverse()
            for j in range(len(to_pop)):
                fits_temp.pop(to_pop[j])
            fits_by_setpoint.append(fits_temp)
            
        #handle the cases were points are not found
        true_setpoints = np.copy(setpoints)
        true_fits_by_setpoint = fits_by_setpoint
        to_pop = []
        false_setpoints = []
        for i in range(len(setpoints)):
            if len(fits_by_setpoint[i]) == 0:
                text = 'no {} nm fits in motortune'.format(setpoints[i])
                informative_text = 'preamp motortune parsing found no points at {} nm. topas_tune will attempt to guess the correct motor points. you may wish to take a look at OPA white light quality.'.format(setpoints[i])
                #_say(text, informative_text)
                false_setpoints.append(setpoints[i])
                to_pop.append(i)
        to_pop.reverse()
        for i in to_pop:
            setpoints = np.delete(setpoints, i)
            fits_by_setpoint.pop(i)
            
        #plot a figure for a particular contour index (for testing purposes)
        '''
        setpoint_index = 23
        plt.figure()
        fig_a.plot(contours[setpoint_index][:,0],contours[setpoint_index][:,1], color = 'k')
        fig_b.plot(contours[setpoint_index][:,0],contours[setpoint_index][:,1], color = 'k')
        for i in range(len(fits_by_setpoint[setpoint_index])):
            c1_point = fits_by_setpoint[setpoint_index][i][0][0]
            d1_point = fits_by_setpoint[setpoint_index][i][0][1]
            amp = fits_by_setpoint[setpoint_index][i][1]
            fig_a.scatter(c1_point, d1_point)
            plt.scatter(c1_point, amp)
            plt.title('c1 vs amp - {} nm'.format(setpoints[setpoint_index]))
        plt.show()
        '''
            
        #fit each setpoint to a gaussian - define chosen c1, d1 along each contour
        preamp_chosen = np.zeros([len(setpoints), 3])
        for i in range(len(setpoints)):
            c1s = np.zeros(len(fits_by_setpoint[i]))
            d1s = np.zeros(len(fits_by_setpoint[i]))
            y = np.zeros(len(fits_by_setpoint[i]))
            for j in range(len(fits_by_setpoint[i])):
                c1s[j] = fits_by_setpoint[i][j][0][0]
                d1s[j] = fits_by_setpoint[i][j][0][1]
                y[j] = fits_by_setpoint[i][j][1]
            
            if len(y) < 4:
                #print i, 'len', len(y)
                #choose by expectation value if you don't have many points (failsafe)
                preamp_chosen[i][0] = setpoints[i]
                preamp_chosen[i][1] = _exp_value(y, c1s)
                preamp_chosen[i][2] = _exp_value(y, d1s)
            else:
                #fit to a guassian
                #c1
                amplitude_guess = max(y)
                center_guess = preamp_old[i, 1]
                sigma_guess = 100.
                p0 = np.array([amplitude_guess, center_guess, sigma_guess])
                try: out_c1 = leastsq(_gauss_residuals, p0, args=(y, c1s))[0]
                except RuntimeWarning: print 'runtime'
                #d1
                amplitude_guess = max(y)
                center_guess = preamp_old[i, 2]
                sigma_guess = 100.
                p0 = np.array([amplitude_guess, center_guess, sigma_guess])
                try: out_d1 = leastsq(_gauss_residuals, p0, args=(y, d1s))[0]
                except RuntimeWarning: print 'runtime'
                #write to preamp_chosen
                preamp_chosen[i][0] = setpoints[i]
                preamp_chosen[i][1] = out_c1[1]
                preamp_chosen[i][2] = out_d1[1]            
            if preamp_chosen[i][1] < c1s.min() or preamp_chosen[i][1] > c1s.max():
                #print i, 'c1', preamp_chosen[i][1], c1.min(), c1.max()
                preamp_chosen[i][0] = setpoints[i]
                preamp_chosen[i][1] = _exp_value(y, c1s)
                preamp_chosen[i][2] = _exp_value(y, d1s)
            elif preamp_chosen[i][2] < d1s.min() or preamp_chosen[i][2] > d1s.max():
                #print i, 'd1', preamp_chosen[i][2], d1.min(), d1.max()
                preamp_chosen[i][0] = setpoints[i]
                preamp_chosen[i][1] = _exp_value(y, c1s)
                preamp_chosen[i][2] = _exp_value(y, d1s)
            else:
                pass

        #plot chosen points
        fig_b.plot(preamp_chosen[:, 1], preamp_chosen[:, 2], color = 'grey', linewidth = 3.5) 
        
        #return setpoints array to default
        setpoints = true_setpoints        
        fits_by_setpoint = true_fits_by_setpoint

        #fit to univariate splines for final smoothness assurance
        c1_spline = UnivariateSpline(preamp_chosen[:, 0], preamp_chosen[:, 1], k=2, s=1000)
        #_say(str(preamp_chosen))
        d1_spline = UnivariateSpline(preamp_chosen[:, 0], preamp_chosen[:, 2], k=2, s=1000)
        preamp_chosen = np.zeros([len(setpoints), 3])  
        for i in range(len(setpoints)):
            preamp_chosen[i][0] = setpoints[i]
            preamp_chosen[i][1] = c1_spline(setpoints[i])
            preamp_chosen[i][2] = d1_spline(setpoints[i])
        false_points = np.zeros([len(false_setpoints), 3])
        for i in range(len(false_setpoints)):
            false_points[i][0] = false_setpoints[i]
            false_points[i][1] = c1_spline(false_setpoints[i])
            false_points[i][2] = d1_spline(false_setpoints[i])
                
        #plot chosen points
        fig_b.plot(preamp_chosen[:, 1], preamp_chosen[:, 2], color = 'k', linewidth = 3.5)
        fig_b.scatter(false_points[:, 1], false_points[:, 2], color = 'red', s = 75, edgecolors = 'grey', zorder = 15) 
        clabel_positions = np.zeros([len(preamp_chosen), 2])
        clabel_positions[:, 0] = preamp_chosen[:, 1] + 50
        clabel_positions[:, 1] = preamp_chosen[:, 2] - 50
        fig_b.clabel(CS, inline=0, fontsize=9, manual=clabel_positions, colors = 'm', fmt = '%1.0f')
        
        #final plot adjustments
        title_text = 'OPA{0} preamp {1}'.format(OPA, time.strftime("  %Y.%m.%d"))
        fig_b.set_xlim(min(preamp_chosen[:, 1])-200, max(preamp_chosen[:, 1])+200)
        fig_b.set_ylim(min(preamp_chosen[:, 2])-200, max(preamp_chosen[:, 2])+200)
        fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
        fig.text(0.5, 0.04, 'c1 (microsteps)', ha='center', va='center')
        fig.text(0.04, 0.5, 'd1 (microsteps)', ha='center', va='center', rotation='vertical')
        fig.subplots_adjust(left = 0.17)
        fig_filepath = fit_filepath.replace('.fit', '.png')
        plt.savefig(fig_filepath, transparent = False, facecolor = 'none')
        plt.close()

        #assemble output curve array
        curve = np.zeros([len(setpoints), 20])
        poweramp_colors = fits_grouped[:, 0, 0]
        poweramp_colors[0] = 1100.0
        c2s = fits_grouped[:, 0, 3] 
        d2s = fits_grouped[:, 0, 4] 
        c2_spline = UnivariateSpline(poweramp_colors, c2s, k=2, s=1000)
        d2_spline = UnivariateSpline(poweramp_colors, d2s, k=2, s=1000)
        for i in range(len(curve)):
            #start with old
            curve[i] = fits_grouped[i][0]
            #replace setpoints
            curve[i, 0] = preamp_chosen[i, 0]
            #replace preamp
            curve[i, 1] = preamp_chosen[i, 1]
            curve[i, 2] = preamp_chosen[i, 2]
            #interpolate poweramp onto new setpoints
            curve[i, 3] = c2_spline(setpoints[i])
            curve[i, 4] = d2_spline(setpoints[i])
            
        #decide if continued tuning is needed, give advice
        continue_tuning = False
        next_range = None
        next_step = None

    
    elif interaction_string == 'poweramp': # - - - - - - - - - - - - - - - - - -

        #discover if c2 or d2 has been scanned
        if np.var(fits_grouped[0, :, 3]) > np.var(fits_grouped[0, :, 4]):
            poweramp_motor = 'c2'
        elif np.var(fits_grouped[0, :, 4]) > 1:
            poweramp_motor = 'd2'
        else:
            print 'fit array incompatible with poweramp'

        #treatment for c2
        if poweramp_motor == 'c2':
            c2_chosen = np.zeros(len(fits_grouped))
            c2_old = np.zeros(len(fits_grouped))
            plt.close()
            fig = plt.figure()
            fig_a = fig.add_subplot(211)
            fig_b = fig.add_subplot(212, sharex=fig_a)
            for i in range(len(fits_grouped)):
                setpoint = fits_grouped[i][0][0]
                c2 = fits_grouped[i, :, 3]
                #create masked signed_mismatch_array
                signed_mismatch = fits_grouped[i, :, 8] - setpoint
                mask = np.zeros(len(signed_mismatch), dtype=bool)
                for j in range(len(signed_mismatch)):
                    if signed_mismatch[j] == np.nan:
                        mask[j] = True
                    elif np.abs(signed_mismatch[j]) > 50:
                        mask[j] = True
                    elif fits_grouped[i, j, 11] < 0.25: #goodness of fit
                        mask[j] = True
                    else:
                        mask[j] = False
                signed_mismatch = np.ma.masked_array(signed_mismatch, mask = mask)
                c2 = np.ma.masked_array(c2, mask = mask)
                #do a linear fit of the mismatch array vs c2
                idx = np.isfinite(signed_mismatch) #why do I have to do this?!?!
                with suppress_stdout_stderr():
                    try: p2, p1, p0 = np.ma.polyfit(signed_mismatch[idx], c2[idx], 2)
                    except: p0 = 0
                    if np.isnan(p0): p0 = 0
                    #add data to c2_chosen
                c2_chosen[i] = int(p0)
                c2_old[i] = np.average(fits_grouped[i, :, 3])
                #plot
                color = _get_color(float(i)/float(len(fits_grouped)))
                fig_b.plot(c2, signed_mismatch, '.-', color=color)
            #polyfit values
            x_poly = []
            y_poly = []
            for i in range(len(c2_chosen)):
                if c2_chosen[i] == 0: pass
                elif i in [0, 1, len(c2_chosen)-1, len(c2_chosen)-2]: pass  #throw away points on edges
                else:
                    x_poly.append(fits_grouped[i, 0, 0])
                    y_poly.append(c2_chosen[i])
            with suppress_stdout_stderr():
                fit_params = np.ma.polyfit(x_poly, y_poly, 3)
            for i in range(len(c2_chosen)):
                setpoint = fits_grouped[i][0][0]
                c2_chosen[i] = int(np.polynomial.polynomial.polyval(setpoint, fit_params[::-1]))        
            #put points on plots
            for i in range(len(c2_chosen)):
                color = _get_color(float(i)/float(len(fits_grouped)))
                fig_a.scatter(c2_old[i], fits_grouped[i, 0, 0], color=color, marker = 'o')
                fig_a.scatter(c2_chosen[i], fits_grouped[i, 0, 0], color=color, marker = 'x', s=100)
                fig_b.plot(c2, signed_mismatch, '.-', color=color)
                fig_b.scatter(c2_chosen[i], 0, marker = 'x', color=color, s=100)
            #finish making plots
            fig_b.set_ylim(-50, 50)
            setp(fig_a.get_xticklabels(), visible=False)
            fig_a.set_ylabel('setpoint (nm)')
            fig_b.set_ylabel('mismatch (nm)')
            fig_b.set_xlabel('crystal two (microsteps)')
            fig_a.grid()
            fig_b.grid()
            title_text = 'OPA{0} Crystal 2 {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
            fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
            plot_filename = fit_filepath.replace('.fit', '.png')
            plt.savefig(plot_filename, transparent = True)
            plt.close()
            #assemble output curve array
            curve = np.zeros([len(fits_grouped), 20])
            for i in range(len(curve)):
                curve[i] = fits_grouped[i][0]
                curve[i, 3] = c2_chosen[i]
            #decide if continued tuning is needed, give advice
            continue_tuning = False
            next_range = None
            next_step = None
            
        #treatment for d2
        elif poweramp_motor == 'd2':
            d2_chosen = np.zeros(len(fits_grouped))
            d2_old = np.zeros(len(fits_grouped))
            plt.close()
            fig = plt.figure()
            fig_a = fig.add_subplot(211)
            fig_b = fig.add_subplot(212, sharex=fig_a)
            for i in range(len(fits_grouped)):
                d2 = fits_grouped[i, :, 4]
                d2_old[i] = np.average(fits_grouped[i, :, 4])
                #create masked amplitude array
                amplitude = fits_grouped[i, :, 9]
                mask = np.zeros(len(amplitude), dtype=bool)
                for j in range(len(amplitude)):
                    if amplitude[j] == np.nan:
                        mask[j] = True
                    elif np.abs(amplitude[j]) > 50:
                        mask[j] = True
                    elif fits_grouped[i, j, 11] < 0.25: #goodness of fit
                        mask[j] = True
                    else:
                        mask[j] = False
                amplitude = np.ma.masked_array(amplitude, mask = mask)
                d2 = np.ma.masked_array(d2, mask = mask)
                #find best d2 by expectation value
                d2_chosen[i] = _exp_value(amplitude, d2)
            #if fail, take old value
            for i in range(len(d2_chosen)):
                if d2_chosen[i] == 0: 
                    d2_chosen[i] = d2_old[i]
            #ensure smoothness with spline
            d2_spline = UnivariateSpline(fits_grouped[:, 0, 0], d2_chosen, k=2, s=1000)      
            d2_rough = np.copy(d2_chosen)
            for i in range(len(d2_chosen)):
                d2_chosen[i] = d2_spline(fits_grouped[i, 0, 0])
            #make plot
            fig_a.plot(fits_grouped[:, 0, 0], d2_old, color = 'k', linewidth = 1)
            fig_a.plot(fits_grouped[:, 0, 0], d2_chosen, color = 'k', linewidth = 4)
            fig_b.plot(fits_grouped[:, 0, 0], d2_rough - d2_old, color = 'grey', linewidth = 4)
            fig_b.plot(fits_grouped[:, 0 ,0], d2_chosen - d2_old, color = 'k', linewidth = 4)
            y = fits_grouped[0, :, 4] - d2_old[0]
            y_step = (y.max() - y.min())/len(y)           
            x = fits_grouped[:, 0, 0]
            x_step = (x.max() - x.min())/len(x)
            z = fits_grouped[:, :, 9].T
            if fits_grouped[0, 0, 4] > fits_grouped[0, -1, 4]:
                z = np.flipud(z) #must be acending array along y axis
            fig_b.imshow(z,
                         origin = 'lower',
                         cmap = wright_colormap,
                         interpolation = 'nearest',
                         aspect = 'auto',
                         extent=[x.min() - x_step/2., 
                                 x.max() + x_step/2., 
                                 y.min() - y_step/2., 
                                 y.max() + y_step/2.])
            setp(fig_a.get_xticklabels(), visible=False)
            #fig_b.set_ylim(-150, 150)
            fig_b.set_xlabel('setpoint (nm)')
            fig_a.set_ylabel('d2 (microsteps)')
            fig_b.set_ylabel('delta d2 (microsteps)')
            fig_a.grid()
            fig_b.grid()
            title_text = 'OPA{0} Delay 2 {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
            fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
            plot_filename = fit_filepath.replace('.fit', '.png')
            plt.savefig(plot_filename, transparent = True)
            plt.close()
            #assemble output curve array
            curve = np.zeros([len(fits_grouped), 20])
            for i in range(len(curve)):
                curve[i] = fits_grouped[i][0]
                curve[i, 4] = d2_chosen[i]
            #decide if continued tuning is needed, give advice
            continue_tuning = False
            next_range = None
            next_step = None
            
    elif interaction_string == 'NON-NON-NON-Idl': #- - - - - - - - - - - - - - -
    
        print 'idler not supported in parse_motortune'
        
    elif interaction_string == 'NON-NON-SH-Idl': # - - - - - - - - - - - - - - -
    
        #ready plot environment
        plt.close()
        fig = plt.figure()
        fig_a = fig.add_subplot(211)
        fig_b = fig.add_subplot(212, sharex=fig_a)
        
        #choose best points        
        m1_chosen = np.zeros(len(fits_grouped))
        m1_old = np.zeros(len(fits_grouped))
        for i in range(len(fits_grouped)):
            m1 = fits_grouped[i, :, 5]
            m1_old[i] = np.average(fits_grouped[i, :, 5])
            #create masked amplitude array
            amplitude = fits_grouped[i, :, 9]
            max_amplitude = np.nanmax(amplitude)
            mask = np.zeros(len(amplitude), dtype=bool)
            for j in range(len(amplitude)):
                if amplitude[j] == np.nan:
                    mask[j] = True
                elif np.abs(amplitude[j]) > 50:
                    mask[j] = True
                elif amplitude[j] < max_amplitude*0.5:
                    mask[j] = True
                else:
                    mask[j] = False
            amplitude = np.ma.masked_array(amplitude, mask = mask)
            m1 = np.ma.masked_array(m1, mask = mask)
            #find best m1 by expectation value
            m1_chosen[i] = _exp_value(amplitude, m1)
        
        #ensure no problems if _exp_value fails
        for i in range(len(m1_chosen)):
            if m1_chosen[i] == 0:
                m1_chosen[i] = m1_old[i]

        #get color for each chosen point
        center_chosen = np.zeros(len(fits_grouped))
        for i in range(len(m1_chosen)):
            m1 = fits_grouped[i, :, 6]
            index = (np.abs(m1 - m1_chosen[i])).argmin()
            center_chosen[i] = fits_grouped[i, index, 8]
        
        #ensure no problems if spline fails
        for i in range(len(center_chosen)):
            if np.isnan(center_chosen[i]):
                center_chosen[i] = fits_grouped[i, 0 ,0]

        #ensure smoothness with spline
        m1_spline = UnivariateSpline(fits_grouped[:, 0, 0], m1_chosen, k=2, s=10000)
        center_spline = UnivariateSpline(fits_grouped[:, 0, 0], center_chosen, k=2, s=10000)
        for i in range(len(fits_grouped[:, 0, 0])):
            setpoint = fits_grouped[i, 0, 0]
            m1_chosen[i] == m1_spline(setpoint)
            center_chosen[i] == center_spline(setpoint)
                
        #make plot
        fig_a.plot(fits_grouped[:, 0, 0], m1_old, color = 'k', linewidth = 1)
        fig_a.plot(center_chosen, m1_chosen, color = 'k', linewidth = 4)
        fig_b.plot(center_chosen, m1_chosen - m1_old, color = 'k', linewidth = 4)
        y = fits_grouped[0, :, 5] - m1_old[0]
        y_step = (y.max() - y.min())/len(y)           
        x = center_chosen
        x_step = (x.max() - x.min())/len(x)
        z = fits_grouped[:, :, 9].T
        if fits_grouped[0, 0, 5] > fits_grouped[0, -1, 5]:
            z = np.flipud(z) #must be acending array along y axis
        fig_b.imshow(z,
                     origin = 'lower',
                     cmap = wright_colormap,
                     interpolation = 'nearest',
                     aspect = 'auto',
                     extent=[x.min() - x_step/2., 
                             x.max() + x_step/2., 
                             y.min() - y_step/2., 
                             y.max() + y_step/2.])
        setp(fig_a.get_xticklabels(), visible=False)
        fig_b.set_xlabel('setpoint (nm)')
        fig_a.set_ylabel('m1 (microsteps)')
        fig_b.set_ylabel('delta m1 (microsteps)')
        fig_a.grid()
        fig_b.grid()
        title_text = 'OPA{0} Mixer 1 (SHI) {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
        fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
        plot_filename = fit_filepath.replace('.fit', '.png')
        plt.savefig(plot_filename, transparent = True)
        plt.close()
        
        #assemble output curve array
        curve = np.zeros([len(fits_grouped), 20])
        for i in range(len(curve)):
            curve[i] = fits_grouped[i][0]
            curve[i, 0] = center_chosen[i]
            curve[i, 6] = m1_chosen[i]
            
        #decide if continued tuning is needed, give advice
        continue_tuning = False
        next_range = None
        next_step = None
            
    
    elif interaction_string == 'NON-SH-NON-Sig': # - - - - - - - - - - - - - - -
    
        #ready plot environment
        plt.close()
        fig = plt.figure()
        fig_a = fig.add_subplot(211)
        fig_b = fig.add_subplot(212, sharex=fig_a)
        
        #choose best points        
        m2_chosen = np.zeros(len(fits_grouped))
        m2_old = np.zeros(len(fits_grouped))
        for i in range(len(fits_grouped)):
            m2 = fits_grouped[i, :, 6]
            m2_old[i] = np.average(fits_grouped[i, :, 6])
            #create masked amplitude array
            amplitude = fits_grouped[i, :, 9]
            max_amplitude = np.nanmax(amplitude)
            mask = np.zeros(len(amplitude), dtype=bool)
            for j in range(len(amplitude)):
                if amplitude[j] == np.nan:
                    mask[j] = True
                elif np.abs(amplitude[j]) > 50:
                    mask[j] = True
                elif amplitude[j] < max_amplitude*0.5:
                    mask[j] = True
                else:
                    mask[j] = False
            amplitude = np.ma.masked_array(amplitude, mask = mask)
            m2 = np.ma.masked_array(m2, mask = mask)
            #find best m1 by expectation value
            m2_chosen[i] = _exp_value(amplitude, m2)
        
        #ensure no problems if _exp_value fails
        for i in range(len(m2_chosen)):
            if m2_chosen[i] == 0:
                m2_chosen[i] = m2_old[i]

        #get color for each chosen point
        center_chosen = np.zeros(len(fits_grouped))
        for i in range(len(m2_chosen)):
            m2 = fits_grouped[i, :, 6]
            index = (np.abs(m2 - m2_chosen[i])).argmin()
            center_chosen[i] = fits_grouped[i, index, 8]
        
        #ensure no problems if spline fails
        for i in range(len(center_chosen)):
            if np.isnan(center_chosen[i]):
                center_chosen[i] = fits_grouped[i, 0 ,0]

        #ensure smoothness with spline
        m2_spline = UnivariateSpline(fits_grouped[:, 0, 0], m2_chosen, k=2, s=10000)
        center_spline = UnivariateSpline(fits_grouped[:, 0, 0], center_chosen, k=2, s=10000)
        for i in range(len(fits_grouped[:, 0, 0])):
            setpoint = fits_grouped[i, 0, 0]
            m2_chosen[i] == m2_spline(setpoint)
            center_chosen[i] == center_spline(setpoint)
                
        #make plot
        fig_a.plot(fits_grouped[:, 0, 0], m2_old, color = 'k', linewidth = 1)
        fig_a.plot(center_chosen, m2_chosen, color = 'k', linewidth = 4)
        fig_b.plot(center_chosen, m2_chosen - m2_old, color = 'k', linewidth = 4)
        y = fits_grouped[0, :, 6] - m2_old[0]
        y_step = (y.max() - y.min())/len(y)           
        x = center_chosen
        x_step = (x.max() - x.min())/len(x)
        z = fits_grouped[:, :, 9].T
        if fits_grouped[0, 0, 6] > fits_grouped[0, -1, 6]:
            z = np.flipud(z) #must be acending array along y axis
        fig_b.imshow(z,
                     origin = 'lower',
                     cmap = wright_colormap,
                     interpolation = 'nearest',
                     aspect = 'auto',
                     extent=[x.min() - x_step/2., 
                             x.max() + x_step/2., 
                             y.min() - y_step/2., 
                             y.max() + y_step/2.])
        setp(fig_a.get_xticklabels(), visible=False)
        fig_b.set_xlabel('setpoint (nm)')
        fig_a.set_ylabel('m2 (microsteps)')
        fig_b.set_ylabel('delta m2 (microsteps)')
        fig_a.grid()
        fig_b.grid()
        title_text = 'OPA{0} Mixer 2 (SHS) {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
        fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
        plot_filename = fit_filepath.replace('.fit', '.png')
        plt.savefig(plot_filename, transparent = True)
        plt.close()
        
        #assemble output curve array
        curve = np.zeros([len(fits_grouped), 20])
        for i in range(len(curve)):
            curve[i] = fits_grouped[i][0]
            curve[i, 0] = center_chosen[i]
            curve[i, 6] = m2_chosen[i]
            
        #decide if continued tuning is needed, give advice
        continue_tuning = False
        next_range = None
        next_step = None
            
    elif interaction_string == 'NON-NON-SF-Idl': # - - - - - - - - - - - - - - -

        #ready plot environment
        plt.close()
        fig = plt.figure()
        fig_a = fig.add_subplot(211)
        fig_b = fig.add_subplot(212, sharex=fig_a)
        
        #choose best points        
        m1_chosen = np.zeros(len(fits_grouped))
        m1_old = np.zeros(len(fits_grouped))
        for i in range(len(fits_grouped)):
            m1 = fits_grouped[i, :, 5]
            m1_old[i] = np.average(fits_grouped[i, :, 5])
            #create masked amplitude array
            amplitude = fits_grouped[i, :, 9]
            max_amplitude = np.nanmax(amplitude)
            mask = np.zeros(len(amplitude), dtype=bool)
            for j in range(len(amplitude)):
                if amplitude[j] == np.nan:
                    mask[j] = True
                elif np.abs(amplitude[j]) > 50:
                    mask[j] = True
                elif amplitude[j] < max_amplitude*0.5:
                    mask[j] = True
                else:
                    mask[j] = False
            amplitude = np.ma.masked_array(amplitude, mask = mask)
            m1 = np.ma.masked_array(m1, mask = mask)
            #find best m1 by expectation value
            m1_chosen[i] = _exp_value(amplitude, m1)

        #ensure no problems if _exp_value fails
        for i in range(len(m1_chosen)):
            if m1_chosen[i] in [0, np.nan]:
                m1_chosen[i] = m1_old[i]
                
        #get color for each chosen point
        center_chosen = np.zeros(len(fits_grouped))
        for i in range(len(m1_chosen)):
            m1 = fits_grouped[i, :, 6]
            index = (np.abs(m1 - m1_chosen[i])).argmin()
            center_chosen[i] = fits_grouped[i, index, 8]

        #ensure no problems if chosen point fails
        for i in range(len(center_chosen)):
            if np.isnan(center_chosen[i]):
                center_chosen[i] = fits_grouped[i, 0 ,0]

        #ensure source colors are correct
        source_colors = 'new'
        old_crv = _crv('read', interaction_string = 'NON-NON-SF-Idl', OPA = OPA)
        old_sources=  old_crv[:, 13]
        
        #monotonic center_chosenz
        m1_chosen = m1_chosen[center_chosen.argsort()]
        center_chosen.sort()

        #ensure smoothness with spline
        #print len(center_chosen), len(m1_chosen), len(old_sources)
        m1_spline = UnivariateSpline(center_chosen, m1_chosen, k=2, s=10000)
        source_spline = UnivariateSpline(center_chosen, old_sources, k=2, s=10000)
        setpoints = np.linspace(min(center_chosen), 610., 10)
        m1_chosen = np.zeros(len(setpoints))
        center_chosen = np.zeros(len(setpoints))
        source_chosen = np.zeros(len(setpoints))
        for i in range(len(setpoints)):
            m1_chosen[i] = m1_spline(setpoints[i])
            source_chosen[i] = source_spline(setpoints[i]) 
            center_chosen[i] = setpoints[i]
        
        #source cannot be out of range
        for i in range(len(source_chosen)):
            if source_chosen[i] > max(old_sources):
                source_chosen[i] = max(old_sources)
            if source_chosen[i] < min(old_sources):
                source_chosen[i] = min(old_sources)

        #make plot
        fig_a.plot(fits_grouped[:, 0, 0], m1_old, color = 'k', linewidth = 1)
        fig_a.plot(center_chosen, m1_chosen, color = 'k', linewidth = 4)
        fig_b.plot(center_chosen, m1_chosen - m1_old, color = 'k', linewidth = 4)
        y = fits_grouped[0, :, 5] - m1_old[0]
        y_step = (y.max() - y.min())/len(y)           
        x = center_chosen
        x_step = (x.max() - x.min())/len(x)
        z = fits_grouped[:, :, 9].T
        if fits_grouped[0, 0, 5] > fits_grouped[0, -1, 5]:
            z = np.flipud(z) #must be acending array along y axis
        fig_b.imshow(z,
                     origin = 'lower',
                     cmap = wright_colormap,
                     interpolation = 'nearest',
                     aspect = 'auto',
                     extent=[x.min() - x_step/2., 
                             x.max() + x_step/2., 
                             y.min() - y_step/2., 
                             y.max() + y_step/2.])
        setp(fig_a.get_xticklabels(), visible=False)
        fig_b.set_xlabel('setpoint (nm)')
        fig_a.set_ylabel('m1 (microsteps)')
        fig_b.set_ylabel('delta m1 (microsteps)')
        fig_a.grid()
        fig_b.grid()
        title_text = 'OPA{0} Mixer 1 (SFI) {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
        fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
        plot_filename = fit_filepath.replace('.fit', '.png')
        plt.savefig(plot_filename, transparent = True)
        plt.close()
        
        #assemble output curve array
        curve = np.zeros([len(setpoints), 20])
        for i in range(len(curve)):
            curve[i] = fits_grouped[i][0]
            curve[i, 0] = center_chosen[i]
            curve[i, 5] = m1_chosen[i]
            curve[i, 13]= source_chosen[i]
            
        #decide if continued tuning is needed, give advice
        continue_tuning = False
        next_range = None
        next_step = None
        
    elif interaction_string == 'NON-NON-SF-Sig': # - - - - - - - - - - - - - - -
    
        #ready plot environment
        plt.close()
        fig = plt.figure()
        fig_a = fig.add_subplot(211)
        fig_b = fig.add_subplot(212, sharex=fig_a)
        
        #choose best points        
        m1_chosen = np.zeros(len(fits_grouped))
        m1_old = np.zeros(len(fits_grouped))
        for i in range(len(fits_grouped)):
            m1 = fits_grouped[i, :, 5]
            m1_old[i] = np.average(fits_grouped[i, :, 5])
            #create masked amplitude array
            amplitude = fits_grouped[i, :, 9]
            max_amplitude = np.nanmax(amplitude)
            mask = np.zeros(len(amplitude), dtype=bool)
            for j in range(len(amplitude)):
                if amplitude[j] == np.nan:
                    mask[j] = True
                elif np.abs(amplitude[j]) > 50:
                    mask[j] = True
                elif amplitude[j] < max_amplitude*0.5:
                    mask[j] = True
                else:
                    mask[j] = False
            amplitude = np.ma.masked_array(amplitude, mask = mask)
            m1 = np.ma.masked_array(m1, mask = mask)
            #find best m1 by expectation value
            m1_chosen[i] = _exp_value(amplitude, m1)
        #if fail, take old value
        for i in range(len(m1_chosen)):
            if m1_chosen[i] == 0: 
                m1_chosen[i] = m1_old[i]
            
        #ensure smoothness with spline
        m1_spline = UnivariateSpline(fits_grouped[:, 0, 0], m1_chosen, k=2, s=10000)
        for i in range(len(fits_grouped[:, 0, 0])):
            setpoint = fits_grouped[i, 0, 0]
            m1_chosen[i] == m1_spline(setpoint)
                
        #make plot
        fig_a.plot(fits_grouped[:, 0, 0], m1_old, color = 'k', linewidth = 1)
        fig_a.plot(fits_grouped[:, 0, 0], m1_chosen, color = 'k', linewidth = 4)
        fig_b.plot(fits_grouped[:, 0 ,0], m1_chosen - m1_old, color = 'k', linewidth = 4)
        y = fits_grouped[0, :, 5] - m1_old[0]
        y_step = (y.max() - y.min())/len(y)           
        x = fits_grouped[:, 0, 0]
        x_step = (x.max() - x.min())/len(x)
        z = fits_grouped[:, :, 9].T
        fig_b.imshow(z,
                     origin = 'lower',
                     cmap = wright_colormap,
                     interpolation = 'nearest',
                     aspect = 'auto',
                     extent=[x.min() - x_step/2., 
                             x.max() + x_step/2., 
                             y.min() - y_step/2., 
                             y.max() + y_step/2.])
        setp(fig_a.get_xticklabels(), visible=False)
        fig_b.set_xlabel('setpoint (nm)')
        fig_a.set_ylabel('m1 (microsteps)')
        fig_b.set_ylabel('delta m1 (microsteps)')
        fig_a.grid()
        fig_b.grid()
        title_text = 'OPA{0} Mixer 1 (SFS) {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
        fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
        plot_filename = fit_filepath.replace('.fit', '.png')
        plt.savefig(plot_filename, transparent = True)
        plt.close()
        
        #assemble output curve array
        curve = np.zeros([len(fits_grouped), 20])
        for i in range(len(curve)):
            curve[i] = fits_grouped[i][0]
            curve[i, 5] = m1_chosen[i]
            
        #decide if continued tuning is needed, give advice
        continue_tuning = False
        next_range = None
        next_step = None
    
    elif interaction_string == 'NON-SH-SH-Idl': #- - - - - - - - - - - - - - - -
        
        #ready plot environment
        plt.close()
        fig = plt.figure()
        fig_a = fig.add_subplot(211)
        fig_b = fig.add_subplot(212, sharex=fig_a)
        
        #choose best points        
        m2_chosen = np.zeros(len(fits_grouped))
        m2_old = np.zeros(len(fits_grouped))
        for i in range(len(fits_grouped)):
            m2 = fits_grouped[i, :, 6]
            m2_old[i] = np.average(fits_grouped[i, :, 6])
            #create masked amplitude array
            amplitude = fits_grouped[i, :, 9]
            max_amplitude = np.nanmax(amplitude)
            mask = np.zeros(len(amplitude), dtype=bool)
            for j in range(len(amplitude)):
                if amplitude[j] == np.nan:
                    mask[j] = True
                elif np.abs(amplitude[j]) > 50:
                    mask[j] = True
                elif amplitude[j] < max_amplitude*0.5:
                    mask[j] = True
                else:
                    mask[j] = False
            amplitude = np.ma.masked_array(amplitude, mask = mask)
            m2 = np.ma.masked_array(m2, mask = mask)
            print amplitude
            #find best m1 by expectation value
            m2_chosen[i] = _exp_value(amplitude, m2)
            
        #ensure smoothness with spline
        m2_spline = UnivariateSpline(fits_grouped[:, 0, 0], m2_chosen, k=2, s=10000)
        for i in range(len(fits_grouped[:, 0, 0])):
            setpoint = fits_grouped[i, 0, 0]
            m2_chosen[i] == m2_spline(setpoint)
                
        #make plot
        fig_a.plot(fits_grouped[:, 0, 0], m2_old, color = 'k', linewidth = 1)
        fig_a.plot(fits_grouped[:, 0, 0], m2_chosen, color = 'k', linewidth = 4)
        fig_b.plot(fits_grouped[:, 0 ,0], m2_chosen - m2_old, color = 'k', linewidth = 4)
        y = fits_grouped[0, :, 6] - m2_old[0]
        y_step = (y.max() - y.min())/len(y)           
        x = fits_grouped[:, 0, 0]
        x_step = (x.max() - x.min())/len(x)
        z = fits_grouped[:, :, 9].T
        fig_b.imshow(z,
                     origin = 'lower',
                     cmap = wright_colormap,
                     interpolation = 'nearest',
                     aspect = 'auto',
                     extent=[x.min() - x_step/2., 
                             x.max() + x_step/2., 
                             y.min() - y_step/2., 
                             y.max() + y_step/2.])
        setp(fig_a.get_xticklabels(), visible=False)
        fig_b.set_xlabel('setpoint (nm)')
        fig_a.set_ylabel('m1 (microsteps)')
        fig_b.set_ylabel('delta m1 (microsteps)')
        fig_a.grid()
        fig_b.grid()
        title_text = 'OPA{0} Mixer 2 (4HI) {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
        fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
        plot_filename = fit_filepath.replace('.fit', '.png')
        plt.savefig(plot_filename, transparent = True)
        plt.close()
        
        #assemble output curve array
        curve = np.zeros([len(fits_grouped), 20])
        for i in range(len(curve)):
            curve[i] = fits_grouped[i][0]
            curve[i, 6] = m2_chosen[i]
            
        #decide if continued tuning is needed, give advice
        continue_tuning = False
        next_range = None
        next_step = None
    
    elif interaction_string == 'SH-SH-NON-Sig': #- - - - - - - - - - - - - - - -

        #ready plot environment
        plt.close()
        fig = plt.figure()
        fig_a = fig.add_subplot(211)
        fig_b = fig.add_subplot(212, sharex=fig_a)
        
        #choose best points        
        m3_chosen = np.zeros(len(fits_grouped))
        m3_old = np.zeros(len(fits_grouped))
        for i in range(len(fits_grouped)):
            m3 = fits_grouped[i, :, 7]
            m3_old[i] = np.average(fits_grouped[i, :, 7])
            #create masked amplitude array
            amplitude = fits_grouped[i, :, 9]
            max_amplitude = np.nanmax(amplitude)
            mask = np.zeros(len(amplitude), dtype=bool)
            for j in range(len(amplitude)):
                if amplitude[j] == np.nan:
                    mask[j] = True
                elif np.abs(amplitude[j]) > 50:
                    mask[j] = True
                elif amplitude[j] < max_amplitude*0.5:
                    mask[j] = True
                else:
                    mask[j] = False
            amplitude = np.ma.masked_array(amplitude, mask = mask)
            m3 = np.ma.masked_array(m3, mask = mask)
            print amplitude
            #find best m1 by expectation value
            m3_chosen[i] = _exp_value(amplitude, m3)
            
        #ensure smoothness with spline
        m3_spline = UnivariateSpline(fits_grouped[:, 0, 0], m3_chosen, k=2, s=10000)
        for i in range(len(fits_grouped[:, 0, 0])):
            setpoint = fits_grouped[i, 0, 0]
            m3_chosen[i] == m3_spline(setpoint)
                
        #make plot
        fig_a.plot(fits_grouped[:, 0, 0], m3_old, color = 'k', linewidth = 1)
        fig_a.plot(fits_grouped[:, 0, 0], m3_chosen, color = 'k', linewidth = 4)
        fig_b.plot(fits_grouped[:, 0 ,0], m3_chosen - m3_old, color = 'k', linewidth = 4)
        y = fits_grouped[0, :, 7] - m3_old[0]
        y_step = (y.max() - y.min())/len(y)           
        x = fits_grouped[:, 0, 0]
        x_step = (x.max() - x.min())/len(x)
        z = fits_grouped[:, :, 9].T
        fig_b.imshow(z,
                     origin = 'lower',
                     cmap = wright_colormap,
                     interpolation = 'nearest',
                     aspect = 'auto',
                     extent=[x.min() - x_step/2., 
                             x.max() + x_step/2., 
                             y.min() - y_step/2., 
                             y.max() + y_step/2.])
        setp(fig_a.get_xticklabels(), visible=False)
        fig_b.set_xlabel('setpoint (nm)')
        fig_a.set_ylabel('m1 (microsteps)')
        fig_b.set_ylabel('delta m1 (microsteps)')
        fig_a.grid()
        fig_b.grid()
        title_text = 'OPA{0} Mixer 3 (4HS) {1}'.format(OPA, time.strftime("  %Y.%m.%d   %H:%M"))
        fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
        plot_filename = fit_filepath.replace('.fit', '.png')
        plt.savefig(plot_filename, transparent = True)
        plt.close()
        
        #assemble output curve array
        curve = np.zeros([len(fits_grouped), 20])
        for i in range(len(curve)):
            curve[i] = fits_grouped[i][0]
            curve[i, 7] = m3_chosen[i]
            
        #decide if continued tuning is needed, give advice
        continue_tuning = False
        next_range = None
        next_step = None
        
    elif interaction_string == 'DF1-NON-NON-Sig': #- - - - - - - - - - - - - - -
        
        print 'DFG not supported in parse_motortune'
    
    else: #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        print 'interaction_string not recognized in parse_motortune'

    #create new curve-----------------------------------------------------------    
    
    new_crv_filepath = None
    if write_crv:
        new_crv_filepath = _crv('write',
                                OPA = OPA,
                                interaction_string = interaction_string,
                                curve = curve,
                                output_filepath_seed = output_filepath_seed,
                                source_colors = source_colors)

    #print result---------------------------------------------------------------

    print new_crv_filepath
    #print continue_tuning
    #print next_range
    #print next_step
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def parse_tune_test(OPA, 
                    interaction_string, 
                    dat_filepath = None,
                    write_crv = True,
                    array = True,
                    force_bounds = False):    
    
    #import dat-----------------------------------------------------------------
    
    if dat_filepath:
        pass
    else:
        os.chdir(DATA_folderpath)
        dat_filepath = max(glob.iglob('*.dat'), key=os.path.getctime)
    
    if array:
        yvar = 'wa'
        zvar = 'mc'
    else:
        yvar = 'wm'
        zvar = 'ai0'
    
    zi, xi, yi = _load_dat(dat_filepath, 'w{}'.format(OPA), yvar, tune_test = True, zvar = zvar)
    
    #get fits-------------------------------------------------------------------
    
    
    
    
    
    #plot-----------------------------------------------------------------------
    
    interaction_text = ''
    
    fig = plt.figure()
    p1 = fig.add_subplot(111)
    p1.contourf(xi, yi, zi, 200, cmap = wright_colormap)
    p1.grid()
    p1.set_xlabel('setpoint (nm)')
    p1.set_ylabel('detuning (nm)')
    title_text = 'OPA{0} {1} tune test {2}'.format(OPA, interaction_text, time.strftime("  %Y.%m.%d   %H:%M"))
    fig.text(0.5, 0.97, title_text, ha='center', va='center', fontsize = 20)
    
    output_filepath = dat_filepath.replace('.dat', '.png')
    plt.savefig(output_filepath)
    plt.close()
    
    #write crv if desired-------------------------------------------------------
    
    if write_crv:
        
        pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _crv(interaction,
         OPA,
         filepath = None,
         interaction_string = None,
         curve = None, 
         source_colors = 'old',
         dummy = False,
         output_filepath_seed = None):
    
    if interaction == 'read': #-------------------------------------------------
    
        #currently hardcoded to import base curves...

        #initialize objects- - - - - - - - - - - - - - - - - - - - - - - - - - -
    
        mcf = _get_motor_conversion_factors(OPA)

        #get filepath if not provided- - - - - - - - - - - - - - - - - - - - - -

        if filepath:
            pass
        else:
            #use currently loaded curve
            if interaction_string in ['preamp', 'poweramp', 'NON-NON-NON-Sig', 'NON-NON-NON-Idl']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 1') #base
            elif interaction_string in ['NON-NON-SF-Sig', 'NON-NON-SH-Idl', 'NON-NON-SF-Idl']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 2') #mixer 1
            elif interaction_string in ['NON-SH-NON-Sig', 'NON-SH-SH-Idl']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 3') #mixer 2
            elif interaction_string in ['SH-SH-NON-Sig', 'DF1-NON-NON-Sig']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 4') #mixer 3
            else:
                print 'error in _crv'

        #import array- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if interaction_string in ['preamp', 'poweramp', 'NON-NON-NON-Sig', 'NON-NON-NON-Idl']:
            lines_between = 7
        elif interaction_string == 'DF1-NON-NON-Sig':
            lines_between = 9
        else:
            lines_between = 10

        crv = open(filepath, 'r')
        crv_lines = crv.readlines()

        #collect a friendly array of the params for each interactions
        num_points = []
        for i in range(len(crv_lines)):
            if 'NON' in crv_lines[i]:
                num_points.append([i+lines_between, crv_lines[i].replace('\n', ''), int(crv_lines[i+lines_between])])
        num_points = np.array(num_points)
        
        #pick out which interaction string you want to pay attention to
        if interaction_string in ['preamp', 'poweramp']:
            num_points = num_points[0]
        else:
            index = np.where(num_points[:, 1] == interaction_string)[0][0]
            num_points = num_points[index]
        
        row = int(num_points[0]) + 1
        num_tune_points = int(num_points[2])
        points = np.zeros([num_tune_points, 20])
        points[:] = np.nan

        #fill into internal format - - - - - - - - - - - - - - - - - - - - - - -        

        if interaction_string in ['preamp', 'poweramp', 'NON-NON-NON-Sig', 'NON-NON-NON-Idl']:        
            #base curves
        
            for i in range(num_tune_points):
                line =  re.split(r'\t+', crv_lines[row])
                points[i, 0] = float(line[1])
                points[i, 1] = int((float(line[3]) - mcf['m0'][1])*mcf['m0'][0])
                points[i, 2] = int((float(line[4]) - mcf['m1'][1])*mcf['m1'][0])
                points[i, 3] = int((float(line[5]) - mcf['m2'][1])*mcf['m2'][0])
                points[i, 4] = int((float(line[6]) - mcf['m3'][1])*mcf['m3'][0])
                points[i, 13] = float(line[0])
                row = row+1
        
        elif interaction_string in ['NON-NON-SH-Idl', 'NON-NON-SF-Idl', 'NON-NON-SF-Sig']:
            #mixer 1 curves
            
            for i in range(num_tune_points):
                line =  re.split(r'\t+', crv_lines[row])
                points[i, 0] = float(line[1])
                points[i, 5] = int((float(line[3]) - mcf['m4'][1])*mcf['m4'][0])
                points[i, 13] = float(line[0])
                row = row+1
                
        elif interaction_string in ['NON-SH-NON-Sig', 'NON-SH-SH-Idl']:
            #mixer 2 curves
            
            for i in range(num_tune_points):
                line =  re.split(r'\t+', crv_lines[row])
                points[i, 0] = float(line[1])
                points[i, 6] = int((float(line[3]) - mcf['m5'][1])*mcf['m5'][0])
                points[i, 13] = float(line[0])
                row = row+1
                
        elif interaction_string in ['SH-SH-NON-Sig', 'DF1-NON-NON-Sig']:
            #mixer 3 curves
            
            for i in range(num_tune_points):
                line =  re.split(r'\t+', crv_lines[row])
                points[i, 0] = float(line[1])
                points[i, 7] = int((float(line[3]) - mcf['m6'][1])*mcf['m6'][0])
                points[i, 13] = float(line[0])
                row = row+1
                
        else:
            
            print 'error filling in _crv read'
            
        return points
            
    elif interaction == 'write': #----------------------------------------------
    
        #initialize objects- - - - - - - - - - - - - - - - - - - - - - - - - - -
    
        to_insert = []
        
        mcf = _get_motor_conversion_factors(OPA)

        #import template filepath- - - - - - - - - - - - - - - - - - - - - - - -
        
        if filepath:
            pass
        else:
            #use currently loaded curve
            if interaction_string in ['preamp', 'poweramp', 'NON-NON-NON-Sig', 'NON-NON-NON-Idl']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 1') #base
            elif interaction_string in ['NON-NON-SF-Sig', 'NON-NON-SH-Idl', 'NON-NON-SF-Idl']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 2') #mixer 1
            elif interaction_string in ['NON-SH-NON-Sig', 'NON-SH-SH-Idl']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 3') #mixer 2
            elif interaction_string in ['SH-SH-NON-Sig', 'DF1-NON-NON-Sig']:
                filepath = _ini_handler('OPA{}'.format(OPA), 
                                        'read',
                                        'Optical Device',
                                        'Curve 4') #mixer 3
            else:
                print 'error in _crv - interaction string =', interaction_string

        old_crv = open(filepath, 'r')
        crv_lines = old_crv.readlines()
        
        if output_filepath_seed:
            pass
        else: 
            output_filepath_seed = filepath

        #decide on output filepath - - - - - - - - - - - - - - - - - - - - - - -

        if dummy:
            if interaction_string in ['preamp', 'poweramp', 'NON-NON-NON-Sig', 'NON-NON-NON-Idl']:
                output_filepath = filepath_of_folder + r'\dummy_curves\base dummy.crv'
            elif interaction_string in ['NON-NON-SF-Sig', 'NON-NON-SH-Idl', 'NON-NON-SF-Idl']:
                output_filepath = filepath_of_folder + r'\dummy_curves\mixer1 dummy.crv'
            elif interaction_string in ['NON-SH-NON-Sig', 'NON-SH-SH-Idl']:
                output_filepath = filepath_of_folder + r'\dummy_curves\mixer2 dummy.crv'
            elif interaction_string in ['SH-SH-NON-Sig', 'DF1-NON-NON-Sig']:
                output_filepath = filepath_of_folder + r'\dummy_curves\mixer3 dummy.crv'
        else:
            output_filepath = output_filepath_seed.split('-', 1)[0]
            output_filepath = output_filepath + '- ' + time.strftime("%Y.%m.%d %H_%M_%S") + '.crv'
        
        #create otput array- - - - - - - - - - - - - - - - - - - - - - - - - - -

        #create to_insert
        #a list where each element is [interaction string, array]
        #here interaction string is formatted as in the CRV
        #array must also be formatted as in the CRV

        if interaction_string in ['preamp', 'poweramp', 'NON-NON-NON-Sig']:
            #must create both signal and idler array
        
            lines_between = 7
            
            #create signal curve from array
            signal_curve = np.zeros([len(curve), 7])
            idler_curve = np.zeros([len(curve), 7])
            for i in range(len(curve)):
                signal_curve[i, 0] = spitfire_output
                signal_curve[i, 1] = curve[i][0]
                signal_curve[i, 2] = 4
                signal_curve[i, 3] = (curve[i][1]/mcf['m0'][0]) + mcf['m0'][1]
                signal_curve[i, 4] = (curve[i][2]/mcf['m1'][0]) + mcf['m1'][1]
                signal_curve[i, 5] = (curve[i][3]/mcf['m2'][0]) + mcf['m2'][1]
                signal_curve[i, 6] = (curve[i][4]/mcf['m3'][0]) + mcf['m3'][1]
                
            #create idler curve
            idler_curve = np.zeros([len(curve), 7])
            for i in range(len(signal_curve)):
                idler_curve[i, 0] = signal_curve[i, 0]
                idler_curve[i, 1] = 1/((1/spitfire_output) - (1/signal_curve[i, 1]))
                idler_curve[i, 2] = signal_curve[i, 2]
                idler_curve[i, 3] = signal_curve[i, 3]
                idler_curve[i, 4] = signal_curve[i, 4]
                idler_curve[i, 5] = signal_curve[i, 5]
                idler_curve[i, 6] = signal_curve[i, 6]
            idler_curve = np.flipud(idler_curve)
                
            #construct to_insert
            to_insert = (['NON-NON-NON-Sig', signal_curve], ['NON-NON-NON-Idl', idler_curve])
                
            #construct image of tuning curve
            plt.close()
            fig_a1 = plt.subplot(211)
            fig_b1 = plt.subplot(212, sharex=fig_a1)
            fig_a2 = fig_a1.twinx()
            fig_b2 = fig_b1.twinx()
            fig_a1.plot(signal_curve[:, 1], signal_curve[:, 3], color = 'b', linewidth = 2.0)
            for tl in fig_a1.get_yticklabels(): tl.set_color('b')
            fig_a1.set_ylabel('c1 (deg)', color = 'b')
            fig_a2.plot(signal_curve[:, 1], signal_curve[:, 4], color = 'r', linewidth = 2.0)
            for tl in fig_a2.get_yticklabels(): tl.set_color('r')
            fig_a2.set_ylabel('d1 (mm)', color = 'r')            
            fig_b1.plot(signal_curve[:, 1], signal_curve[:, 5], color = 'b', linewidth = 2.0)
            for tl in fig_b1.get_yticklabels(): tl.set_color('b')
            fig_b1.set_ylabel('c2 (deg)', color = 'b')              
            fig_b2.plot(signal_curve[:, 1], signal_curve[:, 6], color = 'r', linewidth = 2.0)
            for tl in fig_b2.get_yticklabels(): tl.set_color('r')
            fig_b2.set_ylabel('d2 (deg)', color = 'r')
            fig_a1.grid()
            fig_b1.grid()
            fig_a1.set_xlim(signal_curve[:, 1].min(), signal_curve[:, 1].max())
            setp(fig_a1.get_xticklabels(), visible=False)
            fig_b1.set_xlabel('tunepoint (nm)')
            fig_a1.set_title('OPA{} Signal Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        elif interaction_string == 'NON-NON-NON-Idl':
            
            lines_between = 7
            
            print 'Idler crv creation not yet supported'
            
        elif interaction_string == 'NON-NON-SH-Idl':
            
            lines_between = 10
            
            #import source colors
            old_crv_array = _crv('read', OPA, interaction_string = interaction_string)
            if source_colors == 'old':
                source_colors = old_crv_array[:, 13]
            else:
                source_colors = curve[:, 13]
            
            #construct crv_array
            crv_array = np.zeros([len(curve), 4])
            for i in range(len(curve)):
                crv_array[i, 0] = source_colors[i]
                crv_array[i, 1] = curve[i, 0]
                crv_array[i, 2] = 1
                crv_array[i, 3] = (curve[i, 5]/mcf['m4'][0]) + mcf['m4'][1]
        
            #construct to_insert
            to_insert = [['NON-NON-SH-Idl', crv_array]]
            
            #construct image of tuning curve
            plt.close()
            plt.plot(old_crv_array[:, 0], (old_crv_array[:, 5]/mcf['m4'][0]) + mcf['m4'][1], color = 'k', linewidth = 1)
            plt.plot(crv_array[:, 1], crv_array[:, 3], color = 'k', linewidth = 2.5)
            plt.xlabel('tunepoint (nm)')
            plt.ylabel('mixer 1 (deg)')
            plt.grid()
            plt.title('OPA{} Second Harmonic Idler Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        elif interaction_string == 'NON-SH-NON-Sig':
            
            lines_between = 10
            
            #import source colors
            old_crv_array = _crv('read', OPA, interaction_string = interaction_string)
            if source_colors == 'old':
                source_colors = old_crv_array[:, 13]
            else:
                source_colors = curve[:, 13]
            
            #construct crv_array
            crv_array = np.zeros([len(curve), 4])
            for i in range(len(curve)):
                crv_array[i, 0] = source_colors[i]
                crv_array[i, 1] = curve[i, 0]
                crv_array[i, 2] = 1
                crv_array[i, 3] = (curve[i, 6]/mcf['m5'][0]) + mcf['m5'][1]
        
            #construct to_insert
            to_insert = [['NON-SH-NON-Sig', crv_array]]
            
            #construct image of tuning curve
            plt.close()
            plt.plot(old_crv_array[:, 0], (old_crv_array[:, 6]/mcf['m5'][0]) + mcf['m5'][1], color = 'k', linewidth = 1)
            plt.plot(crv_array[:, 1], crv_array[:, 3], color = 'k', linewidth = 2.5)
            plt.xlabel('tunepoint (nm)')
            plt.ylabel('mixer 1 (deg)')
            plt.grid()
            plt.title('OPA{} Second Harmonic Signal Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        elif interaction_string == 'NON-NON-SF-Idl':
            
            lines_between = 10
            
            #import source colors
            old_crv_array = _crv('read', OPA, interaction_string = interaction_string)
            if source_colors == 'old':
                source_colors = old_crv_array[:, 13]
            else:
                source_colors = curve[:, 13]
            
            #construct crv_array
            crv_array = np.zeros([len(curve), 4])
            for i in range(len(curve)):
                crv_array[i, 0] = source_colors[i]
                crv_array[i, 1] = curve[i, 0]
                crv_array[i, 2] = 1
                crv_array[i, 3] = (curve[i, 5]/mcf['m4'][0]) + mcf['m4'][1]
        
            #construct to_insert
            to_insert = [['NON-NON-SF-Idl', crv_array]]
            
            #construct image of tuning curve
            plt.close()
            plt.plot(old_crv_array[:, 0], (old_crv_array[:, 5]/mcf['m4'][0]) + mcf['m4'][1], color = 'k', linewidth = 1)
            plt.plot(crv_array[:, 1], crv_array[:, 3], color = 'k', linewidth = 2.5)
            plt.xlabel('tunepoint (nm)')
            plt.ylabel('mixer 1 (deg)')
            plt.grid()
            plt.title('OPA{} Sum Frequency Idler Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        elif interaction_string == 'NON-NON-SF-Sig':
            
            lines_between = 10
            
            #import source colors
            old_crv_array = _crv('read', OPA, interaction_string = interaction_string)
            if source_colors == 'old':
                source_colors = old_crv_array[:, 13]
            else:
                source_colors = curve[:, 13]
            
            #construct crv_array
            crv_array = np.zeros([len(curve), 4])
            for i in range(len(curve)):
                crv_array[i, 0] = source_colors[i]
                crv_array[i, 1] = curve[i, 0]
                crv_array[i, 2] = 1
                crv_array[i, 3] = (curve[i, 5]/mcf['m4'][0]) + mcf['m4'][1]
        
            #construct to_insert
            to_insert = [['NON-NON-SF-Sig', crv_array]]
            
            #construct image of tuning curve
            plt.close()
            plt.plot(old_crv_array[:, 0], (old_crv_array[:, 5]/mcf['m4'][0]) + mcf['m4'][1], color = 'k', linewidth = 1)
            plt.plot(crv_array[:, 1], crv_array[:, 3], color = 'k', linewidth = 2.5)
            plt.xlabel('tunepoint (nm)')
            plt.ylabel('mixer 1 (deg)')
            plt.grid()
            plt.title('OPA{} Sum Frequency Signal Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        elif interaction_string == 'NON-SH-SH-Idl':
            
            lines_between = 10
            
            #import source colors
            old_crv_array = _crv('read', OPA, interaction_string = interaction_string)
            if source_colors == 'old':
                source_colors = old_crv_array[:, 13]
            else:
                source_colors = curve[:, 13]
            
            #construct crv_array
            crv_array = np.zeros([len(curve), 4])
            for i in range(len(curve)):
                crv_array[i, 0] = source_colors[i]
                crv_array[i, 1] = curve[i, 0]
                crv_array[i, 2] = 1
                crv_array[i, 3] = (curve[i, 6]/mcf['m5'][0]) + mcf['m5'][1]
        
            #construct to_insert
            to_insert = [['NON-SH-SH-Idl', crv_array]]
            
            #construct image of tuning curve
            plt.close()
            plt.plot(old_crv_array[:, 0], (old_crv_array[:, 6]/mcf['m5'][0]) + mcf['m5'][1], color = 'k', linewidth = 1)
            plt.plot(crv_array[:, 1], crv_array[:, 3], color = 'k', linewidth = 2.5)
            plt.xlabel('tunepoint (nm)')
            plt.ylabel('mixer 1 (deg)')
            plt.grid()
            plt.title('OPA{} Fourth Harmonic Idler Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        elif interaction_string == 'SH-SH-NON-Sig':
            
            lines_between = 10
            
            #import source colors
            old_crv_array = _crv('read', OPA, interaction_string = interaction_string)
            if source_colors == 'old':
                source_colors = old_crv_array[:, 13]
            else:
                source_colors = curve[:, 13]
            
            #construct crv_array
            crv_array = np.zeros([len(curve), 4])
            for i in range(len(curve)):
                crv_array[i, 0] = source_colors[i]
                crv_array[i, 1] = curve[i, 0]
                crv_array[i, 2] = 1
                crv_array[i, 3] = (curve[i, 7]/mcf['m6'][0]) + mcf['m6'][1]
        
            #construct to_insert
            to_insert = [['SH-SH-NON-Sig', crv_array]]
            
            #construct image of tuning curve
            plt.close()
            plt.plot(old_crv_array[:, 0], (old_crv_array[:, 7]/mcf['m6'][0]) + mcf['m6'][1], color = 'k', linewidth = 1)
            plt.plot(crv_array[:, 1], crv_array[:, 3], color = 'k', linewidth = 2.5)
            plt.xlabel('tunepoint (nm)')
            plt.ylabel('mixer 1 (deg)')
            plt.grid()
            plt.title('OPA{} Fourth Harmonic Idler Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        elif interaction_string == 'DF1-NON-NON-Sig':
            
            lines_between = 9
            
            #import source colors
            old_crv_array = _crv('read', OPA, interaction_string = interaction_string)
            if source_colors == 'old':
                source_colors = old_crv_array[:, 13]
            else:
                source_colors = curve[:, 13]
            
            #construct crv_array
            crv_array = np.zeros([len(curve), 4])
            for i in range(len(curve)):
                crv_array[i, 0] = source_colors[i]
                crv_array[i, 1] = curve[i, 0]
                crv_array[i, 2] = 1
                crv_array[i, 3] = (curve[i, 7]/mcf['m6'][0]) + mcf['m6'][1]
        
            #construct to_insert
            to_insert = [['DF1-NON-NON-Sig', crv_array]]
            
            #construct image of tuning curve
            plt.close()
            plt.plot(old_crv_array[:, 0], (old_crv_array[:, 7]/mcf['m6'][0]) + mcf['m6'][1], color = 'k', linewidth = 1)
            plt.plot(crv_array[:, 1], crv_array[:, 3], color = 'k', linewidth = 2.5)
            plt.xlabel('tunepoint (nm)')
            plt.ylabel('mixer 1 (deg)')
            plt.grid()
            plt.title('OPA{} Difference Frequency Tuning Curve'.format(OPA))
            image_filepath = output_filepath.replace('.crv', '.png')
            plt.savefig(image_filepath, transparent = True)
            plt.close()
            
        else:
            
            print 'interaction_string {} not recognized in _crv'.format(interaction_string)

        #insert array(s) into file - - - - - - - - - - - - - - - - - - - - - - -

        for interaction in to_insert:            
            
            to_replace = interaction[0]
            input_points = interaction[1]
            
            #get relevant properties of curve in its old state
            num_points = []
            for i in range(len(crv_lines)):
                if 'NON' in crv_lines[i]:
                    num_points.append([i+lines_between, crv_lines[i].replace('\n', ''), int(crv_lines[i+lines_between])])

            #remove old points
            index = ''
            to_remove = ''
            for i in range(len(num_points)):
                if num_points[i][1] == to_replace:
                    index = num_points[i][0]
                    to_remove = num_points[i][2]
            if index == '':
                print 'interaction {0} not found in {1}'.format(to_replace, filepath)
                return
            del crv_lines[index:index+to_remove+1]
            
            #put in new points (gets done 'backwards')
            input_points = np.flipud(input_points)
            for tune_point in input_points:
                line = ''
                for value in tune_point:
                    #the number of motors must be an integer - so dumb
                    if value == 4:
                        value_as_string = '4'
                    elif value == 1:
                        value_as_string = '1'
                    else:
                        value_as_string = str(np.round(value, decimals=6))
                        portion_before_decimal = value_as_string.split('.')[0]
                        portion_after_decimal = value_as_string.split('.')[1].ljust(6, '0')
                        value_as_string = portion_before_decimal + '.' + portion_after_decimal
                    line = line + value_as_string + '\t'
                line = line + '\n'
                crv_lines.insert(index, line)
            crv_lines.insert(index, str(len(input_points)) + '\n') #length of new curve
        
        #create new file, write to it
        new_crv = open(output_filepath, 'w')
        for line in crv_lines:
            new_crv.write(line)
        new_crv.close()

    #return---------------------------------------------------------------------

    return output_filepath

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _ini_handler(ini_type,
                 interaction,
                 section,
                 option,
                 value = None):
                        
    #handles reading and writing to ini files with ConfigParser package      
    
    #get correct filepath based on ini_type-------------------------------------
    
    ini_filepath = ''
    if ini_type == 'COLORS':
        ini_filepath = COLORS_ini_filepath
    elif ini_type == 'OPA1':
        ini_filepath = OPA1_ini_filepath
    elif ini_type == 'OPA2':
        ini_filepath = OPA2_ini_filepath
    elif ini_type == 'OPA3':
        ini_filepath = OPA3_ini_filepath
    elif ini_type == 'topas_tune':
        ini_filepath = topas_tune_ini_filepath
    else:
        print 'ini_type not recognized in ini_handler'
        return
        
    #clean up filepath
    ini_filepath = ini_filepath.replace('\\', '\\\\')
        
    #check if real
    if not os.path.isfile(ini_filepath):
        print 'ini_filepath {} is not a file!'.format(ini_filepath)
        
    #do action------------------------------------------------------------------
    
    config = ConfigParser.SafeConfigParser()
    
    if interaction == 'read':
        config.read(ini_filepath) 
        return config.get(section, option)
    elif interaction == 'write':
        value = str(value) #ensure 'value' is a string
        config.read(ini_filepath)
        config.set(section, option, value) #update
        with open(ini_filepath, 'w') as configfile: 
            config.write(configfile) #save

if False:
    COLORS_ini_filepath = _ini_handler('topas_tune', 'read', 'general', 'COLORS.ini filepath')
    DATA_folderpath =  _ini_handler('topas_tune', 'read', 'general', 'DATA folderpath')
    OPA1_ini_filepath = _ini_handler('COLORS', 'read', 'OPA', 'OPA1 device').replace('\"', '')
    OPA2_ini_filepath = _ini_handler('COLORS', 'read', 'OPA', 'OPA2 device').replace('\"', '') 
    OPA3_ini_filepath = _ini_handler('COLORS', 'read', 'OPA', 'OPA3 device').replace('\"', '')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _load_fit(filepath):
    
    fit_cols =  {'num':       (0, 0.0, None, 'acquisition number'),
                 'w':         (1, 5.0, 'nm', r'$\mathrm{\bar\nu_1=\bar\nu_m (cm^{-1})}$'),
                 'm0':        (2, 0.0, 'us', r'm0'),
                 'm1':        (3, 0.0, 'us', r'm1'),
                 'm2':        (4, 0.0, 'us', r'm2'),
                 'm3':        (5, 0.0, 'us', r'm3'),
                 'm4':        (6, 0.0, 'us', r'm4'),
                 'm5':        (7, 0.0, 'us', r'm5'),
                 'm6':        (8, 0.0, 'us', r'm6'),
                 'center':    (9, 0.0, 'nm', r'center'),
                 'amplitude': (10, 0.0, 'a.u.', r'amplitude'),
                 'fwhm':      (11, 0.0, 'nm', r'fwhm'),
                 'gof':       (12, 0.0, '%', r'gof'),
                 'mismatch':  (13, 0.0, 'nm', r'mismatch')}

    #load raw array from fit file-----------------------------------------------

    raw_fit = np.loadtxt(filepath)
    raw_fit.T
    
    #construct fit array in topas_tune format-----------------------------------

    fit_array = np.empty([len(raw_fit), 20])
    fit_array[:] = np.nan
    for i in range(len(fit_array)):
        for j in range(13):
            fit_array[i][j] = raw_fit[i][j+1]
            
    #return---------------------------------------------------------------------
    
    return fit_array

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _load_dat(filepath, xvar, yvar, tune_test = False, zvar = 'mc'):
    
    dat_cols =  {'num':  (0, 0.0, None, 'acquisition number'),
                 'w1':   (1, 5.0, 'nm', r'$\mathrm{\bar\nu_1=\bar\nu_m (cm^{-1})}$'),
                 'w2':   (3, 5.0, 'nm', r'$\mathrm{\bar\nu_2=\bar\nu_{2^{\prime}} (cm^{-1})}$'),
                 'w3':   (5, 5.0, 'nm', r'$\mathrm{\bar\nu_3 (cm^{-1})}$'),
                 'wm':   (7, 1.0, 'nm', r'$\bar\nu_m / cm^{-1}$'),
                 'wa':   (8, 1.0, 'nm', r'array'),
                 'ai0':  (16, 0.0, 'V', 'Signal 0'),
                 'ai1':  (17, 0.0, 'V', 'Signal 1'),
                 'ai2':  (18, 0.0, 'V', 'Signal 2'),
                 'ai3':  (19, 0.0, 'V', 'Signal 3'),
                 'ai4':  (20, 0.0, 'V', 'Signal 4'),
                 'mc':   (21, 0.0, 'a.u.', 'array signal')}

    #load raw array-------------------------------------------------------------

    raw_dat = np.genfromtxt(filepath, dtype=np.float)
    dat = raw_dat.T
    
    #grid data------------------------------------------------------------------
    
    grid_factor = 1    
    
    #x
    xlis = dat[dat_cols[xvar][0]]
    xtol = dat_cols[xvar][1]
    xstd = []
    xs = []
    while len(xlis) > 0:
        set_val = xlis[0]
        xi_lis = [xi for xi in xlis if np.abs(set_val - xi) < xtol]
        xlis = [xi for xi in xlis if not np.abs(xi_lis[0] - xi) < xtol]
        xi_lis_average = sum(xi_lis) / len(xi_lis)
        xs.append(xi_lis_average)
        xstdi = sum(np.abs(xi_lis - xi_lis_average)) / len(xi_lis)
        xstd.append(xstdi)
    tol = sum(xstd) / len(xstd)
    tol = max(tol, 1e-1)
    xi = np.linspace(min(xs)+tol,max(xs)-tol, num=(len(xs) + (len(xs)-1)*(grid_factor-1)))    
    
    #y
    if tune_test:
        ylis = dat[dat_cols[yvar][0]] - dat[dat_cols[xvar][0]]
    else:
        ylis = dat[dat_cols[yvar][0]]
    ytol = dat_cols[yvar][1]
    ystd = []
    ys = []
    while len(ylis) > 0:
        set_val = ylis[0]
        yi_lis = [yi for yi in ylis if np.abs(set_val - yi) < ytol]
        ylis = [yi for yi in ylis if not np.abs(yi_lis[0] - yi) < ytol]
        yi_lis_average = sum(yi_lis) / len(yi_lis)
        ys.append(yi_lis_average)
        ystdi = sum(np.abs(yi_lis - yi_lis_average)) / len(yi_lis)
        ystd.append(ystdi)
    tol = sum(ystd) / len(ystd)
    tol = max(tol, 1e-1)
    yi = np.linspace(min(ys)+tol,max(ys)-tol, num=(len(ys) + (len(ys)-1)*(grid_factor-1)))
    
    #z
    xlis = dat[dat_cols[xvar][0]]
    if tune_test:
        ylis = dat[dat_cols[yvar][0]] - dat[dat_cols[xvar][0]]
    else:
        ylis = dat[dat_cols[yvar][0]]
    zlis = dat[dat_cols[zvar][0]]
    zi = scipy.interpolate.griddata((xlis, ylis), zlis, (xi[None,:], yi[:,None]), method='cubic', fill_value = 0.0)    
    
    #return---------------------------------------------------------------------
    
    return zi, xi, yi

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _get_motor_conversion_factors(OPA):
    
    #for any motor
    #  geo = ustep*((1/mcf[motor][0]) + mcf[motor][1])
    #  ustep = (geo - mcf[motor][1])*mcf[motor][0]
    
    OPA_str = 'OPA{}'.format(OPA)    
    motor_conversion_factors = {'m0': (240 , float(_ini_handler(OPA_str, 'read', 'Motors logical parameters', 'Affix 0'))),
                                'm1': (3200, float(_ini_handler(OPA_str, 'read', 'Motors logical parameters', 'Affix 1'))),
                                'm2': (240 , float(_ini_handler(OPA_str, 'read', 'Motors logical parameters', 'Affix 2'))),
                                'm3': (240 , float(_ini_handler(OPA_str, 'read', 'Motors logical parameters', 'Affix 3'))),
                                'm4': (240 , float(_ini_handler(OPA_str, 'read', 'Motors logical parameters', 'Affix 4'))),
                                'm5': (240 , float(_ini_handler(OPA_str, 'read', 'Motors logical parameters', 'Affix 5'))),
                                'm6': (240 , float(_ini_handler(OPA_str, 'read', 'Motors logical parameters', 'Affix 6')))}
    
    return motor_conversion_factors     

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _say(text, informative_text = '', window_title = 'topas_tune'):

    text = str(text)
    informative_text = str(informative_text)
    
    msgBox = QMessageBox()
    msgBox.setWindowTitle(window_title)
    msgBox.setText(text);
    msgBox.setInformativeText(informative_text);  
    msgBox.isActiveWindow()
    msgBox.setFocusPolicy(Qt.StrongFocus)
    msgBox.exec_();
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _get_color(value):
    
    colormap_input = ['#0000FF', #blue
                      '#00FFFF', #aqua
                      '#00FF00', #green
                      '#FFFF00', #yellow
                      '#FF0000', #red
                      '#881111', #burgandy
                      '#0000FF', #blue
                      '#00FFFF', #aqua
                      '#00FF00', #green
                      '#FFFF00', #yellow
                      '#FF0000', #red
                      '#881111', #burgandy
                      '#0000FF', #blue
                      '#00FFFF', #aqua
                      '#00FF00', #green
                      '#FFFF00', #yellow
                      '#FF0000', #red
                      '#881111'] #burgandy
                      
    global rainbow_cmap
    
    rainbow_cmap = mplcolors.LinearSegmentedColormap.from_list('my colormap',colormap_input)

    return rainbow_cmap(value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _gauss_residuals(p, y, x):

    A, mu, sigma = p
    
    err = y-np.abs(A)*np.exp(-(x-mu)**2 / (2*np.abs(sigma)**2))
    
    return np.abs(err)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def _exp_value(y, x):
    
    y_internal = np.ma.copy(y)
    x_internal = np.ma.copy(x)

    #get sum
    sum_y = 0.
    for i in range(len(y_internal)):
        if np.ma.getmask(y_internal[i]) == True:
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            sum_y = sum_y + y_internal[i]
    
    #divide by sum
    for i in range(len(y_internal)):
        if np.ma.getmask(y_internal[i]) == True:
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            y_internal[i] = y_internal[i] / sum_y

    #get expectation value    
    value = 0.
    for i in range(len(x_internal)):
        if np.ma.getmask(y_internal[i]) == True:
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            value = value + y_internal[i]*x_internal[i]
    return value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~