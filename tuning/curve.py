'''
OPA tuning curves.
'''


### import ####################################################################


import os
import copy

import numpy as np

import scipy

import matplotlib
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

from .. import units as wt_units
from .. import kit as wt_kit

debug = False


### interpolation classes #####################################################


class Linear:

    def __init__(self, colors, units, motors):
        '''
        Linear interpolation using scipy.interpolate.interp1d.
        '''
        self.units = units
        self.functions = [scipy.interpolate.interp1d(colors, motor.positions)
                          for motor in motors]
        self.i_functions = [scipy.interpolate.interp1d(motor.positions, colors)
                            for motor in motors]

    def get_motor_positions(self, color):
        return [f(color) for f in self.functions]

    def get_color(self, motor_index, motor_position):
        return self.i_functions[motor_index](motor_position)


class Poly:

    def __init__(self, colors, units, motors):
        '''
        Fourth order polynomial.
        '''
        self.colors = colors
        self.n = 4
        self.fit_params = []
        for motor in motors:
            out = np.polynomial.polynomial.polyfit(colors, motor.positions, self.n, full=True)
            self.fit_params.append(out)
        
    def get_motor_positions(self, color):
        outs = []
        for params in self.fit_params:
            out = np.polynomial.polynomial.polyval(color, params[0])
            outs.append(out)
        return outs
    
    def get_color(self, motor_index, motor_position):
        a = self.fit_params[motor_index][0][::-1].copy()
        a[4] -= motor_position
        roots = np.real(np.roots(a))
        for root in roots:
            if self.colors.min() < root < self.colors.max():
                return root
        return roots[3]


### curve class ###############################################################


class Motor:

    def __init__(self, positions, name):
        '''
        Container class for motor arrays.
        '''
        self.positions = positions
        self.name = name


class Curve:

    def __init__(self, colors, units, motors, name, kind, method=Linear):
        '''
        Central object-type for all OPA tuning curves.
        
        Parameters
        ----------
        colors : array
            The color destinations for the curve.
        units : str
            The color units.
        motors : list of Motor objects
            Motor positions for each color.
        name : str
            Name of curve.
        kind : {'opa800'}
            The kind of curve (for saving).
        method : interpolation class
            The interpolation method to use.
        '''
        self.colors = colors
        self.units = units
        self.motors = motors
        self.name = name
        self.kind = kind
        # set motors as attributes of self
        self.motor_names = [m.name for m in self.motors]
        for obj in self.motors:
            setattr(self, obj.name, obj)
        # initialize function object
        self.method = method
        self.interpolate()

    def coerce_motors(self):
        '''
        Coerce the motor positions to lie exactly along the interpolation
        positions. Can be thought of as 'smoothing' the curve.
        '''
        self.map_colors(self.colors, units='same')
        
    def convert(self, units):
        '''
        Convert the colors.
        
        Parameters
        ----------
        units : str
            The destination units.
        '''
        self.colors = wt_units.converter(self.colors, self.units, units)
        self.units = units

    def copy(self):
        '''
        Copy the object.

        Returns
        -------
        curve
            A deep copy of the curve object.
        '''
        return copy.deepcopy(self)

    def get_color(self, motor_positions, units='same'):
        '''
        Get the color given a set of motor positions.
        
        Parameters
        ----------
        motor_positions : array
            The motor positions.
        units : str (optional)
            The units of the returned color.
            
        Returns
        -------
        float
            The current color.
        '''
        colors = []
        for motor_index, motor_position in enumerate(motor_positions):
            color = self.interpolator.get_color(motor_index, motor_position)
            colors.append(color)
        # TODO: decide how to handle case of disagreement between colors
        print colors
        return colors[0]

    def get_limits(self, units='same'):
        '''
        Get the edges of the curve.

        Parameters
        ----------
        units : str (optional)
            The units to return. Default is same.

        Returns
        -------
        list of floats
            [min, max] in given units
        '''
        if units == 'same':
            return [self.colors.min(), self.colors.max()]
        else:
            units_colors = wt_units.converter(self.colors, self.units, units)
            return [units_colors.min(), units_colors.max()]

    def get_motor_positions(self, color, units='same'):
        '''
        Get the motor positions for a destination color.

        Parameters
        ----------
        color : number
            The destination color.
        units : str (optional)
            The units of the input color.

        Returns
        -------
        list of floats
            The destination motor positions.
        '''
        if units == 'same':
            pass
        else:
            color = wt_units.converter(color, units, self.units)
        return self.interpolator.get_motor_positions(color)
        
    def interpolate(self):
        '''
        Generate the interploator object.
        '''
        self.interpolator = self.method(self.colors, self.units, self.motors)

    def map_colors(self, colors, units='same'):
        '''
        Map the curve onto new tune points using the curve's own interpolation
        method

        Parameters
        ----------
        colors : int or array
            The number of new points (between current limits) or the new points
            themselves.
        units : str (optional.)
            The input units if given as array. Default is same. Units of curve
            object are not changed by map_colors.
        '''
        # get new colors in input units
        if type(colors) == int:
            limits = self.get_limits(units)
            new_colors = np.linspace(limits[0], limits[1], colors)
        elif type(colors) in [list, np.ndarray]:
            new_colors = colors
        else:
            print 'type not recognized in curve.map_points - using old points'
            new_colors = self.colors
        # convert new colors to local units
        if units == 'same':
            units = self.units
        new_colors = wt_units.converter(new_colors, units, self.units)
        new_colors.sort()
        # map motors
        new_motors = []
        for motor_index, motor in enumerate(self.motors):
            positions = self.get_motor_positions(new_colors)[motor_index]
            new_motor = Motor(positions, motor.name)  # new motor objects
            new_motors.append(new_motor)
        # finish
        self.colors = new_colors
        self.motors = new_motors
        self.motor_names = [m.name for m in self.motors]
        for obj in self.motors:
            setattr(self, obj.name, obj)
            
    def offset(self, motor, amount):
        '''
        Offset given motor by some ammount.
        
        Parameters
        ----------
        motor : number or str
            The motor index or name.
        amount : number
            The offset.
        '''
        if type(motor) in [float, int]:
            motor_index = motor
        elif type(motor) == str:
            motor_index = self.motor_names.index(motor)
        else:
            print 'motor type not recognized in curve.offset'
        # offset
        self.motors[motor_index].positions += amount
        self.interpolate()

    def plot(self, autosave=False, save_path=''):
        '''
        Plot the curve.
        '''
        fig = plt.figure(figsize=(8, 2*len(self.motors)))
        axs = grd.GridSpec(3, 1, hspace=0)
        limits = self.get_limits()
        line_points = np.linspace(limits[0], limits[1], 1000) # get interpolated points
        positions = np.array([self.get_motor_positions(c) for c in line_points]).T
        for motor_index, motor in enumerate(self.motors):
            ax = plt.subplot(axs[motor_index])
            ax.scatter(self.colors, motor.positions, c='k')
            ax.plot(line_points, positions[motor_index], c='k')
            plt.grid()
            plt.xticks(rotation=45)
            plt.yticks(ax.get_yticks()[1:-1])
            plt.yticks()
            plt.ylabel(motor.name)
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            if motor_index != len(self.motors)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
        plt.xlabel('color ({})'.format(self.units))
        plt.suptitle(self.name)
        if autosave:
            if save_path[-3:] != 'png':
                image_path = save_path + self.name + '.png'
            else:
                image_path = save_path
            plt.savefig(image_path, transparent=True, dpi=300)
            plt.close(fig)

    def save(self, save_directory=None, plot=True, verbose=True):
        '''
        Save the curve.

        Parameters
        ----------
        save_directory : str (optional)
            The save directory. If not supplied, current working directory is
            used.
        plot : bool (optional)
            Toggle saving plot along with curve. Default is True.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        str
            The filepath of the saved curve.
        '''
        # get save directory
        if save_directory is None:
            save_directory = os.getcwd()
        # save
        if self.kind == 'opa800':
            out_path = to_800_curve(self, save_directory)
        else:
            print 'kind', self.kind, 'does not know how to save!'
        # plot
        if plot:
            image_path = out_path.replace('.curve', '.png')
            self.plot(autosave=True, save_path=image_path)
        # finish
        if verbose:
            print 'curve saved at', out_path
        return out_path


### curve import methods ######################################################

def from_800_curve(filepath):
    arr = np.genfromtxt(filepath).T
    colors = arr[0]
    grating = Motor(arr[1], 'Grating')
    bbo = Motor(arr[2], 'BBO')
    mixer = Motor(arr[3], 'Mixer')
    motors = [grating, bbo, mixer]
    path, name, suffix = wt_kit.filename_parse(filepath)
    curve = Curve(colors, 'wn', motors, name=name, kind='opa800', method=Poly)
    return curve

### curve writing methods #####################################################

def to_800_curve(curve, save_directory):
    # ensure curve is in wn
    curve = curve.copy()
    curve.convert('wn')
    # array
    colors = curve.colors
    motors = curve.motors
    out_arr = np.zeros([4, len(colors)])
    out_arr[0] = colors
    out_arr[1:4] = np.array([motor.positions for motor in motors])
    # filename
    timestamp = wt_kit.get_timestamp()
    out_name = curve.name.split('-')[0] + '- ' + timestamp
    out_path = os.path.join(save_directory, out_name + '.curve')
    # save
    header1 = 'file created:\t' + timestamp
    header2 = 'Color (wn)\tGrating\tBBO\tMixer'
    header = '\n'.join([header1, header2])
    np.savetxt(out_path, out_arr.T, fmt='%.2f',
               delimiter='\t', header=header)
    return out_path
