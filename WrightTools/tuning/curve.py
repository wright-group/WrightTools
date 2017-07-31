"""
OPA tuning curves.
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import copy
import shutil
import collections

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


# --- define --------------------------------------------------------------------------------------


TOPAS_C_motor_names = {0: ['Crystal_1', 'Delay_1', 'Crystal_2', 'Delay_2'],
                       1: ['Mixer_1'],
                       2: ['Mixer_2'],
                       3: ['Mixer_3']}

# [num_between, motor_names]
TOPAS_C_interactions = {'NON-NON-NON-Sig': [8, TOPAS_C_motor_names[0]],
                        'NON-NON-NON-Idl': [8, TOPAS_C_motor_names[0]],
                        'NON-NON-SH-Sig': [11, TOPAS_C_motor_names[1]],
                        'NON-SH-NON-Sig': [11, TOPAS_C_motor_names[2]],
                        'NON-NON-SH-Idl': [11, TOPAS_C_motor_names[1]],
                        'NON-NON-SF-Sig': [11, TOPAS_C_motor_names[1]],
                        'NON-NON-SF-Idl': [11, TOPAS_C_motor_names[1]],
                        'NON-SH-SH-Sig': [11, TOPAS_C_motor_names[2]],
                        'SH-SH-NON-Sig': [11, TOPAS_C_motor_names[3]],
                        'NON-SH-SH-Idl': [11, TOPAS_C_motor_names[2]],
                        'SH-NON-SH-Idl': [11, TOPAS_C_motor_names[3]],
                        'DF1-NON-NON-Sig': [10, TOPAS_C_motor_names[3]]}

TOPAS_800_motor_names = {0: ['Crystal', 'Amplifier', 'Grating'],
                         1: [''],
                         2: [''],
                         3: ['NDFG_Crystal', 'NDFG_Mirror', 'NDFG_Delay']}

# [num_between, motor_names]
TOPAS_800_interactions = {'NON-NON-NON-Sig': [8, TOPAS_800_motor_names[0]],
                          'NON-NON-NON-Idl': [8, TOPAS_800_motor_names[0]],
                          'DF1-NON-NON-Sig': [7, TOPAS_800_motor_names[3]],
                          'DF2-NON-NON-Sig': [7, TOPAS_800_motor_names[3]]}

TOPAS_interaction_by_kind = {'TOPAS-C': TOPAS_C_interactions,
                             'TOPAS-800': TOPAS_800_interactions}


# --- interpolation classes -----------------------------------------------------------------------


class Linear:

    def __init__(self, colors, units, motors):
        """ Linear interpolation using scipy.interpolate.InterpolatedUnivariateSpline.  """
        self.colors = colors
        self.units = units
        self.motors = motors
        self.functions = [wt_kit.Spline(colors, motor.positions, k=1, s=0) for motor in motors]
        self.i_functions = [wt_kit.Spline(motor.positions, colors, k=1, s=0) for motor in motors]

    def get_motor_positions(self, color):
        return [f(color) for f in self.functions]

    def get_color(self, motor_index, motor_position):
        motor = self.motors[motor_index]
        if motor.positions.min() < motor_position < motor.positions.max():
            pass
        else:
            # take closest valid motor position if outside of range
            idx = (np.abs(motor.positions - motor_position)).argmin()
            motor_position = motor.positions[idx]
        return self.i_functions[motor_index](motor_position)


class Poly:

    def __init__(self, colors, units, motors):
        self.colors = colors
        self.n = 8
        self.fit_params = []
        for motor in motors:
            out = np.polynomial.polynomial.polyfit(colors, motor.positions, self.n, full=True)
            self.fit_params.append(out)
        self.linear = Linear(colors, units, motors)

    def get_motor_positions(self, color):
        outs = []
        for params in self.fit_params:
            out = np.polynomial.polynomial.polyval(color, params[0])
            outs.append(out)
        return outs

    def get_color(self, motor_index, motor_position):
        a = self.fit_params[motor_index][0][::-1].copy()
        a[-1] -= motor_position
        roots = np.real(np.roots(a))
        # return root closest to guess from linear interpolation
        guess = self.linear.get_color(motor_index, motor_position)
        idx = (np.abs(roots - guess)).argmin()
        return roots[idx]


class Spline:

    def __init__(self, colors, units, motors):
        """ Linear interpolation using scipy.interpolate.InterpolatedUnivariateSpline.  """
        self.colors = colors
        self.units = units
        self.motors = motors
        self.functions = [scipy.interpolate.UnivariateSpline(
            colors, motor.positions, k=3, s=1000) for motor in motors]
        self.i_functions = [scipy.interpolate.UnivariateSpline(
            motor.positions, colors, k=3, s=1000) for motor in motors]

    def get_motor_positions(self, color):
        return [f(color) for f in self.functions]

    def get_color(self, motor_index, motor_position):
        motor = self.motors[motor_index]
        if motor.positions.min() < motor_position < motor.positions.max():
            pass
        else:
            # take closest valid motor position if outside of range
            idx = (np.abs(motor.positions - motor_position)).argmin()
            motor_position = motor.positions[idx]
        return self.i_functions[motor_index](motor_position)


# --- curve class ---------------------------------------------------------------------------------


class Motor:

    def __init__(self, positions, name):
        """ Container class for motor arrays.  """
        self.positions = positions
        self.name = name


class Curve:

    def __init__(self, colors, units, motors, name, interaction,
                 kind, method=Linear,
                 subcurve=None, source_colors=None):
        """ Central object-type for all OPA tuning curves.

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
        """
        # version
        from .. import __version__
        self.__version__ = __version__
        # inherit
        self.colors = np.array(colors)  # needs to be array for some interpolation methods
        self.units = units
        self.motors = motors
        self.name = name
        self.kind = kind
        self.subcurve = subcurve
        self.source_colors = source_colors
        self.interaction = interaction
        # set motors as attributes of self
        self.motor_names = [m.name for m in self.motors]
        for obj in self.motors:
            setattr(self, obj.name, obj)
        # initialize function object
        self.method = method
        self.interpolate()

    def __repr__(self):
        # when you inspect the object
        outs = []
        outs.append('WrightTools.tuning.curve.Curve object at ' + str(id(self)))
        outs.append('  name: ' + self.name)
        outs.append('  interaction: ' + self.interaction)
        outs.append('  range: {0} - {1} ({2})'.format(self.colors.min(),
                                                      self.colors.max(), self.units))
        outs.append('  number: ' + str(len(self.colors)))
        return '\n'.join(outs)

    def coerce_motors(self):
        """ Coerce the motor positions to lie exactly along the interpolation positions.

        Can be thought of as 'smoothing' the curve.
        """
        self.map_colors(self.colors, units='same')

    def convert(self, units):
        """ Convert the colors.

        Parameters
        ----------
        units : str
            The destination units.
        """
        self.colors = wt_units.converter(self.colors, self.units, units)
        if self.subcurve:
            positions = self.source_colors.positions
            self.source_colors.positions = wt_units.converter(positions, self.units, units)
        self.units = units
        self.interpolate()  # how did it ever work if this wasn't here?  - Blaise 2017-03-22

    def copy(self):
        """ Copy the object.

        Returns
        -------
        curve
            A deep copy of the curve object.
        """
        return copy.deepcopy(self)

    def get_color(self, motor_positions, units='same'):
        """ Get the color given a set of motor positions.

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
        """
        colors = []
        for motor_index, motor_position in enumerate(motor_positions):
            color = self.interpolator.get_color(motor_index, motor_position)
            colors.append(color)
        # TODO: decide how to handle case of disagreement between colors
        return colors[0]

    def get_limits(self, units='same'):
        """ Get the edges of the curve.

        Parameters
        ----------
        units : str (optional)
            The units to return. Default is same.

        Returns
        -------
        list of floats
            [min, max] in given units
        """
        if units == 'same':
            return [self.colors.min(), self.colors.max()]
        else:
            units_colors = wt_units.converter(self.colors, self.units, units)
            return [units_colors.min(), units_colors.max()]

    def get_motor_names(self, full=True):
        if self.subcurve and full:
            subcurve_motor_names = self.subcurve.get_motor_names()
        else:
            subcurve_motor_names = []
        return subcurve_motor_names + [m.name for m in self.motors]

    def get_motor_positions(self, color, units='same', full=True):
        """ Get the motor positions for a destination color.

        Parameters
        ----------
        color : number
            The destination color. May be 1D array.
        units : str (optional)
            The units of the input color.

        Returns
        -------
        np.ndarray
            The motor positions. If color is an array the output shape will
            be (motors, colors).
        """
        # get color in units
        if units == 'same':
            pass
        else:
            color = wt_units.converter(color, units, self.units)
        # color must be array

        def is_numeric(obj):
            attrs = ['__add__', '__sub__', '__mul__', '__pow__']
            return all([hasattr(obj, attr) for attr in attrs] + [not hasattr(obj, '__len__')])
        if is_numeric(color):
            color = np.array([color])
        # evaluate
        if full and self.subcurve:
            out = []
            for c in color:
                source_color = np.array(self.source_color_interpolator.get_motor_positions(c))
                source_motor_positions = np.array(self.subcurve.get_motor_positions(
                    source_color, units=self.units, full=True)).squeeze()
                own_motor_positions = np.array(self.interpolator.get_motor_positions(c)).flatten()
                out.append(np.hstack((source_motor_positions, own_motor_positions)))
            out = np.array(out)
            return out.squeeze().T
        else:
            out = np.array([self.interpolator.get_motor_positions(c) for c in color])
            return out.T

    def get_source_color(self, color, units='same'):
        if not self.subcurve:
            return None
        # color must be array

        def is_numeric(obj):
            attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
            return all([hasattr(obj, attr) for attr in attrs] + [not hasattr(obj, '__len__')])
        if is_numeric(color):
            color = np.array([color])
        # get color in units
        if units == 'same':
            pass
        else:
            color = wt_units.converter(color, units, self.units)
        # evaluate
        return np.array([self.source_color_interpolator.get_motor_positions(c) for c in color])

    def interpolate(self, interpolate_subcurve=True):
        """ Generate the interploator object.  """
        self.interpolator = self.method(self.colors, self.units, self.motors)
        if self.subcurve and interpolate_subcurve:
            self.source_color_interpolator = self.method(
                self.colors, self.units, [self.source_colors])

    def map_colors(self, colors, units='same'):
        """ Map the curve onto new tune points using the curve's own interpolation method

        Parameters
        ----------
        colors : int or array
            The number of new points (between current limits) or the new points
            themselves.
        units : str (optional.)
            The input units if given as array. Default is same. Units of curve
            object are not changed by map_colors.
        """
        # get new colors in input units
        if isinstance(colors, int):
            limits = self.get_limits(units)
            new_colors = np.linspace(limits[0], limits[1], colors)
        else:
            new_colors = colors
        # convert new colors to local units
        if units == 'same':
            units = self.units
        new_colors = sorted(wt_units.converter(new_colors, units, self.units))
        # ensure that motor interpolators agree with current motor positions
        self.interpolate(interpolate_subcurve=True)
        # map own motors
        new_motors = []
        for motor_index, motor in enumerate(self.motors):
            positions = self.get_motor_positions(new_colors, full=False)[motor_index]
            new_motor = Motor(positions, motor.name)  # new motor objects
            new_motors.append(new_motor)
        # map source colors, subcurves
        if self.subcurve:
            new_source_colors = np.array(
                self.source_color_interpolator.get_motor_positions(new_colors)).squeeze()
            self.subcurve.map_colors(new_source_colors, units=self.units)
            self.source_colors.positions = new_source_colors
        # finish
        self.colors = new_colors
        self.motors = new_motors
        self.motor_names = [m.name for m in self.motors]
        for obj in self.motors:
            setattr(self, obj.name, obj)
        self.interpolate(interpolate_subcurve=True)

    def offset_by(self, motor, amount):
        """ Offset a motor by some ammount.

        Parameters
        ----------
        motor : number or str
            The motor index or name.
        amount : number
            The offset.

        See Also
        --------
        offset_to
        """
        # get motor index
        if type(motor) in [float, int]:
            motor_index = motor
        elif isinstance(motor, str):
            motor_index = self.motor_names.index(motor)
        else:
            print('motor type not recognized in curve.offset_by')
        # offset
        self.motors[motor_index].positions += amount
        self.interpolate()

    def offset_to(self, motor, destination, color, color_units='same'):
        """ Offset a motor such that it evaluates to `destination` at `color`.

        Parameters
        ----------
        motor : number or str
            The motor index or name.
        amount : number
            The motor position at color after offseting.
        color : number
            The color at-which to set the motor to amount.
        color_units : str (optional)
            The color units. Default is same.

        See Also
        --------
        offset_by
        """
        # get motor index
        if type(motor) in [float, int]:
            motor_index = motor
        elif isinstance(motor, str):
            motor_index = self.motor_names.index(motor)
        else:
            print('motor type not recognized in curve.offset_to')
        # get offset
        current_positions = self.get_motor_positions(color, color_units, full=False)
        offset = destination - current_positions[motor_index]
        # apply using offset_by
        self.offset_by(motor, offset)

    def plot(self, autosave=False, save_path='', title=None):
        """ Plot the curve.  """
        # count number of subcurves
        subcurve_count = 0
        total_motor_count = len(self.motors)
        current_curve = self
        all_curves = [self]
        while current_curve.subcurve:
            subcurve_count += 1
            total_motor_count += len(current_curve.subcurve.motors)
            current_curve = current_curve.subcurve
            all_curves.append(current_curve)
        all_curves = all_curves[::-1]
        # prepare figure
        num_subplots = total_motor_count + subcurve_count
        fig = plt.figure(figsize=(8, 2 * num_subplots))
        axs = grd.GridSpec(num_subplots, 1, hspace=0)
        # assign subplot indicies
        ax_index = 0
        ax_dictionary = {}
        lowest_ax_dictionary = {}
        for curve_index, curve in enumerate(all_curves):
            for motor_index, motor in enumerate(curve.motors):
                ax_dictionary[motor.name] = axs[ax_index]
                lowest_ax_dictionary[curve.interaction] = axs[ax_index]
                ax_index += 1
            if curve_index != len(all_curves):
                ax_index += 1
        # add scatter
        for motor_index, motor_name in enumerate(self.get_motor_names()):
            ax = plt.subplot(ax_dictionary[motor_name])
            xi = self.colors
            yi = self.get_motor_positions(xi)[motor_index]
            ax.scatter(xi, yi, c='k')
            ax.set_ylabel(motor_name)
            plt.xticks(self.colors)
            plt.setp(ax.get_xticklabels(), visible=False)
        # add lines
        for motor_index, motor_name in enumerate(self.get_motor_names()):
            ax = plt.subplot(ax_dictionary[motor_name])
            limits = curve.get_limits()
            xi = np.linspace(limits[0], limits[1], 1000)
            yi = self.get_motor_positions(xi)[motor_index].flatten()
            ax.plot(xi, yi, c='k')
        # get appropriate source colors
        source_color_arrs = {}
        for curve_index, curve in enumerate(all_curves):
            current_curve = self
            current_arr = self.colors
            for _ in range(len(all_curves) - curve_index - 1):
                current_arr = current_curve.get_source_color(current_arr)
                current_curve = current_curve.subcurve
            source_color_arrs[current_curve.interaction] = np.array(current_arr).flatten()
        # add labels
        for curve in all_curves:
            ax = plt.subplot(lowest_ax_dictionary[curve.interaction])
            plt.setp(ax.get_xticklabels(), visible=True)
            ax.set_xlabel(curve.interaction + ' color ({})'.format(curve.units))
            xtick_positions = self.colors
            xtick_labels = [str(np.around(x, 1)) for x in source_color_arrs[curve.interaction]]
            plt.xticks(xtick_positions, xtick_labels, rotation=45)
        # formatting details
        xmin = self.colors.min() - np.abs(self.colors[0] - self.colors[1])
        xmax = self.colors.max() + np.abs(self.colors[0] - self.colors[1])
        for ax in ax_dictionary.values():
            ax = plt.subplot(ax)
            plt.xlim(xmin, xmax)
            plt.grid()
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            yticks = ax.yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)
            yticks[-1].label1.set_visible(False)
        # title
        if title is None:
            title = self.name
        plt.suptitle(title)
        # save
        if autosave:
            if save_path[-3:] != 'png':
                image_path = save_path + self.name + '.png'
            else:
                image_path = save_path
            plt.savefig(image_path, transparent=True, dpi=300)
            plt.close(fig)

    def save(self, save_directory=None, plot=True, verbose=True, full=False):
        """ Save the curve.

        Parameters
        ----------
        save_directory : str (optional)
            The save directory. If not supplied, current working directory is
            used.
        plot : bool (optional)
            Toggle saving plot along with curve. Default is True.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        full : bool (optional)
            Include all files (if curve is stored in multiple files)

        Returns
        -------
        str
            The filepath of the saved curve.
        """
        # get save directory
        if save_directory is None:
            save_directory = os.getcwd()
        # save
        if self.kind == 'opa800':
            out_path = to_800_curve(self, save_directory)
        elif self.kind == 'poynting':
            out_path = to_poynting_curve(self, save_directory)
        elif self.kind in ['TOPAS-C', 'TOPAS-800']:
            kwargs = {}
            kwargs['old_filepaths'] = self.old_filepaths
            out_path = to_TOPAS_crvs(self, save_directory, self.kind, full=full, **kwargs)
        else:
            error_text = ' '.join(['kind', self.kind, 'does not know how to save!'])
            raise LookupError(error_text)
        # plot
        if plot:
            image_path = os.path.splitext(out_path)[0] + '.png'
            title = os.path.basename(os.path.splitext(out_path)[0])
            self.plot(autosave=True, save_path=image_path, title=title)
        # finish
        if verbose:
            print('curve saved at', out_path)
        return out_path


# --- curve import methods ------------------------------------------------------------------------


def from_800_curve(filepath):
    headers = wt_kit.read_headers(filepath)
    arr = np.genfromtxt(filepath).T
    colors = arr[0]
    grating = Motor(arr[1], 'Grating')
    bbo = Motor(arr[2], 'BBO')
    mixer = Motor(arr[3], 'Mixer')
    motors = [grating, bbo, mixer]
    interaction = headers['interaction']
    path, name, suffix = wt_kit.filename_parse(filepath)
    curve = Curve(colors, 'wn', motors, name=name, interaction=interaction,
                  kind='opa800', method=Spline)
    return curve


def from_poynting_curve(filepath, subcurve=None):
    print('FROM POYNTING CURVE', filepath, subcurve)
    # read from file
    headers = wt_kit.read_headers(filepath)
    arr = np.genfromtxt(filepath).T
    names = headers['name']
    # colors
    colors = arr[0]
    # motors
    motors = []
    for i in range(1, len(headers['name'])):
        motors.append(Motor(arr[i], names[i]))
    # kwargs
    kwargs = {}
    kwargs['interaction'] = headers['interaction']
    kwargs['kind'] = 'poynting'
    kwargs['method'] = Linear
    kwargs['name'] = wt_kit.filename_parse(filepath)[1]
    if subcurve is not None:
        kwargs['subcurve'] = subcurve
        kwargs['source_colors'] = Motor(colors, 'wn')
    # finish
    curve = Curve(colors, 'wn', motors, **kwargs)
    return curve


def from_TOPAS_crvs(filepaths, kind, interaction_string):
    """ Create a curve object from a TOPAS crv file

    Parameters
    ----------
    filepaths : list of str [base, mixer 1, mixer 2, mixer 3]
        Paths to all crv files for OPA. Filepaths may be None if not needed /
        not applicable.
    kind : {'TOPAS-C', 'TOPAS-800'}
        The kind of TOPAS represented.
    interaction_string : str
        Interaction string for this curve, in the style of Light Conversion -
        e.g. 'NON-SF-NON-Sig'.

    Returns
    ------
    WrightTools.tuning.curve.Curve object
    """
    TOPAS_interactions = TOPAS_interaction_by_kind[kind]
    # setup to recursively import data
    interactions = interaction_string.split('-')
    interaction_strings = []  # most subservient tuning curve comes first
    idx = 3
    while idx >= 0:
        if not interactions[idx] == 'NON':
            interaction_strings.append('NON-' * idx + '-'.join(interactions[idx:]))
        idx -= 1
    # create curve objects, starting from most subservient curve
    subcurve = None
    for interaction_string in interaction_strings:
        # open appropriate crv
        interactions = interaction_string.split('-')
        curve_index = next((i for i, v in enumerate(interactions) if v != 'NON'), -1)
        crv_path = filepaths[-(curve_index + 1)]
        with open(crv_path, 'r') as crv:
            crv_lines = crv.readlines()
        # collect information from file
        for i in range(len(crv_lines)):
            if crv_lines[i].rstrip() == interaction_string:
                line_index = i + TOPAS_interactions[interaction_string][0]
                num_tune_points = int(crv_lines[line_index - 1])
        # get the actual array
        lis = []
        for i in range(line_index, line_index + num_tune_points):
            line_arr = np.fromstring(crv_lines[i], sep='\t')
            lis.append(line_arr)
        arr = np.array(lis).T
        # create the curve
        source_colors = Motor(arr[0], 'source colors')
        colors = arr[1]
        motors = []
        for i in range(3, len(arr)):
            motor_name = TOPAS_interactions[interaction_string][1][i - 3]
            motor = Motor(arr[i], motor_name)
            motors.append(motor)
            name = wt_kit.filename_parse(crv_path)[1]
        curve = Curve(colors, 'nm', motors, name, interaction_string,
                      kind, method=Linear,
                      subcurve=subcurve, source_colors=source_colors)
        subcurve = curve.copy()
    # finish
    setattr(curve, 'old_filepaths', filepaths)
    return curve


# --- curve writing methods -----------------------------------------------------------------------


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
    timestamp = wt_kit.TimeStamp()
    out_name = curve.name.split('-')[0] + '- ' + timestamp.path
    out_path = os.path.join(save_directory, out_name + '.curve')
    # save
    headers = collections.OrderedDict()
    headers['file created'] = timestamp.RFC3339
    headers['interaction'] = curve.interaction
    headers['name'] = ['Color (wn)', 'Grating', 'BBO', 'Mixer']
    wt_kit.write_headers(out_path, headers)
    with open(out_path, 'ab') as f:
        np.savetxt(f, out_arr.T, fmt=['%.2f', '%.5f', '%.5f', '%.5f'],
                   delimiter='\t')
    return out_path


def to_poynting_curve(curve, save_directory):
    # ensure curve is in wn
    curve = curve.copy()
    curve.convert('wn')
    # array
    colors = curve.colors
    motors = curve.motors
    out_arr = np.zeros([3, len(colors)])
    out_arr[0] = colors
    out_arr[1:3] = np.array([motor.positions for motor in motors])
    # filename
    timestamp = wt_kit.TimeStamp()
    out_name = curve.name.split('-')[0] + '- ' + timestamp.path
    out_path = os.path.join(save_directory, out_name + '.curve')
    # save
    headers = collections.OrderedDict()
    headers['file created'] = timestamp.RFC3339
    headers['interaction'] = curve.interaction
    headers['name'] = ['Color (wn)', 'Phi', 'Theta']
    wt_kit.write_headers(out_path, headers)
    with open(out_path, 'ab') as f:
        np.savetxt(f, out_arr.T, fmt=['%.2f', '%.0f', '%.0f'],
                   delimiter='\t')
    # save subcurve
    if curve.subcurve:
        curve.subcurve.save(save_directory=save_directory)
    return out_path


def to_TOPAS_crvs(curve, save_directory, kind, full, **kwargs):
    TOPAS_interactions = TOPAS_interaction_by_kind[kind]
    # unpack
    curve = curve.copy()
    curve.convert('nm')
    old_filepaths = kwargs['old_filepaths']
    interaction_string = curve.interaction
    # open appropriate crv
    interactions = interaction_string.split('-')
    curve_index = next((i for i, v in enumerate(interactions) if v != 'NON'), -1)
    curve_index += 1
    curve_index = len(old_filepaths) - curve_index
    crv_path = old_filepaths[curve_index]
    if full:
        # copy other curves over as well
        for i, p in enumerate(old_filepaths):
            print(i, p, curve_index)
            if i == curve_index:
                continue
            if p is None:
                continue
            print(i, p)
            d = os.path.join(save_directory, os.path.basename(p))
            shutil.copy(p, d)
    with open(crv_path, 'r') as crv:
        crv_lines = crv.readlines()
    # collect information from file
    for i in range(len(crv_lines)):
        if crv_lines[i].rstrip() == interaction_string:
            line_index = i + TOPAS_interactions[interaction_string][0]
            num_tune_points = int(crv_lines[line_index - 1])
    # construct to_insert (dictionary of arrays)
    to_insert = collections.OrderedDict()
    if interaction_string == 'NON-NON-NON-Sig':  # must generate idler
        # read spitfire color from crv
        spitfire_output = float(crv_lines[line_index - 4].rstrip())
        # create signal array from curve
        signal_arr = np.zeros([7, len(curve.colors)])
        signal_arr[0] = spitfire_output
        signal_arr[1] = curve.colors
        signal_arr[2] = 4
        for i in range(4):
            signal_arr[3 + i] = curve.motors[i].positions
        # create idler aray
        idler_arr = signal_arr.copy()
        idler_arr[1] = 1 / ((1 / spitfire_output) - (1 / curve.colors))
        # construct to_insert
        to_insert['NON-NON-NON-Sig'] = signal_arr
        to_insert['NON-NON-NON-Idl'] = idler_arr
    elif interaction_string == 'NON-NON-NON-Idl':  # must generate signal
        # read spitfire color from crv
        spitfire_output = float(crv_lines[line_index - 4].rstrip())
        # create idler array from curve
        idler_arr = np.zeros([7, len(curve.colors)])
        idler_arr[0] = spitfire_output
        idler_arr[1] = curve.colors
        idler_arr[2] = 4
        for i in range(4):
            idler_arr[3 + i] = curve.motors[i].positions
        # create idler aray
        signal_arr = idler_arr.copy()
        signal_arr[1] = 1 / ((1 / spitfire_output) - (1 / curve.colors))
        # construct to_insert
        to_insert['NON-NON-NON-Sig'] = signal_arr
        to_insert['NON-NON-NON-Idl'] = idler_arr
    # TOPAS800 DFG (3 motor mier)
    elif interaction_string in ['DF1-NON-NON-Sig', 'DF2-NON-NON-Sig'] and curve.kind == 'TOPAS-800':
        # create array from curve
        arr = np.zeros([6, len(curve.colors)])
        arr[0] = curve.source_colors.positions
        arr[1] = curve.colors
        arr[2] = 3
        arr[3] = curve.motors[0].positions
        arr[4] = curve.motors[1].positions
        arr[5] = curve.motors[2].positions
        to_insert[interaction_string] = arr
    else:  # all single-motor mixer processes
        # create array from curve
        arr = np.zeros([4, len(curve.colors)])
        arr[0] = curve.source_colors.positions
        arr[1] = curve.colors
        arr[2] = 1
        arr[3] = curve.motors[0].positions
        to_insert[interaction_string] = arr
    # generate output
    out_lines = copy.copy(crv_lines)
    for interaction_string, arr in to_insert.items():
        # get current properties of out_lines
        for i in range(len(crv_lines)):
            if crv_lines[i].rstrip() == interaction_string:
                line_index = i + TOPAS_interactions[interaction_string][0]
                num_tune_points = int(crv_lines[line_index - 1])
        # prepare array for addition
        arr = arr.T
        # TOPAS wants curves to be ascending in nm
        #   curves get added 'backwards' here
        #   so arr should be decending in nm
        if arr[0, 1] < arr[-1, 1]:
            arr = np.flipud(arr)
        # remove old points
        del out_lines[line_index - 1:line_index + num_tune_points]
        # add strings to out_lines
        for row in arr:
            line = ''
            for value in row:
                # the number of motors must be written as an integer for TOPAS
                if value in [1, 3, 4]:
                    value_as_string = str(int(value))
                else:
                    value_as_string = '%f.6' % value
                    portion_before_decimal = value_as_string.split('.')[0]
                    portion_after_decimal = value_as_string.split('.')[1].ljust(6, '0')
                    value_as_string = portion_before_decimal + '.' + portion_after_decimal
                line += value_as_string + '\t'
            line += '\n'
            out_lines.insert(line_index - 1, line)
        out_lines.insert(line_index - 1, str(len(curve.colors)) +
                         '\n')  # number of points of new curve
    # filename
    timestamp = wt_kit.TimeStamp().path
    out_name = curve.name.split('-')[0] + '- ' + timestamp
    out_path = os.path.join(save_directory, out_name + '.crv')
    # save
    with open(out_path, 'w') as new_crv:
        new_crv.write(''.join(out_lines).rstrip())
    return out_path
