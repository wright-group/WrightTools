"""
COSET
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import copy
import collections

import numpy as np

import scipy

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 14

from .. import units as wt_units
from .. import kit as wt_kit
from .. import artists as wt_artists

debug = False


# --- coset class ---------------------------------------------------------------------------------


class CoSet:

    def __add__(self, coset):
        # HOW THIS WORKS
        #
        # TODO: proper checks and warnings...
        # copy
        other_copy = coset.__copy__()
        self_copy = self.__copy__()
        # coerce other to own units
        other_copy.convert_control_units(self.control_units)
        other_copy.convert_offset_units(self.offset_units)
        # find new control points
        other_limits = other_copy.get_limits()
        self_limits = self_copy.get_limits()
        min_limit = max(other_limits[0], self_limits[0])
        max_limit = min(other_limits[1], self_limits[1])
        num_points = max(other_copy.control_points.size, self_copy.control_points.size)
        new_control_points = np.linspace(min_limit, max_limit, num_points)
        # coerce to new control points
        other_copy.map_control_points(new_control_points)
        self_copy.map_control_points(new_control_points)
        # add
        self_copy.offset_points += other_copy.offset_points
        return self_copy

    def __copy__(self):
        return copy.deepcopy(self)

    def __init__(self, control_name, control_units, control_points,
                 offset_name, offset_units, offset_points, name='coset'):
        self.control_name = control_name
        self.control_units = control_units
        self.control_points = control_points
        self.offset_name = offset_name
        self.offset_units = offset_units
        self.offset_points = offset_points
        self.name = name
        self.sort()
        self.interpolate()

    def __repr__(self):
        # when you inspect the object
        outs = []
        outs.append('WrightTools.tuning.coset.CoSet object at ' + str(id(self)))
        outs.append('  name: ' + self.name)
        outs.append('  control: ' + self.control_name)
        outs.append('  offset: ' + self.offset_name)
        return '\n'.join(outs)

    def coerce_offsets(self):
        """ Coerce the offsets to lie exactly along the interpolation positions.

        Can be thought of as 'smoothing' the coset.
        """
        self.map_control_points(self.control_points, units='same')

    def convert_control_units(self, units):
        self.control_points = wt_units.converter(self.control_points, self.control_units, units)
        self.sort()
        self.control_units = units
        self.interpolate()

    def convert_offset_units(self, units):
        self.offset_points = wt_units.converter(self.offset_points, self.offset_units, units)
        self.offset_units = units
        self.interpolate()

    def copy(self):
        return self.__copy__()

    def get_limits(self, units='same'):
        """ Get the edges of the coset object.

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
            return [self.control_points.min(), self.control_points.max()]
        else:
            units_points = wt_units.converter(self.control_points, self.control_units, units)
            return [units_points.min(), units_points.max()]

    def get_offset(self, control_position, input_units='same',
                   output_units='same'):
        # get control position in own units
        if not input_units == 'same':
            control_position = wt_units.converter(
                control_position, self.control_units, input_units)
        # get offset in own units using spline
        offset = self.spline(control_position)
        # convert offset to output units
        if not output_units == 'same':
            offset = wt_units.converter(offset, self.offset_units, output_units)
        # finish
        return offset

    def interpolate(self):
        self.spline = scipy.interpolate.InterpolatedUnivariateSpline(
            self.control_points, self.offset_points)

    def map_control_points(self, points, units='same'):
        """ Map the offset points onto new control points using interpolation.

        Parameters
        ----------
        points : int or array
            The number of new points (between current limits) or the new points
            themselves.
        units : str (optional.)
            The input units if given as array. Default is same. Units of coset
            object are not changed.
        """
        # get new points in input units
        if isinstance(points, int):
            limits = self.get_limits(self.control_units)
            new_points = np.linspace(limits[0], limits[1], points)
        else:
            new_points = points
        # convert new points to local units
        if units == 'same':
            units = self.control_units
        new_points = sorted(wt_units.converter(new_points, units, self.control_units))
        new_offsets = self.get_offset(new_points)
        # finish
        self.control_points = new_points
        self.offset_points = new_offsets

    def plot(self, autosave=False, save_path=''):
        fig, gs = wt_artists.create_figure(cols=[1])
        ax = plt.subplot(gs[0])
        xi = self.control_points
        yi = self.offset_points
        ax.plot(xi, yi, c='k', lw=2)
        ax.scatter(xi, yi, c='k')
        ax.grid()
        xlabel = self.control_name + ' (' + self.control_units + ')'
        ax.set_xlabel(xlabel, fontsize=18)
        ylabel = self.offset_name + ' (' + self.offset_units + ')'
        ax.set_ylabel(ylabel, fontsize=18)
        wt_artists._title(fig, self.name)
        if autosave:
            plt.savefig(save_path, dpi=300, transparent=True, pad_inches=1)
            plt.close(fig)

    def save(self, save_directory=None, plot=True, verbose=True):
        if save_directory is None:
            save_directory = os.getcwd()
        time_stamp = wt_kit.TimeStamp()
        file_name = ' - '.join([self.name, time_stamp.path]) + '.coset'
        file_path = os.path.join(save_directory, file_name)
        headers = collections.OrderedDict()
        headers['control'] = self.control_name
        headers['control units'] = self.control_units
        headers['offset'] = self.offset_name
        headers['offset units'] = self.offset_units
        file_path = wt_kit.write_headers(file_path, headers)
        X = np.vstack([self.control_points, self.offset_points]).T
        with open(file_path, 'ab') as f:
            np.savetxt(f, X, fmt=str('%8.6f'), delimiter='\t')
        if plot:
            image_path = file_path.replace('.coset', '.png')
            self.plot(autosave=True, save_path=image_path)
        if verbose:
            print('coset saved at {}'.format(file_path))

    def sort(self):
        """ Control points must be ascending.  """
        idxs = np.argsort(self.control_points)
        self.control_points = self.control_points[idxs]
        self.offset_points = self.offset_points[idxs]


# --- coset load method ---------------------------------------------------------------------------


def from_file(path):
    # get raw information from file
    headers = wt_kit.read_headers(path)
    arr = np.genfromtxt(path).T
    name = os.path.basename(path).split(' - ')[0]
    # construct coset object
    control_name = headers['control']
    control_units = headers['control units']
    control_points = arr[0]
    offset_name = headers['offset']
    offset_units = headers['offset units']
    offset_points = arr[1]
    coset = CoSet(control_name, control_units, control_points, offset_name,
                  offset_units, offset_points, name=name)
    # finish
    return coset
