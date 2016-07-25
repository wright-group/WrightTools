'''
Calibration.
'''


### import ####################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import copy
import collections

import numpy as np

import scipy

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 14

from . import units as wt_units
from . import kit as wt_kit
from . import artists as wt_artists

debug = False


### calibration class #########################################################


class Calibration:

    def __init__(self, points, values, units, name='calibration', note=''):
        self.points = points
        self.values = values
        self.units = units
        self.name = name
        self.note = note
        self.sort()
        self.interpolate()
        
    def convert(self, units):
        self.points = wt_units.converter(self.points, self.units, units)
        self.units = units
        self.sort()
        self.interpolate()
        
    def get_limits(self, units='same'):
        '''
        Get the edges of the calibration object.

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
            return [self.points.min(), self.points.max()]
        else:
            units_points = wt_units.converter(self.points, self.units, units)
            return [units_points.min(), units_points.max()]
            
    def get_value(self, position, units='same'):
        # get control position in own units
        if not units == 'same':
            position = wt_units.converter(position, self.units, units)
        # get offset in own units using spline
        value = self.spline(position)
        return value
        
    def interpolate(self):
        self.spline = wt_kit.Spline(self.points, self.values)
        
    def map_points(self, points, units='same'):
        '''
        Map the points onto new points using interpolation.

        Parameters
        ----------
        points : int or array
            The number of new points (between current limits) or the new points
            themselves.
        units : str (optional.)
            The input units if given as array. Default is same. Units of coset
            object are not changed.
        '''
        # get new points in input units
        if type(points) == int:
            limits = self.get_limits(self.units)
            new_points = np.linspace(limits[0], limits[1], points)
        else:
            new_points = points
        # convert new points to local units
        if units == 'same':
            units = self.control_units
        new_points = wt_units.converter(new_points, units, self.control_units)
        new_points.sort()
        new_values = self.get_offset(new_points)
        # finish
        self.points = new_points
        self.values = new_values
        
    def plot(self, autosave=False, save_path=''):
        fig, gs = wt_artists.create_figure(cols=[1])
        ax = plt.subplot(gs[0])
        xi = self.points
        yi = self.values
        ax.plot(xi, yi, c='k', lw=2)
        ax.scatter(xi, yi, c='k')
        ax.grid()
        xlabel = self.units
        ax.set_xlabel(xlabel, fontsize=18)
        ylabel = 'calibration'
        ax.set_ylabel(ylabel, fontsize=18)
        wt_artists._title(fig, self.name)
        if autosave:
            plt.savefig(save_path, dpi=300, transparent=True, pad_inches=1)
            plt.close(fig)
        
    def save(self, save_directory=None, plot=True, verbose=True):
        if save_directory is None:
            save_directory = os.getcwd()
        file_name = ' - '.join([self.name, wt_kit.get_timestamp()]) + '.calibration'
        file_path = os.path.join(save_directory, file_name)
        headers = collections.OrderedDict()
        headers['name'] = self.name
        headers['file created'] = wt_kit.get_timestamp()
        headers['units'] = self.units
        headers['note'] = self.note
        file_path = wt_kit.write_headers(file_path, headers)
        X = np.vstack([self.points, self.values]).T
        with open(file_path, 'a') as f:
            np.savetxt(f, X, fmt='%8.6f', delimiter='\t')
        if plot:
            image_path = file_path.replace('.calibration', '.png')
            self.plot(autosave=True, save_path=image_path)
        if verbose:
            print('calibration saved at {}'.format(file_path))
    
    def sort(self):
        '''
        Points must be ascending.
        '''
        idxs = np.argsort(self.points)
        self.points = self.points[idxs]
        self.values = self.values[idxs]


### load method ###############################################################
        

def from_file(path):
    # get raw information from file
    headers = wt_kit.read_headers(path)
    arr = np.genfromtxt(path).T
    name = os.path.basename(path).split(' - ')[0]
    # construct calibration object
    points = arr[0]
    values = arr[1]
    units = headers['units']
    name = headers['name']
    note = headers['note']
    cal = Calibration(points=points, values=values, units=units, name=name,
                      note=note)
    # finish
    return cal
