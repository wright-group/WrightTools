"""
Calibration.
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

from . import units as wt_units
from . import kit as wt_kit
from . import artists as wt_artists

debug = False


# --- define --------------------------------------------------------------------------------------


cmap = wt_artists.colormaps['default']
cmap.set_bad([0.75] * 3, 1.)
cmap.set_under([0.75] * 3)


def get_label(name, units):
    # units kind
    units_kind = None
    for dic in wt_units.unit_dicts:
        if units in dic.keys():
            units_kind = dic['kind']

    # units string
    d = getattr(wt_units, units_kind)
    units_string = r'$\mathsf{('
    units_string += d[units][2]
    units_string += r')}$'
    # finish
    out = ' '.join([name, units_string])
    return out


# --- calibration class ---------------------------------------------------------------------------


class Calibration:

    def __init__(self, axis_names, axis_units, points, values, name='calibration',
                 note=''):
        """ Container for unstructured calibration data.

        Parameters
        ----------
        points : list of lists
            List or list-like of lists. May be a 2D numpy array.
        axis_units : list of strings
            Units for each axis.
        values : 1D array-like
            Corrections for each coordinate.
        name : string (optional)
            Name of calibration. Default is 'calibration'
        note : string (optional)
            Note about calibration. Default is an empty string.
        """
        self.axis_names = axis_names
        self.axis_units = axis_units
        self.dimensionality = len(self.axis_names)
        self.points = np.array(points)
        self.values = np.array(values)
        self.name = name
        self.note = note
        self._interpolate()

    def _interpolate(self):
        """ (Re)create the interpolator using the current points and values.  """
        self._sort()
        if self.dimensionality == 1:
            self.interpolator = wt_kit.Spline(self.points[0], self.values, k=1, s=0)
        else:
            self.interpolator = scipy.interpolate.LinearNDInterpolator(self.points.T, self.values)

    def _sort(self):
        """ Sort data by all axes.

        First axis will be strictly ascending, second
        will be ascending within groups sharing the same value in the first
        axis etc.
        """
        ind = np.lexsort((self.points[::-1]))
        self.points = self.points[:, ind]
        self.values = self.values[ind]

    def append(self, points, values, units='same'):
        """ Add new data to the calibration.

        Parameters
        ----------
        points : list of lists
            List or list-like of lists. May be a 2D numpy array.
        values : 1D array-like
            Corrections for each coordinate.
        units : 'same' or list of strings (optional)
            Units of points. Default is 'same'.
        """
        points = np.array(points)
        values = np.array(values)
        if not units == 'same':
            for i, p in enumerate(points):
                points[i] = wt_units.converter(p, units[i], self.axis_units[i])
        self.points = np.hstack([self.points, points])
        self.values = np.hstack([self.values, values])
        self._interpolate()

    def convert(self, axis_units):
        """ Convert axes to new units.

        Parameters
        ----------
        axis_units : list of string
            New units for each axis.
        """
        for i, p in enumerate(self.points):
            self.points[i] = wt_units.converter(p, self.axis_units[i], axis_units[i])
        self.axis_units = axis_units
        self._interpolate()

    def get_positions(self, value, **kwargs):
        """
        Returns
        -------
        list of dictionaries
            All valid coordinates, defined for all axes. List may be empty.
        """
        # TODO: complete documentation
        if self.dimensionality == 1:
            # TODO:
            out = []
        else:
            # TODO: unhack...
            xi = np.linspace(self.points[1].min(), self.points[1].max(), 101)
            yi = np.array([self.get_value([kwargs['color'], x]) for x in xi])
            i = np.argmin(np.abs(yi - value))
            out = [{'color': kwargs['color'], 'angle': xi[i]}]
        return out

    def get_value(self, positions, units='same'):
        """ Get the value at some particular coordinate using linear interpolation.

        Parameters
        ----------
        positions : list of numbers
            Coordinate to evaluate at.
        units : 'same' or list of strings (optional)
            Units of positions. Default is 'same'.
        """
        return self.interpolator(*positions)

    def map_points(self, points, units='same'):
        """ Map the points onto new points using interpolation.

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
            limits = self.get_limits(self.units)
            new_points = np.linspace(limits[0], limits[1], points)
        else:
            new_points = points
        # convert new points to local units
        if units == 'same':
            units = self.control_units
        new_points = sorted(wt_units.converter(new_points, units, self.control_units))
        new_values = self.get_offset(new_points)
        # finish
        self.points = new_points
        self.values = new_values

    def plot(self, autosave=False, save_directory=None, file_name=None):
        """ Plot the calibration.

        Parameters
        ----------
        autosave : boolean (optional)
            Toggle automatic saving. Default is False.
        save_directory : string or None
            Location to save file. If None (default), uses current working
            directory.
        file_name : string (optional)
            Name for output image. If None (default), uses own name and a
            timestamp.

        Returns
        -------
        output
            Save path if autosave is True, fig object if autosave is False.

        .. note:: Only works for one and two-dimensional calibrations at this time.

        """
        if self.dimensionality == 1:
            fig, gs = wt_artists.create_figure(cols=[1])
            ax = plt.subplot(gs[0])
            xi = self.points[0]
            yi = self.values
            ax.scatter(xi, yi, c='k')
            xi = np.linspace(self.points[0].min(), self.points[0].max(), 1000)
            yi = self.interpolator(xi)
            ax.plot(xi, yi, c='k', lw=2)
            ax.grid()
            plt.xticks(rotation=45)
            ax.set_xlabel(get_label(self.axis_names[0], self.axis_units[0]), fontsize=18)

            wt_artists._title(fig, self.name)
        elif self.dimensionality == 2:
            fig, gs = wt_artists.create_figure(cols=[1, 'cbar'])
            ax = plt.subplot(gs[0, 0])
            ax.patch.set_facecolor([0.75] * 3)
            levels = np.linspace(self.values.min(), self.values.max(), 200)
            ax.tricontourf(self.points[0], self.points[1], self.values, cmap=cmap, levels=levels)
            ax.scatter(self.points[0], self.points[1], c='k', s=3, marker='o')
            ax.set_xlim(self.points[0].min(), self.points[0].max())
            ax.set_ylim(self.points[1].min(), self.points[1].max())
            ax.grid()
            plt.xticks(rotation=45)
            ax.set_xlabel(get_label(self.axis_names[0], self.axis_units[0]), fontsize=18)
            ax.set_ylabel(get_label(self.axis_names[1], self.axis_units[1]), fontsize=18)
            cax = plt.subplot(gs[0, 1])
            ticks = np.linspace(self.values.min(), self.values.max(), 11)
            wt_artists.plot_colorbar(cax=cax, ticks=ticks)
            plt.suptitle(self.name, fontsize=20)
        else:
            print('cannot plot---dimensionality too high')
            return
        # save
        if autosave:
            if save_directory is None:
                save_directory = os.getcwd()
            if file_name is None:
                time_stamp = wt_kit.TimeStamp()
                file_name = ' '.join([self.name, time_stamp.path]) + '.png'
            save_path = os.path.join(save_directory, file_name)
            wt_artists.savefig(save_path, fig=fig)
            return save_path
        return fig

    def save(self, save_directory=None, file_name=None, plot=True,
             verbose=True):
        """ Save the calibration.

        Parameters
        ----------
        save_directory : string or None (optional)
            Directory to save to. If None (default), saves to current working
            directory.
        file_name : string or None (optional)
            Name for output file. If None (default), uses own name and a
            timestamp.
        plot : boolean (optional)
            Toggle saving a plot of the calibration as well. Default is True.
        verbose : boolean (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        string
            Full path to saved file.
        """
        time_stamp = wt_kit.TimeStamp()
        # get save path
        if save_directory is None:
            save_directory = os.getcwd()
        if file_name is None:
            file_name = ' - '.join([self.name, time_stamp.path]) + '.calibration'
        file_path = os.path.join(save_directory, file_name)
        # write to file
        headers = collections.OrderedDict()
        headers['name'] = self.name
        headers['file created'] = time_stamp.RFC3339
        headers['axis names'] = self.axis_names
        headers['axis units'] = self.axis_units
        headers['note'] = self.note
        file_path = wt_kit.write_headers(file_path, headers)
        X = np.vstack([self.points, self.values]).T
        with open(file_path, 'ab') as f:
            np.savetxt(f, X, fmt=str('%6e'), delimiter='\t')
        # plot
        if plot:
            save_directory = save_directory
            file_name = os.path.basename(file_path).replace('.calibration', '.png')
            self.plot(autosave=True, save_directory=save_directory, file_name=file_name)
        # finish
        if verbose:
            print('calibration saved at {}'.format(file_path))
        return file_path


# --- load method ---------------------------------------------------------------------------------


def from_file(path):
    # get raw information from file
    headers = wt_kit.read_headers(path)
    arr = np.genfromtxt(path).T
    name = os.path.basename(path).split(' - ')[0]
    # construct calibration object
    points = arr[0:-1]
    values = arr[-1]
    axis_names = headers['axis names']
    axis_units = headers['axis units']
    name = headers['name']
    note = headers['note']
    c = Calibration(axis_names, axis_units, points, values, name=name, note=note)
    # finish
    return c
