"""
fitting tools
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings
import collections
from collections import OrderedDict

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import scipy
from scipy import optimize as scipy_optimize

from . import data as wt_data
from . import kit as wt_kit
from . import artists as wt_artists


# --- helper functions ----------------------------------------------------------------------------


def get_baseline(values, deviations=3):
    """ Guess the baseline for a data set.

    Returns the average of all points in ``values`` less than n ``deviations``
    away from zero.

    Obviously, works best for data in which signal is relatively sparse so that
    noise dominates the standard deviation.

    As a fallback, returns the minimum of ``values``.

    Parameters
    ----------
    values : array-like
        The values to find the baseline of.
    deviations : integer (optional)
        The number of standard deviations away from zero to exclude.

    Returns
    -------
    float
        Baseline guess.
    """
    values_internal = values.copy()
    std = np.nanstd(values)
    values_internal[np.abs(values_internal) >= deviations * std] = np.nan
    baseline = np.nanmean(values_internal)
    if np.isnan(baseline):
        baseline = np.nanmin(values)

    # return np.nanmin(values) # TODO Fix baseline
    return baseline


# --- functions objects ---------------------------------------------------------------------------


class Function:

    def __init__(self, *args, **kwargs):
        pass

    def residuals(self, p, *args):
        return args[0] - self.evaluate(p, *args[1:])

    def fit(self, *args, **kwargs):
        if self.dimensionality == 1:
            args = tuple(wt_kit.remove_nans_1D(args))
            if len(args[0]) == 0:
                return [np.nan] * len(self.params)
        if 'p0' in kwargs:
            p0 = kwargs['p0']
        else:
            p0 = self.guess(*args)
        out = scipy_optimize.leastsq(self.residuals, p0, args=args)
        if out[1] not in [1, 2]:  # solution was not found
            return np.full(len(p0), np.nan)
        return out[0]


class ExpectationValue(Function):

    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 1
        self.params = ['value']
        self.limits = {}
        self.global_cutoff = None
        warnings.warn('ExpectationValue depreciated---use Moments',
                      DeprecationWarning, stacklevel=2)

    def evaluate(self, p, xi):
        """ Returns 1 at expectation value, 0 elsewhere.  """
        out = np.zeros(len(xi))
        out[np.argmin(np.abs(xi - p[0]))] = 1
        return out

    def fit(self, *args, **kwargs):
        y, x = args
        y_internal = np.ma.copy(y)
        x_internal = np.ma.copy(x)
        # apply global cutoff
        if self.global_cutoff is not None:
            y_internal[y < self.global_cutoff] = 0.
        # get sum
        sum_y = 0.
        for i in range(len(y_internal)):
            if np.ma.getmask(y_internal[i]):
                pass
            elif np.isnan(y_internal[i]):
                pass
            else:
                sum_y += y_internal[i]
        # divide by sum
        for i in range(len(y_internal)):
            if np.ma.getmask(y_internal[i]):
                pass
            elif np.isnan(y_internal[i]):
                pass
            else:
                y_internal[i] /= sum_y
        # get expectation value
        value = 0.
        for i in range(len(x_internal)):
            if np.ma.getmask(y_internal[i]):
                pass
            elif np.isnan(y_internal[i]):
                pass
            else:
                value += y_internal[i] * x_internal[i]
        return [value]

    def guess(self, values, xi):
        return 1.


class Exponential(Function):

    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 1
        self.params = ['amplitude', 'tau', 'offset']
        self.limits = {}

    def evaluate(self, p, xi):
        # check the sign convention
        if np.mean(xi) < 0:
            x = xi.copy() * -1
        else:
            x = xi.copy()
        # enforce limits
        for i, name in zip(range(len(p)), self.params):
            if name in self.limits.keys():
                p[i] = np.clip(p[i], *self.limits[name])
        # evaluate
        a, b, c = p
        return a * np.exp(-x / b) + c

    def guess(self, values, xi):
        p0 = np.zeros(3)
        p0[0] = values.max() - values.min()  # amplitude
        idx = np.argmin(abs(np.median(values) - values))
        p0[1] = abs(xi[idx])  # tau
        p0[2] = values.min()  # offset
        return p0


class Gaussian(Function):
    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 1
        self.params = ['mean', 'width', 'amplitude', 'baseline']
        self.limits = {'width': [0, np.inf]}

    def evaluate(self, p, xi):
        # enforce limits
        for i, name in zip(range(len(p)), self.params):
            if name in self.limits.keys():
                p[i] = np.clip(p[i], *self.limits[name])
        # evaluate
        m, w, amp, baseline = p
        out = amp * np.exp(-(xi - m)**2 / (2 * (w**2))) + baseline
        return out

    def guess(self, values, xi):
        values, xi = wt_kit.remove_nans_1D([values, xi])
        if len(values) == 0:
            return [np.nan] * 4
        Use_visible_baseline = False
        if Use_visible_baseline:
            baseline = get_baseline(values)
        else:
            baseline = min(values)
        values -= baseline
        mean = sum(np.multiply(xi, values)) / sum(values)
        width = np.sqrt(abs(sum((xi - mean)**2 * values) / sum(values)))
        amp = max(values)
        p0 = [mean, width, amp, baseline]
        return p0


class TwoD_Gaussian(Function):

    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 2
        self.params = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'baseline']
        self.limits = {'sigma_x': [0, np.inf], 'sigma_y': [0, np.inf]}

    def _Format_input(self, values, x, y):
        """ This function makes sure the values and axis are in an ok format for fitting and free of nans

        Parameters
        ----------
        arrs : list of 1D arrays
            The arrays to remove nans from

        Returns
        -------
        (v, xi, yi, OK)
        v : 1D array
            The values array, flattened if necessary, with any 'nan's removed
        xi : 1D array
            a flattend or meshgrided then flattened array of the x coordinate, nans removed
        yi : 1D array
            a flattend or meshgrided then flattened array of the y coordinate, nans removed
        OK : boolean
            True if the shapes of the inputs were compatable, False otherwise.

        """
        v = values.copy()
        xi = x.copy()
        yi = y.copy()

        if not len(v.shape) == 1 and v.shape == xi.shape and xi.shape == yi.shape:
            v = v.flatten()
            xi = xi.flatten()
            yi = yi.flatten()
        elif len(xi.shape) == 1 and len(yi.shape) == 1:
            if v.shape == (xi.shape[0], yi.shape[0]):
                xi, yi = np.meshgrid(xi, yi, indexing='ij')
                v = v.flatten()
                xi = xi.flatten()
                yi = yi.flatten()
            elif len(v.shape) == 1 and v.shape[0] == xi.shape[0] * yi.shape[0]:
                xi, yi = np.meshgrid(xi, yi, indexing='ij')
                xi = xi.flatten()
                yi = yi.flatten()
            elif v.shape == xi.shape:
                pass
            else:
                return np.nan, np.nan, np.nan, False
        else:
            return np.nan, np.nan, np.nan, False

        try:
            v, xi, yi = wt_kit.remove_nans_1D([v, xi, yi])
            return v, xi, yi, True
        except BaseException:
            return np.nan, np.nan, np.nan, False

    def residuals(self, p, *args):
        return args[0][0] - self.evaluate(p, args[0][1], args[0][2])

    def evaluate(self, p, x, y):
        # enforce limits
        for i, name in zip(range(len(p)), self.params):
            if name in self.limits.keys():
                p[i] = np.clip(p[i], *self.limits[name])
        # evaluate
        amp, xo, yo, sigma_x, sigma_y, theta, baseline = p
        a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
        return amp * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) *
                              (y - yo) + c * ((y - yo)**2))) + baseline

    def guess(self, v, x, y):
        if len(v) == 0:
            return [np.nan] * 4
        # Makes sure xi and yi are already the right size and shape
        values, xi, yi, ok = self._Format_input(v, x, y)

        if not ok:
            print("xi, yi are not the correct size and/or shape")
            return np.full(7, np.nan)

        Use_visible_baseline = False
        if Use_visible_baseline:
            baseline = get_baseline(values)
        else:
            baseline = min(values.flatten())
        values -= baseline
        x0 = (min(xi) + max(xi)) / 2.0  # sum(np.multiply(xi, values))/sum(values)
        y0 = (min(yi) + max(yi)) / 2.0  # sum(np.multiply(yi, values))/sum(values)
        sigma_x = np.sqrt(abs(sum((xi - x0)**2 * values) / sum(values)))
        sigma_y = np.sqrt(abs(sum((yi - y0)**2 * values) / sum(values)))
        theta = 0.0
        amp = max(values)
        p0 = [amp, x0, y0, sigma_x, sigma_y, theta, baseline]
        return p0

    def fit(self, values, x, y, **kwargs):

        if 'p0' in kwargs:
            p0 = kwargs['p0']
        else:
            p0 = self.guess(values, x, y)

        # Makes sure xi and yi are already the right size and shape & remove nans
        v, xi, yi, ok = self._Format_input(values, x, y)

        if not ok:
            print("xi, yi are not the correct size and/or shape")
            return np.full(7, np.nan)

        # optimize
        self.out = scipy_optimize.leastsq(self.residuals, p0, [v, xi, yi])

        if self.out[1] not in [1]:  # solution was not found
            return np.full(len(p0), np.nan)
        return self.out[0]


class Moments(Function):

    def __init__(self, subtract_baseline=False):
        Function.__init__(self)
        self.dimensionality = 1
        self.params = ['integral', 'one', 'two', 'three', 'four', 'baseline']
        self.limits = {}
        self.subtract_baseline = subtract_baseline

    def evaluate(self, p, xi):
        """ Currently just returns nans.  """
        # TODO: fix this (how should it work?!)
        return np.full(xi.shape, np.nan)

    def fit(self, *args, **kwargs):
        y, x = args
        y_internal = np.ma.copy(y)
        x_internal = np.ma.copy(x)
        # x must be ascending here, because of how np.trapz works
        if x_internal[0] > x_internal[-1]:
            y_internal = y_internal[::-1]
            x_internal = x_internal[::-1]
        # subtract baseline
        if self.subtract_baseline:
            baseline = get_baseline(y_internal)
            y_internal -= baseline
        else:
            baseline = np.nan
        # calculate
        # integral
        outs = [np.trapz(y_internal, x_internal)]
        # first moment (expectation value)
        one = np.nansum((x_internal * y_internal) / np.nansum(y_internal))
        outs.append(one)
        # second moment (central) (variance)
        outs.append(np.nansum((x_internal - outs[1]) * y_internal) / np.nansum(y_internal))
        sdev = np.sqrt(outs[2])
        # third and fourth moment (standardized)
        for n in range(3, 5):
            mu = np.nansum(((x_internal - outs[1])**n) *
                           y_internal) / (np.nansum(y_internal) * (sdev**n))
            outs.append(mu)
        # finish
        outs.append(baseline)
        return outs

    def guess(self, values, xi):
        return [0] * 6


# --- fitter --------------------------------------------------------------------------------------


class Fitter:

    def __init__(self, function, data, *args, **kwargs):
        self.function = function
        self.data = data.copy()
        self.axes = args
        self.axis_indicies = [self.data.axis_names.index(name) for name in self.axes]
        # will iterate over axes NOT fit
        self.not_fit_indicies = [self.data.axis_names.index(
            name) for name in self.data.axis_names if name not in self.axes]
        self.fit_shape = [self.data.axes[i].points.shape[0] for i in self.not_fit_indicies]
        print('fitter recieved data to make %d fits' % np.product(self.fit_shape))

    def run(self, channel=0, verbose=True):
        # get channel -----------------------------------------------------------------------------
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, str):
            channel_index = self.data.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        # transpose data --------------------------------------------------------------------------
        # fitted axes will be LAST
        transpose_order = list(range(len(self.data.axes)))
        self.axis_indicies.reverse()
        for i in range(len(self.axes)):
            ai = self.axis_indicies[i]
            ri = list(range(len(self.data.axes)))[-(i + 1)]
            transpose_order[ri], transpose_order[ai] = transpose_order[ai], transpose_order[ri]
        self.axis_indicies.reverse()
        self.data.transpose(transpose_order, verbose=False)
        # create output objects -------------------------------------------------------------------
        # model
        self.model = self.data.copy()
        if self.data.name:
            self.model.name = self.data.name + ' model'
        # outs
        self.outs = self.data.copy()
        for a in self.axes:
            self.outs.collapse(a, method='integrate')
        if self.data.name:
            self.outs.name = self.data.name + ' outs'
        self.outs.channels.pop(channel_index)
        params_channels = []
        for param in self.function.params:
            values = np.full(self.outs.shape, np.nan)
            channel = wt_data.Channel(values, units=None, znull=0, name=param)
            params_channels.append(channel)
        self.outs.channels = params_channels + self.outs.channels
        # do all fitting operations ---------------------------------------------------------------
        axes_points = [axis.points for axis in self.data.axes if axis.name in self.axes]
        timer = wt_kit.Timer(verbose=False)
        with timer:
            for idx in np.ndindex(*self.fit_shape):
                # do fit
                values = self.data.channels[channel_index].values[idx]
                fit_args = [values] + axes_points
                out = self.function.fit(*fit_args)
                # fill outs
                for i in range(len(self.function.params)):
                    self.outs.channels[i].values[idx] = out[i]
                # fill model
                model_data = self.function.evaluate(out, *axes_points)
                self.model.channels[channel_index].values[idx] = model_data
        if verbose:
            print('fitter done in %f seconds' % timer.interval)
        # clean up --------------------------------------------------------------------------------
        # model
        self.model.transpose(transpose_order, verbose=False)
        self.model.channels[channel_index].zmax = np.nanmax(
            self.model.channels[channel_index].values)
        self.model._update()
        # outs
        for i in range(len(self.function.params)):
            # give the data all at once
            channel = self.outs.channels[i]
            values = channel.values
            channel.zmax = np.nanmax(values)
            channel.znull = 0
            channel.zmin = np.nanmin(values)
        self.outs._update()
        return self.outs


# --- MultiPeakFitter -----------------------------------------------------------------------------


class MultiPeakFitter:
    """ Class which allows easy fitting and representation of aborbance data.

    Written by Darien Morrow. darienmorrow@gmail.com & dmorrow3@wisc.edu

    Currently only offers Gaussian and Lorentzian functions.
    Functions are paramaterized by FWHM, height, and center.
    Requires the use of WrightTools.Data objects.

    fittype = 2 may be used to fit a spectrum to maximize smoothness of model remainder.
    fittype = 0 may be used to fit a spectrum to minimize amplitude of model remainder.
    """

    def __init__(self, data, channel=0, name='', fittype=2, intensity_label='OD'):
        """
        Parameters
        ----------
        data : WrightTools.Data object
        channel : int
            channel of data object which has the z values to fit
        name : str
        fittype : int
            specifies order of differentiation to fit to---currently 0, 1, 2 are available
        intensity_label : str
        """
        self.data = data
        self.name = name
        self.fittype = fittype
        self.intensity_label = intensity_label
        # assertions
        if not data.axes[0].units_kind == 'energy':
            raise Exception('Yo, your axes is/are not of the correct kind. Hand me some energy.')
        if len(data.axes) > 1:
            raise Exception('Yo, your data must be 1D. Try again homey.')
        # get channel
        if isinstance(channel, int):
            self.channel_index = channel
        elif isinstance(channel, str):
            self.channel_index = self.data.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        # get diff of data
        self.zi = self.data.channels[self.channel_index].values
        self.diff = wt_kit.diff(self.data.axes[0].points, self.zi, order=self.fittype)

    def build_funcs(self, x, params, kinds, diff_order=0):
        """ Builds a new 1D function of many 1D function of various kinds.

        Parameters
        ----------
        x : array
            1D array containing points over which to return new function
        params : array
            1D array of ordered parameters for new functions---length 3n
        kinds : list
            List of kinds of functions to add---length n
        diff_order : int
            Specifies order of differentiated function to return

        Returns
        -------
        z : array
        """
        # instantiate array
        z = np.zeros(x.size)
        # loop through kinds and add functions to array
        for i, kind in enumerate(kinds):
            z += self.function(x, kind, params[i * 3 + 0], params[i * 3 + 1],
                               params[i * 3 + 2], diff_order=diff_order)
        return z

    def convert(self, destination_units):
        """ Exposes wt.data.convert() method to convert units of data object.

        Parameters
        ----------
        destination_units : str
        """
        self.data.convert(destination_units)

    def encode_params(self, names, kinds, params):
        """ Helper method to encode parameters of fit into an ordered dict of dicts.

        Parameters
        ----------
        names : list
            list of strings of length n
        kinds : list
            list of strings of length n
        params : array
            1D array of floats of length 3n

        Returns
        -------
        dic : ordered dict of dicts
        """
        dic = OrderedDict()
        for i in range(len(names)):
            dic[names[i]] = {'kind': kinds[i], 'FWHM': params[i * 3 + 0],
                             'intensity': params[i * 3 + 1], 'x0': params[i * 3 + 2]}
        return dic

    def extract_params(self, dic):
        """ Takes dictionary of fit parameters and returns tuple of extracted parameters
        that function method can work with.

        Parameters
        ----------
        dic : ordered dictionary of dicts
            Must contain keys: 'FWHM', 'intensity', 'x0', and 'kind'.

        Returns
        -------
        names, kinds, p0 : tuple
            names : list
            kinds : list
            p0 : array
        """
        p0 = np.zeros(len(dic) * 3)  # assumes each individual function is a 3 parameter function.
        names = []
        kinds = []
        i = 0
        for key, value in dic.items():
            names.append(key)
            p0[i] = value['FWHM']
            p0[i + 1] = value['intensity']
            p0[i + 2] = value['x0']
            i += 3
            kinds.append(value['kind'])
        return names, kinds, p0

    def fit(self, verbose=True):
        """ Fitting method that takes data and guesses (class attributes) and instantiates/updates
        fit_results, diff_model, and remainder (class attributes),

        Parameters
        ----------
        verbose : bool
            toggle talkback

        Attributes
        ----------
        fit_results : ordered dict of dict
        diff_model : array
            array offerring the final results of the fit in the native space of the fit
        remainder : array
            array offering the remainder (actual - fit) in the native space of the spectra
        """
        # generate arrays for fitting
        zi_diff = wt_kit.diff(self.data.axes[0].points,
                              self.data.channels[self.channel_index].values,
                              order=self.fittype)
        names, kinds, p0 = self.extract_params(self.guesses)
        # define error function

        def error(p, x, z):
            return z - self.build_funcs(x, p, kinds, diff_order=self.fittype)
        # perform fit
        timer = wt_kit.Timer(verbose=False)
        with timer:
            out = scipy_optimize.leastsq(error, p0, args=(self.data.axes[0].points, zi_diff))
            # write results in dictionary
            self.fit_results = self.encode_params(names, kinds, out[0])
        if verbose:
            print('fitting done in %f seconds' % timer.interval)
        # generate model
        self.diff_model = self.build_funcs(
            self.data.axes[0].points, out[0], kinds, diff_order=self.fittype)
        self.remainder = self.data.channels[self.channel_index].values - \
            self.build_funcs(self.data.axes[0].points, out[0], kinds, diff_order=0)

    def function(self, x, kind, FWHM, intensity, x0, diff_order=0):
        """ Returns a peaked distribution over array x.

        The peaked distributions are characterized by their FWHM, height, and center.
        The distributions are not normalized to the same value given the same parameters!
        Analytic derivatives are provided for sake of computational speed & accuracy.

        Parameters
        ----------
        x : array
            array of values to return function over
        kind : str
            Kind of function to return. Includes 'lorentzian' and 'gaussian'
        FWHM : float
            Full width at half maximum of returned function
        intensity : float
            Center intensity of returned function
        x0 : float
            x value of center of returned function
        diff_order : int
            order of differentiation to return. diff_order = 0 returns merely the function.
        """
        # TODO for order > 2, offer numerical derivatives.
        if kind == 'lorentzian':
            if diff_order == 0:
                return intensity * (0.5 * FWHM)**2 / ((x - x0)**2 + (.5 * FWHM)**2)
            elif diff_order == 1:
                return intensity * (0.5 * FWHM)**2 * (-1) * \
                    (((x - x0)**2 + (0.5 * FWHM)**2))**-2 * (2 * (x - x0))
            elif diff_order == 2:
                return intensity * (0.5 * FWHM)**2 * (2 * ((((x - x0)**2 + (0.5 * FWHM)**2))**-3)
                                                      * (2 * (x - x0))**2 + (-2) * (((x - x0)**2 + (0.5 * FWHM)**2))**-2)
            else:
                print('analytic derivative not pre-calculated')
        elif kind == 'gaussian':
            sigma = FWHM / 2.35482
            if diff_order == 0:
                return intensity * np.exp(-0.5 * ((x - x0) / sigma)**2)
            elif diff_order == 1:
                return intensity * np.exp(-0.5 * ((x - x0) / sigma)**2) * (-(x - x0) / (sigma**2))
            elif diff_order == 2:
                return intensity * np.exp(-0.5 * ((x - x0) / sigma)**2) * \
                    (((x - x0)**2 / (sigma**4)) - sigma**-2)
            else:
                print('analytic derivative not pre-calculated')
        else:
            raise Exception('kind not recognized!')

    def guess(self, guesses):
        """ Creates guess library for use in fitting.

        Parameters
        ----------
        guesses : ordered dict of dicts
            In order to play nicely with the function creation method, the internal
            dict, at minimum, should have entries specifying 'kind', 'FWHM', 'intensity', and 'x0'.
            The contents of guesses are used in the fit method.
        """
        if isinstance(guesses, OrderedDict) != True:
            raise Exception('guesses must be an OrderedDict')
        self.guesses = guesses

    def intensity_label_change(self, intensity_label):
        """ Helper method for changing label present in plot method.

        Parameters
        ----------
        intensity_label : str
        """
        self.intensity_label = intensity_label

    def plot(self,):
        """ Plot fit results.  """
        # get results
        names, kinds, params = self.extract_params(self.fit_results)
        num_funcs = len(kinds)
        # get color map for trace colors
        cm = wt_artists.colormaps['default']
        # create figure
        fig, gs = wt_artists.create_figure(width='single', cols=[1], nrows=2, default_aspect=.5)
        # as-taken
        ax = plt.subplot(gs[0, 0])
        ax.set_title(self.name + ' fittype = ' + str(self.fittype), fontsize=20)
        xi = self.data.axes[0].points
        ax.plot(xi, self.zi, color='k', linewidth=2)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylim(0, self.zi.max() + .005)
        ax.set_xlim(xi.min(), xi.max())
        ax.set_ylabel(self.intensity_label, fontsize=18)
        ax.grid()
        # fit results
        ax.plot(xi, self.remainder, color='k', linestyle='--', linewidth=2)
        for i, kind in enumerate(kinds):
            ax.plot(xi,
                    self.function(xi,
                                  kind,
                                  params[i * 3 + 0],
                                  params[i * 3 + 1],
                                  params[i * 3 + 2],
                                  diff_order=0),
                    color=cm((i + 1) / num_funcs),
                    linewidth=2)
            ax.axvline(x=params[i * 3 + 2], color=cm((i + 1) / num_funcs), linewidth=1)
        # diff
        ax = plt.subplot(gs[1, 0])
        # as-taken
        ax.plot(xi, self.diff, color='k', linewidth=2)
        ax.set_xlim(xi.min(), xi.max())
        ax.grid()
        label = r'$\mathsf{\frac{d^%i \mathrm{OD}}{d (\hslash \omega)^%i}}$' % (
            self.fittype, self.fittype)
        ax.set_ylabel(label, fontsize=18)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel(label, fontsize=18)
        ax.set_xlabel(self.data.axes[0].get_label())
        # fit results
        ax.plot(xi, self.diff_model, color='b', linewidth=2)
        for i, kind in enumerate(kinds):
            ax.axvline(x=params[i * 3 + 2], color=cm((i + 1) / num_funcs), linewidth=1)

    def save(self, path=os.getcwd(), fit_params=True, figure=True, verbose=True):
        """ Saves results and representation of fits. Saved files are timestamped.

        Parameters
        ----------
        path : str
            file path to save to
        fit_params : bool
            toggle to save fit parameters
        figure : bool
            toggle to save figure
        verbose : bool
            toggle talkback
        """
        # instantiate timestamp object
        timestamp = wt_kit.TimeStamp()
        if fit_params:
            params_path = os.path.join(path, ' '.join(
                (self.name, timestamp.path, 'fit_params.txt')))
            headers = collections.OrderedDict()
            headers['transition names'] = list(self.fit_results.keys())
            for state, d in self.fit_results.items():
                for prop, value in d.items():
                    headers[' '.join([state, prop])] = value
            write = wt_kit.write_headers(params_path, headers)
            if verbose:
                print('Parameters saved to:', write)
        if figure:
            fig = self.plot()
            fig_path = os.path.join(path, ' '.join((self.name, timestamp.path, 'fits.png')))
            write = wt_artists.savefig(fig_path, fig=fig, close=True)
            if verbose:
                print('Figure saved to:', write)


def leastsqfitter(p0, datax, datay, function, verbose=False, cov_verbose=False):
    """ Convenience method for using scipy.optmize.leastsq().

    Returns fit parameters and their errors.

    Parameters
    ----------
    p0 : list
        list of guess parameters to pass to function
    datax : array
        array of independent values
    datay : array
        array of dependent values
    function : function
        function object to fit data to. Must be of the callable form function(p, x)
    verbose : bool
        toggles printing of fit time, fit params, and fit param errors
    cov_verbose : bool
        toggles printing of covarience matrix

    Returns
    -------
    pfit_leastsq : list
        list of fit parameters. s.t. the error between datay and function(p, datax) is minimized
    perr_leastsq : list
        list of fit parameter errors (1 std)
    """

    timer = wt_kit.Timer(verbose=False)
    with timer:
        # define error function
        def errfunc(p, x, y): return y - function(p, x)
        # run optimization
        pfit_leastsq, pcov, infodict, errmsg, success = scipy_optimize.leastsq(
            errfunc, p0, args=(datax, datay), full_output=1, epsfcn=0.0001)
        # calculate covarience matrix
        # original idea https://stackoverflow.com/a/21844726
        if (len(datay) > len(p0)) and pcov is not None:
            s_sq = (errfunc(pfit_leastsq, datax, datay)**2).sum() / (len(datay) - len(p0))
            pcov = pcov * s_sq
            if cov_verbose:
                print(pcov)
        else:
            pcov = np.inf
        # calculate and write errors
        error = []
        for i in range(len(pfit_leastsq)):
            try:
                error.append(np.absolute(pcov[i][i])**0.5)
            except BaseException:
                error.append(0.00)
        perr_leastsq = np.array(error)
    # exit
    if verbose:
        print('fit params:       ', pfit_leastsq)
        print('fit params error: ', perr_leastsq)
        print('fitting done in %f seconds' % timer.interval)
    return pfit_leastsq, perr_leastsq

# --- testing -------------------------------------------------------------------------------------


if __name__ == '__main__':

    pass
