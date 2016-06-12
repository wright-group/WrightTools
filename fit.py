'''
fitting tools
'''


### import ####################################################################


import os
import collections

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import scipy
from scipy import optimize as scipy_optimize

import data as wt_data
import kit as wt_kit


### helper functions ##########################################################


def get_baseline(values, deviations=3):
    '''
    Guess the baseline for a data set.

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
    '''
    values_internal = values.copy()
    std = np.std(values)
    values_internal[np.abs(values_internal) <= deviations*std] = np.nan
    baseline = np.average(values_internal)
    if np.isnan(baseline):
        baseline = np.nanmin(values)
    return baseline


### functions objects #########################################################


class Function:

    def __init__(self):
        pass

    def residuals(self, p, *args):
        return args[0] - self.evaluate(p, *args[1:])

    def fit(self, *args, **kwargs):
        if self.dimensionality == 1:
            args = tuple(wt_kit.remove_nans_1D(args))
            if len(args[0]) == 0:
                return [np.nan]*len(self.params)
        if 'p0' in kwargs:
            p0 = kwargs['p0']
        else:
            p0 = self.guess(*args)
        out = scipy_optimize.leastsq(self.residuals, p0, args=args)
        if out[1] not in [1]:  # solution was not found
            return np.full(len(p0), np.nan)
        return out[0]


class ExpectationValue(Function):

    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 1
        self.params = ['value']
        self.limits = {}
        self.global_cutoff = None

    def evaluate(self, p, xi):
        '''
        Returns 1 at expectation value, 0 elsewhere.
        '''
        out = np.zeros(len(xi))
        out[np.argmin(np.abs(xi-p[0]))] = 1
        return out

    def fit(self, *args, **kwargs):
        y, x = args
        y_internal = np.ma.copy(y)
        x_internal = np.ma.copy(x)
        # apply global cutoff
        if self.global_cutoff is not None:
            y_internal[y<self.global_cutoff] = 0.
        # get sum
        sum_y = 0.
        for i in range(len(y_internal)):
            if np.ma.getmask(y_internal[i]) == True:
                pass
            elif np.isnan(y_internal[i]):
                pass
            else:
                sum_y += y_internal[i]
        # divide by sum
        for i in range(len(y_internal)):
            if np.ma.getmask(y_internal[i]) == True:
                pass
            elif np.isnan(y_internal[i]):
                pass
            else:
                y_internal[i] /= sum_y
        # get expectation value
        value = 0.
        for i in range(len(x_internal)):
            if np.ma.getmask(y_internal[i]) == True:
                pass
            elif np.isnan(y_internal[i]):
                pass
            else:
                value += y_internal[i]*x_internal[i]
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
            x = xi.copy()*-1
        else:
            x = xi.copy()
        # enforce limits
        for i, name in zip(range(len(p)), self.params):
            if name in self.limits.keys():
                p[i] = np.clip(p[i], *self.limits[name])
        # evaluate
        a, b, c = p
        return a*np.exp(-x/b)+c

    def guess(self, values, xi):
        p0 = np.zeros(3)
        p0[0] = values.max() - values.min()  # amplitude
        idx = np.argmin(abs(np.median(values)-values))
        p0[1] = abs(xi[idx])  # tau
        p0[2] = values.min()  # offset
        return p0

class Gaussian(Function):
    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 1
        self.params = ['mean','width','amplitude','baseline']
        self.limits = {'width': [0, np.inf]}

    def evaluate(self, p, xi):
        # enforce limits
        for i, name in zip(range(len(p)), self.params):
            if name in self.limits.keys():
                p[i] = np.clip(p[i], *self.limits[name])
        # evaluate
        m, w, amp, baseline = p
        return amp*np.exp(-(xi-m)**2/(2*w**2)) + baseline

    def guess(self, values, xi):
        values, xi = wt_kit.remove_nans_1D([values, xi])
        if len(values) == 0:
            return [np.nan]*4
        Use_visible_baseline = False
        if Use_visible_baseline:
            baseline = get_baseline(values)
        else:
            baseline = min(values)
        values -= baseline
        mean = sum(np.multiply(xi, values))/sum(values)
        width = np.sqrt(abs(sum((xi-mean)**2*values)/sum(values)))
        amp = max(values)
        p0 = [mean, width, amp, baseline]
        return p0

class TwoD_Gaussian(Function):

    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 2
        self.params = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta','baseline']
        self.limits = {'sigma_x': [0, np.inf],'sigma_y':[0,np.inf]}

    def _Format_input(self, values, x, y):
        '''
        This function makes sure the values and axis are in an ok format for fitting and free of nans

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

        '''
        v = values.copy()
        xi = x.copy()
        yi = y.copy()

        if not len(v.shape) == 1 and v.shape == xi.shape and xi.shape == yi.shape:
            v = v.flatten()
            xi = xi.flatten()
            yi = yi.flatten()
        elif len(xi.shape) == 1 and len(yi.shape) == 1:
            if v.shape == (xi.shape[0], yi.shape[0]):
                xi, yi = np.meshgrid(xi,yi,indexing='ij')
                v = v.flatten()
                xi = xi.flatten()
                yi = yi.flatten()
            elif len(v.shape) == 1 and v.shape[0] == xi.shape[0]*yi.shape[0]:
                xi, yi = np.meshgrid(xi,yi,indexing='ij')
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
        except:
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
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        return amp*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2))) + baseline

    def guess(self, v, x, y):
        if len(v) == 0:
            return [np.nan]*4
        # Makes sure xi and yi are already the right size and shape
        values, xi, yi, ok= self._Format_input(v, x, y)

        if not ok:
            print "xi, yi are not the correct size and/or shape"
            return np.full(7, np.nan)

        Use_visible_baseline = False
        if Use_visible_baseline:
            baseline = get_baseline(values)
        else:
            baseline = min(values.flatten())
        values -= baseline
        x0 = (min(xi)+max(xi))/2.0 #sum(np.multiply(xi, values))/sum(values)
        y0 = (min(yi)+max(yi))/2.0 #sum(np.multiply(yi, values))/sum(values)
        sigma_x = np.sqrt(abs(sum((xi-x0)**2*values)/sum(values)))
        sigma_y = np.sqrt(abs(sum((yi-y0)**2*values)/sum(values)))
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
        v, xi, yi, ok= self._Format_input(values, x, y)

        if not ok:
            print "xi, yi are not the correct size and/or shape"
            return np.full(7, np.nan)

        # optimize
        self.out = scipy_optimize.leastsq(self.residuals, p0, [v, xi, yi])

        if self.out[1] not in [1]:  # solution was not found
            return np.full(len(p0), np.nan)
        return self.out[0]


class Moments(Function):

    def __init__(self):
        Function.__init__(self)
        self.dimensionality = 1
        self.params = ['integral', 'one', 'two', 'three', 'four', 'baseline']
        self.limits = {}

    def evaluate(self, p, xi):
        '''
        Currently just returns nans.
        '''
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
        baseline = get_baseline(y_internal)
        y_internal -= baseline
        # calculate
        # integral
        outs = [np.trapz(y_internal, x_internal)]
        # first moment (expectation value)
        outs.append(np.sum((x_internal*y_internal) / np.sum(y_internal)))
        # second moment (central) (variance)
        outs.append(np.sum((x_internal-outs[1])*y_internal) / np.sum(y_internal))
        sdev = np.sqrt(outs[2])
        # third and fourth moment (standardized)
        for n in range(3, 5):
            mu = np.sum(((x_internal-outs[1])**n)*y_internal) / (np.sum(y_internal)*(sdev**n))
            outs.append(mu)
        # finish
        outs.append(baseline)
        return outs

    def guess(self, values, xi):
        return [0]*6


### fitter ####################################################################


class Fitter:

    def __init__(self, function, data, *args, **kwargs):
        self.function = function
        self.data = data.copy()
        self.axes = args
        self.axis_indicies = [self.data.axis_names.index(name) for name in self.axes]
        # will iterate over axes NOT fit
        self.not_fit_indicies = [self.data.axis_names.index(name) for name in self.data.axis_names if name not in self.axes]
        self.fit_shape = [self.data.axes[i].points.shape[0] for i in self.not_fit_indicies]
        print 'fitter recieved data to make %d fits'%np.product(self.fit_shape)

    def run(self, channel=0, verbose=True):
        # get channel ---------------------------------------------------------
        if type(channel) == int:
            channel_index = channel
        elif type(channel) == str:
            channel_index = self.data.channel_names.index(channel)
        else:
            print 'channel type', type(channel), 'not valid'
        # transpose data ------------------------------------------------------
        # fitted axes will be LAST
        transpose_order = range(len(self.data.axes))
        self.axis_indicies.reverse()
        for i in range(len(self.axes)):
            ai = self.axis_indicies[i]
            ri = range(len(self.data.axes))[-(i+1)]
            transpose_order[ri], transpose_order[ai] = transpose_order[ai], transpose_order[ri]
        self.axis_indicies.reverse()
        self.data.transpose(transpose_order, verbose=False)
        # create output objects -----------------------------------------------
        # model
        self.model = self.data.copy()
        self.model.name = self.data.name + ' model'
        # outs
        self.outs = self.data.copy()
        for a in self.axes:
            self.outs.collapse(a, method='integrate')
        self.outs.name = self.data.name + ' outs'
        self.outs.channels.pop(channel_index)
        params_channels = []
        for param in self.function.params:
            values = np.full(self.outs.shape, np.nan)
            channel = wt_data.Channel(values, units=None, znull=0, name=param)
            params_channels.append(channel)
        self.outs.channels = params_channels + self.outs.channels
        # do all fitting operations -------------------------------------------
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
            print 'fitter done in %f seconds'%timer.interval
        # clean up ------------------------------------------------------------
        # model
        self.model.transpose(transpose_order, verbose=False)
        self.model.channels[channel_index].zmax = np.nanmax(self.model.channels[channel_index].values)
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


### testing ###################################################################


if __name__ == '__main__':

    pass
