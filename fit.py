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
            ystdev = np.std(values)
            good = False
            baselines = []
            for i in values:
                if abs(i)<= 3*ystdev:
                    baselines.append(i)
                else: good=True
            if good:
                baseline = np.average(baselines)
            else:
                # No data was rejected, so no visible baseline or no visible peak
                baseline = min(values)
        else:
            baseline = min(values)
        values -= baseline
        mean = sum(np.multiply(xi, values))/sum(values)
        width = np.sqrt(abs(sum((xi-mean)**2*values)/sum(values)))
        amp = max(values)
        p0 = [mean, width, amp, baseline]
        return p0

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
