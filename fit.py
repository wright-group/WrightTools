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
        if 'p0' in kwargs:
            p0 = kwargs['p0']
        else:
            p0 = self.guess(*args)
        out = scipy_optimize.leastsq(self.residuals, p0, args=args)
        if out[1] not in [1]:  # solution was not found
            return np.full(len(p0), np.nan)
        return out[0]


class Exponential(Function):

    def __init__(self):
        Function.__init__(self)
        self.params = ['amplitude', 'tau', 'offset']
        self.limits = {}

    def evaluate(self, p, xi):
        # check the sign convention
        if np.mean(xi) < 0:
            x = xi.copy()*-1
        else:
            x = xi.copy()
        # evaluate
        for i, name in zip(range(len(p)), self.params):
            if name in self.limits.keys():
                p[i] = np.clip(p[i], *self.limits[name])
        a, b, c = p
        return a*np.exp(-x/b)+c

    def guess(self, values, xi):
        p0 = np.zeros(3)
        p0[0] = values.max() - values.min()  # amplitude
        idx = np.argmin(abs(np.median(values)-values))
        p0[1] = abs(xi[idx])  # tau
        p0[2] = values.min()  # offset
        return p0


### fitter ####################################################################


class Fitter:
    
    def __init__(self, function, data, *args):
        self.function = function
        self.data = data.copy()
        self.axes = args
        self.axis_indicies = [self.data.axis_names.index(name) for name in self.axes]
        # will iterate over axes NOT fit
        self.not_fit_indicies = [self.data.axis_names.index(name) for name in self.data.axis_names if name not in self.axes]
        self.fit_shape = [self.data.axes[i].points.shape[0] for i in self.not_fit_indicies]
        print 'fitter recieved data to make %d fits'%np.product(self.fit_shape)
        
    def run(self, channel_index=0, verbose=True):
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
        outs_dimensions = self.data.axis_names
        at = collections.OrderedDict()
        for arg in self.axes:
            axis = getattr(self.data, arg)
            at[arg] = [axis.points[0], axis.units]
            outs_dimensions.remove(arg)
            outs_dimensions
        args = tuple(outs_dimensions) + tuple([at])
        self.outs = self.data.chop(*args, verbose=False)[0]
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
