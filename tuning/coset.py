'''
COSET
'''


### import ####################################################################


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


### coset class ###############################################################


class CoSet:
    
    def __add__(self, coset):
        # TODO: use interpolation to make this work if points are not evaluated
        #   at exactly the same positions...
        # TODO: checks and warnings...
        # TODO: __iadd__ (?)
        copy = self.__copy__()
        copy.offset_points += coset.offset_points
        return copy

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

    def __repr__(self):
        # when you inspect the object
        outs = []
        outs.append('WrightTools.tuning.coset.CoSet object at ' + str(id(self)))
        outs.append('  name: ' + self.name)
        outs.append('  control: ' + self.control_name)
        outs.append('  offset: ' + self.offset_name)
        return '\n'.join(outs)
        
    def copy(self):
        return self.__copy__()
        
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
        file_name = ' - '.join([self.name, wt_kit.get_timestamp()]) + '.coset'
        file_path = os.path.join(save_directory, file_name)
        headers = collections.OrderedDict()
        headers['control'] = self.control_name
        headers['control units'] = self.control_units
        headers['offset'] = self.offset_name
        headers['offset units'] = self.offset_units
        file_path = wt_kit.write_headers(file_path, headers)
        X = np.vstack([self.control_points, self.offset_points]).T
        with open(file_path, 'a') as f: 
            np.savetxt(f, X, fmt='%8.6f', delimiter='\t')
        if plot:
            image_path = file_path.replace('.coset', '.png')
            self.plot(autosave=True, save_path=image_path)
        if verbose:
            print 'coset saved at {}'.format(file_path)


### coset load method #########################################################
        

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
