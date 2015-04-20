import os
import copy
import time
import collections

import pickle

import numpy as np

from scipy.interpolate import griddata, interp1d

import matplotlib
import matplotlib.pyplot as plt

import kit

### data class #################################################################

class Axis:
    
    def __init__(self, points, units,
                 tolerance = None, file_idx = None,
                 name = None, label = None, label_seed = None):
        
        self.name = name
        self.tolerance = tolerance
        self.points = points
        self.units = units
        self.file_idx = file_idx        
        self.label_seed = label_seed        
        self.label = label
        
        #get units kind
        self.units_kind = None
        for dic in kit.unit_dicts:
            if self.units in dic.keys():
                self.units_kind = dic['kind']
        
        self.make_label()        
        
    def convert(self, destination_units):
        
        self.points = kit.unit_converter(self.points, self.units, destination_units)
        self.units = destination_units

    def make_label(self):
        
        self.label = self.name
        
    def min_max_step(self):
        
        _min = self.points.min()
        _max = self.points.max()
        _step = (_max-_min)/(len(self.points)-1)
        
        return _min, _max, _step
        
class Channel:
    
    def __init__(self, units,
                 file_idx,
                 znull = None, zmin = None, zmax = None, signed = False,
                 name = None, label = None, label_seed = None):
                     
        self.name = name
        self.units = units
        self.file_idx = file_idx
        self.signed = signed
        self.znull = znull
        self.zmin = zmin
        self.zmax = zmax

class Data:

    def __init__(self, axes, zis, channels, constants = {}, 
                 znull = None, zmin = None, zmax = None, 
                 name = None, source = None):
        '''
        central object for all data types                              \n
        create data objects by calling the methods of this script
        
        contains axes - a list of axes objects and zis - an array 
        of data gridded to those axes (in order)
        
        contains constants - a dictionary of conjugate variables. example: \n
        w1: [1300., 'nm', 'energy', 'label']
        '''
        
        #axes-------------------------------------------------------------------
        
        #check that no two names are the same
        self.axis_names = []
        for axis in axes:
            self.axis_names.append(axis.name)
        if not len(self.axis_names) == len(set(self.axis_names)):
            print 'init failed: each axis must have a unique name'
            return
                     
        self.axes = axes
        for axis in self.axes:
            setattr(self, axis.name, axis)
        
        #channels---------------------------------------------------------------
        
        self.channels = channels
        for arg in [znull, zmin, zmax]:
            for i in range(len(self.channels)):
                channel = self.channels[i]
                
            
        #other------------------------------------------------------------------
        
        self.zis = zis        
        self.constants = constants 
        
        
        self.name = name
        self.source = source
        
        
        
            
    def chop(self, *args, **kwargs):
        '''
        obtain a subset of the contained data   \n
        all axes must be accounted for          \n
        '''
        
        #organize arguments recieved--------------------------------------------

        axes_args = []
        chopped_constants = {}
        
        for arg in args:
            if type(arg) == str:
                axes_args.append(arg)
            elif type(arg) == dict:
                chopped_constants = arg
                
        verbose = True
        for name, value in kwargs.items():
            if name == 'verbose':
                verbose = value 
        
        #iterate!---------------------------------------------------------------
                
        #find iterated dimensions
        iterated_dimensions = []
        iterated_shape = [1]
        for name in self.axis_names:
            if not name in axes_args and not name in chopped_constants.keys():
                iterated_dimensions.append(name)
                length = len(getattr(self, name).points)
                iterated_shape.append(length)
    
        chopped_constants_everywhere = chopped_constants
        out = []
        for index in np.ndindex(tuple(iterated_shape)):
            
            #get chopped_constants correct for this iteration
            chopped_constants = chopped_constants_everywhere.copy()
            for i in range(len(index[1:])):
                idx = index[1:][i]
                name = iterated_dimensions[i]
                units = getattr(self, name).units
                position = getattr(self, name).points[idx]
                chopped_constants[name] = [position, units]

            #re-order array: [all_chopped_constants, channels, all_chopped_axes]
            
            transpose_order = []
            constant_indicies = []
            
            #handle constants first
            constants = list(self.constants) #copy
            for dim in chopped_constants.keys():
                idx = [idx for idx in range(len(self.axes)) if self.axes[idx].name == dim][0]
                transpose_order.append(idx + 1)
                #get index of nearest value
                val = chopped_constants[dim][0]
                val = kit.unit_converter(val, chopped_constants[dim][1], self.axes[idx].units)
                c_idx = np.argmin(abs(self.axes[idx].points - val))
                constant_indicies.append(c_idx)
                obj = Axis(self.axes[idx].points[c_idx], self.axes[idx].units )
                constants.append(obj)
    
            #now one for the channels
            transpose_order.append(0)
        
            #now handle axes
            axes_chopped = []
            for dim in axes_args:
                idx = [idx for idx in range(len(self.axes)) if self.axes[idx].name == dim][0]
                transpose_order.append(idx + 1)
                axes_chopped.append(self.axes[idx])
                
            #ensure that everything is kosher
            if len(transpose_order) == len(self.zis.shape):
                pass
            else:
                print 'chop failed: not enough dimensions specified'
                return
            if len(transpose_order) == len(set(transpose_order)):
                pass
            else:
                print 'chop failed: same dimension used twice'
                return
    
            #chop
            zis_chopped = self.zis.transpose(transpose_order)
            for idx in constant_indicies: 
                zis_chopped = zis_chopped[idx]
                
            #finish iteration 
            out.append([axes_chopped, zis_chopped, constants])
        
        #return-----------------------------------------------------------------
        
        if verbose:
            print 'chopped data into %d piece(s)'%len(out), 'in', axes_args

        return out
        
    def convert(self, units, verbose = True):
        '''
        convinience method                                                    \n
        converts all compatable constants and axes to units 
        '''
        
        #get kind of units
        for dic in kit.unit_dicts:
            if units in dic.keys():
                units_kind = dic['kind']
        
        for axis in self.axes + self.constants:
            if axis.units_kind == units_kind:
                axis.convert('wn')
                if verbose:
                    print 'axis', axis.name, 'converted'
        
    def copy(self):
        
        return copy.deepcopy(self)
        
    def dOD(self, signal_channel_index, reference_channel_index, 
            method = 'boxcar',
            verbose = True):
        '''
        for differential scans:  convert zi signal from dT to dOD
        '''    
    
        T =  self.zis[reference_channel_index]
        dT = self.zis[signal_channel_index]
        signal_channel = self.channels[signal_channel_index]
        
        if method == 'boxcar':
            #assume data collected with boxcar i.e.
            #sig = 1/2 dT
            #ref = T + 1/2 dT
            dT = 2 * dT
            out = -np.log10((T + dT) / T)
            
        self.zis[signal_channel_index] = out
            
        #reset znull, zmin, zmax------------------------------------------------
        
        signal_channel.znull = 0.
        signal_channel.zmin = out.min()
        signal_channel.zmax = out.max()
        signal_channel.signed = True
        
    def save(self, filepath = None, verbose = True):
        '''
        pickle the data object
        '''
        
        if not filepath:
            chdir = os.getcwd()
            timestamp = time.strftime('%Y.%m.%d %H_%M_%S')
            filepath = os.path.join(chdir, timestamp + ' data.p')
            
        pickle.dump(self, open(filepath, 'wb'))
        
        if verbose:
            print 'data saved at', filepath
        
        return filepath
        
    def zoom(self, factor, order=1, verbose = True):
        '''
        'zoom' the data array using spline interpolation of the requested order. \n
        the number of points along each axis is increased by factor of factor.   \n
        essentially a wrapper for scipy.ndimage.interpolation.zoom
        '''
        import scipy.ndimage
        
        #axes
        for axis in self.axes:
            axis.points = scipy.ndimage.interpolation.zoom(axis.points, factor, order=order)
        
        #data (don't zoom along channel dimension)
        zoomed_zis = []
        for i in range(len(self.zis)):
            zoomed_zis.append(scipy.ndimage.interpolation.zoom(self.zis[i], factor, order=order))
        self.zis = np.array(zoomed_zis)
        
        if verbose:
            print 'data zoomed - new shape:', self.zis.shape
        
        
### data manipulation and usage methods ########################################

def make_fit(**kwargs):
    """
    make a fit file filling in only the arguments specified
    kwargs must be lists or arrays of uniform size and 1D shape
    """
    n = len(kwargs.values()[0])
    out = np.zeros((n, 12))
    #first column is just row number (unless overwritten)
    out[:, 0] = range(n)
    for name, value in kwargs.items():
        #all kwargs have to be the same length to make an intelligable array
        if len(value) == n:
            if name in Fit.cols.keys():
                out[:, Fit.cols[name][0]] = value
            else:
                print 'name {0} is not an appropriate column name'.format(name)
                return
        else:
            print 'Error: not all columns are the same length:  len({0})={1}, len({2}) = {3}'.format(
                kwargs.keys()[0], n, name, len(value))
            return
    return out

def make_tune(obj, set_var, fname=None, amp='int', center='exp_val', fit=True,
              offset=None, write=True):
    """
        function for turning dat scans into tune files using exp value

        takes a dat class object and transforms it into a fit file

        need to specify which axis we need the expectation value from 
        (set_var; either 'x' or 'y'; the other axis will be called int_var)

        amp can measure either amplitude or integrated itensity

        offset:  the a point contained within the set_var range that you wish 
        to be the zero point--if such a point is included, the exp_values will
        be shifted relative to it.  This is convenient in tunetests if you want 
        to have a specific color you want to set zero delay to.
    """
    if set_var not in ['x', 'y', obj.xvar, obj.yvar]:
        print 'Error:  set_var type not supported: {0}'.format(set_var)
    # make sure obj type is appropriate and extract properties
    #zimin = obj.zi.min()
    tempzi = obj.zi - obj.znull
    if set_var in ['y', obj.yvar]:
        int_var = obj.xvar
        set_var = obj.yvar
        set_lis = obj.yi
        #int_lis = obj.xi
        axis = 1
    elif set_var in ['x', obj.xvar]:
        int_var = obj.yvar
        set_var = obj.xvar
        set_lis = obj.xi
        #int_lis = obj.yi
        axis = 0

    # decide what tune type this is
    # if exp value var is delay, call this zerotune, if mono, call it colortune
    if int_var in ['lm', 'wm']:
        fit_type = 'colortune'
    elif int_var in ['d1', 'd2']:
        fit_type = 'zerotune'
    else:
        # not sure what type of fit it is
        fit_type = 'othertune'
    if fit:
        # use least squares fitting to fill in tune values
        plsq = obj.fit_gauss(axis=set_var)
        obj_amp, obj_exp, obj_width, obj_y0 = plsq
    else:
        # use expectation values and explicit measurements to extract values
        # calculate the expectation value to get the peak center
        obj_exp = obj.center(axis=set_var, center=center)
        # calculate the width of the feature using the second moment
        obj_width = obj.exp_value(axis=set_var, moment=2)
        obj_width = np.sqrt(np.abs(obj_width - obj_exp**2))
        # also include amplitude
        if amp == 'int':
            # convert area to max amplitude assuming gaussian form factor
            obj_amp = obj.exp_value(axis=set_var, moment=0, norm=False)
            obj_amp = obj_amp / (np.sqrt(2*np.pi)* obj_width)
        elif amp == 'max':
            obj_amp = tempzi.max(axis=axis) - obj.znull
    # convert obj_width from stdev to fwhm
    obj_width *= 2*np.sqrt(2*np.log(2))
    # offset the values if specified
    if offset is not None:
        f_exp = interp1d(set_lis,obj_exp, kind='linear')
        off = f_exp(offset)
        obj_exp = obj_exp - off
    # convert color to nm for fit file
    if set_var in ['w1', 'w2', 'wm']:
        set_lis = 10**7 / set_lis
    # put wavelength in ascending order
    pts = zip(set_lis, obj_exp, obj_amp)
    pts.sort()
    pts = zip(*pts)
    set_lis, obj_exp, obj_amp = pts
    out = makefit(set_pt=set_lis, mu=obj_exp, amp=obj_amp, sigma=obj_width)
    # make a fit file using the expectation value data
    # first, make sure fname has proper format 
    # append descriptors to filename regardless of whether name is provided
    # emulates how COLORS treats naming
    if fit:
        auto = '{0} {1} fitted'.format(set_var, fit_type)
    elif center == 'exp_val':
        auto = '{0} {1} exp_value center'.format(set_var, fit_type)
    elif center == 'max':
        auto = '{0} {1} max value center'.format(set_var, fit_type)
    else:
        auto = '{0} {1}'.format(set_var, fit_type)
    # suffix:  let me add the .fit filename suffix
    if fname is not None:
        filepath, fname, filesuffix = filename_parse(fname)
        # path:  don't imply path if an absolute path is given
        fname = ' '.join([fname, auto])
        if filepath is None:
            filepath=obj.filepath
    else:
        # use object's filepath as default
        filepath = obj.filepath
        fname = auto
    
    if filepath is not None:
        fname = filepath + '\\' + fname
    fstr = find_name(fname, 'fit')
    if not fstr:
        print 'Could not write file without overwriting an existing file'
        print 'Aborting file write'
        return
    with file(fstr+'.fit', 'a') as exp_file:
        np.savetxt(exp_file, out, delimiter='\t', fmt='%.3f')
    print 'saved as {0}'.format(fstr+'.fit')

### data creation methods ######################################################

def from_COLORS(filepaths, znull = None,
                cols = None, verbose = True,
                name = None):

    #do we have a list of files or just one file?-------------------------------
    
    if type(filepaths) == list:
        file_example = filepaths[0]
    else:
        file_example = filepaths
        filepaths = [filepaths]
    
    #define format of dat file--------------------------------------------------
    
    if cols:
        pass
    else:
        num_cols = len(np.genfromtxt(file_example).T)
        if num_cols == 35:
            cols = 'v2'
        elif num_cols == 18:
            cols = 'v1'
        elif num_cols == 16:
            cols = 'v0'
        if verbose:
            print 'cols recognized as', cols, '(%d)'%num_cols
            
    if cols == 'v2':
        axes = collections.OrderedDict()
        axes['num']  = Axis(None, None, tolerance = 0.5,  file_idx = 0,  name = 'num',  label_seed = ['num'])
        axes['w1']   = Axis(None, 'nm', tolerance = 5.0,  file_idx = 1,  name = 'w1',   label_seed = ['1'])
        axes['w2']   = Axis(None, 'nm', tolerance = 5.0,  file_idx = 3,  name = 'w2',   label_seed = ['2'])
        axes['w3']   = Axis(None, 'nm', tolerance = 5.0,  file_idx = 5,  name = 'w3',   label_seed = ['3'])
        axes['wm']   = Axis(None, 'nm', tolerance = 5.0,  file_idx = 7,  name = 'wm',   label_seed = ['m'])
        axes['wa']   = Axis(None, 'nm', tolerance = 1.0,  file_idx = 8,  name = 'wm',   label_seed = ['a'])
        axes['dref'] = Axis(None, 'fs', tolerance = 25.0, file_idx = 10, name = 'dref', label_seed = ['ref'])
        axes['d1']   = Axis(None, 'fs', tolerance = 3.0,  file_idx = 12, name = 'd1',   label_seed = ['1'])
        axes['d2']   = Axis(None, 'fs', tolerance = 3.0,  file_idx = 14, name = 'd2',   label_seed = ['2'])
        channels = collections.OrderedDict()
        channels['ai0'] = Channel('V',  file_idx = 16, name = 'Signal 0',  label_seed = ['0'])
        channels['ai1'] = Channel('V',  file_idx = 17, name = 'Signal 1',  label_seed = ['1'])
        channels['ai2'] = Channel('V',  file_idx = 18, name = 'Signal 2',  label_seed = ['2'])
        channels['ai3'] = Channel('V',  file_idx = 19, name = 'Signal 3',  label_seed = ['3'])
        channels['ai4'] = Channel('V',  file_idx = 20, name = 'Signal 4',  label_seed = ['4'])
        channels['mc']  = Channel(None, file_idx = 21, name = 'Array Sig', label_seed = ['a'])
    elif cols == 'v1':
        #fix this later
        cols_v1 = {
            'num':  (0,    0.5, None, None, 'acquisition number'),
            'w1':   (1,    5.0, 'nm', 'nm', r'$\mathrm{\bar\nu_1=\bar\nu_m (cm^{-1})}$'),
            'w2':   (3,    5.0, 'nm', 'nm', r'$\mathrm{\bar\nu_2=\bar\nu_{2^{\prime}} (cm^{-1})}$'),
            'wm':   (5,    1.0, 'nm', 'nm', r'$\bar\nu_m / cm^{-1}$'),
            'd1':   (6,    3.0, 'fs', 'fs', r'$\mathrm{\tau_{2^{\prime} 1} (fs)}$'),
            't2p1': (6,    3.0, 'fs', 'fs', r'$\mathrm{\tau_{2^{\prime} 1}(fs)}$'),
            'd2':   (7,    3.0, 'fs', 'fs', r'$\mathrm{\tau_{21} (fs)}$'),
            't21':  (7,    3.0, 'fs', 'fs', r'$\mathrm{\tau_{21} (fs)}$'),
            'ai0':  (8,    0.0, 'V',  'V',  'Signal 0'),
            'ai1':  (9,    0.0, 'V',  'V',  'Signal 1'),
            'ai2':  (10,   0.0, 'V',  'V',  'Signal 2'),
            'ai3':  (11,   0.0, 'V',  'V',  'Signal 3')}
    elif cols == 'v0':
        #fix this later
        cols_v0 = {
            'num':  (0,    0.5,  None, None,  'acquisition number'),
            'w1':   (1,    2.0,  'nm', 'nm',  r'$\mathrm{\bar\nu_1=\bar\nu_m (cm^{-1})}$'),
            'w2':   (3,    2.0,  'nm', 'nm',  r'$\mathrm{\bar\nu_2=\bar\nu_{2^{\prime}} (cm^{-1})}$'),
            'wm':   (5,    0.25, 'nm', 'nm',  r'$\bar\nu_m / cm^{-1}$'),
            'lm':   (5,    0.25, 'nm', 'wn',  r'$\lambda_m / nm$'),
            't2p1': (6,    3.0,  'fs', 'fs',  r'$\mathrm{\tau_{2^{\prime} 1}(fs)}$'),
            'd2':   (8,    3.0,  'fs', 'fs',  r'$\mathrm{\tau_{21} (fs)}$'),
            't21':  (8,    3.0,  'fs', 'fs',  r'$\mathrm{\tau_{21} (fs)}$'),
            'ai0':  (10,   0.0,  'V',  'V',  'Signal 0'),
            'ai1':  (11,   0.0,  'V',  'V',  'Signal 1'),
            'ai2':  (12,   0.0,  'V',  'V',  'Signal 2'),
            'ai3':  (13,   0.0,  'V',  'V',  'Signal 3')}
            
    #import full array----------------------------------------------------------
            
    for i in range(len(filepaths)):
        dat = np.genfromtxt(filepaths[i]).T
        if verbose: print 'dat imported:', dat.shape
        if i == 0:
            arr = dat
        else:      
            arr = np.append(arr, dat, axis = 1)
        
    #recognize dimensionality of data-------------------------------------------
        
    axes_discover = axes.copy()
    for key in ['num', 'dref']: 
        axes_discover.pop(key) #remove dimensions that mess up discovery
    scanned, constant = discover_dimensions(arr, axes_discover)
    
    #get axes points------------------------------------------------------------
    
    #clean this up for readability later - blaise    
    
    for axis in scanned:
        #generate lists from data
        lis = sorted(arr[axis.file_idx])
        tol = axis.tolerance
        # values are binned according to their averages now, so min and max 
        #  are better represented
        xs = []
        # check to see if unique values are sufficiently unique
        # deplete to list of values by finding points that are within 
        #  tolerance
        while len(lis) > 0:
            # find all the xi's that are like this one and group them
            # after grouping, remove from the list
            set_val = lis[0]
            xi_lis = [xi for xi in lis if np.abs(set_val - xi) < tol]
            # the complement of xi_lis is what remains of xlis, then
            lis = [xi for xi in lis if not np.abs(xi_lis[0] - xi) < tol]
            xi_lis_average = sum(xi_lis) / len(xi_lis)
            xs.append(xi_lis_average)
        # create uniformly spaced x and y lists for gridding
        # infinitesimal offset used to properly interpolate on bounds; can
        #  be a problem, especially for stepping axis
        axis.points = np.linspace(min(xs)+1E-06,
                                  max(xs)-1E-06,
                                  len(xs))

    #grid data------------------------------------------------------------------
    
    points = tuple(arr[axis.file_idx] for axis in scanned)
    xi = tuple(np.meshgrid(*[axis.points for axis in scanned]))

    zis = []
    for key in channels.keys():
        channel = channels[key]
        zcol = channel.file_idx
        fill_value = 0.
        grid_i = griddata(points, arr[zcol], xi,
                          method='linear',fill_value=fill_value)
        zis.append(grid_i)
    zis = np.array(zis)
        
    #create data object---------------------------------------------------------

    data = Data(scanned, zis, channels.values(), constant, znull)
    
    #add extra stuff to data object---------------------------------------------

    data.source = filepaths
    
    if not name:
        name = kit.filename_parse(file_example)[1]
    data.name = name
    
    #return---------------------------------------------------------------------
    
    if verbose:
        print 'axis_names:', data.axis_names
        print'zis_shape:', data.zis.shape
        
    return data
    
def from_JASCO(filepath, name = None, verbose = True):
    
    #check filepath-------------------------------------------------------------
    
    if os.path.isfile(filepath):
        if verbose: print 'found the file!'
    else:
        print 'Error: filepath does not yield a file'
        return

    #is the file suffix one that we expect?  warn if it is not!
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'txt':
        should_continue = raw_input('Filetype is not recognized and may not be supported.  Continue (y/n)?')
        if should_continue == 'y':
            pass
        else:
            print 'Aborting'
            return
            
    #import data----------------------------------------------------------------
    
    #now import file as a local var--18 lines are just txt and thus discarded
    data = np.genfromtxt(filepath, skip_header=18).T
    
    #construct data
    x_axis = Axis(data[0], 'nm', name = 'wm')
    zis = np.array([data[1]])
    data = Data([x_axis], zis, 'JASCO')
    
    #return---------------------------------------------------------------------
    
    return data
    
def from_pickle(filepath):
    
    return pickle.load(open(filepath, 'rb'))
    
### other ######################################################################

def discover_dimensions(arr, dimension_cols, verbose = True):
    '''
    Discover the dimensions of array arr. \n
    Watches the indicies contained in dimension_cols. Returns dictionaries of 
    axis objects [scanned, constant]. \n
    Constant objects have their points object initialized. Scanned dictionary is
    in order of scanning (..., zi, yi, xi). Both dictionaries are condensed
    into coscanning / setting.
    '''
    
    #sorry that this method is so convoluted and unreadable - blaise    
    
    input_cols = dimension_cols
    
    #import values--------------------------------------------------------------
    
    dc = dimension_cols 
    di = [dc[key].file_idx for key in dc.keys()]
    dt = [dc[key].tolerance for key in dc.keys()]
    du = [dc[key].units for key in dc.keys()]
    dk = [key for key in dc.keys()]
    dims = zip(di, dt, du, dk)

    #remove nan dimensions and bad dimensions------------------------------------------------------
    
    to_pop = []
    for i in range(len(dims)):
        if np.all(np.isnan(arr[dims[i][0]])):
            to_pop.append(i)

    to_pop.reverse()
    for i in to_pop:
        dims.pop(i)
    
    #which dimensions are equal-------------------------------------------------

    #find
    d_equal = np.zeros((len(dims), len(dims)), dtype=bool)
    d_equal[:, :] = True
    for i in range(len(dims)): #test
        for j in range(len(dims)): #against
            for k in range(len(arr[0])):
                upper_bound = arr[dims[i][0], k] + dims[i][1]
                lower_bound = arr[dims[i][0], k] - dims[i][1]
                test_point =  arr[dims[j][0], k]
                if upper_bound > test_point > lower_bound:
                    pass
                else:
                    d_equal[i, j] = False
                    break

    #condense
    dims_unaccounted = range(len(dims))
    dims_condensed = []
    while dims_unaccounted:
        dim_current = dims_unaccounted[0]
        index = dims[dim_current][0]
        tolerance = [dims[dim_current][1]]
        units = dims[dim_current][2]
        key = [dims[dim_current][3]]
        dims_unaccounted.pop(0)
        indicies = range(len(dims_unaccounted))
        indicies.reverse()
        for i in indicies:
            dim_check = dims_unaccounted[i]
            if d_equal[dim_check, dim_current]:
                tolerance.append(dims[dim_check][1])
                key.append(dims[dim_check][3])
                dims_unaccounted.pop(i)
        tolerance = max(tolerance)
        dims_condensed.append([index, tolerance, units, key])
    dims = dims_condensed
    
    #which dimensions are scanned-----------------------------------------------
    
    #find
    scanned = []
    constant_list = []
    for dim in dims:
        name = dim[3]
        index = dim[0]
        vals = arr[index]
        tolerance = dim[1]
        if vals.max() - vals.min() > tolerance:
            scanned.append([name, index, tolerance, None])
        else:
            constant_list.append([name, index, tolerance, arr[index, 0]])
            
    #order scanned dimensions (..., zi, yi, xi)
    first_change_indicies = []
    for axis in scanned:
        first_point = arr[axis[1], 0]
        for i in range(len(arr[0])):
            upper_bound = arr[axis[1], i] + axis[2]
            lower_bound = arr[axis[1], i] - axis[2]
            if upper_bound > first_point > lower_bound:
                pass
            else:
                first_change_indicies.append(i)
                break
    scanned_ordered = [scanned[i] for i in np.argsort(first_change_indicies)]
    scanned_ordered.reverse()
    
    #return---------------------------------------------------------------------

    #package back into ordered dictionary of objects

    scanned = collections.OrderedDict()
    for axis in scanned_ordered:
        key = axis[0][0]
        obj = input_cols[key]
        obj.name_seed = axis[0]
        scanned[key] = obj
        
    constant = collections.OrderedDict()
    for axis in constant_list:
        key = axis[0][0]
        obj = input_cols[key]
        obj.name_seed = axis[0]
        obj.points = axis[3]
        constant[key] = obj
        
    return scanned.values(), constant.values()
    
    


    
    