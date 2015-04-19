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
    
    def __init__(self, points, input_units, name = None, label = None):
        
        self.name = name
        if not label:
            self.label = self.name
        else:
            self.label = label
        
        self.points = points

        self.units = input_units
        
        self.units_kind = None
        for dic in kit.unit_dicts:
            if self.units in dic.keys():
                self.units_kind = dic['kind']
        
    def convert(self, destination_units):
        
        self.points = kit.unit_converter(self.points, self.units, destination_units)
        self.units = destination_units

class Data:

    def __init__(self, axes, zis, zvars, constants = {}, 
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
        
        #data-------------------------------------------------------------------        
        
        if not znull:            
            self.znull = zis.min()
        else:
            self.znull = znull
            
        if not zmin:            
            self.zmin = zis.min()
        else:
            self.zmin = zmin
            
        if not zmax:            
            self.zmax = zis.max()
        else:
            self.zmax = zmax
            
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
            constants = self.constants.copy()
            for dim in chopped_constants.keys():
                idx = [idx for idx in range(len(self.axes)) if self.axes[idx].name == dim][0]
                transpose_order.append(idx + 1)
                #get index of nearest value
                val = chopped_constants[dim][0]
                val = kit.unit_converter(val, chopped_constants[dim][1], self.axes[idx].units)
                c_idx = np.argmin(abs(self.axes[idx].points - val))
                constant_indicies.append(c_idx)
                constants[dim] = [self.axes[idx].points[c_idx], #add to constants dictionary
                                  self.axes[idx].units, self.axes[idx].units_kind,
                                  self.axes[idx].label] 
    
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
        
    def copy(self):
        
        return copy.deepcopy(self)
        
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

def from_COLORS(filepaths, xvar = None, yvar = None, zvar = None,
                grid_factor = 2, znull = None,
                cols = None, verbose = True,
                name = None):
    '''
    here zvar corresponds to the variable scanned over many .dat files...
    '''
    
    #do we have a list of files or just one file?-------------------------------
    
    if type(filepaths) == list:
        movie = True
        file_example = filepaths[0]
    else:
        movie = False
        file_example = filepaths
        filepaths = [filepaths]
    
    #define format of dat file--------------------------------------------------
    
    cols_v2 = {  #index #tolerance #units #name
        'num':  (0,     0.5,       None, 'acquisition number'),
        'w1':   (1,     5.0,       'nm', '1'),
        'w2':   (3,     5.0,       'nm', '2'),
        'w3':   (5,     5.0,       'nm', '3'),
        'wm':   (7,     1.0,       'nm', 'm'),
        'wa':   (8,     1.0,       'nm', 'array'),
        'dref': (10,    25.0,      'fs', 'ref'),
        'd1':   (12,    3.0,       'fs', '2^{\prime} 1'),
        'd2':   (14,    3.0,       'fs', '2 1'),
        'ai0':  (16,    0.0,       'V',  'Signal 0'),
        'ai1':  (17,    0.0,       'V',  'Signal 1'),
        'ai2':  (18,    0.0,       'V',  'Signal 2'),
        'ai3':  (19,    0.0,       'V',  'Signal 3'),
        'ai4':  (20,    0.0,       'V',  'Signal 4'),
        'mc':   (21,    0.0,       'au', 'Array Signal')}

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
    
    zvars = collections.OrderedDict()
    zvars['ai0'] = None
    zvars['ai1'] = None
    zvars['ai2'] = None
    zvars['ai3'] = None
    
    if cols == 'v2':
        datCols = cols_v2
    elif cols == 'v1':
        datCols = cols_v1
    elif cols == 'v0':
        datCols = cols_v0
    else:
        #guess based on when the file was made
        v1_time = 1349067600.0
        v2_time = 1395723600.0    
        file_date = os.path.getctime(file_example)
        if file_date > v2_time:
            cols = 'v2'
            datCols = cols_v2
        elif file_date > v1_time:
            cols = 'v1'
            datCols = cols_v1
        else:
            # file is older than all other dat versions
            cols = 'v0'
            datCols = cols_v0
        if verbose: print 'cols', cols
    cols=cols
    
    #add array to zvars if version 2 dat file
    if cols == 'v2': zvars['mc'] = None
        
    #recognize dimensionality of data-------------------------------------------
        
    if xvar == None:
        if yvar == None and zvar == None:
            #import data for sake of discover_dimensions
            
            for i in range(len(filepaths)):
                dat = np.genfromtxt(filepaths[i]).T
                if i == 0:
                    arr = dat
                else:      
                    arr = np.append(arr, dat, axis = 1)
                #arr.append(np.genfromtxt(filepath).T)
            #arr = np.array(arr)
            #construct dimension_cols dictionary
            dimension_cols = {
                            'w1':   (1,     5.0,       'nm', '1'),
                            'w2':   (3,     5.0,       'nm', '2'),
                            'w3':   (5,     5.0,       'nm', '3'),
                            'wm':   (7,     1.0,       'nm', 'm'),
                            'wa':   (8,     1.0,       'nm', 'array'),
                            'd1':   (12,    3.0,       'fs', '2^{\prime} 1'),
                            'd2':   (14,    3.0,       'fs', '2 1')}
            var_list, names = discover_dimensions(arr, dimension_cols)
        else:
            print 'define all dimensions or no dimensions'
    
    
    #load data from all files---------------------------------------------------
    
    data_objs = []
    zi = []
    for filepath in filepaths:

        #load raw data from filepath--------------------------------------------
        
        if os.path.isfile(filepath):
            dat = np.genfromtxt(filepath).T
            if verbose: print 'data loaded:', dat.shape
        else:
            print 'filepath', filepath, 'does not yield a file'
            return
            
        #get zvar value if appropriate------------------------------------------
        
        if zvar:
            zcol = datCols[zvar][0]
            zi.append(dat[zcol][0])
        
        #treatment for 1D data--------------------------------------------------
        
        if not yvar:
            
            #define columns
            xcol = datCols[xvar][0]
            
            #define columns
            xcol = datCols[xvar][0]
    
            #get data
            xi = dat[xcol]
            zis = []
            for key in zvars:
                zcol = datCols[key][0]
                zis.append(dat[zcol])
            zis = np.array(zis)
    
            #create data object
            x_axis = Axis(xi, datCols[xvar][2], xvar, datCols[xvar][4])
            x_axis.convert(datCols[xvar][3])
            data_objs.append(Data([x_axis], zis, zvars))
    
        #treatment for 2D data--------------------------------------------------
    
        else:
            
            #define columns
            xcol = datCols[xvar][0]
            ycol = datCols[yvar][0]
            
            #grid data
            
            #generate regularly spaced y and x bins to use for gridding 2d data
            #grid_factor:  multiplier factor for blowing up grid
            #grid all input channels (ai0-ai3) to the set xi and yi attributes
        
            #generate lists from data
            xlis = sorted(dat[xcol])
            xtol = datCols[xvar][1]
            # values are binned according to their averages now, so min and max 
            #  are better represented
            xs = []
            # check to see if unique values are sufficiently unique
            # deplete to list of values by finding points that are within 
            #  tolerance
            while len(xlis) > 0:
                # find all the xi's that are like this one and group them
                # after grouping, remove from the list
                set_val = xlis[0]
                xi_lis = [xi for xi in xlis if np.abs(set_val - xi) < xtol]
                # the complement of xi_lis is what remains of xlis, then
                xlis = [xi for xi in xlis if not np.abs(xi_lis[0] - xi) < xtol]
                xi_lis_average = sum(xi_lis) / len(xi_lis)
                xs.append(xi_lis_average)
            # create uniformly spaced x and y lists for gridding
            # infinitesimal offset used to properly interpolate on bounds; can
            #  be a problem, especially for stepping axis
            xi = np.linspace(min(xs)+1E-06,max(xs)-1E-06,
                             (len(xs) + (len(xs)-1)*(grid_factor-1)))
                                  
            ylis = sorted(dat[ycol])
            ytol = datCols[yvar][1]
            ys = []
            while len(ylis) > 0:
                set_val = ylis[0]
                yi_lis = [yi for yi in ylis if np.abs(set_val - yi) < ytol]
                ylis = [yi for yi in ylis if not np.abs(yi_lis[0] - yi) < ytol]
                yi_lis_average = sum(yi_lis) / len(yi_lis)
                ys.append(yi_lis_average)
            yi = np.linspace(min(ys)+1E-06,max(ys)-1E-06,
                             (len(ys) + (len(ys)-1)*(grid_factor-1)))
        
            x_col = dat[xcol] 
            y_col = dat[ycol]
            # grid each of our signal channels
            zis = []
            for key in zvars:
                zcol = datCols[key][0]
                #make fill value znull right now (instead of average value)
                fill_value = 0. #ugly hack for now #self.znull #self.data[zcol].sum()  / len(self.data[zcol])
                grid_i = griddata((x_col,y_col), dat[zcol], 
                                   (xi[None,:],yi[:,None]),
                                    method='cubic',fill_value=fill_value)
                zis.append(grid_i)
            zis = np.array(zis)
                                
            #create data object
            x_axis = Axis(xi, datCols[xvar][2], xvar, datCols[xvar][4])
            x_axis.convert(datCols[xvar][3])
            y_axis = Axis(yi, datCols[yvar][2], yvar, datCols[yvar][4])
            y_axis.convert(datCols[yvar][3])
            data = Data([x_axis, y_axis], zis, zvars)
    
            #create data object
            x_axis = Axis(xi, datCols[xvar][2], xvar, datCols[xvar][4])
            x_axis.convert(datCols[xvar][3])
            y_axis = Axis(yi, datCols[yvar][2], yvar, datCols[yvar][4])
            y_axis.convert(datCols[yvar][3])
            data_objs.append(Data([y_axis, x_axis], zis, zvars))

    #collapse data objects into one object--------------------------------------

    if not movie:
        data = data_objs[0]
    else:
        #sort data objects in ascending zi (native units)
        zi = np.array(zi)
        data_objs = [data_objs[i] for i in np.argsort(zi)]
        zi.sort()
        #now construct data objct
        x_axis = data_objs[0].axes[1]
        y_axis = data_objs[0].axes[0]
        z_axis = Axis(zi, datCols[zvar][2], zvar, datCols[zvar][4])
        z_axis.convert(datCols[zvar][3])
        zis = []
        for data_obj in data_objs:
            zis.append(data_obj.zis)
        zis = np.array(zis)
        zis = zis.transpose(1, 0, 2, 3) #channel, zi, yi, xi
        data = Data([z_axis, y_axis, x_axis], zis, zvars)
            
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

def discover_dimensions(arr, dimension_cols):
    
    print arr.shape
    
    #import values--------------------------------------------------------------
    
    dc = dimension_cols 
    di = [dc[key][0] for key in dc.keys()]
    dt = [dc[key][1] for key in dc.keys()]
    du = [dc[key][2] for key in dc.keys()]
    dk = [key for key in dc.keys()]
    print dk
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
    print dims
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
    
    #find which are scanned-----------------------------------------------------
    
    scanned = []
    constant = []
    for dim in dims:
        name = dim[3]
        index = dim[0]
        vals = arr[index]
        if vals.max() - vals.min() > dim[1]:
            scanned.append([name, index, None])
        else:
            constant.append([name, index, arr[index, 0]])
            
    print scanned
    print constant
    
    #order scanned dimensions---------------------------------------------------
    
    #to do....
        
    #return---------------------------------------------------------------------

    return scanned, constant
    
    

    
    
    
    
    
    
    
    
    
    
    
    