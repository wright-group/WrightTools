import os
import numpy as np

class data:
    """
        a class for manipulating 2d data objects and their axes
            -plotting
            -decompositions and fitting
            -"slicing" data
            -normal array operations (adding and subtracting, etc.)
    """
    def __init__(self, x, y, z, 
                 znull=None, zmin=None, zmax=None):
        self.x = x
        self.y = y 
        self.z = z
        if znull is None:
            self.znull = 0.
        else: self.znull = znull
        if zmin is None:
            self.zmin = z.min()
        else: self.zmin = zmin
        if zmax is None:
            self.zmax = z.max()
        else: self.zmax = zmax

    def zoom(self, factor, order=1):
        import scipy.ndimage
        self.x = scipy.ndimage.interpolation.zoom(self.x, factor, order=order)
        self.y = scipy.ndimage.interpolation.zoom(self.y, factor, order=order)
        self.z = scipy.ndimage.interpolation.zoom(self.z, factor, order=order)
        
    def svd(self, verbose=False):
        """
            singular value decomposition of gridded data z
        """
        #give feedback on top (normalized) singular values
        U, s, V = np.linalg.svd(self.z)
        if verbose:
            # report significant stats on svd
            plt.figure()
            s_max = s.max()
            plt.scatter(s / s_max)
        return U, s, V

    def center(self, axis=None, center=None):
        if center == 'max':
            print 'listing center as the point of maximum value'
            if axis == 0 or axis in ['x', self.xvar]:
                index = self.zi.argmax(axis=0)
                set_var = self.xi
                max_var = self.yi
                out = np.zeros(self.xi.shape)
            elif axis == 1 or axis in ['y', self.yvar]:
                index = self.zi.argmax(axis=1)
                set_var = self.yi
                max_var = self.xi
                out = np.zeros(self.yi.shape)
            else:
                print 'Input error:  axis not identified'
                return
            for i in range(len(set_var)):
                out[i] = max_var[index[i]]
        else:
            # find center by average value
            out = self.exp_value(axis=axis, moment=1)
        return out
                
    def exp_value(self, axis=None, moment=1, norm=True, noise_filter=None):
        """
            returns the weighted average for fixed points along axis
            specify the axis you want to have exp values for (x or y)
            good for poor-man's 3peps, among other things
            moment argument can be any integer; meaningful ones are:
                0 (area, set norm False)
                1 (average, mu) or 
                2 (variance, or std**2)
            noise filter, a number between 0 and 1, specifies a cutoff for 
                values to consider in calculation.  zi values less than the 
                cutoff (on a normalized scale) will be ignored
            
        """
        if axis == 0:
            # an output for every x var
            z = self.z.copy()
            int_var = self.y
            out = np.zeros(self.x.shape)
        elif axis == 1:
            # an output for every y var
            z = self.z.T.copy()
            int_var = self.x
            out = np.zeros(self.y.shape)
        else:
            print 'Input error:  axis not identified'
            return
        if not isinstance(moment, int):
            print 'moment must be an integer.  recieved {0}'.format(moment)
            return
        for i in range(out.shape[0]):
            # ignoring znull for this calculation, and offseting my slice by min
            z_min = z[:,i].min()
            #zi_max = zi[:,i].max()
            temp_z = z[:,i] - z_min
            if noise_filter is not None:
                cutoff = noise_filter * (temp_z.max() - z_min)
                temp_z[temp_z < cutoff] = 0
            #calculate the normalized moment
            if norm == True:
                out[i] = np.dot(temp_z,int_var**moment) / temp_z.sum()#*np.abs(int_var[1]-int_var[0]) 
            else:
                out[i] = np.dot(temp_z,int_var**moment)
        return out

    def fit_gauss(self, axis=None):
        """
            least squares optimization of traces
            intial params p0 guessed by moments expansion
        """
        if axis == 0:
            # an output for every x var
            z = self.z.copy()
            var = self.y
            #out = np.zeros((len(self.xi), 3))
        elif axis == 1:
            # an output for every y var
            z = self.z.T.copy()
            var = self.x
            #out = np.zeros((len(self.yi), 3))

        # organize the list of initial params by calculating moments
        m0 = self.exp_value(axis=axis, moment=0, norm=False)
        m1 = self.exp_value(axis=axis, moment=1, noise_filter=0.1)
        m2 = self.exp_value(axis=axis, moment=2, noise_filter=0.1)        

        mu_0 = m1
        s0 = np.sqrt(np.abs(m2 - mu_0**2))
        A0 = m0 / (s0 * np.sqrt(2*np.pi))
        offset = np.zeros(m0.shape)
        
        p0 = np.array([A0, mu_0, s0, offset])
        out = p0.copy()
        from scipy.optimize import leastsq
        for i in range(out.shape[1]):
            #print leastsq(gauss_residuals, p0[:,i], args=(zi[:,i], var))
            try:
                out[:,i] = leastsq(gauss_residuals, p0[:,i], args=(z[:,i]-self.znull, var))[0]
            except:
                print 'least squares failed on {0}:  initial guesses will be used instead'.format(i)
                out[:,i] = p0[:,i]
        out[2] = np.abs(out[2])
        return out
        
    def smooth(self, 
               x=0,y=0, 
               window='kaiser',
               debug = False): #smoothes via adjacent averaging            
        """
            convolves the signal with a 2D window function
            currently only equipped for kaiser window
            'x' and 'y', both integers, are the nth nearest neighbor that get 
                included in the window
            Decide whether to perform xaxis smoothing or yaxis by setting the 
                boolean true
        """
        # n is the seed of the odd numbers:  n is how many nearest neighbors 
        # in each direction
        # make sure n is integer and n < grid dimension
        # account for interpolation using grid factor
        nx = x
        ny = y
        # create the window function
        if window == 'kaiser':
            # beta, a real number, is a form parameter of the kaiser window
            # beta = 5 makes this look approximately gaussian in weighting 
            # beta = 5 similar to Hamming window, according to numpy
            # over window (about 0 at end of window)
            beta=5.0
            wx = np.kaiser(2*nx+1, beta)
            wy = np.kaiser(2*ny+1, beta)
        # for a 2D array, y is the first index listed
        w = np.zeros((len(wy),len(wx)))
        for i in range(len(wy)):
            for j in range(len(wx)):
                w[i,j] = wy[i]*wx[j]
        # create a padded array of zi
        # numpy 1.7.x required for this to work
        temp_z = np.pad(self.zi, ((ny,ny), 
                                   (nx,nx)), 
                                    mode='edge')
        from scipy.signal import convolve
        out = convolve(temp_z, w/w.sum(), mode='valid')
        if debug:
            plt.figure()
            sp1 = plt.subplot(131)
            plt.contourf(self.zi, 100)
            plt.subplot(132, sharex=sp1, sharey=sp1)
            plt.contourf(w,100)
            plt.subplot(133)
            plt.contourf(out,100)
        self.z=out
        # reset zmax
        self.zmax = self.z.max()
        self.zmin = self.z.min()

    def intaxis(self, int_axis, filename=None):
         if int_axis == 0: #sum over all x values at fixed y
             out = np.zeros((len(self.y),2))
             for y in range(len(self.y)):
                 out[y][0] = self.y[y]
                 out[y][1] = self.z[y].sum() -  self.znull * len(self.x)

         elif int_axis == 1: #sum over all y values at fixed x
             out = np.zeros((len(self.x),2))
             for x in range(len(self.x)):
                 out[x][0] = self.x[x]
                 for y in range(len(self.y)):
                     out[x][1] += self.z[y][x] - self.znull
         else:
             print 'specified axis is not recognized'
         return out

        
def makefit(**kwargs):
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

def find_name(fname, suffix):
    """
    save the file using fname, and tacking on a number if fname already exists
    iterates until a unique name is found
    returns False if the loop malfunctions
    """
    good_name=False
    # find a name that isn't used by enumerating
    i = 1
    while not good_name:
        try:
            with open(fname+'.'+suffix):
               # file does exist
               # see if a number has already been guessed
               if fname.endswith(' ({0})'.format(i-1)):
                   # cut the old off before putting the new in
                   fname = fname[:-len(' ({0})'.format(i-1))]
               fname += ' ({0})'.format(i)
               i = i + 1
               # prevent infinite loop if the code isn't perfect
               if i > 100:
                   print 'didn\'t find a good name; index used up to 100!'
                   fname = False
                   good_name=True
        except IOError:
            # file doesn't exist and is safe to write to this path
            good_name = True
    return fname

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

def filename_parse(fstr):
    """
    parses a filepath string into it's path, name, and suffix
    """
    split = fstr.split('\\')
    if len(split) == 1:
        file_path = None
    else:
        file_path = '\\'.join(split[0:-1])
    split2 = split[-1].split('.')
    # try and guess whether a suffix is there or not
    # my current guess is based on the length of the final split string
    # suffix is either 3 or 4 characters
    if len(split2[-1]) == 3 or len(split2[-1]) == 4:
        file_name = '.'.join(split2[0:-1])
        file_suffix = split2[-1]
    else:
        file_name = split[-1]
        file_suffix = None
    return file_path, file_name, file_suffix    
    
def gauss_residuals(p, y, x):
    """
    calculates the residual between y and a gaussian with:
        amplitude p[0]
        mean p[1]
        stdev p[2]
    """
    A, mu, sigma, offset = p
    # force sigma to be positive
    err = y-A*np.exp(-(x-mu)**2 / (2*np.abs(sigma)**2)) - offset
    return err


