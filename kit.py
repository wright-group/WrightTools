'''
a collection of small, general purpose objects and methods
'''


### import ####################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import ast
import sys
import copy
import itertools
import linecache
import collections
from time import clock

import numpy as np

from . import units


### file processing ###########################################################


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
    if len(split2[-1]) in [3, 4, 5]:
        file_name = '.'.join(split2[0:-1])
        file_suffix = split2[-1]
    else:
        file_name = split[-1]
        file_suffix = None
    return file_path, file_name, file_suffix


def file_len(fname):
    '''
    Cheaply get the number of lines in a file. File is not entirely loaded 
    into memory.
    '''
    # adapted from http://stackoverflow.com/questions/845058
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


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
                   print('didn\'t find a good name; index used up to 100!')
                   fname = False
                   good_name=True
        except IOError:
            # file doesn't exist and is safe to write to this path
            good_name = True
    return
    
    
class FileSlicer:
    
    def __init__(self, path, skip_headers=True, header_charachter='#'):
        '''
        Access groups of lines from a file quickly, without loading the entire
        file into memory. Lines are accesed from Useful especially in cases where 
        
        Mostly a convinient wrapper around the standard library 'linecache'
        module.
        
        Parameters
        ----------
        path : string
            Path to the file.
        skip_headers : bool (optional)
            Toggle to skip headers at beginning of file. Default is True.
        header_charachter : string (optional)
            Charachter that appears at the beginning of header lines for this
            file. Default is '#'.
        '''
        self.path = path
        self.n = 0
        self.length = file_len(path)
        if skip_headers:
            with open(path) as f:
                while f.readline()[0] == header_charachter:
                    self.n += 1
                    
    def close(self):
        '''
        Clear the cache, attempting to free as much memory as possible.
        '''
        linecache.clearcache()
    
    def get(self, line_count):
        '''
        Get the next group of lines from the file.
        
        Parameters
        ----------
        line_count : int
            The number of lines to read from the file.
            
        Returns
        -------
        list
            List of lines as strings.
        '''
        # calculate indicies
        start = self.n
        stop = self.n + line_count
        if stop > self.length:
            raise IndexError('there are no more lines in the slicer: ' +
                             '(file length {})'.format(self.length))
        # get lines using list comprehension
        # for some reason linecache is 1 indexed >:-(
        out = [linecache.getline(self.path, i) for i in range(start+1, stop+1)]
        # finish
        self.n += line_count
        return out
    
    def skip(self, line_count):
        '''
        Skip the next group of lines from the file.
        
        Parameters
        ----------
        line_count : int
            The number of lines to skip.
        '''
        if self.n + line_count > self.length:
            raise IndexError('there are no more lines in the slicer: ' +
                             '(file length {})'.format(self.length))        
        # finish
        self.n += line_count


def get_box_path():
    '''
    LEGACY METHOD. Use ``get_path_matching(name)`` instead.
    '''
    box_path = get_path_matching('Box Sync')
    return os.path.join(box_path, 'Wright Shared')
    

def get_path_matching(name):
    # first try looking in the user folder
    p = os.path.join(os.path.expanduser('~'), name)
    # then try expanding upwards from cwd
    if not os.path.isdir(p):
        p = None
        drive, folders = os.path.splitdrive(os.getcwd())
        folders = folders.split(os.sep)
        folders.insert(0, os.sep)
        if name in folders:
            p = os.path.join(drive, *folders[:folders.index(name)+1])
    # TODO: something more robust to catch the rest of the cases?
    return p

def get_timestamp():

    import time

    return time.strftime('%Y.%m.%d %H_%M_%S')


def glob_handler(extension, folder=None, identifier=None):
    '''
    returns a list of all files matching specified inputs \n
    if no folder is specified, looks in chdir
    '''

    import glob

    filepaths = []

    if folder:
        # comment out [ and ]...
        folder = folder.replace('[', '?')
        folder = folder.replace(']', '*')
        folder = folder.replace('?', '[[]')
        folder = folder.replace('*', '[]]')
        glob_str = os.path.join(folder, '*' + extension)
    else:
        glob_str = '*' + extension + '*'

    for filepath in glob.glob(glob_str):
        if identifier:
            if identifier in filepath:
                filepaths.append(filepath)
        else:
            filepaths.append(filepath)

    return filepaths


def plot_dats(folder=None, transpose=True):
    '''
    Convinience function to plot raw data from COLORS
    '''

    import data
    import artists

    if folder:
        pass
    else:
        folder = os.getcwd()

    files = glob_handler('.dat', folder = folder)

    for _file in files:

        print(' ')

        try:

            dat_data = data.from_COLORS(_file)

            fname = filename_parse(_file)[1]

            dat_data.convert('wn')

            #1D
            if len(dat_data.axes) == 1:
                artist = artists.mpl_1D(dat_data, dat_data.axes[0].name)
                artist.plot(0, autosave = True, output_folder = folder, fname = fname)

            #2D
            elif len(dat_data.axes) == 2:
                if transpose: dat_data.transpose()
                artist = artists.mpl_2D(dat_data, dat_data.axes[0].name, dat_data.axes[1].name)
                artist.plot(0, pixelated = True, contours = 0, xbin = True, ybin = True,
                            autosave = True, output_folder = folder, fname = fname)

            else:
                print('error! - dimensionality of data ({}) not recognized'.format(len(dat_data.axes)))

        except:
            import sys
            print('dat {} not recognized as plottible in plot_dats'.format(filename_parse(_file)[1]))
            print(sys.exc_info()[0])
            pass


def read_headers(filepath):
    '''
    Read 'Wright group formatted' headers from given path.
    
    Parameters
    ----------
    filepath : str
        Path of file.
        
    Returns
    -------
    OrderedDict
        Dictionary containing header information.
    '''
    headers = collections.OrderedDict()
    for line in open(filepath):
        if line[0] == '#':
            split = line.split(':')
            key = split[0][2:]
            item = split[1].split('\t')
            if split[1][0:3] == ' [[':  # case of multidimensional arrays
                arr = string2array(split[1][1:])
                headers[key] = arr
            else:
                if item[0] == '':
                    item = [item[1]]
                item = [i.strip() for i in item]  # remove dumb things
                item = [i if i is not '' else 'None' for i in item]  # handle empties
                # handle lists
                is_list = False
                list_chars = ['[', ']']
                for item_index, item_string in enumerate(item):
                    if item_string == '[]':
                        continue
                    for char in item_string:
                        if char in list_chars:
                            is_list = True
                    for char in list_chars:
                        item_string = item[item_index]
                        item[item_index] = item_string.replace(char, '')
                # eval contents
                item = [i.strip() for i in item]  # remove dumb things
                item = [ast.literal_eval(i) for i in item]
                if len(item) == 1 and not is_list:
                    item = item[0]
                headers[key] = item
        else:
            break  # all header lines are at the beginning
    return headers


def write_headers(filepath, dictionary):
    '''
    Write 'Wright Group formatted' headers to given file. Headers written can
    be read again using read_headers.

    Paramters
    ---------
    filepath : str
        Path of file. File must not exist.
    dictionary : dict or OrderedDict
        Dictionary of header items.

    Returns
    -------
    str
        Filepath of file.
    '''
    if sys.version[0] == '2':
        string_type = basestring  # recognize unicode and string types
    else:
        string_type = str  # newer versions of python don't have unicode type
    dictionary = copy.deepcopy(dictionary)
    header_items = []
    for key, value in dictionary.items():
        header_item = key + ':'
        if isinstance(value, string_type):
            header_item += '\t' + '\'' + value + '\''
        elif type(value) == list:
            for i in range(len(value)):
                if isinstance(value, string_type):
                    value[i] = '\'' + value[i] + '\''
                else:
                    value[i] = str(value[i])
            header_item += ' [' + '\t'.join(value) + ']'
        elif type(value).__module__ == np.__name__:  # anything from numpy
            if hasattr(value, 'shape'):
                string = array2string(value)
                header_item += ' ' + string
            else:
                header_item += ' [' + '\t'.join([str(i) for i in value]) + ']'
        else:
            header_item += '\t' + str(value)
        header_items.append(header_item)
    # write header
    header_str = ''
    for item in header_items:
        header_str += item + '\n'
    header_str = header_str[:-1]  # remove final newline charachter
    np.savetxt(filepath, [], header=header_str)
    # return
    return filepath


### array and math ############################################################


def closest_pair(arr, give='indicies'):
    '''
    Find the pair of indices corresponding to the closest elements in an array.
    If multiple pairs are equally close, both pairs of indicies are returned.
    Optionally returns the closest distance itself.

    I am sure that this could be written as a cheaper operation. I
    wrote this as a quick and dirty method because I need it now to use on some
    relatively small arrays. Feel free to refactor if you need this operation
    done as fast as possible. - Blaise 2016.02.07

    Parameters
    ----------
    arr : numpy.ndarray
        The array to search.
    give : {'indicies', 'distance'} (optional)
        Toggle return behavior. If 'distance', returns a single float - the
        closest distance itself. Default is indicies.

    Returns
    -------
    list of lists of two tuples
        List containing lists of two tuples: indicies the nearest pair in the
        array.

    Examples
    --------
    >>> arr = np.array([0, 1, 2, 3, 3, 4, 5, 6, 1])
    >>> closest_pair(arr)
    [[(1,), (8,)], [(3,), (4,)]]
    '''
    idxs = [idx for idx in np.ndindex(arr.shape)]
    outs = []
    min_dist = arr.max() - arr.min()
    for idxa in idxs:
        for idxb in idxs:
            if idxa == idxb:
                continue
            dist = abs(arr[idxa]-arr[idxb])
            if dist == min_dist:
                if not [idxb, idxa] in outs:
                    outs.append([idxa, idxb])
            elif dist < min_dist:
                min_dist = dist
                outs = [[idxa, idxb]]
    if give == 'indicies':
        return outs
    elif give == 'distance':
        return min_dist
    else:
        raise KeyError('give not recognized in closest_pair')


def diff(xi, yi, order = 1):
    '''
    numpy.diff is a convinient method but it only works for evenly spaced data \n
    this method does the same but for an arbitrary 1D data slice \n
    returns numpy array [xi, yi_out]. edge points are padded.
    '''
    import numpy as np

    # grid data to be even ----------------------------------------------------

    # get function that describes data
    import scipy
    f = scipy.interpolate.interp1d(xi, yi, kind = 'linear')

    xi_even = np.linspace(min(xi), max(xi), len(xi))
    yi_even = f(xi_even)

    # call numpy.diff ---------------------------------------------------------

    yi_out_even = np.diff(yi_even, n = order)
    yi_out_even = np.pad(yi_out_even, order, mode = 'edge')
    yi_out_even = np.delete(yi_out_even, range(order))

    # put data back onto original xi points -----------------------------------

    xi_even += xi_even[1] - xi_even[0]  # offset by half step...

    fdiff = scipy.interpolate.interp1d(xi_even, yi_out_even,
                                       kind = 'linear', bounds_error = False)

    yi_out = fdiff(xi)

    return np.array([xi, yi_out])


def mono_resolution(grooves_per_mm, slit_width, focal_length, output_color, output_units='wn'):
    '''
    slit width mm, focal_length mm, output_color nm
    '''
    d_lambda = 1e6*slit_width/(grooves_per_mm*focal_length)  # nm
    upper = output_color + d_lambda/2  # nm
    lower = output_color - d_lambda/2  # nm
    return abs(units.converter(upper, 'nm', output_units) - 
               units.converter(lower, 'nm', output_units))


def remove_nans_1D(arrs):
    '''
    Remove nans in a list of 1D arrays. Removes indicies in all arrays if any
    array is nan at that index. All input arrays must have the same size.
    
    Parameters
    ----------
    arrs : list of 1D arrays
        The arrays to remove nans from
        
    Returns
    -------
    list
        List of 1D arrays in same order as given, with nan indicies removed.
    '''
    # find all indicies to keep
    bads = np.array([])
    for arr in arrs:
        bad = np.array(np.where(np.isnan(arr))).flatten()
        bads = np.hstack((bad, bads))
    goods = [i for i in np.arange(len(arrs[0])) if i not in bads]
    # apply
    return [a[goods] for a in arrs]


def share_nans(arrs1):
    # Written by DJM. darienmorrow@gmail.com. January 15, 2016.    
    '''
    Takes a list of nD arrays and returns a new list of nD arrays. 
    The new list is in the same order as the old list. 
    If one indexed element in an old array is nan then every element for that
    index in all new arrays in the list is then nan.
    
    Parameters
    ----------
    arrs1 : list of nD arrays
        The arrays to syncronize nans from
        
    Returns
    -------
    list
        List of nD arrays in same order as given, with nan indicies syncronized.
    '''
    
    nans = np.zeros((arrs1[0].shape))
    
    for arr in arrs1:
        nans *= arr       
 
    arrs2 = [a + nans for a in arrs1]

    return arrs2


def smooth_1D(arr, n=10):
    '''
    smooth 1D data by 'running average'\n
    int n smoothing factor (num points)
    '''
    for i in range(n, len(arr)-n):
        window = arr[i-n:i+n].copy()
        arr[i] = window.mean()
    return arr


class Spline:

    def __call__(self, *args, **kwargs):
        return self.true_spline(*args, **kwargs)

    def __init__(self, xi, yi, k=2, s=1000, ignore_nans=True):
        '''
        Wrapper class for scipy.UnivariateSpline, made to be slightly less
        finicky with things like decending xi arrays and nans.
        '''
        # import
        from scipy.interpolate import UnivariateSpline
        xi_internal = xi.copy()
        yi_internal = yi.copy()
        # nans
        if ignore_nans:
            l = [xi_internal, yi_internal]
            xi_internal, yi_internal = remove_nans_1D(l)
        # UnivariateSpline needs ascending xi
        sort = np.argsort(xi_internal)
        xi_internal = xi_internal[sort]
        yi_internal = yi_internal[sort]
        # create true spline
        self.true_spline = UnivariateSpline(xi_internal, yi_internal, k=k, s=s)
    
    
def unique(arr, tolerance=1e-6):
    '''
    Return unique elements in 1D array, within tolerance.
    
    Parameters
    ----------
    arr : array_like
        Input array. This will be flattened if it is not already 1D.
    tolerance : number (optional)
        The tolerance for uniqueness.
        
    Returns
    -------
    array
        The sorted unique values.
    '''
    arr = list(arr.flatten())
    arr.sort()
    unique = []
    while len(arr) > 0:
        current = arr[0]
        lis = [xi for xi in arr if np.abs(current - xi) < tolerance]
        arr = [xi for xi in arr if not np.abs(lis[0] - xi) < tolerance]
        xi_lis_average = sum(lis) / len(lis)
        unique.append(xi_lis_average)
    return np.array(unique)


### uncategorized #############################################################


def array2string(array, sep='\t'):
    '''
    Generate a string from an array with useful formatting. Great for writing
    arrays into single lines in files.
    
    See Also
    --------
    string2array
    '''
    np.set_printoptions(threshold=array.size)
    string = np.array2string(array, separator=sep)
    string = string.replace('\n', sep)
    string = re.sub(r'({})(?=\1)'.format(sep), '', string)
    return string
    
    
def flatten_list(l):
    '''
    Flatten an irregular list. Works generally but may be slower than it could
    be if you can make assumptions about your list.
    
    Adapted from http://stackoverflow.com/questions/2158395
    
    Example
    -------
    >>> l = [[[1, 2, 3], [4, 5]], 6]
    >>> wt.kit.flatten_list(l)
    [1, 2, 3, 4, 5, 6]
    '''
    listIsNested = True
    while listIsNested:  # outer loop
        keepChecking = False
        Temp = []
        for element in l:  # inner loop
            if isinstance(element,list):
                Temp.extend(element)
                keepChecking = True
            else:
                Temp.append(element)
        listIsNested = keepChecking  # determine if outer loop exits
        l = Temp[:]
    return l


def get_methods(the_class, class_only=False, instance_only=False,
                exclude_internal=True):
    '''
    get a list of strings corresponding to the names of the methods
    of an object.
    '''
    import inspect

    def acceptMethod(tup):
        # internal function that analyzes the tuples returned by getmembers
        # tup[1] is the actual member object
        is_method = inspect.ismethod(tup[1])
        if is_method:
            bound_to = tup[1].im_self
            internal = tup[1].im_func.func_name[:2] == '__' and tup[1].im_func.func_name[-2:] == '__'
            if internal and exclude_internal:
                include = False
            else:
                include = (bound_to == the_class and not instance_only) or (bound_to == None and not class_only)
        else:
            include = False
        return include

    # filter to return results according to internal function and arguments
    tups = filter(acceptMethod, inspect.getmembers(the_class))
    return [tup[0] for tup in tups]
    

def intersperse(lst, item):
    '''
    Put item between each existing item in list. \n
    From http://stackoverflow.com/a/5921708
    '''
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


identity_operators = ['=', '+', '-', '*', '/', 'F']
def parse_identity(string):
    '''
    Parse an identity string into its components.
    
    Returns
    -------
    tuple of lists
        (names, operators)
    '''
    names = re.split("[=F]+", string)
    operators = [c for c in list(string) if c in identity_operators]
    return names, operators


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    from http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions

    with wt.kit.suppress_stdout_stderr():
        rogue_function()
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        

def string2array(string, sep='\t'):
    '''
    Generate an array from a string created using array2string.
    
    See Also
    --------
    array2string
    '''
    # discover size
    size = string.count('\t')+1 
    # discover dimensionality
    dimensionality = 0
    while string[dimensionality] == '[':
        dimensionality += 1
    # discover shape
    shape = []       
    for i in range(1, dimensionality+1)[::-1]:
        to_match = '['*(i-1)
        count_positive = string.count(to_match + ' ')
        cout_negative = string.count(to_match + '-')
        shape.append(count_positive + cout_negative)
    shape[-1] = size / shape[-2]
    for i in range(1, dimensionality-1)[::-1]:
        shape[i] = shape[i] / shape[i-1]
    shape = tuple(shape)
    # import list of floats
    l = string.split(' ')
    l = flatten_list([i.split('-') for i in l])  # annoyingly series of negative values get past previous filters
    for i, item in enumerate(l):
        bad_chars = ['[', ']', '\t', '\n']
        for bad_char in bad_chars:
            item = item.replace(bad_char, '')
        l[i] = item
    for i in range(len(l))[::-1]:
        try:
            l[i] = float(l[i])
        except ValueError:
            l.pop(i)
    # create and reshape array
    arr = np.array(l)
    arr.shape = shape
    # finish
    return arr


unicode_dictionary = collections.OrderedDict()
unicode_dictionary['Alpha'] = u'\u0391'
unicode_dictionary['Beta'] = u'\u0392'
unicode_dictionary['Gamma'] = u'\u0392'
unicode_dictionary['Delta'] = u'\u0394'
unicode_dictionary['Epsilon'] = u'\u0395'
unicode_dictionary['Zeta'] = u'\u0396'
unicode_dictionary['Eta'] = u'\u0397'
unicode_dictionary['Theta'] = u'\u0398'
unicode_dictionary['Iota'] = u'\u0399'
unicode_dictionary['Kappa'] = u'\u039A'
unicode_dictionary['Lamda'] = u'\u039B'
unicode_dictionary['Mu'] = u'\u039C'
unicode_dictionary['Nu'] = u'\u039D'
unicode_dictionary['Xi'] = u'\u039E'
unicode_dictionary['Omicron'] = u'\u039F'
unicode_dictionary['Pi'] = u'\u03A0'
unicode_dictionary['Rho'] = u'\u03A1'
unicode_dictionary['Sigma'] = u'\u03A3'
unicode_dictionary['Tau'] = u'\u03A4'
unicode_dictionary['Upsilon'] = u'\u03A5'
unicode_dictionary['Phi'] = u'\u03A6'
unicode_dictionary['Chi'] = u'\u03A7'
unicode_dictionary['Psi'] = u'\u03A8'
unicode_dictionary['Omega'] = u'\u03A9'
unicode_dictionary['alpha'] = u'\u03B1'
unicode_dictionary['beta'] = u'\u03B2'
unicode_dictionary['gamma'] = u'\u03B3'
unicode_dictionary['delta'] = u'\u03B4'
unicode_dictionary['epsilon'] = u'\u03B5'
unicode_dictionary['zeta'] = u'\u03B6'
unicode_dictionary['eta'] = u'\u03B7'
unicode_dictionary['theta'] = u'\u03B8'
unicode_dictionary['iota'] = u'\u03B9'
unicode_dictionary['kappa'] = u'\u03BA'
unicode_dictionary['lamda'] = u'\u03BB'
unicode_dictionary['mu'] = u'\u03BC'
unicode_dictionary['nu'] = u'\u03BD'
unicode_dictionary['xi'] = u'\u03BE'
unicode_dictionary['omicron'] = u'\u03BF'
unicode_dictionary['pi'] = u'\u03C0'
unicode_dictionary['rho'] = u'\u03C1'
unicode_dictionary['sigma'] = u'\u03C3'
unicode_dictionary['tau'] = u'\u03C4'
unicode_dictionary['upsilon'] = u'\u03C5'
unicode_dictionary['phi'] = u'\u03C6'
unicode_dictionary['chi'] = u'\u03C7'
unicode_dictionary['psi'] = u'\u03C8'
unicode_dictionary['omega'] = u'\u03C9'


def update_progress(progress, carriage_return=False, length=50):
    '''
    prints a pretty progress bar to the console     \n
    accepts 'progress' as a percentage              \n
    bool carriage_return toggles overwrite behavior \n
    '''
    # make progress bar string
    text =  '\r'
    num_oct = int(progress * (length/100.))
    text += '[{0}{1}]'.format('#'*num_oct, ' '*(length-num_oct))
    text += ' {}%'.format(np.round(progress, decimals = 2))
    if carriage_return:
        text += '\r\n'
    sys.stdout.write(text)
    if progress == 100.:
        print('\n')
    else:
        sys.stdout.flush()


class Timer:
    '''
    with Timer(): your_code()
    '''

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self, progress=None):
        self.start = clock()

    def __exit__(self, type, value, traceback):
        self.end = clock()
        self.interval = self.end - self.start
        if self.verbose:
            print('elapsed time: {0} sec'.format(self.interval))
