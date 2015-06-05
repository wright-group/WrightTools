'''
a collection of small, general purpose objects and methods
'''

import os

### files ######################################################################
 
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
    return
    
def get_box_path():
    box_path = os.path.join(os.path.expanduser('~'), 'Box Sync', 'Wright Shared')
    return box_path

def get_timestamp():
    
    import time
    
    return time.strftime('%Y.%m.%d %H_%M_%S')

def glob_handler(extension, folder = None, identifier = None):
    '''
    returns a list of all files matching specified inputs \n
    if no folder is specified, looks in chdir
    '''
    
    import glob

    filepaths = []
    
    if folder:
        glob_str = folder + '\\' + '*' + extension + '*'
    else:
        glob_str = '*' + extension + '*'

    for filepath in glob.glob(glob_str):
        if identifier:
            if identifier in filepath:
                filepaths.append(filepath)
        else:
            filepaths.append(filepath)

    return filepaths
    
def plot_dats(folder = None, transpose = True):
    '''
    convinience function to plot raw data
    '''
    
    import data
    import artists
    
    if folder:
        pass
    else:
        folder = os.getcwd()

    files = glob_handler('.dat', folder = folder)
    
    for _file in files:

        print ' '

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
                print 'error! - dimensionality of data ({}) not recognized'.format(len(dat_data.axes))
            
        except:
            import sys            
            print 'dat {} not recognized as plottible in plot_dats'.format(filename_parse(_file)[1])
            print sys.exc_info()[0]
            pass
        
### math #######################################################################
        
def diff(xi, yi, order = 1):
    '''
    numpy.diff is a convinient method but it only works for evenly spaced data \n
    this method does the same but for an arbitrary 1D data slice \n
    returns numpy array [xi, yi_out]. edge points are padded.
    '''
    import numpy as np
    
    #grid data to be even-------------------------------------------------------

    #get function that describes data
    import scipy    
    f = scipy.interpolate.interp1d(xi, yi, kind = 'linear')
    
    xi_even = np.linspace(min(xi), max(xi), len(xi))
    yi_even = f(xi_even)

    #call numpy.diff------------------------------------------------------------   
    
    yi_out_even = np.diff(yi_even, n = order)
    yi_out_even = np.pad(yi_out_even, order, mode = 'edge')
    yi_out_even = np.delete(yi_out_even, range(order))
    
    #put data back onto original xi points--------------------------------------
    
    xi_even += xi_even[1] - xi_even[0] #offset by half step...
    
    fdiff = scipy.interpolate.interp1d(xi_even, yi_out_even, 
                                       kind = 'linear', bounds_error = False)
    
    yi_out = fdiff(xi)
    
    return np.array([xi, yi_out])

### units ######################################################################
         
#units are stored in dictionaries of like kind. format:
#    unit : to native, from native, units_symbol, units_label
         
#angle units (native: rad)
angle = {'kind': 'angle',
         'rad': ['x', 'x', r'\phi', r'rad']}
         
#energy units (native: nm)
energy = {'kind': 'energy',
          'nm': ['x', 'x', r'\lambda', r'nm'],
          'wn': ['1e7/x', '1e7/x', r'\bar\nu', r'cm^{-1}'],
          'eV': ['1240./x', '1240./x', r'E', r'eV']} 
     
#time units (native: s)
time = {'kind': 'time',
        'fs': ['x/1e15', 'x*1e15', r'\tau', r'fs'],
        'ps': ['x/1e12', 'x*1e12', r'\tau', r'ps'],
        'ns': ['x/1e9', 'x*1e9', r'\tau', r'ns'],
        'us': ['x/1e6', 'x*1e6', r'\tau', r'us'],
        'ms': ['x/1000.', 'x*1000.', r'\tau', r'ms'],
        's':  ['x', 'x', r'\tau', r's'],
        'm':  ['x*60.', 'x/60.', r'\tau', r'm'],
        'h':  ['x*3600.', 'x/3600.', r'\tau', r'h'],
        'd':  ['x*86400.', 'x/86400.', r'\tau', r'd']}
            
#position units (native: mm)
position = {'kind': 'position',
            'nm_p': ['x/1e6', '1e6/x'], #can't have same name as nm for energy
            'um': ['x/1000.', '1000/x.'],
            'mm': ['x', 'x'],
            'cm': ['10.*x', 'x/10.'],
            'in': ['x*0.039370', '0.039370*x']}

#pulse width units (native: FWHM)
pulse_width = {'kind': 'pulse_width',
               'FWHM': ['x', 'x', r'\sigma', r'FWHM']}        
            
#fluence units (native: uJ per sq. cm)
fluence = {'kind': 'fluence', 
           'uJ per sq. cm': ['x', 'x', r'\mathcal{F}', r'\frac{\mu J}{cm^{2}}']}
       
unit_dicts = [angle, energy, time, position, pulse_width, fluence] 
            
def unit_converter(val, current_unit, destination_unit):

    x = val
    
    for dic in unit_dicts:
        if current_unit in dic.keys() and destination_unit in dic.keys():
            native = eval(dic[current_unit][0])
            x = native
            out = eval(dic[destination_unit][1])
            return out

    #if all dictionaries fail
    print 'conversion not valid: returning input'
    return val
    
### uncategorized ##############################################################

def update_progress(progress, carriage_return = True, length = 50):
    '''
    prints a pretty progress bar to the console \n
    accepts 'progress' as a percentage          \n
    carriage_return toggles overwrite behavior  \n
    '''
    #make progress bar string
    progress_bar = ''
    num_oct = int(progress * (length/100.))
    progress_bar = progress_bar + '[{0}{1}]'.format('#'*num_oct, ' '*(length-num_oct))
    progress_bar = progress_bar + ' {}%'.format(np.round(progress, decimals = 2))
    if carriage_return:
        progress_bar = progress_bar + '\r'
        print progress_bar,
        return
    if progress == 100:
        progress_bar[-2:] = '\n'
    print progress_bar
    
class Timer:
    def __enter__(self, progress=None, verbose=True):
        self.verbose = verbose        
        self.start = clock()
    def __exit__(self, type, value, traceback):
        self.end = clock()
        self.interval = self.end - self.start
        if self.verbose:
            print 'elapsed time: {0} sec'.format(self.interval)
