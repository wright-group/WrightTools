'''
unit and label handling in WrightTools
'''

### units ######################################################################
         
#units are stored in dictionaries of like kind. format:
#    unit : to native, from native, units_symbol, units_label
         
#angle units (native: rad)
angle = {'kind': 'angle',
         'rad': ['x', 'x', r'rad']}
         
#energy units (native: nm)
energy = {'kind': 'energy',
          'nm': ['x', 'x', r'nm'],
          'wn': ['1e7/x', '1e7/x', r'cm^{-1}'],
          'eV': ['1240./x', '1240./x', r'eV']} 
     
#time units (native: s)
time = {'kind': 'time',
        'fs': ['x/1e15', 'x*1e15', r'fs'],
        'ps': ['x/1e12', 'x*1e12', r'ps'],
        'ns': ['x/1e9', 'x*1e9', r'ns'],
        'us': ['x/1e6', 'x*1e6', r'us'],
        'ms': ['x/1000.', 'x*1000.', r'ms'],
        's':  ['x', 'x', r's'],
        'm':  ['x*60.', 'x/60.', r'm'],
        'h':  ['x*3600.', 'x/3600.', r'h'],
        'd':  ['x*86400.', 'x/86400.', r'd']}
            
#position units (native: mm)
position = {'kind': 'position',
            'nm_p': ['x/1e6', '1e6/x'], #can't have same name as nm for energy
            'um': ['x/1000.', '1000/x.'],
            'mm': ['x', 'x'],
            'cm': ['10.*x', 'x/10.'],
            'in': ['x*0.039370', '0.039370*x']}

#pulse width units (native: FWHM)
pulse_width = {'kind': 'pulse_width',
               'FWHM': ['x', 'x', r'FWHM']}        
            
#fluence units (native: uJ per sq. cm)
fluence = {'kind': 'fluence', 
           'uJ per sq. cm': ['x', 'x', r'\frac{\mu J}{cm^{2}}']}
       
unit_dicts = [angle, energy, time, position, pulse_width, fluence] 
            
def converter(val, current_unit, destination_unit):

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
    
### symbol #####################################################################
  
class symbol_dict(dict):
    #subclass dictionary to get at __missing__ method
    def __missing__(self, key):
        return self['default']

#color
color = symbol_dict()
color['default'] = r'E'
color['nm']      = r'\lambda'
color['wn']      = r'\bar\nu'
color['eV']      = r'\hslash\omega'

#delay
delay = symbol_dict()
delay['default'] = r'\tau'
        
#fluence
fluence = symbol_dict()
fluence['default'] = r'\mathcal{F}'

#pulse width
pulse_width = symbol_dict()
pulse_width['default'] = r'\sigma'

#catch all
none = symbol_dict()
none['default'] = ''

def get_default_symbol_type(units_str):
    if units_str in ['nm', 'wn', 'eV']:
        return 'color'
    elif units_str in ['fs', 'ps', 'ns']:
        return 'delay'
    elif units_str in ['uJ per sq. cm']:
        return 'fluence'
    elif units_str in ['FWHM']:
        return 'pulse_width'
    else:
        return 'none'
