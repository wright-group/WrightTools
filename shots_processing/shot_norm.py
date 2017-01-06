from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
#import scipy.optimize as opt

def process(shots, names, kinds):
    '''
    Return the average and vairance of the pyro-normalized output signal.
    Choppers are ignored.

    Parameters
    ----------
    shots : ndarray
        A 2D ndarray (input index, shot index)
    names : list of str
        A list of input names
    kinds : list of {'channel', 'chopper'}
        Kind of each input

    Returns
    -------
    list
        [ndarray (channels), list of channel names]

    '''

# Each Channel Averaged

    channel_indicies = [i for i, x in enumerate(kinds) if x == 'channel']
    out = np.full(len(channel_indicies)*2+2, np.nan)
    out_index = 0
    out_names = []
    for i in channel_indicies:
        out[out_index] = np.mean(shots[i])
        out_names.append(names[i] + '_mean')
        out_index += 1
        out[out_index] = np.std(shots[i])
        out_names.append(names[i] + '_std')
        out_index += 1

# Shot-Normalized Signal

    needed_channels = ['signal','pyro1','pyro2']
    optional_channels = ['pyro3']
    index_dict = dict()
    for c in needed_channels:
        if c in names:
            index_dict[c]=names.index(c)
        else:
            # a needed channel isn't avalible
            # g.logger.log('error', 'Additional signal channels are needed to normalize signal!')
            return [out, out_names]
    for c in optional_channels:
        if c in names:
            index_dict[c]=names.index(c)
            needed_channels.append(c)

    arr = np.ones(shots.shape[1])
    for c in needed_channels:
        if c == 'signal':
            arr = arr*shots[index_dict[c]]
        else:
            arr = arr/shots[index_dict[c]]
    out[-2] = np.mean(arr)
    out[-1] = np.var(arr)
    out_names.append('p_norm_sig_mean')
    out_names.append('p_norm_sig_var')
    
    return [out, out_names]
