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
    chopper_indicies = [len(kinds)-1]#[i for i, x in enumerate(kinds) if x == 'chopper']
    out = np.full(len(channel_indicies)*2+2+3, np.nan)
    out_index = 0
    out_names = []
    sig_zero_adj=0.0015
    for i in channel_indicies:
        #out[out_index] = np.mean(shots[i]*shots[chopper_indicies[0]])*2.
        #out_names.append(names[i] + '_cmean')
        #out_index += 1
        #out[out_index] = np.std(shots[i]*shots[chopper_indicies[0]])
        #out_names.append(names[i] + '_cstd')
        #out_index += 1
        if names[i] == 'signal':
            zero_adj = sig_zero_adj
        else:
            zero_adj = 0.0
        out[out_index] = np.mean(shots[i])+zero_adj
        out_names.append(names[i] + '_mean')
        out_index += 1
        out[out_index] = np.std(shots[i])
        out_names.append(names[i] + '_std')
        out_index += 1
        

# Shot-Normalized Signal - Pyro3 only
    try:
        needed_channels = ['signal']#,'pyro1','pyro2']
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
                arr = arr*(shots[index_dict[c]]+sig_zero_adj)
            else:
                arr = arr/shots[index_dict[c]]
        out[-5] = np.mean(arr)
        out[-4] = np.var(arr)
        if out[-5]<-1:
            out[-5]=np.nan
        out_names.append('py3_n_sig_mean')
        out_names.append('py3_n_sig_var')
    except:
        pass
    
    # Raw and (Pyro3) Normalized Photon Counting
    try:
        high_baseline = 0.002
        low_baseline = -0.004
        #one_photon_max = .02
        num_shots = len(shots[0])
        photon_count = [0,0,0] # 0 = 0 phontons, 1 = 1 photon, -1 = dark count
        for idx in range(len(shots[0])):
            p = shots[0][idx]
            if low_baseline < p < high_baseline:
                photon_count[0]+=1
            elif p >= high_baseline:
                photon_count[1] +=1
            elif p <= low_baseline:
                photon_count[-1]+=1
        # Fish Stats
        # First, use the 0-counts
        zero_adj = (2*photon_count[-1]+photon_count[0])/float(num_shots)
        if zero_adj >= 0.04:
            null_mean = -np.log(zero_adj)
        else:
            null_mean = np.nan
        # The future may hold many things, including more sophisticated algorithems
        # Including using multiple photon counts, etc.
        norm_fac = out[out_names.index('pyro3_mean')]
        out_names.append('raw_0_photon')    
        out_names.append('calc_pp100s')
        out_names.append('py3_n_pps')
        out[-3] = photon_count[0]+photon_count[-1]
        out[-2] = null_mean*100
        out[-1] = null_mean/norm_fac
    except:
        pass
    return [out, out_names]
