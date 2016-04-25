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
    chopper_indicies = []
    out = np.full(len(channel_indicies)*2+3+3, np.nan)
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
        

# Fit the power to a 2nd order polynomial
    try:
        Sig_idx = names.index('signal')
        Norm_idx = names.index('pyro3')
        
        fit = np.polyfit(shots[Norm_idx],shots[Sig_idx],2,full=True)
        
        out[-6] = fit[0][0]
        out[-5] = fit[0][1]
        out[-4] = fit[0][2]
        out[-3] = fit[3][0]
        out[-2] = fit[3][1]
        out[-1] = fit[3][2]
    except Exception as inst:
        print type(inst)
    
    out_names.append('square')
    out_names.append('linear')
    out_names.append('constant')
    out_names.append('square_var')
    out_names.append('linear_var')
    out_names.append('constant_var')

    return [out, out_names]
