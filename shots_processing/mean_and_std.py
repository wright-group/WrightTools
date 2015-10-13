import numpy as np

def process(shots, names, kinds):
    '''
    Return the mean and the variance of every channel. Choppers are ignored.
    
    Parameters
    ----------
    shots : ndarray
        A ndarray (inputs, shots)
    names : list of str
        A list of input names
    kinds : list of {'channel', 'chopper'}
        Kind of each input
        
    Returns
    -------
    list
        [ndarray (channels), list of channel names]
    '''
    channel_indicies = [i for i, x in enumerate(kinds) if x == 'channel']
    out = np.full(len(channel_indicies)*2, np.nan)
    out_index = 0
    out_names = []
    for i in channel_indicies:
        out[out_index] = np.mean(shots[i])
        out_names.append(names[i] + '_mean')
        out_index += 1
        out[out_index] = np.std(shots[i])
        out_names.append(names[i] + '_std')
        out_index += 1
    return [out, out_names]
    