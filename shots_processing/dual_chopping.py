import numpy as np

def process(shots, names, kinds):
    '''
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
    chopper_indicies = [i for i, x in enumerate(kinds) if x == 'chopper']
    out = np.full(len(channel_indicies)+1, np.nan)
    channel_indicies.pop(0)
    out_names = []
    # signal diff
    #            A B C D
    # chopper 1: - + + -
    # chopper 2: - - + +
    #   we want A-B+C-D
    al = []
    bl = []
    cl = []
    dl = []    
    for i in range(shots.shape[1]):
        c1 = shots[chopper_indicies[0], i]
        c2 = shots[chopper_indicies[1], i]
        if c1 == -1 and c2 == -1:
            al.append(shots[0, i])
        elif c1 == 1 and c2 == -1:
            bl.append(shots[0, i])
        elif c1 == 1 and c2 == 1:
            cl.append(shots[0, i])
        elif c1 == -1 and c2 == 1:
            dl.append(shots[0, i])
    a = np.mean(al)
    b = np.mean(bl)
    c = np.mean(cl)
    d = np.mean(dl)
    if False:
        print [len(l) for l in [al, bl, cl, dl]]
        print a, b, c, d
    out[0] = a-b+c-d
    out_names.append('signal_diff')
    # signal mean
    out[1] = np.mean(shots[0])
    out_names.append('signal_mean')
    # others
    for i in channel_indicies:
        out[i+1] = np.mean(shots[i])
        out_names.append(names[i])
    # finish
    return [out, out_names]
    