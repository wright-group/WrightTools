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

    chopped = ['signal','pyro1']

    channel_indicies = [i for i, x in enumerate(kinds) if x == 'channel']
    chopper_indicies = [len(kinds)-1]#[i for i, x in enumerate(kinds) if x == 'chopper']
    if len(chopper_indicies) == 1:
        chpr_idx = chopper_indicies[0]
    
    mask = shots[chpr_idx]
    for i in range(len(mask)):
        if mask[i]<0:
            mask[i] = 0
        if shots[chpr_idx][i]==0:
            shots[chpr_idx][i]=-1

    out = np.full(len(channel_indicies)*2+2+3+4, np.nan)
    out_index = 0
    out_names = []
    for i in channel_indicies:
        if names[i] in chopped:
            out[out_index] = np.mean(shots[i]*shots[chopper_indicies[0]])*2.
            out_names.append(names[i] + '_cmean')
            out_index += 1
            baseline = [shots[i][k] for k in range(len(shots[i])) if not mask[k]]
            real = [shots[i][k] for k in range(len(shots[i])) if mask[k]]
            out[out_index] = np.sqrt(np.power(np.std(baseline),2)+np.power(np.std(real),2))
            out_names.append(names[i] + '_cstd')
            out_index += 1
            if names[i] == 'signal':
                out[out_index] = np.mean(shots[i])
                out_names.append(names[i] + '_mean')
                out_index += 1
                out[out_index] = np.std(shots[i])
                out_names.append(names[i] + '_std')
                out_index += 1
        if names[i] not in chopped:
            out[out_index] = np.mean(shots[i])
            out_names.append(names[i] + '_mean')
            out_index += 1
            out[out_index] = np.std(shots[i])
            out_names.append(names[i] + '_std')
            out_index += 1
        

#### Shot-Normalized Signal
    try:
    # Step 1: get pyro channels
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
    # Step 2: Create a chopper mask
        sig_z_mask = [0.0 for i in range(len(shots[0]))]
        for i in range(len(shots[index_dict['signal']])):
            if shots[index_dict['signal']][i]>.004 or shots[index_dict['signal']][i]<-.004:
                sig_z_mask[i] = 1.0
    
    # Step 3: find "chi" for 2w1+w3
        off_sig_z =  np.array([sig_z_mask[i] for i in range(len(sig_z_mask)) if not mask[i]])
        off_signal = np.array([shots[index_dict['signal']][i] for i in range(len(shots[index_dict['signal']])) if not mask[i]])
        off_pyro2  = np.array([shots[index_dict['pyro2']][i] for i in range(len(shots[index_dict['pyro2']])) if not mask[i]])
        off_pyro3  = np.array([shots[index_dict['pyro3']][i] for i in range(len(shots[index_dict['pyro3']])) if not mask[i]])
        
        ave_sig = np.mean(off_signal*off_sig_z)
        chi = np.mean(off_signal*off_sig_z/(off_pyro2*off_pyro2*off_pyro3))
        if chi < 0:
            chi = 0
        
        out_names.append('chi_2w2+w3')
        out[out_index] = chi
        out_index +=1
    
    # Step 4: Normalize after (not) subtracting the appropreate chi*p1*p1*p3 value
        on_sig_z =  np.array([sig_z_mask[i] for i in range(len(sig_z_mask)) if mask[i]])
        on_signal = np.array([shots[index_dict['signal']][i] for i in range(len(shots[index_dict['signal']])) if mask[i]])
        on_pyro1  = np.array([shots[index_dict['pyro1']][i]  for i in range(len(shots[index_dict['pyro1']]))  if mask[i]])
        on_pyro2  = np.array([shots[index_dict['pyro2']][i]  for i in range(len(shots[index_dict['pyro2']]))  if mask[i]])
        on_pyro3  = np.array([shots[index_dict['pyro3']][i]  for i in range(len(shots[index_dict['pyro3']]))  if mask[i]])
        # I'm not sure which method here is better. Either way, we overestimate the presence of 2w1+w3 because of decreased w3 power in the presence of w2.
        # Probably doesn't matter. The right way is probably based on stats, not the Int equation.
        norm_signal_clever = np.mean(on_signal*on_sig_z/(on_pyro1*on_pyro2*on_pyro3)-chi*on_pyro2*on_pyro2*on_pyro3)
        norm_signal_simple = np.mean(on_signal*on_sig_z/(on_pyro1*on_pyro2*on_pyro3))-ave_sig
        
        out_names.append('chi_py_norm_signal_chop')
        out_names.append('ave_norm_signal_chop')
        out[out_index] =  norm_signal_clever
        out_index +=1
        out[out_index] = norm_signal_simple
        out_index +=1
        
    # Raw and (Bulk) Normalized Photon Counting
        
        high_baseline = 0.004
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
        norm_fac = out[out_names.index('pyro1_cmean')]*out[out_names.index('pyro2_mean')]*out[out_names.index('pyro3_mean')]
        norm_fac = np.abs(norm_fac)
        out_names.append('pmt_dark_counts')
        out_names.append('raw_0_photon')    
        out_names.append('calc_pp100s_0')
        out_names.append('py_norm_pps_0')
        out[-4] = photon_count[-1]
        out[-3] = photon_count[0]+photon_count[-1]
        out[-2] = null_mean*100
        out[-1] = null_mean/norm_fac
    except:
        pass
    
    return [out, out_names]
