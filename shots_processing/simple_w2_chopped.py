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

    chopped = ['signal','pyro2']
    num_shots = len(shots[0])

    channel_indicies = [i for i, x in enumerate(kinds) if x == 'channel']
    chopper_indicies = [len(kinds)-1]#[i for i, x in enumerate(kinds) if x == 'chopper']
    if len(chopper_indicies) == 1:
        chpr_idx = chopper_indicies[0]
    
    mask = shots[chpr_idx]
    for i in range(num_shots):
        if mask[i]==0:
            mask[i] = -1

    out = []
    out_names = []
    for i in channel_indicies:
        if names[i] in chopped:
            out.append(np.mean(shots[i]*mask)*2.)
            out_names.append(names[i] + '_cmean')
            baseline = [shots[i][k] for k in range(num_shots) if mask[k]==-1]
            real = [shots[i][k] for k in range(num_shots) if mask[k]==1]
            out.append(np.sqrt(np.power(np.std(baseline),2)+np.power(np.std(real),2)))
            out_names.append(names[i] + '_cstd')

        out.append(np.mean(shots[i]))
        out_names.append(names[i] + '_mean')
        out.append(np.std(shots[i]))
        out_names.append(names[i] + '_std')

#### Shot-Normalized Signal
# Step 1: get pyro channels
    try:
        needed_channels = ['signal','pyro1','pyro2','pyro3']
        index_dict = dict()
        for c in needed_channels:
            if c in names:
                index_dict[c]=names.index(c)
            else:
                # a needed channel isn't avalible
                #TODO: make logger work
                # g.logger.log('error', 'Additional signal channels are needed to normalize signal!')
                return [out, out_names]
    # Step 2: Create a signal mask
        sig_z_mask = [0.0 for i in range(len(shots[0]))]
        for i in range(len(shots[index_dict['signal']])):
            if shots[index_dict['signal']][i]>.004 or shots[index_dict['signal']][i]<-.004:
                sig_z_mask[i] = 1.0
    
    # Step 3: find "chi" for 2w1+w3 (i.e. w2 is "off")
        no_w2_sig_z =  np.array([sig_z_mask[i] for i in range(len(sig_z_mask)) if not mask[i]])
        no_w2_signal = np.array([shots[index_dict['signal']][i] for i in range(num_shots) if mask[i]==-1])
        no_w2_pyro1  = np.array([shots[index_dict['pyro1' ]][i] for i in range(num_shots) if mask[i]==-1])
        no_w2_pyro3  = np.array([shots[index_dict['pyro3' ]][i] for i in range(num_shots) if mask[i]==-1])
        
        ave_sig = np.mean([no_w2_signal[i]*no_w2_sig_z[i] for i in range(num_shots)])
        chi = np.mean([no_w2_signal[i]*no_w2_sig_z[i]/(no_w2_pyro1[i]*no_w2_pyro1[i]*no_w2_pyro3[i]) for i in range (len(no_w2_sig_z))])
        if chi < 0:
            chi = 0
        
        out_names.append('Signal_w2_off')
        out.append(ave_sig)
        out_names.append('chi_2w1+w3')
        out.append(chi)
    
    # Step 4: Normalize after (not) subtracting the appropreate chi*p1*p1*p3 value (when w2 is "on")
        w2_on_sig_z =  np.array([sig_z_mask[i] for i in range(len(sig_z_mask)) if mask[i]])
        w2_on_signal = np.array([shots[index_dict['signal']][i] for i in range(num_shots) if mask[i]==1])
        w2_on_pyro1  = np.array([shots[index_dict['pyro1' ]][i] for i in range(num_shots) if mask[i]==1])
        w2_on_pyro2  = np.array([shots[index_dict['pyro2' ]][i] for i in range(num_shots) if mask[i]==1])
        w2_on_pyro3  = np.array([shots[index_dict['pyro3' ]][i] for i in range(num_shots) if mask[i]==1])
        # I'm not sure which method here is better. Either way, we overestimate the presence of 2w1+w3 because of decreased w3 power in the presence of w2.
        # Probably doesn't matter. The right way is probably based on stats, not the Int equation.
        norm_signal_clever = np.mean([w2_on_signal[i]*w2_on_sig_z[i]/(w2_on_pyro1[i]*w2_on_pyro2[i]*w2_on_pyro3[i])-chi*w2_on_pyro1[i]*w2_on_pyro1[i]*w2_on_pyro3[i] for i in range(len(no_w2_sig_z))])
        norm_signal_simple = np.mean([w2_on_signal[i]*w2_on_sig_z[i]/(w2_on_pyro1[i]*w2_on_pyro2[i]*w2_on_pyro3[i]) for i in range(len(no_w2_sig_z))])-ave_sig
        
        out_names.append('chi_py_norm_signal_chop')
        out_names.append('ave_norm_signal_chop')
        out.append(norm_signal_clever)
        out.append(norm_signal_simple)
    except:
        pass
    
    return [np.array(out), out_names]
