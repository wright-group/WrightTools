### import ####################################################################


import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
plt.close('all')

import scipy
from scipy.optimize import leastsq
from scipy.interpolate import griddata, interp1d, interp2d, UnivariateSpline

import numpy as np

import WrightTools as wt


### define ####################################################################





### signal and idler motortune ################################################


if False:
    data_path = r'MOTORTUNE [w2_Grating, w2_BBO] 2015.10.15 17_38_22.data'
    # this data is sufficiently old that we have to process it manually :-(
    # get values from file
    headers = wt.kit.read_headers(data_path)
    arr = np.genfromtxt(data_path).T
    # extract arrays
    grating_index = headers['name'].index('w2_Grating')
    bbo_index = headers['name'].index('w2_BBO')
    signal_index = headers['name'].index('pyro2_mean')
    gra = arr[grating_index]
    gra.shape = (-1, 401)
    gra = gra[:, 0]
    bbo = arr[bbo_index]
    bbo.shape = (-1, 401)
    bbo = bbo[0]
    sig = arr[signal_index]
    sig.shape = (-1, 401)
    sig -= sig.min()
    sig /= sig.max()
    sig = sig.T
    # prepare plot
    fig = plt.figure(figsize=[8, 6])
    gs = grd.GridSpec(1, 2, hspace=0.05, wspace=0.05, width_ratios=[20, 1])
    # pcolor
    cmap = wt.artists.colormaps['default']
    ax = plt.subplot(gs[0])
    X, Y, Z = wt.artists.pcolor_helper(gra, bbo, sig)    
    cax = plt.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(Z), cmap=cmap)
    ax.set_xlim(22, gra.max())
    ax.set_ylim(bbo.min(), bbo.max())
    # labels
    ax.set_xlabel('Grating (mm)', fontsize=16)
    ax.set_ylabel('BBO (mm)', fontsize=16)
    ax.grid()
    ax.axvline(34.6, c='k', alpha=0.5, lw=2)
    ax.axhline(39.2, c='k', alpha=0.5, lw=2)
    # on-plot labels
    distance = 0.05
    wt.artists.corner_text('II signal', ax=ax, corner='UL', fontsize=16, distance=distance)
    wt.artists.corner_text('II idler', ax=ax, corner='UR', fontsize=16, distance=distance)
    wt.artists.corner_text('III signal', ax=ax, corner='LL', fontsize=16, distance=distance)
    wt.artists.corner_text('III idler', ax=ax, corner='LR', fontsize=16, distance=distance)
    # colorbar
    plt.colorbar(cax, cax=plt.subplot(gs[1]))
    # finish
    plt.savefig('signal_and_idler_motortune.png', dpi=300, transparent=True)
    
    
### DFG Mixer Motortune #######################################################


if False:
    data_path = r'MOTORTUNE [w2, w2_Mixer] 2015.10.16 17_59_58.data'
    # this data is sufficiently old that we have to process it manually :-(
    # get values from file
    headers = wt.kit.read_headers(data_path)
    arr = np.genfromtxt(data_path).T
    # extract arrays
    w2_index = headers['name'].index('w2')
    mixer_index = headers['name'].index('w2_Mixer')
    signal_index = headers['name'].index('pyro2_mean')
    w2 = arr[w2_index]
    w2.shape = (-1, 501)
    w2 = w2[:, 0]
    mix = arr[mixer_index]
    mix.shape = (-1, 501)
    mix = mix[0]
    sig = arr[signal_index]
    sig.shape = (-1, 501)
    sig -= sig.min()
    sig /= sig.max()
    sig = sig.T
    # prepare plot
    fig = plt.figure(figsize=[8, 6])
    gs = grd.GridSpec(1, 2, hspace=0.05, wspace=0.05, width_ratios=[20, 1])
    # pcolor
    cmap = wt.artists.colormaps['default']
    ax = plt.subplot(gs[0])
    X, Y, Z = wt.artists.pcolor_helper(w2, mix, sig)    
    cax = plt.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(Z), cmap=cmap)
    ax.set_xlim(w2.min(), w2.max())
    ax.set_ylim(mix.min(), mix.max())
    ax.grid()
    # axis labels
    ax.set_xlabel('w2 (wn)', fontsize=16)
    ax.set_ylabel('Grating (mm)', fontsize=16)
    # colorbar
    plt.colorbar(cax, cax=plt.subplot(gs[1]))
    # finish
    plt.savefig('DFG_mixer_motortune.png', dpi=300, transparent=True)
