### import ####################################################################


import itertools

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

import scipy
from scipy.optimize import leastsq
from scipy.interpolate import griddata, interp1d, interp2d, UnivariateSpline

import numpy as np

import WrightTools as wt


### define ####################################################################


### amplitude and center (preamp) #############################################


if False:
    # THIS METHOD IS BROKEN - BJT 2016.01.17
    # get data from file
    motortune_path = r'MOTORTUNE [w1_Crystal_1, w1_Delay_1, wa] 2016.01.13 19_00_00.data'
    curves = [r'OPA1 (10743) base - 2015.11.10 09_25_25.crv',
              r'OPA1 (10743) mixer1 - 2015.11.09.crv',
              r'OPA1 (10743) mixer2 - 2015.10.26 12_39_55.crv',
              r'OPA1 (10743) mixer3 - 2013.06.01.crv']
    headers = wt.kit.read_headers(motortune_path)
    c1_index = headers['name'].index('w1_Crystal_1')
    d1_index = headers['name'].index('w1_Delay_1')
    wa_index = headers['name'].index('wa')
    zi_index = headers['name'].index('array')
    # this array is very very very large (~2.6 billion lines)
    # it cannot be imported directly into memory
    # instead I load chunks and fit them to Gaussians as I go
    outs = np.full((10201, 4), np.nan)
    i = len(headers)
    j = 0
    n = sum(1 for line in open(motortune_path, 'r'))
    function = wt.fit.Gaussian()
    while i <= n:
        # get data from file
        with open(motortune_path, 'r') as f:
            lines = [line for line in itertools.islice(f, i, i+256)]
            arr = np.array([np.fromstring(line, sep='\t') for line in lines]).T
        # fit data, record
        out = function.fit(arr[zi_index], arr[wa_index])
        outs[j] = out
        # finish
        i += 256
        j += 1
        wt.kit.update_progress(100.*j/10201)
    c1 = np.array(headers['w1_Crystal_1 points'])
    d1 = np.array(headers['w1_Delay_1 points'])
    outs.shape = (101, 101, 4)
    np.savez('amp_and_cen.npz', c1=c1, d1=d1, outs=outs)


if False:
    # plot data
    # import from npz
    npz = np.load('amp_and_cen.npz')
    c1 = npz['c1']
    d1 = npz['d1']
    outs = npz['outs']
    outs = outs.T
    cen = outs[0].flatten()
    wid = outs[1].flatten()
    amp = outs[2].flatten()
    # grid out c1, d1
    c1_grid, d1_grid = np.meshgrid(c1, d1, indexing='xy')
    c1_list = c1_grid.flatten()
    d1_list = d1_grid.flatten()
    # remove points with amplitudes that are ridiculous
    amp[amp<0.1] = np.nan
    amp[amp>4] = np.nan
    # remove points with centers that are ridiculous
    cen[cen<1150] = np.nan
    cen[cen>1650] = np.nan
    # remove points with widths that are ridiculous
    wid[wid<5] = np.nan
    wid[wid>500] = np.nan
    amp, cen, wid, c1_list, d1_list = wt.kit.remove_nans_1D([amp, cen, wid, c1_list, d1_list])
    # grid data
    xi = tuple(np.meshgrid(c1, d1, indexing='xy'))
    c1_grid, d1_grid = xi
    points = tuple([c1_list, d1_list])
    amp = griddata(points, amp, xi, method='cubic')
    amp /= np.nanmax(amp)
    cen = griddata(points, cen, xi, method='cubic')
    # prepare plot
    fig = plt.figure(figsize=[14, 6])
    gs = grd.GridSpec(1, 5, hspace=0.05, wspace=0.05, width_ratios=[20, 1, 5, 20, 1])
    # intensity
    cmap = wt.artists.colormaps['default']
    cmap.set_under([0.75]*3, 1)
    ax0 = plt.subplot(gs[0])
    X, Y, Z = wt.artists.pcolor_helper(c1, d1, amp)
    cax = ax0.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(Z), cmap=cmap)
    ax0.set_xlim(c1.min(), c1.max())
    ax0.set_ylim(1.35, 1.8)
    ax0.set_xlabel('C1 (degrees)', fontsize=16)
    ax0.set_ylabel('D1 (mm)', fontsize=16)
    ax0.grid()
    plt.colorbar(cax, plt.subplot(gs[1]))
    wt.artists.corner_text('intensity (a.u.)', ax=ax0, fontsize=16)
    ax0.contour(c1, d1, amp, 5, colors='k')
    # color
    cmap = wt.artists.colormaps['rainbow']
    cmap.set_under([0.75]*3, 1)
    ax1 = plt.subplot(gs[3])
    X, Y, Z = wt.artists.pcolor_helper(c1, d1, cen)
    cax = ax1.pcolor(X, Y, Z, vmin=np.nanmin(Z), vmax=np.nanmax(Z), cmap=cmap)
    ax1.set_xlim(c1.min(), c1.max())
    ax1.set_ylim(1.35, 1.8)
    ax1.set_xlabel('C1 (degrees)', fontsize=16)
    ax1.set_ylabel('D1 (mm)', fontsize=16)
    ax1.grid()
    plt.colorbar(cax, plt.subplot(gs[4]))
    wt.artists.corner_text('color (nm)', ax=ax1, fontsize=16)
    ax1.contour(c1, d1, cen, 25, colors='k')
    # finish
    plt.savefig('amplitude_and_center.png', transparent=True, dpi=300)
    plt.close('all')


### amplitude and center (poweramp) ###########################################


if True:
    motortune_path = r'MOTORTUNE [w1, w1_Crystal_2, w1_Delay_2, wa] 2016.01.15 09_50_46.data'
    n = wt.kit.file_len(motortune_path)
    # for some reason, read headers fails for this file...
    #headers = wt.kit.read_headers(motortune_path)
    with open(motortune_path) as f:
        for i, l in enumerate(f):
            if '# name' in l:
                break

    w1_index = 5
    C2_index = 8
    D2_index = 9
    wa_index = 28
    zi_index = 29

    w1_len = 25
    C2_len = 51
    D2_len = 51 

    w1 = np.full(65025, np.nan)
    C2 = np.full(65025, np.nan)
    D2 = np.full(65025, np.nan)
    outs = np.full((65025, 4), np.nan)
    function = wt.fit.Gaussian()
    j = 0
    i += 1
    while i <= n:
        # get data from file
        with open(motortune_path, 'r') as f:
            lines = [line for line in itertools.islice(f, i, i+256)]
            arr = np.array([np.fromstring(line, sep='\t') for line in lines]).T
        # fit data, record
        out = function.fit(arr[zi_index], arr[wa_index])
        outs[j] = out
        # record others
        w1[j] = arr[w1_index, 0]
        C2[j] = arr[C2_index, 0]
        D2[j] = arr[D2_index, 0]
        # finish
        i += 256
        j += 1
        wt.kit.update_progress(100.*j/65025)
        
    np.savez('poweramp.npz', w1=w1, c2=C2, d2=D2, outs=outs)


    






