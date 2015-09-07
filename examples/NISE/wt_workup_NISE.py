import os

import numpy as np

import NISE
from NISE.lib.misc.__init__ import NISE_path
import NISE.lib.measure as m
import NISE.experiments.trive as trive
import NISE.hamiltonians.H0 as H_1
import NISE.hamiltonians.params.inhom as inhom

import WrightTools as wt

if False:
    # generate data (this might take a minute)

    # set dimensions
    trive.exp.set_coord(trive.d1, 0.)
    trive.exp.set_coord(trive.d2, 0.)
    trive.exp.set_coord(trive.ss, 45.)
    w1 = trive.w1
    w2 = trive.w2
    ws = trive.ws
    d2 = trive.d2
    d1 = trive.d1
    w1.points = np.linspace(6000, 8000, num=11)
    w2.points = np.linspace(6000, 8000, num=11)
    trive.exp.timestep = 2.0
    # set all coherence times the same
    tau = 25.
    H_1.Omega.tau_ag = tau
    H_1.Omega.tau_2aa = tau
    H_1.Omega.tau_2ag = tau
    H = H_1.Omega()

    # run
    scan = trive.exp.scan(w1, w2, H=H)
    scan.run(autosave=False, mp=False, chunk=False)

    # measure
    slitwidth = 120.
    measure = m.Measure(scan, m.Mono, m.SLD)
    m.Mono.slitwidth = slitwidth
    measure.run(save=False)
    data = wt.data.from_NISE(measure)
    data.save('data.p')

if True:
    # plot data
    data = wt.data.from_pickle('data.p')
    data.zoom(3)
        
    artist = wt.artists.mpl_2D(data, 0, 1)
    artist.plot()
