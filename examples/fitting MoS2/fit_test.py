import numpy as np

import matplotlib.pyplot as plt
plt.close('all')

import WrightTools as wt

if False:
    # import data
    files = wt.kit.glob_handler('.dat', folder='data_files', identifier='Wigner')
    data = wt.data.from_COLORS(files, name='MoS2')
    data.convert('eV')
    data.level(0, 'd2', -3)
    data.scale(0, 'amplitude')
    data.normalize(0)
    data.flip('d2')
    data.save('data.p')

data = wt.data.from_pickle('data.p')

if True:
    # plot data
    artist = wt.artists.mpl_2D(data, 'w1', 'w2')
    artist.plot()


data = data.split('d2', -50)[1]


    

if False:
    # do the fit
    
    function = wt.fit.Exponential()
    function.limits['amplitude'] = [0, 1]
    function.limits['tau'] = [50, 5000]
    function.limits['offset'] = [0, 0]
    
    fit = wt.fit.Fitter(function, data, 'd2')
    outs = fit.run()
    model = fit.model
    model.heal(method='linear')
    
    model.save('model.p')
    outs.save('outs.p')

if False:
    # plot model

    model = wt.data.from_pickle('model.p')
    
    artist = wt.artists.mpl_2D(model, 'w1', 'w2')
    artist.plot(0, output_folder='figures', contours=0, facecolor='grey')
    
if False:
    # test offset method
    data = wt.data.from_pickle('data.p')
    data.flip('d2')
    data.channels = [data.channels[0]]
    data.clip(zmin=0, zmax=1, replace='val')
    data = data.chop('w1', 'd2')[15]
    
    points = np.linspace(1.7, 2.1, 10)
    values = np.linspace(-100., -300., 10)
    
    with wt.kit.Timer():
        if True:
            data.offset(points, values, along='w1', offset_axis='d2',
                        method='linear', mode='valid')
        
    artist = wt.artists.mpl_2D(data, 'w1', 'd2')
    artist.plot(facecolor='grey')
    
if False:
    # plot difference
    
    model = wt.data.from_pickle('model.p')
    
    artist = wt.artists.difference_2D(data, model, 'w1', 'w2')
    artist.plot(output_folder='figures',)

if False:
    # plot outs

    outs = wt.data.from_pickle('outs.p')
    outs.clip(1, zmin=0, zmax=2000, replace='nan')
    outs.heal(0, fill_value=outs.channels[1].znull)
    outs.heal(1, fill_value=outs.channels[1].znull)
    
    artist = wt.artists.mpl_2D(outs, 'w1', 'w2')
    artist.plot(0, output_folder='figures', contours=9, pixelated=True,
                facecolor='grey', local=False, autosave=True)
