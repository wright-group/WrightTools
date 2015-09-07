import WrightTools as wt

files = wt.kit.glob_handler(extension='', folder='data_files')

if True:
    # import data
    data = wt.data.from_KENT(files, name='LDS TSF')
    data.save('data.p')

if True:
    # plot data
    data = wt.data.from_pickle('data.p')
    artist = wt.artists.mpl_2D(data)
    artist.plot()
