import WrightTools as wt

if True:
    # import data
    filepaths = wt.kit.glob_handler('.dat', folder='data_files', identifier = 'high res')
    data = wt.data.from_COLORS(filepaths, name = 'PbSe')
    data.save('data.p')
    # import absorbance
    absorbance = wt.data.from_JASCO(r'data_files\DK-2014.09.17-PbSe 025.4.a.truncated.txt')
    absorbance.save('absorbance.p')

if True:
    # plot data
    data = wt.data.from_pickle('data.p')
    absorbance = wt.data.from_pickle('absorbance.p')

    artist = wt.artists.mpl_2D(data)
    artist.sideplot(absorbance)
    artist.plot(output_folder='images_out', xbin=True, ybin=True)
