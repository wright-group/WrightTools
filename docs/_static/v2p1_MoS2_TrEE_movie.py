# import
import WrightTools as wt
from WrightTools import datasets

# create
p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
data = wt.open(p)
# cleanup
data.level("ai0", "d2", -3)
data.scale()
data.convert("eV")
data.name = "MoS2"
data.flip("d2")
# plot
artist = wt.artists.mpl_2D(data, "w1", "w2")
ps = artist.plot(fname="MoS2", output_folder="v2p1_MoS2_TrEE_movie", autosave=True)
# stitch
wt.artists.stitch_to_animation(ps, outpath="v2p1_MoS2_TrEE_movie.gif")
