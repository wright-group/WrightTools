"""
Split
=====

Some examples of how splitting works.
"""

from matplotlib import pyplot as plt
import WrightTools as wt
from WrightTools import datasets


# p = datasets.PyCMDS.w2_w1_000
# d = wt.data.from_PyCMDS(p)

# d.convert("wn", convert_variables=True)

# # A simple split along an axis
# a = d.split("w2", [7000, 8000])
# # A more complicated split along some diagonal
# b = d.split("w1+w2+7000", [20400, 23000], units="wn")
# # A particularly strange split
# d.create_variable("strange", values=d.channels[0].points / d.channels[0].max())
# c = d.split("strange", [0.2, 0.4])

# Plot the splits in columns
fig, gs = wt.artists.create_figure(nrows=len(c), cols=[1, 1, 1])
# for j, (col, title) in enumerate(zip([a, b, c], ["Simple", "Medium", "Advanced"])):
#     for i, da in enumerate(col.values()):
#         ax = plt.subplot(gs[i, j])
#         if i == 0:
#             ax.set_title(title)
#         ax.pcolor(da)
#         ax.set_xlim(d.axes[0].min(), d.axes[0].max())
#         ax.set_ylim(d.axes[1].min(), d.axes[1].max())

# wt.artists.set_fig_labels(xlabel=d.axes[0].label, ylabel=d.axes[1].label)
