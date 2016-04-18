### import ####################################################################


import matplotlib.pyplot as plt
plt.close('all')

from colorspacious import cspace_converter

import numpy as np

import WrightTools as wt


### define ####################################################################


### colorbars #################################################################


if True:
    height = len(wt.artists.colormaps)*0.5
    fig, axes = plt.subplots(nrows=len(wt.artists.colormaps)*3, figsize=[8, height])
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax_index = 0
    for name, cmap in wt.artists.colormaps.items():
        # space
        ax_index +=1
        # greyscale
        rgb = cmap(gradient[0])[np.newaxis,:,:3]
        lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
        L = lab[0,:,0]
        L = np.float32(np.vstack((L, L, L)))
        axes[ax_index].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100)  # greyscale
        pos_grey = list(axes[ax_index].get_position().bounds)
        ax_index += 1
        # color
        axes[ax_index].imshow(gradient, aspect='auto', cmap=cmap, vmin=0., vmax=1.)
        pos_color = list(axes[ax_index].get_position().bounds)
        # name
        x_text = pos_color[0] - 0.01
        y_text = pos_color[1] + pos_grey[3]
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=12)
        # space
        ax_index += 1
    for ax in axes:
        ax.set_axis_off()
    axes[0].set_title('WrightTools colormaps')
    plt.savefig('colormaps.png', dpi=300, transparent=True)


### cubehelix components ######################################################


if False:
    cmap = wt.artists.make_cubehelix(gamma=0.5, s=0.25, r=-6/6., h=1.25, 
                                     lum_rev=False, darkest=0.7, plot=True)
    plt.savefig('colormap_components.png', dpi=300, transparent=True)



