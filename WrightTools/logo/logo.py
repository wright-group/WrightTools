"""Generate WrightTools logo."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

import h5py

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import WrightTools as wt


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))

cmap = wt.artists.colormaps["default"]
cmap.set_under(color="w", alpha=0)

text_effects = [PathEffects.withStroke(linewidth=5, foreground="w")]


# --- logo ----------------------------------------------------------------------------------------


wt.artists.apply_rcparams("publication")
matplotlib.rcParams["font.monospace"] = "DejaVu Sans Mono"
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["text.usetex"] = False


# get arrays
p = os.path.join(here, "peak.h5")
h5 = h5py.File(p, "r")
xi = np.array(h5["yi"])
yi = np.array(h5["xi"])
zi = np.transpose(np.array(h5["zi"]))

# process
zi = np.log10(zi)
zi = zi - np.nanmin(zi)
zi = zi / np.nanmax(zi)
xi, yi, zi = wt.kit.zoom2D(xi, yi, zi)

# create figure
fig = plt.figure()
ax = plt.subplot(aspect="equal")

cutoff = 0.58
levels = np.linspace(cutoff, np.nanmax(zi), 10)

# contours
ax.contourf(xi, yi, zi, cmap=cmap, alpha=1, vmin=cutoff, vmax=np.nanmax(zi), levels=levels)
ax.contour(xi, yi, zi, levels=levels, colors="k", lw=5, alpha=0.5)
ax.contour(xi, yi, zi, levels=[cutoff], colors="k", lw=5, alpha=1)

# decorate
ax.set_xlim(5500, 8500)
ax.set_ylim(5500, 8500)
wt.artists.set_ax_spines(ax=ax, lw=0)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# text
ax.text(6250, 7500, "ω", ha="center", va="center", fontsize=200, path_effects=text_effects)
ax.text(7750, 6500, "τ", ha="center", va="center", fontsize=200, path_effects=text_effects)

# save
plt.savefig("logo.png", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
plt.close(fig)


# --- favicon -------------------------------------------------------------------------------------


# create figure
fig = plt.figure()
ax = plt.subplot(aspect="equal")

# decorate
ax.set_xlim(6000, 8000)
ax.set_ylim(6000, 8000)
wt.artists.set_ax_spines(ax=ax, lw=0)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# text
ax.text(7000, 7000, "ωτ", ha="center", va="center", fontsize=200, path_effects=text_effects)

# save
plt.savefig("favicon.png", dpi=19, bbox_inches="tight", pad_inches=0, transparent=True)
plt.close(fig)
