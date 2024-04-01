"""
use the right combination of colorbar and matplotlib norms
to communicate your data!
"""

import WrightTools as wt
from WrightTools import datasets

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from matplotlib import colormaps as colormaps

import numpy as np


# --- colormaps -----------------------------------------------------------------------------------
unsigned_cmap = [
    wt.artists.colormaps["default"].copy(),
    colormaps["cubehelix_r"],
    colormaps["magma_r"],
    colormaps["viridis_r"],
][0]
unsigned_cmap.set_under([0.9, 0.9, 0.9, 1])

cyclic_cmap = "twilight"

signed_cmap = ["twilight_shifted", "RdBu", "coolwarm", "seismic"][1]


# --- data ----------------------------------------------------------------------------------------
p = datasets.wt5.v1p0p0_perovskite_TA
signed = wt.open(p).at(d2=[-15, "fs"]).split("w1", [1.6])[1]
signed.bring_to_front("signal_diff")

unsigned = wt.data.from_PyCMDS(r"https://osf.io/75vny/download")
unsigned.bring_to_front("signal_diff")
unsigned.convert("eV")
unsigned.signal_diff.normalize()


# --- plot ----------------------------------------------------------------------------------------
label_kwargs = dict(fontsize=14, corner="LR", background_alpha=0.6)
fig, gs = wt.artists.create_figure(width=8, cols=[1, 1], nrows=4, wspace=1.3)


# --- unsigned data -------------------------------------------------------------------------------
# --- --- linear ----------------------------------------------------------------------------------
ax00 = fig.add_subplot(gs[0, 0], label="00 - unsigned linear")
wt.artists.corner_text("linear", **label_kwargs)
art = ax00.pcolormesh(unsigned, cmap=unsigned_cmap, autolabel="y")
fig.colorbar(art, ax=ax00, extend="min")

ax00.set_title("unsigned data")

# --- --- sqrt ------------------------------------------------------------------------------------
ax10 = fig.add_subplot(gs[1, 0], label="10 - unsigned sqrt")
wt.artists.corner_text("sqrt", **label_kwargs)
norm = mpl_colors.PowerNorm(gamma=0.5, vmin=0)
"""
NOTE: a bug with PowerNorm makes the colorbar extend arrow incorrect color
as recently as mpl 3.8.3.
a fix has been applied upstream; we just have to wait for it
https://github.com/matplotlib/matplotlib/pull/27589
in the meantime, we can define a custom norm to get past the issue
"""
_forward = lambda x: np.sign(x) * np.sqrt(np.abs(x))
_inverse = lambda x: np.sign(x) * x**2
norm = mpl_colors.FuncNorm((_forward, _inverse), vmin=0, clip=False)
art = ax10.pcolormesh(unsigned, norm=norm, cmap=unsigned_cmap, autolabel="y")
cb = fig.colorbar(art, ax=ax10, extend="min")

# --- --- log10 -----------------------------------------------------------------------------------
ax20 = fig.add_subplot(gs[2, 0], label="20 - unsigned log")
wt.artists.corner_text("log10", **label_kwargs)
norm = mpl_colors.LogNorm(vmin=5e-4, clip=False)
art = ax20.pcolormesh(unsigned, norm=norm, cmap=unsigned_cmap, autolabel="y")
cblog = fig.colorbar(art, ax=ax20, extend="min")

# --- --- log10, decadic cycles -------------------------------------------------------------------
# currently a bit hackish, but works.  We could make tools for this
ax30 = fig.add_subplot(gs[3, 0], label="30 - unsigned log cyclic")
wt.artists.corner_text("log10, cyclic", **label_kwargs)
mantissa = lambda x: np.mod(np.log10(x), -1)
unsigned.create_channel("mantissa", values=10 ** mantissa(unsigned.signal_diff[:]))
norm = mpl_colors.LogNorm()
art = ax30.pcolormesh(unsigned, channel="mantissa", norm=norm, cmap=cyclic_cmap, autolabel="both")
cb = fig.colorbar(art, ax=ax30)
cb.ax.yaxis.set_minor_formatter("{x:0.1f}")
cb.ax.set_yticks([0.1, 0.2, 0.5], minor=True)  # , labels=["0.2", "0.5"], minor=True)

# --- plot signed data ---------------------------------------------------------------------------
# --- --- linear norm ----------------------------------------------------------------------------
ax01 = fig.add_subplot(gs[0, 1], label="01 - signed linear")
ax01.set_title("signed data")
wt.artists.corner_text("linear", **label_kwargs)
art = ax01.pcolormesh(signed, cmap=signed_cmap, autolabel="y")
fig.colorbar(art, ax=ax01)

# --- --- bilinear norm --------------------------------------------------------------------------
ax11 = fig.add_subplot(gs[1, 1], label="11 - signed bilinear")
wt.artists.corner_text("bilinear", **label_kwargs)
norm = mpl_colors.TwoSlopeNorm(vcenter=signed.signal_diff.null)
art = ax11.pcolormesh(signed, norm=norm, cmap=signed_cmap, autolabel="y")
cb = fig.colorbar(art, ax=ax11)
cb.ax.set_yscale("linear")

# --- --- asinh norm ------------------------------------------------------------------------------
ax21 = fig.add_subplot(gs[2, 1], label="21 - signed asinh")
wt.artists.corner_text("asinh", **label_kwargs)
norm = mpl_colors.AsinhNorm(
    linear_width=1e-3, vmin=-signed.signal_diff.mag(), vmax=signed.signal_diff.mag()
)
art = ax21.pcolormesh(signed, norm=norm, cmap=signed_cmap, autolabel="y")
fig.colorbar(art, ax=ax21)

# --- --- symlog norm -----------------------------------------------------------------------------
ax31 = fig.add_subplot(gs[3, 1], label="31 - signed symlog")
wt.artists.corner_text("symLog", **label_kwargs)
norm = mpl_colors.SymLogNorm(
    linthresh=1e-3, vmin=-signed.signal_diff.mag(), vmax=signed.signal_diff.mag()
)
art = ax31.pcolormesh(signed, norm=norm, cmap=signed_cmap, autolabel="both")
fig.colorbar(art, ax=ax31)


# --- final decorations ---------------------------------------------------------------------------

for ax in fig.axes:
    label = ax.get_label()
    if label == "<colorbar>":
        continue
    ax.grid(c="k", lw=1, ls="-.", alpha=0.5)
    if label[0] != "3":  # not bottom row
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), visible=False)
    ax.set_xticks(ax.get_yticks())
    if label[1] == "0":  # unsigned
        ax.set_xlim(unsigned.w2.min(), unsigned.w2.max())
    else:
        ax.set_xlim(signed.w1.min(), signed.w1.max())
