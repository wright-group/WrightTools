# -*- coding: utf-8 -*-
"""
WMELs: TRIVE population transfer
================================

Draw WMELs for TRIVE population transfer.
"""

import matplotlib.pyplot as plt

import WrightTools.diagrams.WMEL as WMEL

artist = WMEL.Artist(
    size=[4, 3],
    energies=[0.0, 0.4, 0.5, 0.8, 0.9, 1.0],
    number_of_interactions=6,
    state_names=["g", "1S", "1P", "2x 1S", "1S+1P", "2x 1P"],
)

artist.label_rows([r"$\mathrm{\alpha}$", r"$\mathrm{\beta}$", r"$\mathrm{\gamma}$"])
artist.label_columns(["diag before", "cross before", "diag after", "cross after"], font_size=8)

artist.clear_diagram([1, 2])
artist.clear_diagram([2, 2])

# diag before alpha
artist.add_arrow([0, 0], 0, [0, 2], "ket", "-2")
artist.add_arrow([0, 0], 1, [2, 0], "ket", "2'")
artist.add_arrow([0, 0], 2, [0, 2], "ket", "1")
artist.add_arrow([0, 0], 3, [2, 0], "out")

# diag before beta
artist.add_arrow([0, 1], 0, [0, 2], "ket", "-2")
artist.add_arrow([0, 1], 1, [0, 2], "bra", "2'")
artist.add_arrow([0, 1], 2, [2, 5], "ket", "1")
artist.add_arrow([0, 1], 3, [5, 2], "out")

# diag before gamma
artist.add_arrow([0, 2], 0, [0, 2], "ket", "-2")
artist.add_arrow([0, 2], 1, [0, 2], "bra", "2'")
artist.add_arrow([0, 2], 2, [2, 0], "bra", "1")
artist.add_arrow([0, 2], 3, [2, 0], "out")

# cross before alpha
artist.add_arrow([1, 0], 0, [0, 2], "ket", "-2")
artist.add_arrow([1, 0], 1, [2, 0], "ket", "2'")
artist.add_arrow([1, 0], 2, [0, 1], "ket", "1")
artist.add_arrow([1, 0], 3, [1, 0], "out")

# cross before beta
artist.add_arrow([1, 1], 0, [0, 2], "ket", "-2")
artist.add_arrow([1, 1], 1, [0, 2], "bra", "2'")
artist.add_arrow([1, 1], 2, [2, 4], "ket", "1")
artist.add_arrow([1, 1], 3, [4, 2], "out")

# diag after alpha
artist.add_arrow([2, 0], 0, [0, 2], "ket", "-2")
artist.add_arrow([2, 0], 1, [2, 0], "ket", "2'")
artist.add_arrow([2, 0], 4, [0, 2], "ket", "1")
artist.add_arrow([2, 0], 5, [2, 0], "out")

# diag after beta
artist.add_arrow([2, 1], 0, [0, 2], "ket", "-2")
artist.add_arrow([2, 1], 1, [0, 2], "bra", "2'")
artist.add_arrow([2, 1], 2, [2, 1], "ket")
artist.add_arrow([2, 1], 3, [2, 1], "bra")
artist.add_arrow([2, 1], 4, [1, 4], "ket", "1")
artist.add_arrow([2, 1], 5, [4, 1], "out")

# cross after alpha
artist.add_arrow([3, 0], 0, [0, 2], "ket", "-2")
artist.add_arrow([3, 0], 1, [2, 0], "ket", "2'")
artist.add_arrow([3, 0], 4, [0, 1], "ket", "1")
artist.add_arrow([3, 0], 5, [1, 0], "out")

# cross after beta
artist.add_arrow([3, 1], 0, [0, 2], "ket", "-2")
artist.add_arrow([3, 1], 1, [0, 2], "bra", "2'")
artist.add_arrow([3, 1], 2, [2, 1], "ket")
artist.add_arrow([3, 1], 3, [2, 1], "bra")
artist.add_arrow([3, 1], 4, [1, 3], "ket", "1")
artist.add_arrow([3, 1], 5, [3, 1], "out")

# cross after gamma
artist.add_arrow([3, 2], 0, [0, 2], "ket", "-2")
artist.add_arrow([3, 2], 1, [0, 2], "bra", "2'")
artist.add_arrow([3, 2], 2, [2, 1], "ket")
artist.add_arrow([3, 2], 3, [2, 1], "bra")
artist.add_arrow([3, 2], 4, [1, 0], "bra", "1")
artist.add_arrow([3, 2], 5, [1, 0], "out")

artist.plot()
plt.show()
