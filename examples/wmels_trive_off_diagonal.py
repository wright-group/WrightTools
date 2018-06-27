# -*- coding: utf-8 -*-
"""
WMELs: TRIVE off diagonal
=========================

Draw WMELs for TRIVE off diagonal.
"""

import matplotlib.pyplot as plt

import WrightTools.diagrams.WMEL as WMEL

artist = WMEL.Artist(
    size=[6, 2], energies=[0., 0.43, 0.57, 1.], state_names=["g", "a", "b", "a+b"]
)

artist.label_rows([r"$\mathrm{\alpha}$", r"$\mathrm{\beta}$", r"$\mathrm{\gamma}$"])
artist.label_columns(["I", "II", "III", "IV", "V", "VI"])

# pw1 alpha
artist.add_arrow([0, 0], 0, [0, 1], "ket", "1")
artist.add_arrow([0, 0], 1, [0, 2], "bra", "-2")
artist.add_arrow([0, 0], 2, [2, 0], "bra", "2'")
artist.add_arrow([0, 0], 3, [1, 0], "out")

# pw1 beta
artist.add_arrow([0, 1], 0, [0, 1], "ket", "1")
artist.add_arrow([0, 1], 1, [0, 2], "bra", "-2")
artist.add_arrow([0, 1], 2, [1, 3], "ket", "2'")
artist.add_arrow([0, 1], 3, [3, 2], "out")

# pw2 alpha
artist.add_arrow([1, 0], 0, [0, 1], "ket", "1")
artist.add_arrow([1, 0], 1, [1, 3], "ket", "2'")
artist.add_arrow([1, 0], 2, [3, 1], "ket", "-2")
artist.add_arrow([1, 0], 3, [1, 0], "out")

# pw2 beta
artist.add_arrow([1, 1], 0, [0, 1], "ket", "1")
artist.add_arrow([1, 1], 1, [1, 3], "ket", "2'")
artist.add_arrow([1, 1], 2, [0, 2], "bra", "-2")
artist.add_arrow([1, 1], 3, [3, 2], "out")

# pw3 alpha
artist.add_arrow([2, 0], 0, [0, 2], "bra", "-2")
artist.add_arrow([2, 0], 1, [0, 1], "ket", "1")
artist.add_arrow([2, 0], 2, [2, 0], "bra", "2'")
artist.add_arrow([2, 0], 3, [1, 0], "out")

# pw3 beta
artist.add_arrow([2, 1], 0, [0, 2], "ket", "-2")
artist.add_arrow([2, 1], 1, [0, 1], "ket", "1")
artist.add_arrow([2, 1], 2, [1, 3], "bra", "2'")
artist.add_arrow([2, 1], 3, [3, 2], "out")

# pw4 alpha
artist.add_arrow([3, 0], 0, [0, 2], "ket", "2'")
artist.add_arrow([3, 0], 1, [2, 3], "ket", "1")
artist.add_arrow([3, 0], 2, [3, 1], "ket", "-2")
artist.add_arrow([3, 0], 3, [1, 0], "out")

# pw4 beta
artist.add_arrow([3, 1], 0, [0, 2], "ket", "2'")
artist.add_arrow([3, 1], 1, [2, 3], "ket", "1")
artist.add_arrow([3, 1], 2, [0, 2], "bra", "-2")
artist.add_arrow([3, 1], 3, [3, 2], "out")

# pw5 alpha
artist.add_arrow([4, 0], 0, [0, 2], "bra", "-2")
artist.add_arrow([4, 0], 1, [2, 0], "bra", "2'")
artist.add_arrow([4, 0], 2, [0, 1], "ket", "1")
artist.add_arrow([4, 0], 3, [1, 0], "out")

# pw5 beta
artist.add_arrow([4, 1], 0, [0, 2], "bra", "-2")
artist.add_arrow([4, 1], 1, [0, 2], "ket", "2'")
artist.add_arrow([4, 1], 2, [2, 3], "ket", "1")
artist.add_arrow([4, 1], 3, [3, 2], "out")

# pw6 alpha
artist.add_arrow([5, 0], 0, [0, 2], "ket", "2'")
artist.add_arrow([5, 0], 1, [2, 0], "ket", "-2")
artist.add_arrow([5, 0], 2, [0, 1], "ket", "1")
artist.add_arrow([5, 0], 3, [1, 0], "out")

# pw6 beta
artist.add_arrow([5, 1], 0, [0, 2], "ket", "2'")
artist.add_arrow([5, 1], 1, [0, 2], "bra", "-2")
artist.add_arrow([5, 1], 2, [2, 3], "ket", "1")
artist.add_arrow([5, 1], 3, [3, 2], "out")

artist.plot()
plt.show()
