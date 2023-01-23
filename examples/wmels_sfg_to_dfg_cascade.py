# -*- coding: utf-8 -*-
"""
WMELs: SFG to DFG Cascade
=========================

Draw WMELs for SFG->DFG cascading process.
"""
import WrightTools.diagrams.WMEL as WMEL
import matplotlib.pyplot as plt


artist = WMEL.Artist(size=[1,1],
                     energies=[0.01,0.3,0.6,0.9])
artist.add_cascade([0, 0],
                   2,
                   number_of_interactions=3)
# First cascade
artist.add_cascade_arrow([0, 0], 0, 0, [0, 2], 'ket')
artist.add_cascade_arrow([0, 0], 0, 1, [2, 3], 'ket')
artist.add_cascade_arrow([0, 0], 0, 2, [3, 0], 'out')

# Second cascade
artist.add_cascade_arrow([0, 0], 1, 0, [0, 3], 'out')
artist.add_cascade_arrow([0, 0], 1, 1, [0, 1], 'bra')
artist.add_cascade_arrow([0, 0], 1, 2, [3, 1], 'out')

artist.plot()
plt.show()
