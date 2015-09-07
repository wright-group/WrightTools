'''
this is something brute force I made to have some diagrams of 2D delay space \n
perhaps something better should be made eventually \n
- Blaise 2015.09.07
'''

import numpy as np
import matplotlib.pyplot as plt

plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([-1, 1], [-1, 1], color = 'k', linewidth = 2)
ax.axhline(0, color = 'k', linewidth = 2)
ax.axvline(0, color = 'k', linewidth = 2)

ax.text(-0.5, 0.5, 'I', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(0.25, 0.6, 'II', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(-0.6, -0.25, 'III', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(0.6, 0.25, 'IV', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(-0.25, -0.6, 'V', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(0.5, -0.5, 'VI', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')

ax.set_xlabel(r'd1 $\mathrm{(\tau_{22^{\prime}})}$', fontsize = 15)
ax.set_ylabel(r'd2 $\mathrm{(\tau_{21})}$', fontsize = 15)
ax.set_title('ultimate representation')

ax.tick_params(axis='both',          
               which='both',      
               bottom='off',      
               top='off',
               left='off',
               right='off',
               labelleft='off',
               labelbottom='off') 

ax.set_aspect(1.)

plt.savefig('TRIEE_delay_space.png', transparent = True)

plt.close()


# as collected ----------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([-1, 1], [1, -1], color = 'k', linewidth = 2)
ax.axhline(0, color = 'k', linewidth = 2)
ax.axvline(0, color = 'k', linewidth = 2)

ax.text(0.5, 0.5, 'I', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(-0.25, 0.6, 'II', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(0.6, -0.25, 'III', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(-0.6, 0.25, 'IV', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(0.25, -0.6, 'V', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')
ax.text(-0.5, -0.5, 'VI', fontsize = 30, verticalalignment = 'center', horizontalalignment = 'center')

ax.set_xlabel(r'd1 $\mathrm{(\tau_{2^{\prime}2})}$', fontsize = 15)
ax.set_ylabel(r'd2 $\mathrm{(\tau_{21})}$', fontsize = 15)
ax.set_title('as collected')

ax.tick_params(axis='both',          
               which='both',      
               bottom='off',      
               top='off',
               left='off',
               right='off',
               labelleft='off',
               labelbottom='off') 

ax.set_aspect(1.)

plt.savefig('TRIEE_delay_space_as_collected.png', transparent = True)
