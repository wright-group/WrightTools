import numpy as np

import matplotlib.pyplot as plt

import WrightTools as wt

def is_odd(num):
    return num & 0x1

#-------------------------------------------------------------------------------

xi = np.array(range(5))
yi = np.array(range(5))

#zi
zi = np.zeros((5, 5))

for index in np.ndindex(zi.shape):
    if is_odd(index[0]):
        if is_odd(index[1]):
            zi[index] = 1
        else:
            zi[index] = 0
    else:
        if not is_odd(index[1]):
            zi[index] = 1
        else:
            zi[index] = 0
            
    print index, zi[index]
            
x_axis = wt.data.Axis(xi, None, name = 'x')
y_axis = wt.data.Axis(yi, None, name = 'y')
channel = wt.data.Channel(zi, 'au')

data = wt.data.Data([y_axis, x_axis], [channel])

artist = wt.artists.mpl_2D(data)
artist.plot(0, contours = 0)
artist.plot(0, pixelated = True, contours = 0)