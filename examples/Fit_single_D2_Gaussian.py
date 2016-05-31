# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:54:55 2016

@author: Nathan
"""

import WrightTools as wt
import numpy as np

f = wt.fit.TwoD_Gaussian()
# Create x and y indices
x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
xi, yi = np.meshgrid(x, y)

#create data
data = f.evaluate([3, 95, 115, 20, 35, .3, 10],xi, yi)

# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data.reshape(201, 201))
plt.colorbar()

# add some noise to the data and try to fit the data generated beforehand
data_noisy = data + 0.2*np.random.normal(size=data.shape)

f.fit(data_noisy,x,y)

# And plot the results:

data_fitted = f.evaluate(f.out[0],xi, yi)

fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
plt.show()