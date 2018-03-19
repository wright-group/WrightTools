import os
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))

data = wt.data.from_PyCMDS('TA ultra high res.data', name='perovskite_TA')
data.convert('eV', convert_variables=True)
data.signal_diff.signed = True

dOD = -np.log10((data.signal_mean[:]+data.signal_diff[:])/data.signal_mean[:])
data.create_channel('dOD', values=dOD, signed=True)
data.bring_to_front('dOD')
clip = 0.05
data.dOD.clip(min=-clip, max=clip)

data.transform('w1=wm', 'w2', 'd2')

wt.artists.quick2D(data, at={'d2': [0, 'fs']})
plt.show()
