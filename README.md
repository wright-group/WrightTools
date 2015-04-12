# WrightTools

Dependencies 
- matplotlib
- numpy
- scipy
- os
- sys
- collections

<hr>

Simple example for plotting a 2D dat file:
```python
import WrightTools as wt
Artist = wt.artists.mpl_2D

data = wt.data.from_dat(filepath, 'w1', 'w2')

artist = Artist()
artist.plot(data, xaxis = 0, yaxis = 1, channel = 0)
```
