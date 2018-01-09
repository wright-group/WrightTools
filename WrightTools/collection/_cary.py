"""Cary."""


# --- import --------------------------------------------------------------------------------------


import os
import re

import numpy as np

from .. import exceptions as wt_exceptions
from ._collection import Collection


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_Cary']


# --- from function -------------------------------------------------------------------------------


def from_Cary(filepath, parent=None, verbose=True):
    """Create a collection object from a Cary UV VIS absorbance file.
    
    We hope to support as many Cary instruments and datasets as possible.
    This function has been tested with data collected on a Cary50 UV/VIS spectrometer.
    If any alternate instruments are found not to work as expected, please
    submit a bug report on our `issue tracker`__.

    __issue tracker: github.com/wright-group/WrightTools/issues

    .. plot::

        >>> import WrightTools as wt
        >>> from WrightTools import datasets
        >>> p = datasets.Cary.CuPCtS_H2O_vis
        >>> data = wt.data.from_Cary(p)[0]
        >>> artist = wt.artists.Quick1D(data)
        >>> artist.plot()

    Parameters
    ----------
    filepath : string
        Path to Cary output file (.csv).
    parent : WrightTools.Collection
        A collection object in which to place a collection of Data objects.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object.
    """
    # check filepath
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'csv':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'csv')
    # import array
    lines = []
    with open(filepath, 'r') as f:
        header = f.readline()
        columns = f.readline()
        while True:
            line = f.readline()
            if line == '\n' or line == '':
                break
            else:
                clean = line[:-2]  # lines end with ',/n'
                lines.append(np.fromstring(clean, sep=','))
    header = header.split(',')
    columns = columns.split(',')
    #lines = [i for i in lines if len(i) > 0]
    arr = np.array(lines).T
    # chew through all scans
    datas = Collection(name='cary', parent=parent, edit_local=parent is not None)
    for i in range(0, len(header)-1, 2):
        print(columns[i], columns[i+1])
        r = re.compile("[ \t\(\)]+")
        spl = r.split(columns[i])
        ax = spl[0].lower() if len(spl) > 0 else None
        units = spl[1].lower() if len(spl) > 1 else None
        dat = datas.create_data(header[i], kind='Cary', source=filepath)
        dat.create_variable(ax, arr[i], units)
        dat.create_channel(columns[i+1].lower(), arr[i+1], label=columns[i+1].lower())
        dat.transform([ax])
    # finish
    if verbose:
        print('{0} data objects successfully created from Cary file:'.format(len(datas)))
        for i, data in enumerate(datas):
            print('  {0}: {1}'.format(i, data))
    return datas
