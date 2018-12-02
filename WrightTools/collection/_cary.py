"""Cary."""


# --- import --------------------------------------------------------------------------------------


import os
import pathlib
import re

import numpy as np

from .. import exceptions as wt_exceptions
from ._collection import Collection


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_Cary"]


# --- from function -------------------------------------------------------------------------------


def from_Cary(filepath, name=None, parent=None, verbose=True):
    """Create a collection object from a Cary UV VIS absorbance file.

    We hope to support as many Cary instruments and datasets as possible.
    This function has been tested with data collected on a Cary50 UV/VIS spectrometer.
    If any alternate instruments are found not to work as expected, please
    submit a bug report on our `issue tracker`__.

    __ github.com/wright-group/WrightTools/issues

    .. plot::

        >>> import WrightTools as wt
        >>> from WrightTools import datasets
        >>> p = datasets.Cary.CuPCtS_H2O_vis
        >>> data = wt.collection.from_Cary(p)[0]
        >>> wt.artists.quick1D(data)

    Parameters
    ----------
    filepath : path-like
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
    filestr = os.fspath(filepath)
    filepath = pathlib.Path(filepath)

    if ".csv" not in filepath.suffixes:
        wt_exceptions.WrongFileTypeWarning.warn(filepath, "csv")
    if name is None:
        name = "cary"
    # import array
    lines = []
    ds = np.DataSource(None)
    with ds.open(filestr, "rt", encoding="iso-8859-1") as f:
        header = f.readline()
        columns = f.readline()
        while True:
            line = f.readline()
            if line == "\n" or line == "" or line == "\r\n":
                break
            else:
                # Note, it is necessary to call this twice, as a single call will
                # result in something like ',,,,' > ',nan,,nan,'.
                line = line.replace(",,", ",nan,")
                line = line.replace(",,", ",nan,")
                # Ensure that the first column has nan, if necessary
                if line[0] == ",":
                    line = "nan" + line
                clean = line[:-2]  # lines end with ',/n'
                lines.append(np.fromstring(clean, sep=","))
    lines = [line for line in lines if len(line) > 0]
    header = header.split(",")
    columns = columns.split(",")
    arr = np.array(lines).T
    duplicate = len(header) // 2 == len(set(header) - {""})
    # chew through all scans
    datas = Collection(name=name, parent=parent, edit_local=parent is not None)
    units_dict = {"°c": "deg_C", "°f": "deg_F"}
    for i in range(0, len(header) - 1, 2):
        r = re.compile(r"[ \t\(\)]+")
        spl = r.split(columns[i])
        ax = spl[0].lower() if len(spl) > 0 else None
        units = spl[1].lower() if len(spl) > 1 else None
        units = units_dict.get(units, units)
        if duplicate:
            name = "{}_{:03d}".format(header[i], i // 2)
        else:
            name = header[i]
        dat = datas.create_data(name, kind="Cary", source=filestr)
        dat.create_variable(ax, arr[i][~np.isnan(arr[i])], units=units)
        dat.create_channel(
            columns[i + 1].lower(), arr[i + 1][~np.isnan(arr[i + 1])], label=columns[i + 1].lower()
        )
        dat.transform(ax)
    # finish
    if verbose:
        print("{0} data objects successfully created from Cary file:".format(len(datas)))
        for i, data in enumerate(datas):
            print("  {0}: {1}".format(i, data))
    return datas
