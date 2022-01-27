"""Generic open method for wt5 files."""


# --- import -------------------------------------------------------------------------------------


import os
import tempfile
import weakref

import h5py
import numpy as np

from . import collection as wt_collection
from . import data as wt_data
from . import _group as wt_group


# --- define -------------------------------------------------------------------------------------


__all__ = ["open"]


# --- functions ----------------------------------------------------------------------------------

_open = open


def open(filepath, edit_local=False):
    """Open any wt5 file, returning the top-level object (data or collection).

    Parameters
    ----------
    filepath : path-like
        Path to file.
        Can be either a local or remote file (http/ftp).
        Can be compressed with gz/bz2, decompression based on file name.
    edit_local : boolean (optional)
        If True, the file itself will be opened for editing. Otherwise, a
        copy will be created. Default is False.

    Returns
    -------
    WrightTools Collection or Data
        Root-level object in file.
    """
    filepath = os.fspath(filepath)
    ds = np.DataSource(None)
    if edit_local is False:
        tf = tempfile.mkstemp(prefix="", suffix=".wt5")
        with _open(tf[1], "w+b") as tff:
            with ds.open(str(filepath), "rb") as f:
                tff.write(f.read())
        filepath = tf[1]
    f = h5py.File(filepath, "r")
    class_name = f["/"].attrs["class"]
    name = f["/"].attrs["name"]
    f.close()
    if class_name == "Data":
        obj = wt_data.Data(filepath=str(filepath), name=name, edit_local=True)
    elif class_name == "Collection":
        obj = wt_collection.Collection(filepath=str(filepath), name=name, edit_local=True)
    else:
        obj = wt_group.Group(filepath=str(filepath), name=name, edit_local=True)

    if edit_local is False:
        setattr(obj, "_tmpfile", tf)
        weakref.finalize(obj, obj.close)
    return obj
