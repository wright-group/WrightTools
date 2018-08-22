"""Generic open method for wt5 files."""


# --- import -------------------------------------------------------------------------------------


import posixpath

import h5py

from . import collection as wt_collection
from . import data as wt_data
from . import _group as wt_group


# --- define -------------------------------------------------------------------------------------


__all__ = ["open"]


# --- functions ----------------------------------------------------------------------------------


def open(filepath, edit_local=False):
    """Open any wt5 file, returning the top-level object (data or collection).

    Parameters
    ----------
    filepath : string
        Path to file.
    edit_local : boolean (optional)
        If True, the file itself will be opened for editing. Otherwise, a
        copy will be created. Default is False.

    Returns
    -------
    WrightTools Collection or Data
        Root-level object in file.
    """
    f = h5py.File(filepath)
    class_name = f[posixpath.sep].attrs["class"]
    name = f[posixpath.sep].attrs["name"]
    if class_name == "Data":
        return wt_data.Data(filepath=filepath, name=name, edit_local=edit_local)
    elif class_name == "Collection":
        return wt_collection.Collection(filepath=filepath, name=name, edit_local=edit_local)
    else:
        return wt_group.Group(filepath=filepath, name=name, edit_local=edit_local)
