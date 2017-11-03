"""Generic open method for wt5 files."""


# --- import -------------------------------------------------------------------------------------

import h5py

from . import collection as wt_collection
from . import data as wt_data


# --- define -------------------------------------------------------------------------------------


__all__ = ['open']


# --- functions ----------------------------------------------------------------------------------


def open(filepath, edit_local=False):
    """Open any wt5 file, returning the top-level object (data or collection).
    """
    f = h5py.File(filepath)
    class_name = f['/'].attrs['class']
    if class_name == 'Data':
        return wt_data.Data(filepath=filepath, edit_local=edit_local)
    elif class_name == 'Collection':
        return wt_collection.Collection(filepath=filepath, edit_local=edit_local)
    else:
        return wt_collection.Group(filepath=filepath, edit_local=edit_local)
