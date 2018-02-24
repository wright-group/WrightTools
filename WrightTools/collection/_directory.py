"""Cary."""


# --- import --------------------------------------------------------------------------------------


import fnmatch
import queue
import os
import posixpath

import numpy as np

from .. import exceptions as wt_exceptions
from ._collection import Collection


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_directory']


# --- from function -------------------------------------------------------------------------------


def from_directory(filepath, from_methods, *, name=None,  parent=None, verbose=True):
    """Create a WrightTools Collection from a directory of source files."""
    if name is None:
        name = os.path.basename(os.path.abspath(filepath))

    if verbose:
        print('Creating Collection at root:', name)

    root = Collection(name=name, parent=parent)

    q = queue.Queue()

    for i in os.listdir(filepath):
        q.put((filepath, i, root))

    while not q.empty():
        path, fname, parent = q.get()
        for pattern, func in from_methods.items():
            if fnmatch.fnmatch(fname, pattern):
                if func is not None:
                    func(os.path.join(path, fname), name=os.path.splitext(fname)[0], parent=parent,
                         verbose=verbose)
                break
        else:
            if os.path.isdir(os.path.join(path, fname)):
                if verbose:
                    print('Creating Collection at', posixpath.join(parent.name, fname))
                col = parent.create_collection(name=fname)
                for i in os.listdir(os.path.join(path, fname)):
                    q.put((os.path.join(path, fname), i, col))
    return root


