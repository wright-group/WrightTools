"""Cary."""


# --- import --------------------------------------------------------------------------------------


import fnmatch
import queue
import os
import posixpath

from ._collection import Collection


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_directory"]


# --- from function -------------------------------------------------------------------------------


def from_directory(filepath, from_methods, *, name=None, parent=None, verbose=True):
    """Create a WrightTools Collection from a directory of source files.

    Parameters
    ----------
    filepath: str
        Path to the directory on the file system
    from_methods: dict<str, callable>
        Dictionary which maps patterns (using Unix-like glob wildcard patterns)
        to functions which take a filepath, plus the keyword arguments
        ['name', 'parent', and 'verbose'].
        (e.g. most from_<kind> methods within WrightTools)
        The value can be `None` which results in that item being ignored.
        The *first* matching pattern encountered will be used.
        Therefore, if multiple patterns will match the same file, use and `OrderedDict`.
        Patterns are matched on the file name level, not using the full path.

    Keyword Arguments
    -----------------
    name: str
        Name to use for the root data object. Default is the directory name.
    parent: Collection
        Parent collection to insert the directory structure into. Default is a new
        collection in temp file.
    verbose: bool
        Print information as objects are created. Passed to the functions.

    Examples
    --------
    >>> from_dict = {'*.data':wt.data.from_PyCMDS,
    ...              '*.csv':wt.collections.from_Cary,
    ...              'unused':None,
    ...             }
    >>> col = wt.collection.from_directory('path/to/folder', from_dict)
    """
    if name is None:
        name = os.path.basename(os.path.abspath(filepath))

    if verbose:
        print("Creating Collection:", name)

    root = Collection(name=name, parent=parent)

    q = queue.Queue()

    for i in os.listdir(filepath):
        q.put((filepath, i, root))

    while not q.empty():
        path, fname, parent = q.get()
        for pattern, func in from_methods.items():
            if fnmatch.fnmatch(fname, pattern):
                if func is not None:
                    func(
                        os.path.join(path, fname),
                        name=os.path.splitext(fname)[0],
                        parent=parent,
                        verbose=verbose,
                    )
                break
        else:
            if os.path.isdir(os.path.join(path, fname)):
                if verbose:
                    print("Creating Collection at", posixpath.join(parent.name, fname))
                col = parent.create_collection(name=fname)
                for i in os.listdir(os.path.join(path, fname)):
                    q.put((os.path.join(path, fname), i, col))
    return root
