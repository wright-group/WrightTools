"""Collection."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

from .. import data as wt_data
from .. import exceptions as wt_exceptions
from .._group import Group


# --- define --------------------------------------------------------------------------------------


__all__ = ["Collection"]


# --- classes -------------------------------------------------------------------------------------


class Collection(Group):
    """Nestable Collection of Data objects."""

    class_name = "Collection"

    def __iter__(self):
        self.__n = 0
        return self

    def __len__(self):
        return len(self.item_names)

    def __next__(self):
        if self.__n < len(self):
            out = self.item_names[self.__n]
            self.__n += 1
        else:
            raise StopIteration
        return out

    def __repr__(self):
        return "<WrightTools.Collection '{0}' {1} at {2}>".format(
            self.natural_name, self.item_names, "::".join([self.filepath, self.name])
        )

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.item_names[key]
        if key == "":
            return None
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        raise NotImplementedError

    @property
    def _leaf(self):
        return self.natural_name

    def _print_branch(self, prefix, depth, verbose):
        for i, name in enumerate(self.item_names):
            item = self[name]
            if i + 1 == len(self.item_names):
                s = prefix + "└── {0}: {1}".format(i, item._leaf)
                p = prefix + "    "
            else:
                s = prefix + "├── {0}: {1}".format(i, item._leaf)
                p = prefix + "│   "
            print(s)
            if depth > 1 and hasattr(item, "_print_branch"):
                item._print_branch(p, depth=depth - 1, verbose=verbose)

    def create_collection(self, name="collection", position=None, **kwargs):
        """Create a new child colleciton.

        Parameters
        ----------
        name : string
            Unique identifier.
        position : integer (optional)
            Location to insert. Default is None (append).
        kwargs
            Additional arguments to child collection instantiation.

        Returns
        -------
        WrightTools Collection
            New child.
        """
        if name in self.item_names:
            wt_exceptions.ObjectExistsWarning.warn(name)
            return self[name]
        collection = Collection(
            filepath=self.filepath, parent=self.name, name=name, edit_local=True, **kwargs
        )
        if position is not None:
            self.attrs["item_names"] = np.insert(
                self.attrs["item_names"][:-1], position, collection.natural_name.encode()
            )
        setattr(self, name, collection)
        return collection

    def create_data(self, name="data", position=None, **kwargs):
        """Create a new child data.

        Parameters
        ----------
        name : string
            Unique identifier.
        position : integer (optional)
            Location to insert. Default is None (append).
        kwargs
            Additional arguments to child data instantiation.

        Returns
        -------
        WrightTools Data
            New child.
        """
        if name in self.item_names:
            wt_exceptions.ObjectExistsWarning.warn(name)
            return self[name]

        if name == "":
            data = None
            natural_name = "".encode()
        else:
            data = wt_data.Data(
                filepath=self.filepath, parent=self.name, name=name, edit_local=True, **kwargs
            )
            natural_name = data.natural_name.encode()
        if position is not None:
            self.attrs["item_names"] = np.insert(
                self.attrs["item_names"][:-1], position, natural_name
            )
        setattr(self, name, data)
        return data

    def index(self):
        """Index."""
        raise NotImplementedError

    def print_tree(self, depth=9, *, verbose=False):
        """Print a ascii-formatted tree representation of the collection contents.

        Parameters
        ----------
        depth : integer (optional)
            Number of layers to include in the tree. Default is 9.
        verbose : boolean (optional)
            Toggle inclusion of extra information. Default is True.
        """
        print("{0} ({1})".format(self.natural_name, self.filepath))
        self._print_branch("", depth=depth, verbose=verbose)

    def flush(self):
        """Ensure contents are written to file."""
        for name in self.item_names:
            item = self[name]
            item.flush()
        self.file.flush()
