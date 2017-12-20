"""Collection."""


# --- import --------------------------------------------------------------------------------------


import os
import shutil

import numpy as np

from .. import data as wt_data
from .._group import Group


# --- define --------------------------------------------------------------------------------------


__all__ = ['Collection']


# --- classes -------------------------------------------------------------------------------------


class Collection(Group):
    """Nestable Collection of Data objects."""

    class_name = 'Collection'

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
        return '<WrightTools.Collection \'{0}\' {1} at {2}>'.format(self.natural_name,
                                                                    self.item_names,
                                                                    '::'.join([self.filepath,
                                                                               self.name]))

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.item_names[key]
        if key == "":
            return None
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        raise NotImplementedError

    @property
    def item_names(self):
        """Item names."""
        if 'item_names' not in self.attrs.keys():
            self.attrs['item_names'] = np.array([], dtype='S')
        return [s.decode() for s in self.attrs['item_names']]

    def create_collection(self, name='collection', position=None, **kwargs):
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
        collection = Collection(filepath=self.filepath, parent=self.name, name=name,
                                edit_local=True, **kwargs)
        if position is None:
            self._items.append(collection)
            self.attrs['item_names'] = np.append(self.attrs['item_names'],
                                                 collection.natural_name.encode())
        else:
            self._items.insert(position, collection)
            self.attrs['item_names'] = np.insert(self.attrs['item_names'], position,
                                                 collection.natural_name.encode())
        setattr(self, name, collection)
        return collection

    def create_data(self, name='data', position=None, **kwargs):
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
        if name == '':
            data = None
            natural_name = "".encode()
        else:
            data = wt_data.Data(filepath=self.filepath, parent=self.name, name=name,
                                edit_local=True, **kwargs)
            natural_name = data.natural_name.encode()
        if position is None:
            self._items.append(data)
            self.attrs['item_names'] = np.append(self.attrs['item_names'], natural_name)
        else:
            self._items.insert(position, data)
            self.attrs['item_names'] = np.insert(self.attrs['item_names'], position, natural_name)
        setattr(self, name, data)
        return data

    def index(self):
        """Index."""
        raise NotImplementedError

    def flush(self):
        """Ensure contents are written to file."""
        for item in self._items:
            item.flush()
        self.file.flush()

    def save(self, filepath=None, verbose=True):
        """Save collection as root of a new file.

        Parameters
        ----------
        filepath : string (optional)
            Filepath to write. If None, file is created using natural_name.
        verbose : boolean (optional)
            Toggle talkback. Default is True

        Returns
        -------
        str
            Written filepath.
        """
        self.flush()  # ensure all changes are written to file
        if filepath is None:
            filepath = os.path.join(os.getcwd(), self.natural_name + '.wt5')
        elif len(os.path.basename(filepath).split('.')) == 1:
            filepath += '.wt5'
        filepath = os.path.expanduser(filepath)
        shutil.copyfile(src=self.filepath, dst=filepath)
        if verbose:
            print('file saved at', filepath)
        return filepath
