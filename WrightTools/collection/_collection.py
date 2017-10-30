"""Collection."""


# --- import --------------------------------------------------------------------------------------


import os
import shutil
import posixpath

import numpy as np

import h5py

from .. import data as wt_data
from .. import kit as wt_kit
from .._base import Group


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
            out = self[self._n]
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
        out = h5py.Group.__getitem__(self, key)
        if 'class' in out.attrs.keys():
            if out.attrs['class'] == 'Data':
                return wt_data.Data(filepath=self.filepath, parent=self.name, name=key,
                                    edit_local=True)
            elif out.attrs['class'] == 'Collection':
                return Collection(filepath=self.filepath, parent=self.name, name=key,
                                  edit_local=True)
        else:
            return out

    def __setitem__(self, key, value):
        raise NotImplementedError

    @property
    def item_names(self):
        if 'item_names' not in self.attrs.keys():
            self.attrs['item_names'] = np.array([], dtype='S')
        return [s.decode() for s in self.attrs['item_names']]

    def create_collection(self, name='collection', position=None, **kwargs):
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
        data = wt_data.Data(filepath=self.filepath, parent=self.name, name=name, edit_local=True,
                            **kwargs)
        if position is None:
            self._items.append(data)
            self.attrs['item_names'] = np.append(self.attrs['item_names'],
                                                 data.natural_name.encode())
        else:
            self._items.insert(position, data)
            self.attrs['item_names'] = np.insert(self.attrs['item_names'], position,
                                                 data.natural_name.encode())
        setattr(self, name, data)
        return data

    def index(self):
        raise NotImplementedError

    def flush(self):
        for item in self._items:
            item.flush()
        self.file.flush()

    def save(self, filepath=None, verbose=True):
        # TODO: documentation
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

    def close(self):
        print("Closing: ", self.filepath)
        try:
            self.file.close()
            self.__tmpfile.close()
        except RuntimeError:
            pass
