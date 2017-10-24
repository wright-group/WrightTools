"""Collection."""


# --- import --------------------------------------------------------------------------------------


import os
import shutil
import tempfile
import weakref

import numpy as np

import h5py

from .. import data as wt_data
from .. import kit as wt_kit
from .. import macros as wt_macros


# --- define --------------------------------------------------------------------------------------


__all__ = ['Collection']


# --- classes -------------------------------------------------------------------------------------


@wt_macros.group_singleton
class Collection(h5py.Group):
    """Nestable Collection of Data objects."""
    instances = {}

    def __init__(self, filepath=None, parent=None, name=None, edit_local=False, **kwargs):
        """Create a ``Collection`` object.

        Parameters
        ----------
        channels : list
            A list of Channel objects. Channels are also inherited as
            attributes using the channel name: ``data.ai0``, for example.
        axes : list
            A list of Axis objects. Axes are also inherited as attributes using
            the axis name: ``data.w1``, for example.
        constants : list
            A list of Axis objects, each with exactly one point.
        **kwargs
            Additional keyword arguments are added to the attrs dictionary
            and to the natural namespace of the object (if possible).
        """
        # TODO: redo docstring
        # parse / create file
        self.__tmpfile = None
        if edit_local and filepath is None:
            raise Exception  # TODO: better exception
        if not edit_local:
            self.__tmpfile = tempfile.NamedTemporaryFile(prefix='', suffix='.wt5')
            self.filepath = self.__tmpfile.name
            if filepath:
                shutil.copyfile(src=filepath, dst=self.filepath)
        elif edit_local and filepath:
            self.filepath = filepath
        # parse / create group
        if parent is None:
            p = '/'
        else:
            p = parent + '/' + name
        file = h5py.File(self.filepath, 'a')
        if '__version__' not in file.attrs.keys():
            file.attrs['__version__'] = '0.0.0'
        file.require_group(p)
        h5py.Group.__init__(self, file[p].id)
        # assign
        self._n = 0
        self.source = kwargs.pop('source', None)  # TODO
        if name is None:
            name = self.attrs.get('name', 'collection')
        self.attrs.update(kwargs)
        self.attrs['class'] = 'Collection'
        self.attrs['name'] = name
        # load from file
        self._items = []
        for name in self.item_names:
            self._items.append(self[name])
        self.__version__  # assigns, if it doesn't already exist

        if self.__tmpfile is not None:
            weakref.finalize(self, self.__tmpfile.close)


    def __iter__(self):
        self._n = 0
        return self

    def __len__(self):
        return len(self.item_names)

    def __next__(self):
        if self._n < len(self):
            out = self[self._n]
            self._n += 1
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
        else:
            return out

    def __setitem__(self, key, value):
        raise NotImplementedError

    @property
    def __version__(self):
        return self.file.attrs['__version__']

    @property
    def natural_name(self):
        return self.attrs['name']

    @property
    def item_names(self):
        if 'item_names' not in self.attrs.keys():
            self.attrs['item_names'] = np.array([], dtype='S')
        return [s.decode() for s in self.attrs['item_names']]

    @property
    def fullpath(self):
        return self.filepath + '::' + self.name

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

    def index():
        raise NotImplementedError

    def flush():
        self.file.flush()

    def save(self, filepath=None, verbose=True):
        # TODO: documentation
        self.file.flush()  # ensure all changes are written to file
        if filepath is None:
            filepath = os.path.join(os.getcwd(), self.natural_name + '.wt5')
        elif len(os.path.basename(filepath).split('.')) == 1:
            filepath += '.wt5'
        filepath = os.path.expanduser(filepath)
        shutil.copyfile(src=self.filepath, dst=filepath)
        if verbose:
            print(filepath)
        return filepath

    def close(self):
        print("Closing: ", self.filepath)
        try:
            self.file.close()
            self.__tmpfile.close()
        except RuntimeError:
            pass
