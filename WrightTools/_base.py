"""WrightTools base classes and associated."""


# --- import --------------------------------------------------------------------------------------


import shutil
import os
import weakref
import tempfile
import posixpath

import numpy as np

import h5py


# --- define --------------------------------------------------------------------------------------


wt5_version = '0.0.0'


# --- classes -------------------------------------------------------------------------------------


class Dataset(h5py.Dataset):
    instances = {}
    class_name = 'Dataset'

    def __getitem__(self, i):
        if hasattr(i, '__iter__'):
            indices = list(i)
        else:
            indices = [i]
        while len(indices) <= self.dimensionality:
            indices.append(slice(None))
        indices = [indices[i] for i in self.transposition]
        transposition = self.transposition[[not isinstance(idx, int) for idx in indices]]
        transposition -= min(transposition)
        # TODO: handle transposition properly
        return super().__getitem__(tuple(indices))#.transpose(self.transposition)

    def __repr__(self):
        return '<WrightTools.{0} \'{1}\' at {2}>'.format(self.class_name, self.natural_name,
                                                         self.fullpath)

    @property
    def fullpath(self):
        return self.parent.fullpath + posixpath.sep + self.natural_name

    @property
    def natural_name(self):
        return self.attrs['name']

    @property
    def parent(self):
        return self._parent

    @property
    def transposition(self):
        if 'transposition' not in self.attrs.keys():
            self.attrs['transposition'] = np.arange(len(self.shape))
        return self.attrs['transposition']

    @property
    def units(self):
        return self.attrs.get('units', None)

    @units.setter
    def units(self, value):
        if value is None:
            if 'units' in self.attrs.keys():
                self.attrs.pop('units')
        else:
            self.attrs['units'] = value.encode()

    def transpose(self, axes):
        if axes is None:
            new = self.transposition[::-1]
        else:
            new = self.transposition
            for i, axis in enumerate(axes):
                new[i] = self.transposition[axis]
        self.attrs['transposition'] = new


class Group(h5py.Group):
    instances = {}
    class_name = 'Group'

    def __init__(self, filepath=None, parent=None, name=None, **kwargs):
        if filepath is None:
            return
        if parent == '':
            parent = posixpath.sep
        # file
        self.filepath = filepath
        path = parent + posixpath.sep + name
        file = h5py.File(self.filepath, 'a')
        file.require_group(parent)
        file.require_group(path)
        h5py.Group.__init__(self, bind=file[path].id)
        self.__n = 0
        self.fid = self.file.fid
        if name is not None:
            self.attrs['name'] = name
        self.attrs.update(kwargs)
        self.attrs['class'] = self.class_name
        # load from file
        self._items = []
        for name in self.item_names:
            self._items.append(self[name])
            setattr(self, name, self[name])
        # kwargs
        self.attrs.update(kwargs)
        # the following are populated if not already recorded
        self.__version__
        self.natural_name

    def __getitem__(self, key):
        from .collection import Collection
        from .data._data import Channel, Data, Variable
        out = h5py.Group.__getitem__(self, key)
        if 'class' in out.attrs.keys():
            if out.attrs['class'] == 'Channel':
                return Channel(parent=self, id=out.id)
            elif out.attrs['class'] == 'Collection':
                return Collection(filepath=self.filepath, parent=self.name, name=key,
                                  edit_local=True)
            elif out.attrs['class'] == 'Data':
                return Data(filepath=self.filepath, parent=self.name, name=key,
                            edit_local=True)
            if out.attrs['class'] == 'Variable':
                return Variable(parent=self, id=out.id)
            else:
                return Group(filepath=self.filepath, parent=self.name, name=key,
                             edit_local=True)
        else:
            return out

    def __new__(cls, *args, **kwargs):
        # extract
        filepath = args[0] if len(args) > 0 else kwargs.get('filepath', None)
        parent = args[1] if len(args) > 1 else kwargs.get('parent', None)
        name = args[2] if len(args) > 2 else kwargs.get('name', cls.class_name.lower())
        edit_local = args[3] if len(args) > 3 else kwargs.get('edit_local', False)
        if isinstance(parent, h5py.Group):
            filepath = parent.filepath
            parent = parent.name
        # tempfile
        tmpfile = None
        if edit_local and filepath is None:
            raise Exception  # TODO: better exception
        if not edit_local:
            tmpfile = tempfile.mkstemp(prefix='', suffix='.wt5')
            p = tmpfile[1]
            if filepath:
                shutil.copyfile(src=filepath, dst=p)
        elif edit_local and filepath:
            p = filepath
        # construct fullpath
        if parent is None:
            parent = ''
            name = '/'
        fullpath = p + '::' + parent + name
        # create and/or return
        if fullpath not in cls.instances.keys():
            kwargs['filepath'] = p
            kwargs['parent'] = parent
            kwargs['name'] = name
            instance = super(Group, cls).__new__(cls)
            cls.__init__(instance, **kwargs)
            cls.instances[fullpath] = instance
            if tmpfile:
                setattr(instance, '_tmpfile', tmpfile)
                weakref.finalize(instance, instance.close)
            return instance
        instance = cls.instances[fullpath]
        return instance

    @property
    def __version__(self):
        if '__version__' not in self.file.attrs.keys():
            self.file.attrs['__version__'] = wt5_version
        return self.file.attrs['__version__']

    @property
    def fullpath(self):
        return self.filepath + '::' + self.name

    @property
    def item_names(self):
        if 'item_names' not in self.attrs.keys():
            self.attrs['item_names'] = np.array([], dtype='S')
        return self.attrs['item_names']

    @property
    def natural_name(self):
        if 'name' not in self.attrs.keys():
            self.attrs['name'] = self.__class__.default_name
        return self.attrs['name']

    @natural_name.setter
    def natural_name(self, value):
        self.attrs['name'] = value

    @property
    def parent(self):
        from .collection import Collection
        group = super().parent
        parent = group.parent.name
        if parent == posixpath.sep:
            parent = None
        return Collection(self.filepath, parent=parent, name=group.attrs['name'])

    def close(self):
        if(self.fid.valid > 0):
            self.__class__.instances.pop(self.fullpath)
            self.file.flush()
            self.file.close()
            if hasattr(self, '_tmpfile'):
                os.close(self._tmpfile[0])
                os.remove(self._tmpfile[1])

    def flush(self):
        self.file.flush()
