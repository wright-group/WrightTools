"""WrightTools base classes and associated."""


# --- import --------------------------------------------------------------------------------------


import shutil
import weakref
import tempfile

import h5py


# --- define --------------------------------------------------------------------------------------


wt5_version = '0.0.0'


# --- dataset -------------------------------------------------------------------------------------


class Dataset(h5py.Dataset):
    instances = {}


# --- group ---------------------------------------------------------------------------------------


class Group(h5py.Group):
    instances = {}
    default_name = 'group'

    def __init__(self, *args, **kwargs):
        h5py.Group.__init__(self, *args, **kwargs)
        # the following are populated if not defined
        self.__version__
        self.natural_name

    def __new__(cls, *args, **kwargs):
        # extract
        filepath = args[0] if len(args) > 0 else kwargs.get('filepath', None)
        parent = args[1] if len(args) > 1 else kwargs.get('parent', None)
        name = args[2] if len(args) > 2 else kwargs.get('name', cls.default_name)
        edit_local = args[3] if len(args) > 3 else kwargs.get('edit_local', False)
        # tempfile
        tmpfile = None
        if edit_local and filepath is None:
            raise Exception  # TODO: better exception
        if not edit_local:
            tmpfile = tempfile.NamedTemporaryFile(prefix='', suffix='.wt5')
            p = tmpfile.name
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
                weakref.finalize(instance, tmpfile.close())
        return cls.instances[fullpath]

    @property
    def __version__(self):
        if '__version__' not in self.file.attrs.keys():
            self.file.attrs['__version__'] = wt5_version
        return self.file.attrs['__version__']

    @property
    def fullpath(self):
        return self.filepath + '::' + self.name

    @property
    def natural_name(self):
        if not 'name' in self.attrs.keys():
            self.attrs['name'] = self.__class__.default_name
        return self.attrs['name']

    @property
    def parent(self):
        from .collection import Collection
        group = super().parent
        parent = group.parent.name
        if parent == posixpath.sep:
            parent = None
        return Collection(self.filepath, parent=parent, name=group.attrs['name'])
