"""Dataset base class."""


# --- import --------------------------------------------------------------------------------------


import posixpath

import h5py


# --- class ---------------------------------------------------------------------------------------


class Dataset(h5py.Dataset):
    """Array-like data container."""

    instances = {}
    class_name = 'Dataset'

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        lis = [min(s - 1, i) if not isinstance(i, slice) else i for s, i in zip(self.shape, index)]
        return super().__getitem__(tuple(lis))

    def __new__(cls, parent, id, **kwargs):
        """New object formation handler."""
        fullpath = parent.fullpath + h5py.h5i.get_name(id).decode()
        if fullpath in cls.instances.keys():
            return cls.instances[fullpath]
        else:
            instance = super(Dataset, cls).__new__(cls)
            cls.__init__(instance, parent, id, **kwargs)
            cls.instances[fullpath] = instance
            return instance

    def __repr__(self):
        return '<WrightTools.{0} \'{1}\' at {2}>'.format(self.class_name, self.natural_name,
                                                         self.fullpath)

    @property
    def fullpath(self):
        """Full path: file and internal structure."""
        return self.parent.fullpath + posixpath.sep + self.natural_name

    @property
    def natural_name(self):
        """Natural name of the dataset. May be different from name."""
        try:
            assert self._natural_name is not None
        except (AssertionError, AttributeError):
            self._natural_name = self.attrs['name']
        finally:
            return self._natural_name

    @natural_name.setter
    def natural_name(self, value):
        self.attrs['name'] = value
        self._natural_name = None

    @property
    def parent(self):
        """Parent."""
        return self._parent

    @property
    def units(self):
        """Units."""
        if 'units' in self.attrs.keys():
            return self.attrs['units'].decode()

    @units.setter
    def units(self, value):
        """Set units."""
        if value is None:
            if 'units' in self.attrs.keys():
                self.attrs.pop('units')
        else:
            self.attrs['units'] = value.encode()
