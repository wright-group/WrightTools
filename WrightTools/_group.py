"""Group base class."""


# --- import --------------------------------------------------------------------------------------


import shutil
import os
import pathlib
import weakref
import tempfile
import posixpath
import warnings

import numpy as np

import h5py

from ._dataset import Dataset
from . import kit as wt_kit
from . import exceptions as wt_exceptions
from . import __wt5_version__


# --- define --------------------------------------------------------------------------------------


wt5_version = __wt5_version__


# --- class ---------------------------------------------------------------------------------------


class MetaClass(type(h5py.Group)):
    def __call__(cls, *args, **kwargs):
        """Bypass normal construction."""
        return cls.__new__(cls, *args, **kwargs)


class Group(h5py.Group, metaclass=MetaClass):
    """Container of groups and datasets."""

    _instances = {}
    class_name = "Group"

    def __init__(self, file=None, parent=None, name=None, **kwargs):
        from .collection import Collection

        if file is None:
            return
        # parent
        if parent is None:
            parent = ""
        if parent == "":
            parent = pathlib.PurePosixPath("/")
            path = pathlib.PurePosixPath("/")
        else:
            path = pathlib.PurePosixPath(parent) / name
        name = path.name if path.name else name
        parent = path.parent
        self.filepath = file.filename
        if name != "" and parent.name != "":
            # Ensure that the parent Collection object is made first
            Collection(file, parent=parent.parent, name=parent.name, edit_local=True)
        file.require_group(str(path))
        h5py.Group.__init__(self, bind=file[str(path)].id)
        self.__n = 0
        self.fid = self.file.id
        self.natural_name = name
        # attrs
        if "class" not in self.attrs.keys():
            self.attrs["class"] = self.class_name
        if "created" not in self.attrs.keys():
            self.attrs["created"] = wt_kit.TimeStamp().RFC3339
        for key, value in kwargs.items():
            try:
                if isinstance(value, pathlib.Path):
                    value = str(value)
                elif (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], (str, os.PathLike))
                ):
                    value = np.array(value, dtype="S")
                else:
                    try:
                        value = os.fspath(value)
                    except TypeError:
                        pass  # Not all things that can be stored have fspath
                self.attrs[key] = value
            except TypeError:
                # some values have no native HDF5 equivalent
                message = "'{}' not included in attrs because its Type ({}) cannot be represented"
                message = message.format(key, type(value))
                warnings.warn(message)
        # the following are populated if not already recorded
        self.__version__
        self.item_names

        parent = file[str(parent)]
        if parent.name == self.name:
            pass  # at root, dont add to item_names
        elif self.natural_name.encode() not in parent.attrs["item_names"]:
            parent.attrs["item_names"] = np.append(
                parent.attrs["item_names"], self.natural_name.encode()
            )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __getattr__(self, key):
        """Gets called if attribute not in self.__dict__.

        See __getattribute__.
        """
        if key in self.keys():
            value = self[key]
            setattr(self, key, value)
            return self[key]
        else:
            message = "{0} has no attribute {1}".format(self.class_name, key)
            raise AttributeError(message)

    def __getitem__(self, key):
        from .collection import Collection
        from .data._data import Channel, Data, Variable

        out = super().__getitem__(key)
        if "class" in out.attrs.keys():
            if out.attrs["class"] == "Channel":
                return Channel(parent=self, id=out.id)
            elif out.attrs["class"] == "Collection":
                return Collection(
                    filepath=self.filepath, parent=self.name, name=key, edit_local=True
                )
            elif out.attrs["class"] == "Data":
                return Data(filepath=self.filepath, parent=self.name, name=key, edit_local=True)
            elif out.attrs["class"] == "Variable":
                return Variable(parent=self, id=out.id)
            else:
                return Group(filepath=self.filepath, parent=self.name, name=key, edit_local=True)
        else:
            return out

    def __new__(cls, *args, **kwargs):
        """New object formation handler."""
        # extract
        filepath = args[0] if len(args) > 0 else kwargs.pop("filepath", None)
        parent = args[1] if len(args) > 1 else kwargs.get("parent", None)
        natural_name = args[2] if len(args) > 2 else kwargs.get("name", cls.class_name.lower())
        edit_local = args[3] if len(args) > 3 else kwargs.pop("edit_local", False)
        file = None
        if isinstance(filepath, h5py.File):
            file = filepath
            filepath = file.filename
        tmpfile = None
        if isinstance(parent, h5py.Group):
            filepath = parent.filepath
            file = parent.file
            if hasattr(parent, "_tmpfile"):
                tmpfile = parent._tmpfile
            parent = parent.name
            edit_local = True
        if edit_local and filepath is None:
            raise Exception  # TODO: better exception
        if not edit_local:
            tmpfile = tempfile.mkstemp(prefix="", suffix=".wt5")
            p = tmpfile[1]
            if filepath:
                shutil.copyfile(src=str(filepath), dst=p)
        elif edit_local and filepath:
            p = filepath
        p = str(p)
        for i in Group._instances.keys():
            if i.startswith(os.path.abspath(p) + "::"):
                file = Group._instances[i].file
                if hasattr(Group._instances[i], "_tmpfile"):
                    tmpfile = Group._instances[i]._tmpfile
                break
        if file is None:
            file = h5py.File(p, "a")
        # construct fullpath
        if parent is None:
            parent = ""
            name = "/"
        else:
            parent = str(parent)
            if not parent.endswith("/"):
                parent += "/"
            name = natural_name
        fullpath = p + "::" + parent + name
        # create and/or return
        try:
            instance = Group._instances[fullpath]
        except KeyError:
            kwargs["file"] = file
            kwargs["parent"] = parent
            kwargs["name"] = natural_name
            instance = super(Group, cls).__new__(cls)
            cls.__init__(instance, **kwargs)
            Group._instances[fullpath] = instance
            if tmpfile:
                setattr(instance, "_tmpfile", tmpfile)
                weakref.finalize(instance, instance.close)
        return instance

    @property
    def __version__(self):
        if "__version__" not in self.file.attrs.keys():
            self.file.attrs["__version__"] = wt5_version
        return self.file.attrs["__version__"]

    @property
    def created(self):
        return wt_kit.timestamp_from_RFC3339(self.attrs["created"])

    @property
    def fullpath(self):
        """Full path: file and internal structure."""
        return self.filepath + "::" + self.name

    @property
    def item_names(self):
        """Item names."""
        if "item_names" not in self.attrs.keys():
            self.attrs["item_names"] = np.array([], dtype="S")
        return tuple(n.decode() for n in self.attrs["item_names"])

    @property
    def natural_name(self):
        """Natural name."""
        try:
            assert self._natural_name is not None
        except (AssertionError, AttributeError):
            self._natural_name = self.attrs["name"]
        finally:
            return self._natural_name

    @natural_name.setter
    def natural_name(self, value):
        """Set natural name."""
        if value is None:
            value = ""
        if hasattr(self, "_natural_name") and self.name != "/":
            keys = [k for k in self._instances.keys() if k.startswith(self.fullpath)]
            dskeys = [k for k in Dataset._instances.keys() if k.startswith(self.fullpath)]
            self.parent.move(self._natural_name, value)
            for k in keys:
                obj = self._instances.pop(k)
                self._instances[obj.fullpath] = obj
            for k in dskeys:
                obj = Dataset._instances.pop(k)
                Dataset._instances[obj.fullpath] = obj
        self._natural_name = value
        if self.file.mode is not None and self.file.mode != "r":
            self.attrs["name"] = value

    @property
    def parent(self):
        """Parent."""
        try:
            assert self._parent is not None
        except (AssertionError, AttributeError):
            key = posixpath.dirname(self.fullpath)
            if key.endswith("::"):
                key += posixpath.sep
            self._parent = Group._instances[key]
        finally:
            return self._parent

    def close(self):
        """Close the file that contains the Group.

        All groups which are in the file will be closed and removed from the
        _instances dictionaries.
        Tempfiles, if they exist, will be removed
        """
        path = os.path.abspath(self.filepath) + "::"
        for key in list(Group._instances.keys()):
            if key.startswith(path):
                Group._instances.pop(key, None)

        if self.fid.valid > 0:
            # for some reason, the following file operations sometimes fail
            # this stops execution of the method, meaning that the tempfile is never removed
            # the following try case ensures that the tempfile code is always executed
            # ---Blaise 2018-01-08
            try:
                self.file.flush()
                try:
                    # Obtaining the file descriptor must be done prior to closing
                    fd = self.fid.get_vfd_handle()
                except (NotImplementedError, ValueError):
                    # only available with certain h5py drivers
                    # not needed if not available
                    pass

                self.file.close()
                try:
                    if fd:
                        os.close(fd)
                except OSError:
                    # File already closed, e.g.
                    pass
            except SystemError as e:
                warnings.warn("SystemError: {0}".format(e))
            finally:
                if hasattr(self, "_tmpfile"):
                    try:
                        os.close(self._tmpfile[0])
                        os.remove(self._tmpfile[1])
                    except OSError:
                        # File already closed, e.g.
                        pass

    def copy(self, parent=None, name=None, verbose=True):
        """Create a copy under parent.

        All children are copied as well.

        Parameters
        ----------
        parent : WrightTools Collection (optional)
            Parent to copy within. If None, copy is created in root of new
            tempfile. Default is None.
        name : string (optional)
            Name of new copy at destination. If None, the current natural
            name is used. Default is None.
        verbose : boolean (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        Group
            Created copy.
        """
        if name is None:
            name = self.natural_name
        if parent is None:
            new = Group()  # root of new tempfile
            # attrs
            new.attrs.update(self.attrs)
            # children
            for k, v in self.items():
                super().copy(v, new, name=v.natural_name)
            new.flush()
            # Converts to appropriate Data/Collection object
            new = new["/"]
            new.natural_name = name
        else:
            # copy
            self.file.copy(self.name, parent, name=name)
            new = parent[name]
        # finish
        if verbose:
            print("{0} copied to {1}".format(self.fullpath, new.fullpath))
        return new

    def flush(self):
        """Ensure contents are written to file."""
        self.file.flush()

    def save(self, filepath=None, overwrite=False, verbose=True):
        """Save as root of a new file.

        Parameters
        ----------
        filepath : Path-like object (optional)
            Filepath to write. If None, file is created using natural_name.
        overwrite : boolean (optional)
            Toggle overwrite behavior. Default is False.
        verbose : boolean (optional)
            Toggle talkback. Default is True

        Returns
        -------
        str
            Written filepath.
        """
        if filepath is None:
            filepath = pathlib.Path(".") / self.natural_name
        else:
            filepath = pathlib.Path(filepath)
        filepath = filepath.with_suffix(".wt5")
        filepath = filepath.absolute().expanduser()
        if filepath.exists():
            if overwrite:
                filepath.unlink()
            else:
                raise wt_exceptions.FileExistsError(filepath)

        # copy to new file
        h5py.File(filepath, "w")
        new = Group(filepath=filepath, edit_local=True)
        # attrs
        for k, v in self.attrs.items():
            new.attrs[k] = v
        # children
        for k, v in self.items():
            super().copy(v, new, name=v.natural_name)
        # finish
        new.flush()
        new.close()
        del new
        if verbose:
            print("file saved at", filepath)
        return str(filepath)
