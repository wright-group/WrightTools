"""Aramis."""


# --- import --------------------------------------------------------------------------------------


import os
import struct
import pathlib
import warnings

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ["from_Aramis"]


# --- from function -------------------------------------------------------------------------------


def from_Aramis(filepath, name=None, parent=None, verbose=True) -> Data:
    """Create a data object from Horiba Aramis ngc binary file.

    Parameters
    ----------
    filepath : path-like
        Path to .ngc file.
        Can be either a local or remote file (http/ftp).
        Can be compressed with gz/bz2, decompression based on file name.
    name : string (optional)
        Name to give to the created data object. If None, name is extracted from file.
        Default is None.
    parent : WrightTools.Collection (optional)
        Collection to place new data object within. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object(s).
    """
    # parse filepath
    filestr = os.fspath(filepath)
    filepath = pathlib.Path(filepath)

    if not ".ngc" in filepath.suffixes:
        wt_exceptions.WrongFileTypeWarning.warn(filepath, ".ngc")
    ds = np.DataSource(None)
    f = ds.open(filestr, "rb")
    header = f.readline()
    if header != b"NGSNextGen\x01\x00\x00\x00\x01\x00\x00\x00\n":
        warnings.warn(f"Unexpected Header {header}, Aramis parsing may not be valid")
    header = f.read(10)
    if header != b"DataMatrix":
        warnings.warn(f"Unexpected Header {header}, Aramis parsing may not be valid")
    instr = _readstr(f)
    iname = _readstr(f)
    # parse name
    if not name:
        name = iname
    # create data
    kwargs = {"name": name, "kind": "Aramis", "source": filestr}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)

    # array
    f.seek(4 * 4, 1)  # skip 4 integers [~=size, 0, 8, -1] is expected
    asize = struct.unpack("<i", f.read(4))[0]
    ndim = struct.unpack("<h", f.read(2))[0]
    shape = struct.unpack(f"<{'i'*ndim}", f.read(4 * ndim))
    f.seek(2 + 4, 1)  # skip '0xffff', size in bytes
    arr = np.fromfile(f, "<f4", np.prod(shape))
    arr.shape = shape
    f.read((asize - arr.size) * 4)
    while f.read(1) == b"\0":
        pass
    f.seek(-1, 1)
    nlab = struct.unpack("<h", f.read(2))[0]
    labels = [_readstr(f) for _ in range(nlab)]
    nunit = struct.unpack("<h", f.read(2))[0]
    unit_trans = {"1/cm": "wn", "µm": "um", "sec": "s_t", "": None}
    units = [_readstr(f) for _ in range(nunit)]
    units = [unit_trans.get(u, u) for u in units]
    skip = struct.unpack("<h", f.read(2))[0]
    f.seek(skip, 1)  # skip values that were all zero in test data
    # Which index in the shape aligns whith which label/unit
    nidx = struct.unpack("<h", f.read(2))[0]
    idx = struct.unpack(f"<{'h'*2*nidx}", f.read(4 * nidx))[::2]

    chidx = idx.index(ndim)
    data.create_channel(labels[chidx], arr, units=units[chidx])

    # Endpoints of axes, needed if full array unavailable
    nend = struct.unpack("<h", f.read(2))[0]
    end = struct.unpack(f"<{'f'*nend}", f.read(4 * nend))
    # Unknown what value means, other than nonzero seems to indicate array present
    nunk = struct.unpack("<h", f.read(2))[0]
    unk = struct.unpack(f"<{'i'*nunk}", f.read(4 * nunk))
    for i, u in enumerate(unk):
        if idx[i] < ndim:
            if u != 0:
                axissize = struct.unpack("<h", f.read(2))[0] // 4
                arr = np.fromfile(f, "<f4", axissize)
            else:
                arr = np.linspace(end[2 * i], end[2 * i + 1], shape[idx[i]])
            sh = [1] * ndim
            sh[idx[i]] = shape[idx[i]]
            arr.shape = tuple(sh)
            data.create_variable(labels[i], arr, units=units[i], label=labels[i])

    data.transform(*[labels[i] for i, ix in enumerate(idx) if ix < ndim])
    # finish
    f.close()
    if verbose:
        print("data created at {0}".format(data.fullpath))
        print("  axes: {0}".format(data.axis_names))
        print("  shape: {0}".format(data.shape))
    return data


def _readstr(f):
    return f.read(ord(f.read(1))).decode("iso-8859-1")
