"""Datasets."""


# --- import --------------------------------------------------------------------------------------


import os

from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- container class -----------------------------------------------------------------------------


class DatasetContainer(object):
    def _from_files(self, dirname, prefix=""):
        """Add datasets from files in a directory.

        Parameters
        ----------
        dirname : string
            Directory name.
        prefix : string
            Prefix.
        """
        ps = [os.path.join(here, dirname, p) for p in os.listdir(os.path.join(here, dirname))]
        for p in ps:
            n = prefix + wt_kit.string2identifier(os.path.basename(p).split(".")[0])
            setattr(self, n, p)

    def _from_directory(self, dirname, prefix=""):
        """Add dataset from files in a directory.

        Parameters
        ----------
        dirname : string
            Directory name.
        prefix : string
            Prefix.
        """
        ps = [os.path.join(here, dirname, p) for p in os.listdir(os.path.join(here, dirname))]
        n = prefix + wt_kit.string2identifier(os.path.basename(dirname))
        setattr(self, n, ps)


# --- fill ----------------------------------------------------------------------------------------


BrunoldrRaman = DatasetContainer()
BrunoldrRaman._from_files(os.path.join(here, "BrunoldrRaman"))

Cary = DatasetContainer()
Cary._from_files("Cary")

COLORS = DatasetContainer()
COLORS._from_files(os.path.join(here, "COLORS", "v0.2"), prefix="v0p2_")
COLORS._from_files(os.path.join(here, "COLORS", "v2.2"), prefix="v2p2_")

JASCO = DatasetContainer()
JASCO._from_files("JASCO")

KENT = DatasetContainer()
KENT._from_directory(os.path.join(here, "KENT", "LDS821 TRSF"))
KENT._from_directory(os.path.join(here, "KENT", "LDS821 DOVE"))
KENT._from_directory(os.path.join(here, "KENT", "PbSe 2D delay B"))

ocean_optics = DatasetContainer()
ocean_optics._from_files("ocean_optics")

PyCMDS = DatasetContainer()
PyCMDS._from_files("PyCMDS")

Shimadzu = DatasetContainer()
Shimadzu._from_files("Shimadzu")

Solis = DatasetContainer()
Solis._from_files("Solis")

spcm = DatasetContainer()
spcm._from_files("spcm")

Tensor27 = DatasetContainer()
Tensor27._from_files("Tensor27")

wt5 = DatasetContainer()
wt5._from_files(os.path.join(here, "wt5", "v1.0.0"), prefix="v1p0p0_")
wt5._from_files(os.path.join(here, "wt5", "v1.0.1"), prefix="v1p0p1_")


# --- pretty namespace ----------------------------------------------------------------------------


__all__ = [
    "BrunoldrRaman",
    "Cary",
    "COLORS",
    "JASCO",
    "KENT",
    "ocean_optics",
    "PyCMDS",
    "Solis",
    "Tensor27",
    "wt5",
]
