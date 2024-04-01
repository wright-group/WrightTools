"""Datasets."""

# --- import --------------------------------------------------------------------------------------


import pathlib

from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


here = pathlib.Path(__file__).parent.resolve()


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
        for p in (here / dirname).iterdir():
            n = prefix + wt_kit.string2identifier(p.name.split(".")[0])
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
        ps = list((here / dirname).iterdir())
        n = prefix + wt_kit.string2identifier(dirname.name)
        setattr(self, n, ps)


BrunoldrRaman = DatasetContainer()
Cary = DatasetContainer()
COLORS = DatasetContainer()
JASCO = DatasetContainer()
KENT = DatasetContainer()
LabRAM = DatasetContainer()
ocean_optics = DatasetContainer()
PyCMDS = DatasetContainer()
Shimadzu = DatasetContainer()
Solis = DatasetContainer()
spcm = DatasetContainer()
Tensor27 = DatasetContainer()
wt5 = DatasetContainer()


# --- fill ----------------------------------------------------------------------------------------


def _populate_containers():
    BrunoldrRaman._from_files(here / "BrunoldrRaman")

    Cary._from_files("Cary")

    COLORS._from_files(here / "COLORS" / "v0.2", prefix="v0p2_")
    COLORS._from_files(here / "COLORS" / "v2.2", prefix="v2p2_")

    JASCO._from_files("JASCO")

    KENT._from_directory(here / "KENT" / "LDS821 TRSF")
    KENT._from_directory(here / "KENT" / "LDS821 DOVE")
    KENT._from_directory(here / "KENT" / "PbSe 2D delay B")

    LabRAM._from_files(here / "LabRAM")

    ocean_optics._from_files("ocean_optics")

    PyCMDS._from_files("PyCMDS")

    Shimadzu._from_files("Shimadzu")

    Solis._from_files("Solis")

    spcm._from_files("spcm")

    Tensor27._from_files("Tensor27")

    wt5._from_files(here / "wt5" / "v1.0.0", prefix="v1p0p0_")
    wt5._from_files(here / "wt5" / "v1.0.1", prefix="v1p0p1_")


_populate_containers()

# --- pretty namespace ----------------------------------------------------------------------------


__all__ = [
    "BrunoldrRaman",
    "Cary",
    "COLORS",
    "JASCO",
    "KENT",
    "LabRAM",
    "ocean_optics",
    "PyCMDS",
    "Shimadzu",
    "Solis",
    "spcm",
    "Tensor27",
    "wt5",
]
