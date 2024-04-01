"""Datasets."""

# --- import --------------------------------------------------------------------------------------


import pathlib
from types import SimpleNamespace

from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


here = pathlib.Path(__file__).parent.resolve()


# --- container class -----------------------------------------------------------------------------


BrunoldrRaman = SimpleNamespace()
Cary = SimpleNamespace()
COLORS = SimpleNamespace()
JASCO = SimpleNamespace()
KENT = SimpleNamespace()
LabRAM = SimpleNamespace()
ocean_optics = SimpleNamespace()
PyCMDS = SimpleNamespace()
Shimadzu = SimpleNamespace()
Solis = SimpleNamespace()
spcm = SimpleNamespace()
Tensor27 = SimpleNamespace()
wt5 = SimpleNamespace()


# --- fill ----------------------------------------------------------------------------------------


def _populate_containers():

    def _from_files(obj, dirname, prefix=""):
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
            setattr(obj, n, p)

    def _from_directory(obj, dirname, prefix=""):
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
        setattr(obj, n, ps)


    _from_files(BrunoldrRaman, here / "BrunoldrRaman")

    _from_files(Cary, "Cary")

    _from_files(COLORS, here / "COLORS" / "v0.2", prefix="v0p2_")
    _from_files(COLORS, here / "COLORS" / "v2.2", prefix="v2p2_")

    _from_files(JASCO, "JASCO")

    _from_directory(KENT, here / "KENT" / "LDS821 TRSF")
    _from_directory(KENT, here / "KENT" / "LDS821 DOVE")
    _from_directory(KENT, here / "KENT" / "PbSe 2D delay B")

    _from_files(LabRAM, here / "LabRAM")

    _from_files(ocean_optics, "ocean_optics")

    _from_files(PyCMDS, "PyCMDS")

    _from_files(Shimadzu, "Shimadzu")

    _from_files(Solis, "Solis")

    _from_files(spcm, "spcm")

    _from_files(Tensor27, "Tensor27")

    _from_files(wt5, here / "wt5" / "v1.0.0", prefix="v1p0p0_")
    _from_files(wt5, here / "wt5" / "v1.0.1", prefix="v1p0p1_")


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
