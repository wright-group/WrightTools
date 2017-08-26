"""Datasets."""


# --- import --------------------------------------------------------------------------------------


import os

from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- container class -----------------------------------------------------------------------------


class DatasetContainer(object):

    def from_files(self, dirname, prefix=''):
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
            n = prefix + wt_kit.string2identifier(os.path.basename(p).split('.')[0])
            setattr(self, n, p)

    def from_directory(self, dirname, prefix=''):
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


COLORS = DatasetContainer()
COLORS.from_files(os.path.join(here, 'COLORS', 'v0.2'), prefix='v0p2_')
COLORS.from_directory(os.path.join(here, 'COLORS', 'v2.1', 'MoS2 TrEE movie'), prefix='v2p1_')

JASCO = DatasetContainer()
JASCO.from_files('JASCO')

KENT = DatasetContainer()
KENT.from_directory(os.path.join(here, 'KENT', 'LDS821 TRSF'))
KENT.from_directory(os.path.join(here, 'KENT', 'PbSe 2D delay A'))
KENT.from_directory(os.path.join(here, 'KENT', 'PbSe 2D delay B'))

BrunoldrRaman = DatasetContainer()
BrunoldrRaman.from_files(os.path.join(here, 'BrunoldrRaman'))

PyCMDS = DatasetContainer()
PyCMDS.from_files('PyCMDS')


# --- pretty namespace ----------------------------------------------------------------------------


__all__ = ['COLORS', 'JASCO', 'KENT', 'BrunoldrRaman']
