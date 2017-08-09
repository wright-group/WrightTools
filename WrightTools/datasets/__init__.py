"""
"""


# --- import --------------------------------------------------------------------------------------


import os


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- container class -----------------------------------------------------------------------------


def clean_name(n, prefix=''):
    illegals = [' ', '[', ']']
    for c in illegals:
        n = n.replace(c, '_')
    return prefix + n


class DatasetContainer(object):

    def from_files(self, dirname, prefix=''):
        ps = [os.path.join(here, dirname, p) for p in os.listdir(os.path.join(here, dirname))]
        for p in ps:
            n = clean_name(os.path.basename(p).split('.')[0], prefix=prefix)
            setattr(self, n, p)

    def from_directory(self, dirname, prefix=''):
        ps = [os.path.join(here, dirname, p) for p in os.listdir(os.path.join(here, dirname))]
        n = clean_name(os.path.basename(dirname), prefix=prefix)
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


# --- pretty namespace ----------------------------------------------------------------------------


__all__ = ['COLORS', 'JASCO', 'KENT']
