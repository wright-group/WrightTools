"""test process_wigner"""


# --- import --------------------------------------------------------------------------------------


import os
import shutil

import WrightTools as wt


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


def make_clean_directory(name):
    dir = os.path.join(here, name)
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    return dir


# --- test ----------------------------------------------------------------------------------------


def test_SDC_0():
    p = os.path.join(here, 'SDC 0.data')
    data = wt.data.from_PyCMDS(p)
    dir = make_clean_directory('SDC 0')
    wt.tuning.spectral_delay_correction.process_wigner(data, 'signal_mean',
                                                       control_name='w2',
                                                       offset_name='d2',
                                                       coset_name='d2_w2',
                                                       save_directory=dir)
    # exceptions will be raised if the above fails


def test_SDC_1():
    p = os.path.join(here, 'SDC 1.data')
    data = wt.data.from_PyCMDS(p)
    dir = make_clean_directory('SDC 1')
    wt.tuning.spectral_delay_correction.process_wigner(data, 'signal_mean',
                                                       control_name='w1',
                                                       offset_name='d2',
                                                       coset_name='d2_w1',
                                                       save_directory=dir)
    # exceptions will be raised if the above fails
