# !/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='WrightTools',
    packages=find_packages(),
    install_requires=['h5py',
                      'matplotlib', 
		      'numpy',
                      'python-dateutil',
                      'pytz',
                      'scipy'],
    version='2.13.0',
    description='A package for processing multidimensional specroscopy data.',
    author='Blaise Thompson',
    license='MIT',
    author_email='blaise@untzag.com',
    url='https://github.com/wright-group/WrightTools',
    download_url='https://github.com/wright-group/WrightTools/archive/2.13.0.tar.gz'
)
