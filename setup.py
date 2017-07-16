# !/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='WrightTools',
    packages=find_packages(),
    install_requires=['h5py', 'matplotlib', 'numpy', 'python-dateutil', 'pytz', 'scipy'],
    version='2.13.1',
    description='A package for processing multidimensional specroscopy data.',
    author='Blaise Thompson',
    license='MIT',
    author_email='blaise@untzag.com',
    url='https://github.com/wright-group/WrightTools',
    download_url='https://github.com/wright-group/WrightTools/archive/2.13.0.tar.gz',
    keywords='spectrosopy science multidimensional visualization',
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering']
)
