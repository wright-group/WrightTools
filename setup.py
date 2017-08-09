# !/usr/bin/env python

import os
from setuptools import setup, find_packages

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('WrightTools/datasets')

setup(
    name='WrightTools',
    packages=find_packages(),
    package_data={'': extra_files},
    install_requires=['h5py', 'matplotlib', 'numpy', 'python-dateutil', 'pytz', 'scipy'],
    version='2.13.4',
    description='Tools for loading, processing, and plotting multidimensional spectroscopy data.',
    author='Blaise Thompson',
    license='MIT',
    author_email='blaise@untzag.com',
    url='https://github.com/wright-group/WrightTools',
    download_url='https://github.com/wright-group/WrightTools/archive/2.13.0.tar.gz',
    keywords='spectroscopy science multidimensional visualization',
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
