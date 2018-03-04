#! /usr/bin/env python3

import os
from setuptools import setup, find_packages


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


here = os.path.abspath(os.path.dirname(__file__))

extra_files = package_files(os.path.join(here, 'WrightTools', 'datasets'))
extra_files.append(os.path.join(here, 'CONTRIBUTORS'))
extra_files.append(os.path.join(here, 'LICENSE'))
extra_files.append(os.path.join(here, 'README.rst'))
extra_files.append(os.path.join(here, 'requirements.txt'))
extra_files.append(os.path.join(here, 'VERSION'))
extra_files.append(os.path.join(here, 'WT5_VERSION'))

with open(os.path.join(here, 'requirements.txt')) as f:
    required = f.read().splitlines()

with open(os.path.join(here, 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name='WrightTools',
    packages=find_packages(),
    package_data={'': extra_files},
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov',
                   'sphinx==1.6.5', 'sphinx-gallery==0.1.12', 'sphinx-rtd-theme'],
    install_requires=required,
    extras_require={'docs': ['sphinx-gallery==0.1.12']},
    version=version,
    description='Tools for loading, processing, and plotting multidimensional spectroscopy data.',
    author='Blaise Thompson',
    author_email='blaise@untzag.com',
    license='MIT',
    url='http://wright.tools',
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
