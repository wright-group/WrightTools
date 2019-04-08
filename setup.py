#! /usr/bin/env python3

import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


def read(fname):
    return open(os.path.join(here, fname)).read()


extra_files = {
    "WrightTools": [
        "datasets",
        "datasets/*",
        "datasets/*/*",
        "datasets/*/*/*",
        "datasets/*/*/*/*",
        "CITATION",
        "VERSION",
        "WT5_VERSION",
    ]
}

with open(os.path.join(here, "WrightTools", "VERSION")) as version_file:
    version = version_file.read().strip()

setup(
    name="WrightTools",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data=extra_files,
    python_requires=">=3.6",
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov", "sphinx", "sphinx-gallery==0.1.12", "sphinx-rtd-theme"],
    install_requires=[
        "h5py",
        "imageio",
        "matplotlib>=3.0",
        "numexpr",
        "numpy>=1.15.0",
        "python-dateutil",
        "scipy",
        "tidy_headers>=1.0.0",
    ],
    extras_require={
        "docs": ["sphinx", "sphinx-gallery>=0.3.0", "sphinx-rtd-theme"],
        "dev": ["black", "pre-commit", "pydocstyle"],
    },
    version=version,
    description="Tools for loading, processing, and plotting multidimensional spectroscopy data.",
    long_description=read("README.rst"),
    author="WrightTools Developers",
    license="MIT",
    url="http://wright.tools",
    keywords="spectroscopy science multidimensional visualization",
    entry_points={"console_scripts": ["wt-tree=WrightTools.__main__:wt_tree"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
)
