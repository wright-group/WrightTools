#! /usr/bin/env python3

import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


def read(fname):
    with open(os.path.join(here, fname)) as f:
        return f.read()


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

docs_require = ["sphinx<8.0", "sphinx-gallery==0.8.2", "sphinx-rtd-theme"]

setup(
    name="WrightTools",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data=extra_files,
    python_requires=">=3.7",
    install_requires=[
        "h5py",
        "imageio>=2.28.0",
        "matplotlib>=3.4.0",
        "numexpr",
        "numpy>=1.15.0",
        "pint",
        "python-dateutil",
        "scipy",
        "click",
        "tidy_headers>=1.0.4",
        "rich",
    ],
    extras_require={
        "docs": docs_require,
        "dev": [
            "black",
            "pre-commit",
            "pydocstyle",
            "pytest",
            "pytest-cov",
            "databroker>=1.2",
            "msgpack",
        ]
        + docs_require,
    },
    version=version,
    description="Tools for loading, processing, and plotting multidimensional spectroscopy data.",
    description_content_type="text/plain",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    author="WrightTools Developers",
    license="MIT",
    url="http://wright.tools",
    keywords="spectroscopy science multidimensional visualization",
    entry_points={
        "console_scripts": [
            "wt5=WrightTools.cli._wt5:cli",
            "wt-convert=WrightTools.cli._units:cli",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Framework :: Matplotlib",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
)
