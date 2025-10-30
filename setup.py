#! /usr/bin/env python3


from setuptools import setup, find_packages


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

setup(
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data=extra_files,
)
