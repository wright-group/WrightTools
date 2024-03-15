import pathlib
from os import PathLike
from typing import Union, List, Iterator
from ..data import Data
from ..collection import Collection
from .._open import open


__all__ = [
    "describe_wt5",
    "filter_wt5s",
    "glob_handler",
    "glob_wt5s",
    "search_for_attrs",
]


def describe_wt5(path: Union[str, PathLike]) -> dict:
    """report useful general information about a wt5 file"""
    wt5 = open(path)
    desc = dict()
    desc["name"] = wt5.natural_name
    try:
        desc["created"] = wt5.created.human
    except:  # likely an old timestamp that cannot be parsed
        desc["created"] = wt5.attrs["created"]

    if isinstance(wt5, Data):
        desc["shape"] = wt5.shape
        desc["axes"] = wt5.axis_expressions
        desc["nvars"] = len(wt5.variables)
        desc["nchan"] = len(wt5.channels)
    elif isinstance(wt5, Collection):
        for k in ["shape", "axes", "nvars", "nchan"]:
            desc[k] = "---"
    wt5.close()
    return desc


def glob_wt5s(directory: Union[str, PathLike, None] = None, recursive=True) -> Iterator:
    """glob all wt5 files in a directory"""
    if directory is None:
        directory = pathlib.Path.cwd()
    pattern = "**/*.wt5" if recursive else f"*.wt5"
    return pathlib.Path(directory).glob(pattern)


def glob_handler(extension, directory=None, identifier=None, recursive=True) -> List[pathlib.Path]:
    """Return a list of all files matching specified inputs.

    Parameters
    ----------
    extension : string
        File extension.
    directory : string (optional)
        Folder to search within. Default is None (current working
        directory).
    identifier : string
        Unique identifier. Default is None.
    recursive : bool
        When true, searches folder and all subfolders for identifier

    Returns
    -------
    list of pathlib.Path objects
        path objects for matching files.
    """
    if directory is None:
        directory = pathlib.Path.cwd()
    pattern = f"**/*.{extension}" if recursive else f"*.{extension}"
    return [
        x for x in filter(lambda x: identifier in str(x), pathlib.Path(directory).glob(pattern))
    ]


def search_for_attrs(
    directory: Union[str, PathLike, None] = None, recursive=True, **kwargs
) -> List[pathlib.Path]:
    """
    Find wt5 file(s) by matching data attrs items.

    Parameters
    ----------
    directory : path-like
        directory to search.  Defaults to cwd.
    recursive : boolean (default True)
        whether or not recursively search the directory

    kwargs
    ------
    key value pairs to filter the attrs with

    Returns
    -------
    paths : list
        list of pathlib.Path objects that match

    Example
    -------
    To find a scan from scan parameters in the bluesky-cmds:
    >>> search_wt5_by_attr(os.environ["WT5_DATA_DIR"], name="primary", shape=[136,101], )
    """
    return filter_wt5s(glob_wt5s(directory, recursive), **kwargs)


def filter_wt5s(paths: List[Union[str, PathLike]], **kwargs) -> List[Union[str, PathLike]]:
    """fillter a list of wt5 paths by attrs"""
    return [p for p in filter(_gen_filter_by_attrs(**kwargs), paths)]


def _gen_filter_by_attrs(**kwargs):
    def filter(x):
        attrs = open(x).attrs
        if all([key in attrs for key in kwargs.keys()]):
            return all([attrs[key] == kwargs[key] for key in kwargs])
        else:
            return False

    return filter
