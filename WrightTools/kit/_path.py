"""Filepath functions."""

# --- import --------------------------------------------------------------------------------------


import pathlib


# --- define --------------------------------------------------------------------------------------


__all__ = ["get_path_matching"]


# --- functions -----------------------------------------------------------------------------------


def get_path_matching(name: str) -> pathlib.Path:
    """
    Non-recursive search for path to the folder "name".
    Searches the user directory, then looks up the cwd for a parent folder that matches.

    Parameters
    ----------
    name : string
        name of directory to search for.

    Returns
    -------
    pathlib.Path
        Full filepath to directory name.

    """
    # first try looking in the user folder
    p = pathlib.Path.home() / name
    # then try expanding upwards from cwd
    if not p.is_dir():
        p = None
        drive, *folders = pathlib.Path.cwd().parts
        if name in folders:
            p = pathlib.Path(drive).joinpath(*folders[: folders.index(name) + 1])
    # TODO: something more robust to catch the rest of the cases?
    return p
