"""Filepath functions."""


# --- import --------------------------------------------------------------------------------------


import os
import glob


# --- define --------------------------------------------------------------------------------------


__all__ = ["get_path_matching", "glob_handler"]


# --- functions -----------------------------------------------------------------------------------


def get_path_matching(name):
    """Get path matching a name.

    Parameters
    ----------
    name : string
        Name to search for.

    Returns
    -------
    string
        Full filepath.
    """
    # first try looking in the user folder
    p = os.path.join(os.path.expanduser("~"), name)
    # then try expanding upwards from cwd
    if not os.path.isdir(p):
        p = None
        drive, folders = os.path.splitdrive(os.getcwd())
        folders = folders.split(os.sep)
        folders.insert(0, os.sep)
        if name in folders:
            p = os.path.join(drive, *folders[: folders.index(name) + 1])
    # TODO: something more robust to catch the rest of the cases?
    return p


def glob_handler(extension, folder=None, identifier=None):
    """Return a list of all files matching specified inputs.

    Parameters
    ----------
    extension : string
        File extension.
    folder : string (optional)
        Folder to search within. Default is None (current working
        directory).
    identifier : string
        Unique identifier. Default is None.

    Returns
    -------
    list of strings
        Full path of matching files.
    """
    filepaths = []
    if folder:
        # comment out [ and ]...
        folder = folder.replace("[", "?")
        folder = folder.replace("]", "*")
        folder = folder.replace("?", "[[]")
        folder = folder.replace("*", "[]]")
        glob_str = os.path.join(folder, "*" + extension)
    else:
        glob_str = "*" + extension + "*"
    for filepath in glob.glob(glob_str):
        if identifier:
            if identifier in filepath:
                filepaths.append(filepath)
        else:
            filepaths.append(filepath)
    return filepaths
