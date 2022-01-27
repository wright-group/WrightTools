"""Function to close all open wt5 files."""


# --- import -------------------------------------------------------------------------------------


from . import _group as wt_group


# --- define -------------------------------------------------------------------------------------


__all__ = ["close"]


# --- functions ----------------------------------------------------------------------------------


def close():
    """Close all open wt5 files.

    Warning
    -------
    This will make any open objects unusable and delete unsaved temporary files.
    """
    while len(wt_group.Group._instances) > 0:
        wt_group.Group._instances.popitem()[1].close()
