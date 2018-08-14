"""Tools for working with lists."""


# --- import --------------------------------------------------------------------------------------

import itertools

# --- define --------------------------------------------------------------------------------------


__all__ = ["flatten_list", "intersperse", "get_index", "pairwise"]


# --- functions -----------------------------------------------------------------------------------


def flatten_list(items, seqtypes=(list, tuple), in_place=True):
    """Flatten an irregular sequence.

    Works generally but may be slower than it could
    be if you can make assumptions about your list.

    `Source`__

    __ https://stackoverflow.com/a/10824086

    Parameters
    ----------
    items : iterable
        The irregular sequence to flatten.
    seqtypes : iterable of types (optional)
        Types to flatten. Default is (list, tuple).
    in_place : boolean (optional)
        Toggle in_place flattening. Default is True.

    Returns
    -------
    list
        Flattened list.

    Examples
    --------
    >>> l = [[[1, 2, 3], [4, 5]], 6]
    >>> wt.kit.flatten_list(l)
    [1, 2, 3, 4, 5, 6]
    """
    if not in_place:
        items = items[:]
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i : i + 1] = items[i]
    return items


def intersperse(lis, value):
    """Put value between each existing item in list.

    Parameters
    ----------
    lis : list
        List to intersperse.
    value : object
        Value to insert.

    Returns
    -------
    list
        interspersed list
    """
    out = [value] * (len(lis) * 2 - 1)
    out[0::2] = lis
    return out


def get_index(lis, argument):
    """Find the index of an item, given either the item or index as an argument.

    Particularly useful as a wrapper for arguments like channel or axis.

    Parameters
    ----------
    lis : list
        List to parse.
    argument : int or object
        Argument.

    Returns
    -------
    int
        Index of chosen object.
    """
    # get channel
    if isinstance(argument, int):
        if -len(lis) <= argument < len(lis):
            return argument
        else:
            raise IndexError("index {0} incompatible with length {1}".format(argument, len(lis)))
    else:
        return lis.index(argument)


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ...

    Originally from `itertools docs`__

    __ https://docs.python.org/3/library/itertools.html#itertools-recipes

    Parameters
    ----------
    iterable : iterable
        Iterable from which to produce pairs

    Returns
    -------
    generator
        Generator which producis pairwise tuples
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
