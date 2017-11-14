"""Tools for interacting with files."""


# --- import --------------------------------------------------------------------------------------


import linecache


# --- define --------------------------------------------------------------------------------------


__all__ = ['file_len', 'FileSlicer']


# --- functions -----------------------------------------------------------------------------------


def file_len(fname):
    """Get the number of lines in a file, without loading into memory."""
    # adapted from http://stackoverflow.com/questions/845058
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# --- class ---------------------------------------------------------------------------------------


class FileSlicer:
    """Access groups of lines from a file quickly, without loading the entire file into memory.

    Mostly a convinient wrapper around the standard library linecache
    module.
    """

    def __init__(self, path, skip_headers=True, header_character='#'):
        """Create a ``FileSlicer`` object.

        Parameters
        ----------
        path : string
            Path to the file.
        skip_headers : bool (optional)
            Toggle to skip headers at beginning of file. Default is True.
        header_character : string (optional)
            Character that appears at the beginning of header lines for this
            file. Default is '#'.
        """
        self.path = path
        self.n = 0
        self.length = file_len(path)
        if skip_headers:
            with open(path) as f:
                while f.readline()[0] == header_charachter:
                    self.n += 1

    def close(self):
        """Clear the cache, attempting to free as much memory as possible."""
        linecache.clearcache()

    def get(self, line_count):
        """Get the next group of lines from the file.

        Parameters
        ----------
        line_count : int
            The number of lines to read from the file.

        Returns
        -------
        list
            List of lines as strings.
        """
        # calculate indicies
        start = self.n
        stop = self.n + line_count
        if stop > self.length:
            raise IndexError('there are no more lines in the slicer: ' +
                             '(file length {})'.format(self.length))
        # get lines using list comprehension
        # for some reason linecache is 1 indexed >:-(
        out = [linecache.getline(self.path, i) for i in range(start + 1, stop + 1)]
        # finish
        self.n += line_count
        return out

    def skip(self, line_count):
        """Skip the next group of lines from the file.

        Parameters
        ----------
        line_count : int
            The number of lines to skip.
        """
        if self.n + line_count > self.length:
            raise IndexError('there are no more lines in the slicer: ' +
                             '(file length {})'.format(self.length))
        # finish
        self.n += line_count
