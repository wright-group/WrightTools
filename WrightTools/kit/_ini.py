"""Tools for interacting with ini files."""


# --- import --------------------------------------------------------------------------------------


import configparser

import tidy_headers


# --- define --------------------------------------------------------------------------------------


__all__ = ["INI"]


# --- class ---------------------------------------------------------------------------------------


class INI:
    """Handle communication with an INI file."""

    def __init__(self, filepath):
        """Create an INI handler object.

        Parameters
        ----------
        filepath : string
            Filepath.
        """
        self.filepath = filepath
        self.config = configparser.ConfigParser()

    def add_section(self, section):
        """Add section.

        Parameters
        ----------
        section : string
            Section to add.
        """
        self.config.read(self.filepath)
        self.config.add_section(section)
        with open(self.filepath, "w") as f:
            self.config.write(f)

    def clear(self):
        """Remove all contents from file. Use with extreme caution.

        .. warning:: This is a destructive action.
        """
        with open(self.filepath, "w"):
            pass
        self.config = configparser.ConfigParser()

    @property
    def dictionary(self) -> dict:
        """Get a python dictionary of contents."""
        self.config.read(self.filepath)
        return self.config._sections

    def get_options(self, section) -> list:
        """List the options in a section.

        Parameters
        ----------
        section : string
            The section to investigate.

        Returns
        -------
        list of strings
            The options within the given section.
        """
        return list(self.dictionary[section].keys())

    def has_option(self, section, option) -> bool:
        """Test if file has option.

        Parameters
        ----------
        section : string
            Section.
        option : string
            Option.

        Returns
        -------
        boolean
        """
        self.config.read(self.filepath)
        return self.config.has_option(section, option)

    def has_section(self, section) -> bool:
        """Test if file has section.

        Parameters
        ----------
        section : string
            Section.

        Returns
        -------
        boolean
        """
        self.config.read(self.filepath)
        return self.config.has_section(section)

    def read(self, section, option):
        """Read from file.

        Parameters
        ----------
        section : string
            Section.
        option : string
            Option.

        Returns
        -------
        string
            Value.
        """
        self.config.read(self.filepath)
        raw = self.config.get(section, option)
        out = tidy_headers._parse_item.string2item(raw, sep=", ")
        return out

    @property
    def sections(self) -> list:
        """List of sections."""
        self.config.read(self.filepath)
        return self.config.sections()

    def write(self, section, option, value):
        """Write to file.

        Parameters
        ----------
        section : string
            Section.
        option : string
            Option.
        value : string
            Value.
        """
        self.config.read(self.filepath)
        string = tidy_headers._parse_item.item2string(value, sep=", ")
        self.config.set(section, option, string)
        with open(self.filepath, "w") as f:
            self.config.write(f)
