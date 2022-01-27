"""Timestamp class and associated."""


# --- import --------------------------------------------------------------------------------------


import time

import dateutil
from dateutil import tz
import datetime

import numpy as np


# --- define --------------------------------------------------------------------------------------


__all__ = ["TimeStamp", "timestamp_from_RFC3339"]


# --- functions -----------------------------------------------------------------------------------


def timestamp_from_RFC3339(RFC3339):
    """Generate a Timestamp object from a RFC3339 formatted string.

    `Link to RFC3339`__

    __ https://www.ietf.org/rfc/rfc3339.txt

    Parameters
    ----------
    RFC3339 : string
        RFC3339 formatted string.

    Returns
    -------
    WrightTools.kit.TimeStamp
    """
    dt = dateutil.parser.parse(RFC3339)
    if hasattr(dt.tzinfo, "_offset"):
        timezone = dt.tzinfo._offset.total_seconds()
    else:
        timezone = "utc"
    timestamp = TimeStamp(at=dt.timestamp(), timezone=timezone)
    return timestamp


# --- class ---------------------------------------------------------------------------------------


class TimeStamp:
    """Class for representing a moment in time."""

    def __init__(self, at=None, timezone="local"):
        """Create a ``TimeStamp`` object.

        Parameters
        ----------
        at : float (optional)
            Seconds since epoch (unix time). If None, current time will be
            used. Default is None.
        timezone : string or integer (optional)
            String (one in {'local', 'utc'} or seconds offset from UTC. Default
            is local.

        Attributes
        ----------
        unix : float
            Seconds since epoch (unix time).
        date : string
            Date.
        hms : string
            Hours, minutes, seconds.
        human : string
            Representation of the timestamp meant to be human readable.
        legacy : string
            Legacy WrightTools timestamp representation.
        RFC3339 : string
            `RFC3339`__ representation (recommended for most applications).

            __ https://www.ietf.org/rfc/rfc3339.txt
        RFC5322 : string
            `RFC5322`__ representation.

            __ https://tools.ietf.org/html/rfc5322#section-3.3
        path : string
            Representation of the timestamp meant for inclusion in filepaths.


        """
        # get timezone
        if timezone == "local":
            self.tz = tz.tzlocal()
        elif timezone == "utc":
            self.tz = tz.UTC
        elif type(timezone) in [int, float]:
            self.tz = tz.tzoffset(None, timezone)
        else:
            raise KeyError
        # get unix timestamp
        if at is None:
            self.unix = time.time()
        else:
            self.unix = at
        # get now
        if at is None:
            self.datetime = datetime.datetime.now(self.tz)
        else:
            self.datetime = datetime.datetime.fromtimestamp(at, self.tz)

    def __repr__(self):
        """Unambiguous representation."""
        return "<WrightTools.kit.TimeStamp object '%s'>" % self.human

    def __str__(self):
        """Readable representation."""
        return self.RFC3339

    def __eq__(self, other):
        return self.datetime == other.datetime

    @property
    def date(self):
        """year-month-day."""
        return self.datetime.strftime("%Y-%m-%d")

    @property
    def hms(self):
        """Get time formated.

        ``HH:MM:SS``
        """
        return self.datetime.strftime("%H:%M:%S")

    @property
    def human(self):
        """Human-readable timestamp."""
        # get timezone offset
        delta_sec = time.timezone
        m, s = divmod(delta_sec, 60)
        h, m = divmod(m, 60)
        # create output
        format_string = "%Y-%m-%d %H:%M:%S"
        out = self.datetime.strftime(format_string)
        return out

    @property
    def RFC3339(self):
        """RFC3339.

        `Link to RFC3339.`__

        __ https://www.ietf.org/rfc/rfc3339.txt
        """
        return self.datetime.isoformat()

    @property
    def RFC5322(self):
        """RFC5322.

        `Link to RFC5322.`__

        __ https://tools.ietf.org/html/rfc5322#section-3.3
        """
        return self.datetime.astimezone(tz=tz.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")

    @property
    def path(self):
        """Timestamp for placing into filepaths."""
        out = self.datetime.strftime("%Y-%m-%d")
        out += " "
        ssm = (
            self.datetime - self.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        ).total_seconds()
        out += str(int(ssm)).zfill(5)
        return out
