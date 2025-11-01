"""
Helpers specific to deal with data structures from the Wright Group's Bluesky deployment
https://github.com/wright-group/bluesky-in-a-box/
"""

import re
import json
import datetime
import pathlib
import logging
from typing import NamedTuple

from .._open import open as wt5_open


__folder_parts__ = [
    r"(?P<date>\d\d\d\d-\d\d-\d\d)",
    r"(?P<time>" + r"\d{5}" + ")",
    r"(?P<plan>\w*)",
    r"(?P<name>[\s\w\d.-]*)",  # not great...
    r"(?P<uid>\w{8})",
]
__folder_seed__ = " ".join(__folder_parts__)
__datetime_seed__ = re.compile(" ".join(__folder_parts__[:3]))
__fmtseed__ = "{date} {time} {plan} {name} {uid}"


class BlueskyFolder:
    """container class for Bluesky acquisitions"""

    def __init__(self, folder_path: str | pathlib.Path):
        self.path = pathlib.Path(folder_path)
        # DDK: better to extract the information from the data inside, rather than relying on the name
        # self.info = parse_folder_contents(folder_path.name)
        self.info = parse_folder_name(folder_path.name)
        if self.info is None:
            return

        self._primary = None
        self._baseline = None
        self.logger = logging.getLogger(self.info.uid)
        self.logger.info(self.info)

    @property
    def primary(self):
        """open procedure based on plan"""
        if self._primary is None:
            # TODO: open procedure based on plan
            if self.info.plan == "gridscan_wp":
                self._primary = wt5_open(self.path / "primary.wt5")
            else:
                raise NotImplementedError(f"plan {self.info.plan}")
        return self._primary

    @property
    def baseline(self):
        if self._baseline is None:
            self._baseline = wt5_open(self.path / "primary.wt5")
        return self._baseline

    @property
    def baseline_tree(self) -> str:
        return (self.path / "baseline tree.txt").read_text()

    @property
    def primary_tree(self) -> str:
        return (self.path / "primary tree.txt").read_text()

    @property
    def start(self) -> dict:
        path = self.path / "bluesky_docs" / "start.json"
        return json.load(path.open())

    @property
    def stop(self) -> dict:
        path = self.path / "bluesky_docs" / "stop.json"
        return json.load(path.open())

    @property
    def primary_descriptor(self) -> dict:
        path = self.path / "bluesky_docs" / "primary descriptor.json"
        return json.load(path.open())

    @property
    def baseline_descriptor(self) -> dict:
        path = self.path / "bluesky_docs" / "baseline descriptor.json"
        return json.load(path.open())


class FolderInfo(NamedTuple):
    """Object representation of bluesky folder names"""

    date: datetime.date
    time: datetime.time
    plan: str
    name: str
    uid: str

    @property
    def folder(self):
        return __fmtseed__.format(
            date=self.date.strftime("%Y-%m-%d"),
            time=int(
                datetime.timedelta(
                    minutes=self.time.minute, seconds=self.time.second, hours=self.time.hour
                ).total_seconds()
            ),
            plan=self.plan,
            name=self.name,
            uid=self.uid,
        )


def match_identifier(dir: pathlib.Path, **bluesky_identifier) -> list[BlueskyFolder]:
    """
    walk a directory to find bluesky folder names that match the specified identifiers

    Parameters
    ----------

    dir: path-like
        the directory to iterate through

    kwargs
    ------

    bluesky_identifiers
        keys corresponding to FolderInfo properties (e.g. date, plan).

    Returns
    -------
    matches: list of BlueskyFolder objects
        BlueskyFolders corresponding to full matches with the bluesky_identifiers
    """
    for key in bluesky_identifier.keys():
        assert key in FolderInfo._fields

    keep = []

    for info in map(
        lambda item: parse_folder_name(item.name),
        filter(
            lambda item: item.is_dir() and re.fullmatch(__folder_seed__, item.name), dir.iterdir()
        ),
    ):
        idict = info._asdict()
        if all(idict[k] == bluesky_identifier[k] for k in bluesky_identifier.keys()):
            keep.append(BlueskyFolder(dir / info.folder))

    return keep


def parse_folder_contents(folder: str) -> FolderInfo | None:
    """
    (Coming soon!)
    Parse key identifiers from a bluesky folder based on the data inside.

    See also
    --------
    parse_folder_name:
        Use the folder name, rather than data inside the folder.
    """
    raise NotImplementedError


def parse_folder_name(folder: str) -> FolderInfo | None:
    """
    Convert a bluesky-formatted folder name into a structured dictonary-like format.

    Parameters
    ----------
    folder : string
        the folder name

    Returns
    -------
    FolderInfo | None
        if the name is parsed, returns a FolderInfo object.
        otherwise, returns None.
    """
    if ((uid_match := re.fullmatch(r"(?P<uid>\w{8})", folder.split()[-1])) is not None) and (
        (datetime_match := __datetime_seed__.match(folder)) is not None
    ):
        matchdict = uid_match.groupdict() | datetime_match.groupdict()
        matchdict["name"] = " ".join(folder.split()[3:-1])
        return _to_object(matchdict)
    else:
        return None


def _to_object(mdict: dict) -> FolderInfo:
    """convert re match dictionary into FolderInfo object"""
    date = datetime.date.fromisoformat(mdict.pop("date"))
    ts = int(mdict.pop("time"))  # total seconds since date start
    time = datetime.time(hour=ts // 3600, minute=(ts % 3600) // 60, second=ts % 60)
    return FolderInfo(date=date, time=time, **mdict)
