"""Test timestamp."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_now():
    wt.kit.TimeStamp()  # exception will be raised upon failure


def test_utc():
    wt.kit.timestamp_from_RFC3339('2017-11-13 16:09:17Z')  # exception will be raised upon failure


def test_date():
    ts = wt.kit.timestamp_from_RFC3339('2017-11-13 16:09:17-6')
    assert len(ts.date) == 10


def test_hms():
    ts = wt.kit.timestamp_from_RFC3339('2017-11-13 16:33:44-6')
    assert len(ts.hms) == 8


def test_human():
    ts = wt.kit.TimeStamp()
    assert len(ts.human) == 19


def test_RFC3339():
    ts = wt.kit.TimeStamp()
    assert ts.RFC3339


def test_RFC5322():
    ts = wt.kit.TimeStamp()
    assert ts.RFC5322


def test_path():
    ts = wt.kit.TimeStamp()
    assert ts.path
