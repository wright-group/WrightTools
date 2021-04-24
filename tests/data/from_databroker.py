import pathlib

from databroker.v2 import temp
import msgpack

import WrightTools as wt

catalog = temp()

__here__ = pathlib.Path(__file__).parent

for child in (__here__ / "bluesky_data").iterdir():
    with child.open("rb") as f:
        unpack = msgpack.Unpacker(f)
        for item in unpack:
            catalog.v1.insert(*item)


def test_2d_data():
    run = catalog["3dbdd402-434b-4aac-b004-447a2f026d73"]
    data = wt.data.from_databroker(run)
    assert data.shape == (10, 11)
    assert "d1_readback" in data
    assert "d1_setpoint" in data
    assert data.channel_names == ("daq_random_walk",)
    assert data.d1_readback.min() == 0
    assert data.d1_readback.max() == 1
