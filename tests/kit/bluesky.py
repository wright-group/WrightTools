import WrightTools as wt
import logging
from WrightTools import datasets

logging.basicConfig(level=logging.INFO)


def test_compress_sensor_mappings():
    data = wt.open(datasets.wt5.v1p0p3_shg_andor)
    wt.kit.bluesky.compress_sensor_mappings(data, "andor_camera")
    assert data["andor_camera_x_index"].shape == (1, 1, 60)
    assert data["andor_camera_y_index"].shape == (1, 60, 1)


def test_folderinfo():
    names = [
        "2025-10-27 52433 count 2 beam PL spot2 post d7f183b5",
        "2025-10-27 00622 grid_scan_wp spot 3 spectral 6a45457c",
        "2025-10-27 00010 custom_plan +-5 --testing-- p3uyqhr7",
    ]

    fis = [wt.kit.bluesky.parse_folder_name(ni) for ni in names]

    for name, fi in zip(names, fis):
        assert fi is not None
        assert fi.strf == name

    assert fis[0].name == "2 beam PL spot2 post"
    assert fis[1].name == "spot 3 spectral"
    assert fis[0].plan == "count"
    assert fis[1].plan == "grid_scan_wp"


def test_filter():
    name1 = "2025-10-27 52433 count 2 beam PL spot2 post d7f183b5"
    name2 = "2025-10-27 54622 grid_scan_wp spot 3 spectral 6a45457c"

    gridscans = [x for x in wt.kit.bluesky.filter_bluesky([name1, name2], plan="grid_scan_wp")]
    assert len(gridscans) == 1
    assert str(gridscans[0]) == name2


if __name__ == "__main__":
    test_compress_sensor_mappings()
    test_folderinfo()
    test_filter()
