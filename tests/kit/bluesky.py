import WrightTools as wt


def test_folderinfo():
    name1 = "2025-10-27 52433 count 2 beam PL spot2 post d7f183b5"
    name2 = "2025-10-27 54622 grid_scan_wp spot 3 spectral 6a45457c"

    fi1 = wt.kit.bluesky.parse_folder_name(name1)
    fi2 = wt.kit.bluesky.parse_folder_name(name2)

    for name, fi in [[name1, fi1], [name2, fi2]]:
        assert fi is not None
        assert fi.folder == name

    assert fi1.name == "2 beam PL spot2 post"
    assert fi2.name == "spot 3 spectral"
    assert fi1.plan == "count"
    assert fi2.plan == "grid_scan_wp"


def test_filter():
    name1 = "2025-10-27 52433 count 2 beam PL spot2 post d7f183b5"
    name2 = "2025-10-27 54622 grid_scan_wp spot 3 spectral 6a45457c"

    gridscans = [x for x in wt.kit.bluesky.filter_bluesky([name1, name2], plan="grid_scan_wp")]
    assert len(gridscans) == 1
    assert str(gridscans[0]) == name2


if __name__ == "__main__":
    test_folderinfo()
    test_filter()
