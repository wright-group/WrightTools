"""test flatten_list"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_list():
    input = [[[1, 2, 3], [4, 5]], 6]
    output = wt.kit.flatten_list(input)
    assert output == [1, 2, 3, 4, 5, 6]
