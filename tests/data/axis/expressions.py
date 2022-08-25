import WrightTools as wt


def test_space_character():
    d = wt.Data()
    d.create_variable("w1")
    d.create_variable("w2")
    d.transform("w1 + w2")
    assert d.axis_names == ("w1__p__w2",)
    assert d.axis_expressions == ("w1+w2",)

if __name__ == "__main__":
    test_space_character()
