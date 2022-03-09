import WrightTools as wt
from WrightTools.datasets import LabRAM


def test_import():
    d = wt.data.from_LabRAM(LabRAM.raman_linescan)
    d.print_tree()
    d.close()


if __name__ == "__main__":
    test_import()
