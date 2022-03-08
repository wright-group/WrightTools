import WrightTools as wt
from WrightTools.datasets import LabRAM


def test_import():
    d = wt.data.from_LabRAM(LabRAM.raman_linescan)
    d.close()


if __name__ == "__main__":
    test_import()
