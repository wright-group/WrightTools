import WrightTools as wt
from WrightTools.datasets import LabRAM


def test_spectral_units():
    d = wt.data.from_LabRAM(LabRAM.spectrum_nm)
    assert d.wm.units == "nm"
    d.close()
    d = wt.data.from_LabRAM(LabRAM.spectrum_wn)
    assert d.wm.units == "wn"
    d.close()


def test_import_1D():
    d = wt.data.from_LabRAM(LabRAM.spectrum_nm)
    d.close()


def test_import_2D():
    d = wt.data.from_LabRAM(LabRAM.raman_linescan)
    d.close()


def test_import_2D_survey():
    d = wt.data.from_LabRAM(LabRAM.survey_nm)
    d.close()


def test_import_3D():
    d = wt.data.from_LabRAM(LabRAM.map_nm)
    d.close()


if __name__ == "__main__":
    test_spectral_units()
    test_import_1D()
    test_import_2D()
    test_import_2D_survey()
    test_import_3D()
