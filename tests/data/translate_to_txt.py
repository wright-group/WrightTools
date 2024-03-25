"""Test transform."""

# --- import --------------------------------------------------------------------------------------


import pathlib
import WrightTools as wt

from WrightTools import datasets
from tempfile import NamedTemporaryFile


# --- tests ---------------------------------------------------------------------------------------


def test_datasets_mos2():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie).at(w2=[18000, "wn"])[:5]

    with NamedTemporaryFile(delete=False) as tmp:
        d.translate_to_txt(tmp.name, verbose=True)

        with open(tmp.name, "r") as f:
            for i in range(100):
                f.readline()
            datum_txt = f.readline().split("|")
            id1, id2 = [_ for _ in map(int, datum_txt[0].split())]
            values = [_ for _ in map(float, datum_txt[1].split())]
            datum_wt5 = d[int(id1), int(id2)]
            for i, vari in enumerate(d.variable_names):
                # print(vari, values[i], datum_wt5[vari][:])
                assert (values[i] - datum_wt5[vari][:]) ** 2 <= (1e-4 * values[i]) ** 2

    d.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_datasets_mos2()
