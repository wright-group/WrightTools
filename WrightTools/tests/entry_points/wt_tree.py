import hashlib
import WrightTools as wt
import subprocess
import os


def test_no_change():
    d = wt.Data()
    d.save("test_no_change", overwrite=True)
    d.close()
    before = hashlib.sha1()
    with open("test_no_change.wt5", "rb") as f:
        before.update(f.read())
    subprocess.call(["wt-tree", "test_no_change.wt5"])
    after = hashlib.sha1()
    with open("test_no_change.wt5", "rb") as f:
        after.update(f.read())
    os.remove("test_no_change.wt5")
    assert before.digest() == after.digest()
