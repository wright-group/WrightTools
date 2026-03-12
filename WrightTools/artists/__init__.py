"""Artists."""

# flake8: noqa

import pathlib
for x in pathlib.Path(__file__).parent.walk():
    print(x)


from ._animate import *
from ._base import *
from ._colors import *
from ._helpers import *
from ._interact import *
from ._quick import *
