.. _cli:

Command Line Interface (CLI)
============================

Most of the CLI interface is new as of v3.5.2 and under active depelopment.
The commands here may change.
Currently, there are two supported CLI calls to use:  `wt-convert`, and `wt5`.
For usage hints, use the `--help` argument.

wt-convert
----------

Use `wt-convert`` to explore the WrightTools units system and the conversions of units.

.. code-block:: shell
    > wt-units 1330 nm wn
    7692.3 wn
    0.95372 eV
    953.72 meV
    2.306096e+14 Hz
    230.61 THz
    230610 GHz

wt5
---

The wt5 command is meant to provide easy access to wt5 files and some basic wt5 properties.

Use `wt5 glob` to quickly probe a folder for wt5s contained inside
.. code-block:: shell

    > wt5 glob -d path/to/datasets
    tests\dataset\max_cached.wt5
    WrightTools\datasets\wt5\v1.0.0\perovskite_TA.wt5
    WrightTools\datasets\wt5\v1.0.1\MoS2_TrEE_movie.wt5

Use `wt5 explore` to see details of wt5s within a folder, as the option of quick load a wt5.
.. code-block:: shell

    > wt5 explore -d path/to/data/folders
    "
    ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
    ┃   ┃ path                                                ┃ size (MB) ┃ created             ┃ name          ┃ shape        ┃ axes                  ┃ variables ┃ channels ┃
    ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
    │ 0 │ tests\dataset\max_cached.wt5                        │    0.0    │ 2020-11-16 23:49:27 │ data          │ (3,)         │ ()                    │ 1         │ 0        │
    │ 1 │ WrightTools\datasets\wt5\v1.0.0\perovskite_TA.wt5   │    3.4    │ 2016.03.28 21_15_20 │ perovskite_TA │ (52, 52, 13) │ ('w1=wm', 'w2', 'd2') │ 27        │ 10       │
    │ 2 │ WrightTools\datasets\wt5\v1.0.1\MoS2_TrEE_movie.wt5 │    2.3    │ 2018-06-11 16:41:47 │ _001_dat      │ (41, 41, 23) │ ('w2', 'w1=wm', 'd2') │ 7         │ 6        │
    └───┴─────────────────────────────────────────────────────┴───────────┴─────────────────────┴───────────────┴──────────────┴───────────────────────┴───────────┴──────────┘
    Specify an index to load that entry. Use `t` to rerender table. Use no argument to exit.
    "

Use `wt5 load` to quickly open an interactive python console with your wt5 data pre-loaded.
.. code-block:: shell
    > wt5 load path\to\data_file.wt5
    --- INTERACTING --- (to continue, call exit() or quit())
    >>> import WrightTools as wt
    >>> import matplotlib.pyplot as plt
    >>> d = wt.open(r'path/to/data_file.wt5')
    >>>


Use `wt5 tree` to see a quick tree structure of a wt5 file.

.. code-block:: shell
    > wt5 tree path\to\data
    "
    / (...\felu75fe.wt5) (25, 256)
    ├── axes (2): w2 (wn), w2-wa (wn)
    ├── constants (0):
    └── channels (1): array_signal
    "

