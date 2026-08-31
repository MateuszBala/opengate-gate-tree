# opengate-gate-tree

`opengate-gate-tree` extracts trees from [GATE](https://github.com/OpenGATE/Gate)
output ROOT files and converts them into the format that suits the analysis
which follows.

It works both as a command line tool and as a library.

## Supported GATE Versions

The package targets the C++ line of GATE, version 9.4.2 and newer. GATE 10, the
Python implementation, is not supported.

## What It Is For

Files written by this package are not meant to be read back by GATE. A typical
run takes the ROOT file produced by a simulation, keeps the branches that
matter for the analysis, and writes them in one of three formats:

| Format | Typical consumer |
| --- | --- |
| `root` | further analysis in C++ with the ROOT framework |
| `hdf5` | analysis in Python or MATLAB, large datasets, columnar access |
| `csv` | quick inspection, spreadsheets, plain `pandas.read_csv` |

```{toctree}
:caption: User guide
:maxdepth: 2

guide/quickstart
guide/library
guide/loading
guide/hits
guide/events
guide/positronium
guide/merging
guide/representations
guide/statistics
guide/export
guide/cli
```

```{toctree}
:caption: API reference
:maxdepth: 2

api/io
api/tree
api/cli
```

```{toctree}
:caption: Project conventions
:maxdepth: 1

ROADMAP
CONTRIBUTION
CODING_CONVENTIONS
TESTING_CONVENTIONS
COMMIT_CONVENTIONS
VERSION_UPGRADE_CONVENTIONS
```
