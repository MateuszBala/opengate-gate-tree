# Quickstart

## Requirements

- Python 3.11 or newer
- a GATE output ROOT file produced by GATE 9.4.2 or newer, C++ line

GATE 10, the Python implementation, is not supported.

## Installation

From PyPI:

```bash
pip3 install opengate-gate-tree
```

From source:

```bash
make init
make install
```

## First Run From The Command Line

Extract the whole `Hits` tree and write it as CSV:

```bash
opengate-gate-tree \
	--input-gate-root-file ./data/simulation.root \
	--output-dir ./out \
	--output-file-title patient_01 \
	--gate-tree Hits \
	--output-file-format csv
```

The run writes `./out/patient_01.csv`. The output directory is created if it
does not exist, and an existing output file is overwritten without a prompt.

Add `--branches-to-extract` to keep only what the analysis needs:

```bash
opengate-gate-tree \
	--input-gate-root-file ./data/simulation.root \
	--output-dir ./out \
	--output-file-title patient_01 \
	--gate-tree Hits \
	--output-file-format hdf5 \
	--branches-to-extract eventID edep posX posY posZ
```

Every option is described in [Command line](cli.md).

## First Script

The same run from Python:

```python
from pathlib import Path

from opengate_gate_tree import GateTree, OutputFileFormat, read_tree, write_tree

data = read_tree(
    Path("data/simulation.root"),
    GateTree.HITS,
    ["eventID", "edep", "posX", "posY", "posZ"],
)
print(data.entry_count, data.branch_names)

write_tree(data, Path("out/patient_01.hdf5"), OutputFileFormat.HDF5)
```

[Using the package as a library](library.md) covers this in full.

## Where To Go Next

- [Loading files](loading.md) — what the package reads and what it refuses
- [Data representations](representations.md) — NumPy arrays and data frames
- [Exporting files](export.md) — choosing a format and reading it back
