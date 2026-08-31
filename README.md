# opengate-gate-tree

[![CI](https://github.com/MateuszBala/opengate-gate-tree/actions/workflows/ci.yaml/badge.svg)](https://github.com/MateuszBala/opengate-gate-tree/actions/workflows/ci.yaml)
[![Version](https://img.shields.io/badge/version-0.3.0-informational)](https://github.com/MateuszBala/opengate-gate-tree/releases)
[![Standard Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)]()
[![Standard Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)]()
[![Standard Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)]()
[![Documentation Status](https://readthedocs.org/projects/opengate-gate-tree/badge/?version=stable)](https://opengate-gate-tree.readthedocs.io/en/stable/?badge=stable)
[![license](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

`opengate-gate-tree` is a utility for processing [GATE 9](https://github.com/OpenGATE/Gate) output ROOT file with trees (Hits, Singles,Coincidences) 

## Supported GATE Versions

The package targets the C++ line of GATE, version 9.4.2 and newer.
GATE 10, the Python implementation, is not supported.

Output files are not meant to be read back by GATE. They are conversions of the
simulation output into whichever format suits the analysis that follows:

| Format | Typical consumer |
| --- | --- |
| `root` | further analysis in C++ with the ROOT framework |
| `hdf5` | analysis in Python or MATLAB, large datasets, columnar access |
| `csv` | quick inspection, spreadsheets, plain `pandas.read_csv` |

## Documentation

Full documentation, tutorials, and API reference are available on [Read the Docs](https://opengate-gate-tree.readthedocs.io/).

## Quick Start

### Install From PyPI

```bash
pip3 install opengate-gate-tree
```

### Install From Source

```bash
make init
make install
```

## Command-Line Options

The CLI accepts the following options:

| Option | Type | Required | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `--input-gate-root-file` | path | yes | file with `.root` extension | Path to the GATE ROOT input file. |
| `--output-dir` | path | yes | existing directory or new path | Directory where output file will be saved. If it does not exist, it is created automatically. |
| `--output-file-title` | string | yes | file name without directories | Title of the output file. The file is named `<title>.<tree>.<format>`, for example `patient_01.hits.csv`. |
| `--gate-tree` | enum-like string | yes | `Hits`, `Singles`, `Coincidences` | Name of the tree to process from the input ROOT file. |
| `--output-file-format` | enum-like string | yes | `root`, `hdf5`, `csv` | Output file format. |
| `--branches-to-extract` | list of strings | no | branch names valid for selected tree | Space-separated list of branches to extract. |
| `--input-tree-name` | string | no | name of a tree in the input file | Tree to read, when it is not named after the selected tree or when the file holds several trees of hits. |
| `--merge-hits-trees` | flag | no | — | Read every tree of hits in the input file as a single dataset. Only for `--gate-tree Hits`, and not together with `--input-tree-name`. |
| `--statistics` | flag | no | — | Write a report describing the extracted data next to the output file, as `<title>.<tree>.stats.json`. |
| `--skip-hits-validation` | flag | no | — | Extract the branches without recognising and checking the structure of the "Hits" tree. |

Validation behavior:

- the input file must exist, end with `.root` and be readable as a ROOT file
- the output directory is created if it does not exist
- an existing output file is overwritten without a prompt
- all required options above must be provided
- the selected tree must be present in the input file; if it is not, the error
  lists the trees the file actually holds
- branch names are validated against the branches present in the input file
- branches whose length varies per entry are reported as unsupported
- the structure of the "Hits" tree is recognised and checked before anything is
  read; a file from a GATE build the package does not know is still extracted
  with `--skip-hits-validation`
- hits stored under another name are found by their structure, so the
  `GateToTree` output, whose tree is called `tree`, needs no extra option
- a file holding hits in several trees, one per run or one per sensitive
  detector, reports them and is read either one tree at a time
  (`--input-tree-name`) or as one dataset (`--merge-hits-trees`)

The structures the "Hits" tree can have, how the package tells them apart and
what it checks are described in the
[guide](https://opengate-gate-tree.readthedocs.io/en/latest/guide/hits.html),
which also lists the branches of every supported structure.

The output file is named after the title, the tree it holds and the format it
is written in: a run extracting the hits into `csv` under the title
`patient_01` writes `patient_01.hits.csv`. The tree is part of the name because
one input file holds several trees, and extracting two of them should not land
on the same file.

The output file holds the extracted tree only. Histograms stored next to the
trees in a GATE file are not copied over.

Fixed-width array branches, such as `volumeID`, keep their shape in the `root`
and `hdf5` output. CSV has no cell for an array, so they are written there as
one column per component, named `volumeID_0` to `volumeID_9`.

Two branch names cannot be carried by every format, and are refused rather than
written as something else. A name holding a bracket, such as the `volumeID[0]`
of the `GateToTree` output, cannot go into a `root` file: uproot reads the
bracket as an array dimension and writes a file that cannot be read back. A
name holding a slash cannot go into an `hdf5` file, where it would create a
nested group instead of a dataset. Both layouts reach the other formats
unchanged.

Examples:

```bash
opengate-gate-tree \
	--input-gate-root-file ./data/simulation.root \
	--output-dir ./out \
	--output-file-title patient_01 \
	--gate-tree Hits \
	--output-file-format csv
```

```bash
opengate-gate-tree \
	--input-gate-root-file ./data/simulation.root \
	--output-dir ./out \
	--output-file-title patient_01 \
	--gate-tree Singles \
	--output-file-format hdf5 \
	--branches-to-extract eventID trackID edep posX
```

## Library Usage

Besides the command-line interface, the package can be used directly from
Python code. Everything the command line does is reachable from
`opengate_gate_tree`.

### Loading And Exporting Files

```python
from pathlib import Path

from opengate_gate_tree import (
    GateTree,
    OutputFileFormat,
    read_tree,
    write_tree,
)

# Load selected branches of the "Hits" tree from a GATE ROOT file.
data = read_tree(
    Path("simulation.root"),
    GateTree.HITS,
    ["eventID", "edep", "posX", "posY", "posZ"],
)

print(data.entry_count, data.branch_names)

# Work with the data as NumPy arrays or as a pandas.DataFrame.
energies = data["edep"]
frame = data.to_dataframe()

# Export to the format that fits the downstream analysis.
write_tree(data, Path("out/hits.hdf5"), OutputFileFormat.HDF5)
```

Omit the branch list to read every branch of the tree:

```python
data = read_tree(Path("simulation.root"), GateTree.HITS)
```

When several trees come from the same file, open it once with `RootFile`:

```python
from opengate_gate_tree import RootFile

with RootFile(Path("simulation.root")) as root_file:
    print(root_file.tree_names)
    hits = root_file.read(GateTree.HITS, ["eventID", "edep"])
```

### Reading A Split File And Summarising It

GATE can write the hits of one simulation into several trees, one per run or
one per sensitive detector. They are read as a single dataset, with a column
recording which tree every row came from:

```python
from opengate_gate_tree import (
    SOURCE_TREE_BRANCH,
    compute_statistics,
    format_statistics,
    read_hits_trees,
    write_statistics,
)

data = read_hits_trees(Path("simulation.root"))
print(set(data[SOURCE_TREE_BRANCH]))
```

Identifiers stay as GATE wrote them, so an event is told apart by its run and
its event identifier together: `eventID` repeats between runs, and repeats for
one decay recorded in two detectors.

A summary of what was extracted can be printed, saved, or both:

```python
statistics = compute_statistics(data)
print(format_statistics(statistics))
write_statistics(statistics, Path("out/hits.stats.json"))
```

The structure of a tree can also be asked about on its own:

```python
from opengate_gate_tree import RootFile, describe_hits_tree

with RootFile(Path("simulation.root")) as root_file:
    detection = root_file.detect_hits_tree()
    print(describe_hits_tree(detection))
```

Failures while reading or writing files are reported through a subclass of
`GateTreeError`, so one `except` clause covers them. Malformed arguments, such as
an empty branch name, raise `ValueError` instead:

```python
from opengate_gate_tree import GateTreeError, TreeNotFoundError

try:
    data = read_tree(Path("simulation.root"), GateTree.SINGLES)
except TreeNotFoundError as error:
    print(f"tree missing: {error}")
except GateTreeError as error:
    print(f"could not process the file: {error}")
```

The package does not configure logging on import. Applications that want the
defaults used by the command line can ask for them:

```python
from opengate_gate_tree.logging_setup import configure_logging

configure_logging()
```

The package ships a `py.typed` marker, so type checkers see its annotations.

## Available Package Capabilities (Cumulative)

This section is append-only.
Add a capability entry only when its roadmap stage status changes from `planned` to `completed`.

Current development stage: version-0.3.0

Available capabilities:

- 0.1.0: project structure initialized and minimal buildable package code added.
- 0.2.0: GATE ROOT files can be loaded and validated, trees and branches extracted into a NumPy-backed representation with a pandas view, and written to ROOT, HDF5 or CSV. Usable both as a command-line tool and as a library, with user documentation on ReadTheDocs.
- 0.3.0: the structure of the "Hits" tree is recognised and validated against the schema of the variant it holds, hits stored under another name are found by their structure, a file split into one tree per run or per sensitive detector is read as one dataset, statistics are computed and saved beside the data, and output files are named after the tree they hold.



## Development

Common development commands:

```bash
make lint
make format
make typecheck
make test
make check
```

### Pre-commit setup

You can install and activate `pre-commit` in two supported ways.

#### Option A (recommended): use `uv` in this repository

```bash
uv add --dev pre-commit
uv sync
uv run pre-commit install --hook-type pre-commit
```

Optional one-time verification on all files:

```bash
uv run pre-commit run --all-files
```

#### Option B: install `pre-commit` from Debian packages

```bash
sudo apt update
sudo apt install -y pre-commit
pre-commit --version
pre-commit install --hook-type pre-commit
```

Optional one-time verification on all files:

```bash
pre-commit run --all-files
```

The configured hook runs `make check` before each commit and blocks the commit if validation fails.

### Documentation

The user documentation is built with Sphinx:

```bash
make docs        # build docs/_build/html
make docs-check  # build with warnings treated as errors, as ReadTheDocs does
```

Project conventions and contribution standards:

- [Coding conventions](docs/CODING_CONVENTIONS.md)
- [Testing conventions](docs/TESTING_CONVENTIONS.md)
- [Commit conventions](docs/COMMIT_CONVENTIONS.md)
- [Contribution guide](docs/CONTRIBUTION.md)

## License

MIT License

Contact: [GitHub](https://github.com/MateuszBala)

## Author

The project was designed and implemented by Mateusz Jakub Bała.

Contact: [GitHub](https://github.com/MateuszBala)


## Contribution

To contribute new functionality:

- create a branch from `develop`
- follow the [commit conventions](docs/COMMIT_CONVENTIONS.md)
- open a PR using the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
- follow the [contribution guide](docs/CONTRIBUTION.md)