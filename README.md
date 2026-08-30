# opengate-gate-tree

[![CI](https://github.com/MateuszBala/opengate-gate-tree/actions/workflows/ci.yaml/badge.svg)](https://github.com/MateuszBala/opengate-gate-tree/actions/workflows/ci.yaml)
[![Version](https://img.shields.io/badge/version-0.1.1-informational)](https://github.com/MateuszBala/opengate-gate-tree/releases)
[![Standard Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)]()
[![Standard Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)]()
[![Standard Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)]()
[![Standard Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)]()
[![license](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

`opengate-gate-tree` is a utility for processing [GATE 9](https://github.com/OpenGATE/Gate) output ROOT file with trees (Hits, Singles,Coincidences) 


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
| `--output-file-title` | string | yes | non-empty string | Base name of the output file (without extension). |
| `--gate-tree` | enum-like string | yes | `Hits`, `Singles`, `Coincidences` | Name of the tree to process from the input ROOT file. |
| `--output-file-format` | enum-like string | yes | `root`, `hdf5`, `csv` | Output file format. |
| `--branches-to-extract` | list of strings | no | branch names valid for selected tree | Space-separated list of branches to extract. |

Validation behavior:

- the input file must exist and end with `.root`
- the output path must resolve to a directory
- all required options above must be provided
- branch names are validated against the selected tree

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

## Available Package Capabilities (Cumulative)

This section is append-only.
Add a capability entry only when its roadmap stage status changes from `planned` to `completed`.

Current development stage: version-0.1.1

Available capabilities:

- 0.1.0: project structure initialized and minimal buildable package code added.



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