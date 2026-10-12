# Tutorials

Worked examples of what `opengate-gate-tree` does, written as Jupyter notebooks
against real GATE output. They are documentation meant to be read by a person,
and the last check that a feature works: a tutorial that stops running is a bug
report.

## Table Of Contents

- [What Is Here](#what-is-here)
- [Quick Start](#quick-start)
- [Make Targets](#make-targets)
- [The Tutorials](#the-tutorials)
- [Writing A Tutorial](#writing-a-tutorial)
- [Where Files Belong](#where-files-belong)

## What Is Here

```text
tutorials/
├── data/                  # what a notebook reads: small ROOT files and images
│   ├── input-examples/    # ROOT files of 500 kB or less, carried in the repository
│   ├── imgs/              # pictures a notebook or a page shows
│   └── README.md          # every file above, described
├── notebooks/
│   ├── .tmp/              # what a notebook writes; not carried in the repository
│   ├── tools/             # helpers the notebooks import instead of repeating
│   ├── template.ipynb     # the shape every tutorial takes
│   ├── tutorial-NN-<name>.ipynb
│   └── README.md          # the index of tutorials
├── simulations/
│   ├── .tmp/              # what a simulation writes besides its ROOT file
│   ├── sim-NN-<name>/     # the macros of one simulation, and what it is for
│   └── README.md
├── .gitignore
├── Makefile
└── README.md
```

## Quick Start

From this directory:

```bash
make init      # an environment of its own, with this package installed from the repository
make kernel    # register that environment with Jupyter
make lab       # open JupyterLab in notebooks/
```

`make init` needs [uv](https://github.com/astral-sh/uv), which this repository
already builds on. The environment is separate from the one at the root of the
repository, because notebooks pull in JupyterLab and matplotlib and the package
itself pulls in neither.

The package is installed from the working tree, not from PyPI, so a notebook
run after a change to `src/` sees that change. This is what makes a tutorial a
test of the package rather than a description of it.

## Make Targets

| Target | What it does |
| --- | --- |
| `make init` | create `.venv` and install what the notebooks need |
| `make install` | install the requirements again, into an existing `.venv` |
| `make kernel` | register the environment with Jupyter as `opengate-gate-tree-tutorials` |
| `make lab` | start JupyterLab in `notebooks/` |
| `make verify` | run every tutorial start to finish, each from an empty output directory, and fail on the first error |
| `make clean` | remove `.venv`, the checkpoints and everything under `.tmp/` |

`make verify` is the author's job before pushing, and again after every change
made in answer to a review. It runs the notebooks with the kernel of `.venv`
itself, so it needs no `make kernel` and always checks the code of this clone. It does not run in CI: executing a notebook needs
data that CI does not have, and a green build that skipped the notebooks would
say more than it knows.

## The Tutorials

The index, with what each one shows, is in
[notebooks/README.md](notebooks/README.md):

- `tutorial-01-tools-plotting.ipynb` - the plots `tools.plotting` draws, and
  which form suits a discrete column and which a continuous one;
- `tutorial-02-reading-writing-root-file.ipynb` - what a GATE ROOT file holds,
  reading chosen branches, and writing a selection of rows to a ROOT file;
- `tutorial-03-representations-and-formats.ipynb` - what a `TreeData` holds
  beside its pandas view, and what ROOT, HDF5 and CSV keep of it;
- `tutorial-04-command-line.ipynb` - the command line: every option, its output
  files and exit codes, and the same run in Python;
- `tutorial-05-hits-tree-variants.ipynb` - which structure a "Hits" tree has,
  and how a tree is checked against it;
- `tutorial-06-split-hits-trees.ipynb` - a file whose hits are split per run or
  per detector, read as one dataset;
- `tutorial-07-statistics-report.ipynb` - the statistics report written beside
  the data;
- `tutorial-08-positronium-source-values.ipynb` - what the values a
  `PositroniumSource` writes mean;
- `tutorial-09-selecting-rows.ipynb` - selecting rows: masks, ranges and
  chains;
- `tutorial-10-shapes-and-detector-geometry.ipynb` - solid shapes and the
  geometry of a cylindricalPET scanner;
- `tutorial-11-events-and-tracks.ipynb` - events and tracks, and why an event
  is the pair `(runID, eventID)`;
- `tutorial-12-units.ipynb` - the units GATE writes and the conversions to the
  ones an analysis reads;
- `tutorial-13-vectors-from-columns.ipynb` - vectors read from three columns,
  and the angles and components they give;
- `tutorial-14-compton-scattering.ipynb` - Compton scattering from the hits of
  a track to the polarization;
- `tutorial-15-angle-ranges.ipynb` - angle ranges, and how a range is part of
  the question;
- `tutorial-16-rows-without-an-answer.ipynb` - rows a calculation cannot
  answer, and what the package reports about them;
- `tutorial-17-whole-analysis-in-one-chain.ipynb` - a whole analysis in one
  chain;
- `tutorial-18-events-tracks-and-hits.ipynb` - a "Hits" tree read as events,
  the tracks in them and the hits they left;
- `tutorial-19-analysis-as-a-description.ipynb` - an analysis as a description;
- `tutorial-20-order-of-operations-and-positions.ipynb` - the order of operations, and positions;
- `tutorial-21-combining-analyses.ipynb` - combining analyses.

## Writing A Tutorial

Copy `notebooks/template.ipynb` to `notebooks/tutorial-NN-<name>.ipynb`, where
`NN` is the next number in order. The template carries every section a tutorial
needs, in the order they belong in, and the conventions behind them are in
[Tutorial conventions](../docs/TUTORIALS_CONVENTIONS.md).

Then:

1. write it, in English;
2. add a row to the index in `notebooks/README.md`;
3. describe any file you added to `data/` in `data/README.md`, and any
   simulation in `simulations/sim-NN-<name>/README.md`;
4. run `make verify`;
5. leave the outputs in the notebook - somebody reading it on GitHub has no
   environment, and a notebook without outputs shows them nothing.

## Where Files Belong

| What | Where | Carried in the repository |
| --- | --- | --- |
| a ROOT file of 500 kB or less | `data/input-examples/` | yes |
| a picture a notebook shows | `data/imgs/` | yes |
| the macros of a simulation | `simulations/sim-NN-<name>/` | yes |
| the ROOT file that simulation writes | wherever the macros put it | no |
| anything a notebook writes | `notebooks/.tmp/tutorial-NN-<name>/` | no |

A ROOT file above 500 kB does not belong in `data/input-examples/`. Give the
macros that produce it instead, under `simulations/`, so that a reader can
generate it and can see what was simulated. A pre-commit hook refuses larger
files under `tutorials/data/`.
