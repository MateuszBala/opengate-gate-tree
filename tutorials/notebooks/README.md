# Notebooks

One notebook per subject, each one readable start to finish without an
environment: the outputs are kept in the file, so GitHub shows what the code
answered.

## The Tutorials

| Notebook | Version | What it shows | Simulation | Data |
| --- | --- | --- | --- | --- |
| [`tutorial-01-tools-plotting.ipynb`](tutorial-01-tools-plotting.ipynb) | 2 | every plot `tools.plotting` draws, and which form suits a discrete column and which a continuous one | — | `positronium-source-all-variants.root` |
| [`tutorial-02-reading-writing-root-file.ipynb`](tutorial-02-reading-writing-root-file.ipynb) | 1 | what a GATE ROOT file holds, reading chosen branches, selecting rows in pandas, writing them to a ROOT file and reading that file back | — | `positronium-source-all-variants.root` |
| [`tutorial-03-representations-and-formats.ipynb`](tutorial-03-representations-and-formats.ipynb) | 1 | what a `TreeData` holds beside its pandas view - array branches, `select` without copying, why `==` checks identity - and which of ROOT, HDF5 and CSV keeps `volumeID` and the 32-bit types | — | `hits-a2-cylindrical-pet.root` |
| [`tutorial-04-command-line.ipynb`](tutorial-04-command-line.ipynb) | 1 | every option of `opengate-gate-tree` run on the example files - output naming, `--statistics`, exit codes, files with several "Hits" trees, `--skip-hits-validation` - the same run in Python giving identical files, and a batch loop | — | `positronium-source-all-variants.root`, `hits-a1-no-system.root`, `hits-a2-cylindrical-pet.root`, `hits-one-tree-per-run.root`, `hits-one-tree-per-detector.root` |

Every new notebook adds a row. A tutorial nobody can find is a tutorial nobody
reads.

## Writing One

Copy [`template.ipynb`](template.ipynb) to `tutorial-NN-<name>.ipynb`, where
`NN` is the next number in order. The template carries every section a tutorial
needs, in the order they belong in: what it is about, what a reader will learn,
the simulation if there was one, the theory if it is needed, the tutorial
itself, a summary, a bibliography and a changelog.

The conventions behind those sections are in
[Tutorial conventions](../../docs/TUTORIALS_CONVENTIONS.md).

## Where A Notebook Writes

Nowhere except its own directory under `.tmp/`:

```python
from tools import paths

output = paths.output_dir("tutorial-01-reading-a-tree")
frame.to_csv(output / "hits.csv")
```

`output_dir` makes the directory the first time it is asked for. Nothing under
`.tmp/` is carried in the repository, so a notebook that writes anywhere else
puts its working files in front of the next reader.

## Tools

`tools/` holds what a notebook would otherwise repeat: where the files are, what
version of the package is installed, and the look of the plots. Each module is
described in [tools/README.md](tools/README.md).

They are imported by name, because a notebook runs with its own directory on
the path:

```python
from tools import environment, paths, plotting
```

## Before Pushing

Run `make verify` from `tutorials/`. It executes every notebook here start to
finish and stops at the first error. A tutorial that does not run is a bug
report about the package, the data, or the notebook - and it is better read
before a reader finds it.
