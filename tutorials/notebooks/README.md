# Notebooks

One notebook per subject, each one readable start to finish without an
environment: the outputs are kept in the file, so GitHub shows what the code
answered.

## The Tutorials

| Notebook | Version | What it shows | Simulation | Data |
| --- | --- | --- | --- | --- |
| [`tutorial-01-tools-plotting.ipynb`](tutorial-01-tools-plotting.ipynb) | 1 | every plot `tools.plotting` draws, and which form suits a discrete column and which a continuous one | — | `positronium-source-all-variants.root` |

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
