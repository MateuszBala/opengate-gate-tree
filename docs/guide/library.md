# Using The Package As A Library

Everything the command line does is reachable from `opengate_gate_tree`. The
command line only parses arguments, configures logging and turns failures into
exit codes.

## Public API

Import from the package root; the submodules are an implementation detail.

```python
from opengate_gate_tree import (
    BranchNotFoundError,
    ExportError,
    GateTree,
    GateTreeError,
    OutputFileFormat,
    RootFile,
    RootFileError,
    TreeData,
    TreeNotFoundError,
    UnsupportedBranchTypeError,
    parse_gate_tree,
    parse_output_file_format,
    read_tree,
    write_tree,
)
```

The package ships a `py.typed` marker, so type checkers see its annotations
without extra configuration.

## Reading A Tree

```python
from pathlib import Path

from opengate_gate_tree import GateTree, read_tree

data = read_tree(Path("simulation.root"), GateTree.HITS, ["eventID", "edep"])
```

Omit the branch list to read every branch:

```python
data = read_tree(Path("simulation.root"), GateTree.HITS)
```

`read_tree` opens the file, reads one tree and closes the file again. When
several trees come from the same file, open it once instead:

```python
from opengate_gate_tree import RootFile

with RootFile(Path("simulation.root")) as root_file:
    print(root_file.tree_names)
    hits = root_file.read(GateTree.HITS, ["eventID", "edep"])
    singles = root_file.read(GateTree.SINGLES)
```

## Working With The Data

`read_tree` returns a {class}`~opengate_gate_tree.tree.treedata.TreeData`:

```python
data.entry_count        # number of entries
data.branch_names       # branch names, in order
data["edep"]            # one branch, as a NumPy array
data.select(["edep"])   # a new TreeData with fewer branches
frame = data.to_dataframe()
```

[Data representations](representations.md) explains both views in detail.

## Writing A File

```python
from opengate_gate_tree import OutputFileFormat, write_tree

write_tree(data, Path("out/hits.hdf5"), OutputFileFormat.HDF5)
```

The parent directory is created when missing, and an existing file is
overwritten. See [Exporting files](export.md) for what each format preserves.

## Handling Failures

Failures while reading or writing files derive from `GateTreeError`, so one
`except` clause covers them:

```python
from opengate_gate_tree import GateTreeError, TreeNotFoundError

try:
    data = read_tree(Path("simulation.root"), GateTree.SINGLES)
except TreeNotFoundError as error:
    print(f"tree missing: {error}")
except GateTreeError as error:
    print(f"could not process the file: {error}")
```

| Error | Raised when |
| --- | --- |
| `RootFileError` | the path is not a readable ROOT file |
| `TreeNotFoundError` | the requested tree is not in the file |
| `BranchNotFoundError` | a requested branch is not in the tree |
| `UnsupportedBranchTypeError` | a branch type has no supported representation |
| `ExportError` | the output file cannot be written |

`GateTreeError` deliberately does not derive from `ValueError`: a file that
cannot be opened is not an invalid value, and merging the two would make
argument mistakes indistinguishable from input and output failures. Malformed
arguments, such as an empty branch name, still raise `ValueError`.

## Logging

The package does not configure logging on import, and its logger carries a
`NullHandler`, so nothing reaches the console until an application asks for it.

Applications that want the defaults used by the command line can call:

```python
from opengate_gate_tree.logging_setup import configure_logging

configure_logging()
```

Everything else is ordinary `logging` configuration on the
`opengate_gate_tree` logger.
