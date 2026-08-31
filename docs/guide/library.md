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

## Structures, Merging And Statistics

The structure of a "Hits" tree is recognised on reading and can be asked about
on its own:

```python
from opengate_gate_tree import RootFile, describe_hits_tree, summarise_hits_tree

with RootFile(path) as root_file:
    detection = root_file.detect_hits_tree()

print(summarise_hits_tree(detection))   # one line
print(describe_hits_tree(detection))    # every branch with its type
```

`HitsTreeVariant` names the structures, `expected_branches` states what each of
them holds, and `validate_hits_tree` checks a branch list against one. See
[The "Hits" Tree](hits.md).

Hits split into several trees are read as one dataset with `read_hits_trees`,
which records where every row came from. See [Merging Trees](merging.md).

`compute_statistics`, `format_statistics` and `write_statistics` summarise what
was extracted. See [Statistics](statistics.md).

Four more errors join the hierarchy in this version:
`UnknownHitsVariantError` when a structure is not a supported one,
`HitsTreeValidationError` when a tree does not match the structure it was
recognised as, `AmbiguousTreeError` when several trees hold hits and none was
named, and `TreeMergeError` when trees cannot be read as one dataset.

## Filtering And Selecting

The pandas view is where rows are picked out of a tree. The package names the
questions pandas has no name for, as functions and as a `gate` namespace
registered on a column and on a frame when the package is imported:

```python
frame = data.to_dataframe()

in_the_ring = frame.gate.in_cylinder((0, 0), radius=500.0, inner_radius=409.0)
energies = in_the_ring["edep"].gate.in_range(0.2, 0.511)
```

Every filter exists twice: `is_*` and `has_*` answer with a boolean column,
which combines with `&` and `|`, and the other name of the pair answers with
the rows themselves, which chains. Shapes (`in_box`, `in_sphere`,
`in_cylinder`) are described by where they sit and how big they are, `by_run`
and `by_event` name rows by the identifiers GATE wrote, and
`select_by_source_type`, `select_by_decay_type`, `select_by_gamma_type` and
`select_by_process` take the meaning of a code rather than its number. See
[Filtering And Selecting](filtering.md).

## PositroniumSource Branches

`SourceType`, `DecayType` and `GammaType` name the values of the three
branches a `PositroniumSource` fills, and their members are the integers GATE
wrote, so a column compares against them as it was read. `decode_positronium_value` and
`decode_positronium_column` read values as names, `positronium_enum` says which class
describes a branch, `has_positronium_metadata` reads `decayIndex` to tell rows
carrying the decay metadata of such a source from the rest, and
`POSITRONIUM_BRANCHES` maps each branch to the class describing it. See
[PositroniumSource Data](positronium.md).
