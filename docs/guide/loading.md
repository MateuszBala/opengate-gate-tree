# Loading Files

## What The Package Reads

A GATE output file holds more than trees. Alongside `Hits`, `Singles` and
`Coincidences` it typically stores histograms such as `latest_event_ID` and
`total_nb_primaries`, and trees outside the scope of this package, such as
`pet_data` or `OpticalData`.

**Only trees are read.** Objects that are not trees are ignored when the file
contents are inspected, and they are never copied into an output file.

## Inspecting A File

```python
from pathlib import Path

from opengate_gate_tree import GateTree, RootFile

with RootFile(Path("simulation.root")) as root_file:
    print(root_file.tree_names)
    print(root_file.has_tree(GateTree.HITS))
    print(root_file.branch_names(GateTree.HITS))
```

`tree_names` reports the trees the file holds, with the ROOT cycle suffix
(`Hits;1`) removed.

## How A Tree Is Found

The tree to read is resolved in four steps:

1. a name given by the caller is used as it is;
2. the keys of the file are matched exactly against the requested tree;
3. they are matched again without regard to case;
4. for hits, the trees are examined and one holding them is used.

The last step is what makes the `GateToTree` output readable: it calls its tree
`tree`, and nothing in the file is named `Hits`. It also covers a file that
stores hits per sensitive detector, where no tree carries the standard name
either.

```python
with RootFile(Path("simulation.root")) as root_file:
    print(root_file.hits_tree_names())
    data = root_file.read(GateTree.HITS, tree_name="Hits_run1")
```

When nothing matches, the error names the trees the file actually holds:

```text
TreeNotFoundError: Tree 'Singles' is not present in file: simulation.root.
Trees available in the file: ['pet_data', 'Hits', 'OpticalData'].
```

That list is the fastest way to see whether a simulation wrote the tree at all.

A file can hold hits in several trees, one per run or one per sensitive
detector. Reading one of them and saying nothing would hide the rest, so the
package either warns or refuses to choose. See [Merging Trees](merging.md).

## What Reading Checks

Reading the "Hits" tree recognises which structure it has and checks the tree
against it, before any data is loaded. A file whose structure the package does
not know is reported rather than half read, and can still be extracted with
`validate=False`. See [The "Hits" Tree](hits.md).

## Which Branch Types Are Supported

| Branch kind | Example | Supported |
| --- | --- | --- |
| scalar, numeric or boolean | `eventID`, `edep` | yes |
| text | `processName`, `comptVolName` | yes |
| fixed-width array | `volumeID`, shape `(entries, 10)` | yes |
| varying length per entry | — | no |

A branch whose length varies per entry has no representation in the supported
output formats, so it raises `UnsupportedBranchTypeError` naming the branches
in question.

The check runs on the branch type recorded in the file, before any data is
read. Text branches and branches of varying length both load as arrays of
Python objects and cannot be told apart afterwards, so an unsupported branch is
rejected without loading it into memory.

## Selecting Branches

An empty selection means every branch of the tree. A repeated name is read
once, at the position of its first occurrence, and an empty name is rejected.

```python
from opengate_gate_tree import read_tree

read_tree(path, GateTree.HITS)                      # every branch
read_tree(path, GateTree.HITS, ["edep", "eventID"])  # these two, in this order
```

Whether a branch exists is decided by the file being read, not by a list built
into the package. A missing branch raises `BranchNotFoundError` listing both
what was missing and what the tree holds.

## Memory

A tree is read into memory in full. For the file sizes GATE produces this is
usually fine, but a large simulation output can be several gigabytes; reading
one branch at a time keeps the footprint down:

```python
with RootFile(path) as root_file:
    energies = root_file.read(GateTree.HITS, ["edep"])
```

Reading in chunks is not available yet.
