# Merging Trees

GATE can write the hits of one simulation into several trees of one file: one
per run, or one per sensitive detector. Each of them is a whole tree of the
same structure.

## Reading A Split File

```python
from pathlib import Path

from opengate_gate_tree import RootFile, read_hits_trees

with RootFile(Path("simulation.root")) as root_file:
    print(root_file.hits_tree_names())

data = read_hits_trees(Path("simulation.root"))
```

`hits_tree_names` reports the trees holding hits, whatever they are called: a
tree is recognised by its branches, not by its name. `read_hits_trees` reads
them all, in file order, as one dataset. On the command line that is
`--merge-hits-trees`.

One tree at a time is read by naming it:

```python
from opengate_gate_tree import GateTree, read_tree

first_run = read_tree(Path("simulation.root"), GateTree.HITS, tree_name="Hits")
```

On the command line that is `--input-tree-name Hits`.

## What The Package Does On Its Own

A file whose hits sit in a tree named `Hits`, next to `Hits_run1` and
`Hits_run2`, is read as that one tree and a warning names the others:

```text
File simulation.root holds hits in 2 more tree(s) besides 'Hits':
['Hits_run1', 'Hits_run2']. Only 'Hits' is read.
```

Reading one run out of three without a word would be the worst outcome
available, so the run says what it left out.

A file where no tree is named `Hits` — one per sensitive detector, for instance
— is not read at all until the caller decides:

```text
AmbiguousTreeError: File simulation.root holds hits in 2 trees:
['Hits_DET_INNER', 'Hits_DET_OUTER']. None of them is named 'Hits', so the one
to read cannot be chosen.
```

Choosing a detector is not something the package can do for an analysis.

## Where A Row Came From

A merged dataset carries an added text column, `sourceTreeName`, naming the
tree every row was read from:

```python
from opengate_gate_tree import SOURCE_TREE_BRANCH

print(set(data[SOURCE_TREE_BRANCH]))
```

It is not a convenience. In a file split per sensitive detector, both trees
hold the same run and share their event identifiers, so once the rows are
together, nothing else in the data says which detector recorded a deposit.

The column can be left out when the result has to match the structure exactly:

```python
data = read_hits_trees(Path("simulation.root"), add_source_branch=False)
```

Written to a file and read back, that column is reported as one the structure
does not describe, which is a warning rather than an error. Reading such a file
through `read_hits_trees` again needs `add_source_branch=False`: recording
where the rows came from a second time would overwrite the column that already
says it.

## What Merging Does Not Do

Merging is a concatenation, and nothing else:

- **rows are not sorted.** GATE writes hits in the order of the tracks within
  an event, so a file is not ordered by time to begin with, and sorting would
  restore no original order;
- **identifiers are not renumbered.** See [Event Identifiers](events.md);
- **rows sharing a run and an event are not collapsed.** That is one decay
  recorded in two detectors, which is the reason to merge in the first place.

Merging is refused when the trees do not hold the same structure: other
branches, the same branches in another order, or a branch stored with another
type or width. Concatenating an `int32` column with an `int64` one would
quietly rewrite both.
