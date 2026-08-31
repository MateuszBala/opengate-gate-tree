# Data Representations

An extracted tree is held as a {class}`~opengate_gate_tree.tree.treedata.TreeData`.
It stores NumPy arrays, which every output format can be built from, and offers
a `pandas.DataFrame` view for analysis.

## The NumPy View

```python
data = read_tree(path, GateTree.HITS)

data.entry_count            # 54956
data.branch_names           # ('PDGEncoding', 'trackID', ...)
data.dtypes["edep"]         # dtype('float32')
data.array_branches         # {'volumeID': 10}
data["edep"]                # numpy.ndarray of shape (54956,)
len(data)                   # same as entry_count
```

Two kinds of column exist:

- **scalar branches**, one-dimensional arrays of shape `(entries,)`
- **fixed-width array branches**, two-dimensional arrays of shape
  `(entries, width)`

`volumeID` is the array branch GATE writes for hits; it carries the detector
identifier hierarchy, so it is kept as an array rather than flattened.

`array_branches` maps each array branch to its width, which is the quickest way
to tell the two kinds apart.

## Selecting Branches

`select` returns a new instance sharing the same arrays:

```python
reduced = data.select(["edep", "eventID"])
reduced.branch_names        # ('edep', 'eventID')
```

The order is the one the caller asked for. A repeated name is kept once, at the
position of its first occurrence. An unknown name raises `BranchNotFoundError`.

## Immutability And Memory

`TreeData` is frozen and its column mapping is read-only, so branches cannot be
added or replaced after construction.

The arrays themselves are referenced rather than copied. Building a `TreeData`,
or calling `select`, does not duplicate the data. The arrays stay writable, so
modifying one in place also changes what any other reference to it sees.

Equality is deliberately not defined. Comparing NumPy arrays with `==` yields
arrays rather than booleans, so a generated comparison would raise instead of
answering. Compare the fields that matter instead:

```python
import numpy as np

np.array_equal(first["edep"], second["edep"])
```

## The Data Frame View

```python
frame = data.to_dataframe()
```

The result is a plain `pandas.DataFrame`, not a subclass, so every pandas
feature works on it unchanged.

A data frame holds scalar cells, so a fixed-width array branch is expanded into
one column per component, named `<branch>_<index>` with indices counted from
zero, placed where the original branch stood:

| `TreeData` | `to_dataframe()` |
| --- | --- |
| `eventID`, shape `(N,)` | `eventID` |
| `volumeID`, shape `(N, 10)` | `volumeID_0` … `volumeID_9` |

The expansion is one way. Reading such a frame back keeps the columns separate:

```python
restored = TreeData.from_dataframe(GateTree.HITS, frame)
restored.array_branches     # {} — volumeID_0 ... volumeID_9 are scalar branches
```

`from_dataframe` drops the index and rejects data frames with `MultiIndex`
columns or repeated column labels. A repeated label makes pandas return a data
frame rather than a series, which would silently turn two scalar branches into
one array branch.

## Building Data Yourself

`TreeData` can be built directly, which is useful for tests and for writing
derived results:

```python
import numpy as np

from opengate_gate_tree import GateTree, TreeData

data = TreeData(
    GateTree.HITS,
    {
        "eventID": np.arange(3, dtype=np.int32),
        "edep": np.array([0.1, 0.5, 1.2], dtype=np.float32),
    },
)
```

Construction checks that branch names carry a value, that every column is a
NumPy array of one or two dimensions, and that all columns agree on the number
of entries. Anything else raises `ValueError`.

## Values That Stand For Something

Some branches hold whole numbers that are not quantities: the four branches a
`PositroniumSource` writes say what a gamma was and where it came from. All
four stay integer columns; three of them — `sourceType`, `decayType` and
`gammaType` — have an enum class describing what their values mean, whose
members are those integers. The fourth, `decayIndex`, holds component numbers
whose meaning depends on how the source was configured, so it has none:

```python
from opengate_gate_tree import GammaType

prompt = data["gammaType"] == GammaType.PROMPT
```

Nothing is converted, and nothing is copied: the comparison runs on the column
as it was read. See [PositroniumSource Data](positronium.md).
