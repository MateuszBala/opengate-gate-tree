# Exporting Files

## Choosing A Format

Output files are not read back by GATE. Pick the format the next step of the
analysis reads most comfortably:

| Format | Choose it when | Extension |
| --- | --- | --- |
| `root` | the analysis is C++ code using the ROOT framework | `.root` |
| `hdf5` | the analysis is Python or MATLAB, or the data is large | `.hdf5` |
| `csv` | the file is for inspection, a spreadsheet, or a quick `read_csv` | `.csv` |

```python
from opengate_gate_tree import OutputFileFormat, write_tree

write_tree(data, Path("out/hits.root"), OutputFileFormat.ROOT)
```

The parent directory is created when missing. An existing file is overwritten
without a prompt.

## How An Output File Is Named

A run of the command line names its output `<title>.<tree>.<format>`:
`--output-file-title patient_01 --gate-tree Hits --output-file-format csv`
writes `patient_01.hits.csv`. The tree is part of the name because one input
file holds several of them, and extracting two should not land on the same
file. A report, when one is asked for, sits next to the data as
`patient_01.hits.stats.json`.

Code using the package builds the same names without repeating the rule:

```python
from opengate_gate_tree import GateTree, OutputFileFormat, build_output_file_path

path = build_output_file_path(Path("out"), "patient_01", GateTree.HITS, OutputFileFormat.CSV)
```

## Branch Names A Format Cannot Carry

Two kinds of branch name are refused rather than written as something else,
because both backends accept them and then store something other than what was
asked for:

| Format | Refused name | What it would become |
| --- | --- | --- |
| `root` | holds `[` or `]`, such as `volumeID[0]` | uproot reads the bracket as an array dimension and writes branches that cannot be read back |
| `hdf5` | holds `/`, such as `cylindricalPET/gantryID` | h5py creates a nested group instead of a dataset |

The first affects the `GateToTree` layout, which splits `volumeID` into ten
branches named that way: it reaches `csv` and `hdf5` unchanged, but not `root`.
The second affects `GateToTree` output of a simulation using more than one
system, which prefixes identifier branches with the name of their system.

## Reading Your Own Output Back

An output file holding a **selection** of branches is no longer a whole hits
structure, so reading it back needs the structure check turned off:

```python
from opengate_gate_tree import GateTree, read_tree

subset = read_tree(Path("out/patient_01.hits.root"), GateTree.HITS, validate=False)
```

A merged dataset written out and read back is a different case: the added
`sourceTreeName` column is reported as one the structure does not describe,
which is a warning, so `read_tree` reads the file as it is. Reading it through
`read_hits_trees` needs `add_source_branch=False`, because recording where the
rows came from would otherwise overwrite the column that already says it.

The output file holds the extracted tree only. Histograms stored next to the
trees in the input file are not copied over.

## What Each Format Preserves

The formats differ in one place that matters: fixed-width array branches such
as `volumeID`.

| | `root` | `hdf5` | `csv` |
| --- | --- | --- | --- |
| scalar branches | kept | kept | kept |
| text branches | kept | kept | kept |
| array branch `(N, 10)` | kept as an array | kept as an array | expanded into 10 columns |
| numeric data types | kept | kept | re-inferred on read |

CSV has no cell for an array, so `volumeID` becomes `volumeID_0` through
`volumeID_9`. Reading that file back gives ten scalar columns, not one array.
Use `root` or `hdf5` when the branch structure has to survive.

## ROOT Output

The tree is written as a `TTree` named after the tree it came from, so ordinary
analysis code reads it:

```cpp
#include "TFile.h"
#include "TTree.h"

TFile file("out/hits.root");
auto* tree = file.Get<TTree>("Hits");

Int_t eventID;
Float_t edep;
tree->SetBranchAddress("eventID", &eventID);
tree->SetBranchAddress("edep", &edep);

for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
    tree->GetEntry(i);
    // ...
}
```

`RDataFrame` and `TTree::Draw` work on it as well.

A tree without branches cannot be expressed in ROOT and raises `ExportError`. A
tree with branches but no entries is written normally, with every branch
declared and no data.

## HDF5 Output

The tree becomes one group holding one dataset per branch:

```text
/Hits
  ├── eventID        (entries,)      int32
  ├── edep           (entries,)      float32
  ├── processName    (entries,)      variable-length UTF-8 string
  ├── volumeID       (entries, 10)   int32
  └── attributes: gate_tree, entries, branches, package_version
```

The group records the branch order, which HDF5 would otherwise list
alphabetically.

```python
import h5py

with h5py.File("out/hits.hdf5") as stored:
    group = stored["Hits"]
    print(dict(group.attrs))
    energies = group["edep"][:]
    volume_ids = group["volumeID"][:]        # shape (entries, 10)
    processes = group["processName"].asstr()[:]
```

Text datasets are stored as variable-length UTF-8 strings, so read them through
`asstr()` to get Python strings rather than bytes.

## CSV Output

```python
import pandas as pd

frame = pd.read_csv("out/hits.csv")
```

The header carries the branch names in order, with array branches expanded. A
tree without entries produces a file holding only the header.
