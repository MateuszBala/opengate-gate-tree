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
