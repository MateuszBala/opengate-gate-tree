# Statistics

A summary answers what a file holds before anything is done with it: how many
entries, how many events behind them, what range each branch covers, and which
process names appear and how often.

## From The Command Line

```bash
opengate-gate-tree \
	--input-gate-root-file ./data/simulation.root \
	--output-dir ./out \
	--output-file-title patient_01 \
	--gate-tree Hits \
	--output-file-format hdf5 \
	--statistics
```

The run writes `./out/patient_01.hits.hdf5` and, beside it,
`./out/patient_01.hits.stats.json`. The same summary goes to the log in a form
meant for reading.

## From Python

```python
from pathlib import Path

from opengate_gate_tree import (
    GateTree,
    RootFile,
    compute_statistics,
    format_statistics,
    write_statistics,
)

with RootFile(Path("simulation.root")) as root_file:
    detection = root_file.detect_hits_tree()
    data = root_file.read(GateTree.HITS)

statistics = compute_statistics(data, detection)
print(format_statistics(statistics))
write_statistics(statistics, Path("out/patient_01.hits.stats.json"))
```

Passing the recognised structure is optional; it adds the variant, the tree
name and the identifier scheme to the summary.

## What A Report Holds

```json
{
  "tree": "Hits",
  "entries": 1500,
  "structure": {
    "variant": "No system",
    "reference": "A1",
    "tree_name": "Hits",
    "branches": 40
  },
  "hits": {
    "event_key": ["runID", "eventID"],
    "events": 458,
    "runs": 3,
    "tracks": 2,
    "total_edep": 467.565,
    "time_min": 1.59e-06,
    "time_max": 0.0417,
    "source_trees": ["Hits", "Hits_run1", "Hits_run2"]
  },
  "branches": [
    {"name": "edep", "type": "float32", "kind": "float", "entries": 1500,
     "minimum": 0.0006, "maximum": 0.511, "mean": 0.3117, "std": 0.1706,
     "not_finite": 0},
    {"name": "processName", "type": "text", "kind": "text", "entries": 1500,
     "minimum": null, "maximum": null, "mean": null, "std": null,
     "distinct_values": 2,
     "most_frequent": [{"value": "PhotoElectric", "count": 915},
                       {"value": "Compton", "count": 585}]}
  ]
}
```

The numbers come from the file the package is tested against, read with
`read_hits_trees`: three runs merged into one dataset. Its 1500 entries carry
only 154 distinct event identifiers, and 458 distinct pairs of a run and an
event — which is what `events` counts.

The format is JSON because a report is meant to feed whatever comes next as
much as to be read: one run compared against another, a table of process
counts, a check that a file holds what a simulation was supposed to produce.

Three details are worth knowing:

- **events are counted by run and event identifier together**, and `event_key`
  names the branches that were used. A selection without `runID` counts by the
  event identifier alone and says so. See [Event Identifiers](events.md);
- **values that are not finite are counted, not averaged.** JSON can write
  neither `NaN` nor an infinity, so both are counted as `not_finite` and left
  out of the range, the mean and the deposited energy; a branch holding
  nothing else reports `null`;
- **the physics part is filled in from the branches that were extracted.** A
  selection without `edep` reports `null` for the deposited energy rather than
  failing.

The part describing branches is not specific to hits, so the other trees of a
GATE file will be summarised the same way once they are supported.
