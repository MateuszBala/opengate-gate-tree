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
  "entries": 4441,
  "structure": {
    "variant": "System",
    "reference": "A2",
    "tree_name": "Hits",
    "branches": 46,
    "system_scheme": ["cylindricalPET", "OPET"],
    "system_depth": 6
  },
  "hits": {
    "event_key": ["runID", "eventID"],
    "events": 1980,
    "runs": 1,
    "tracks": 2,
    "total_edep": 120.374,
    "time_min": 1.59e-06,
    "time_max": 0.00208,
    "source_trees": ["Hits", "Hits_run1"]
  },
  "branches": [
    {"name": "edep", "type": "float32", "kind": "float", "entries": 4441,
     "minimum": 0.0, "maximum": 0.511, "mean": 0.271, "std": 0.18,
     "not_a_number": 0},
    {"name": "processName", "type": "text", "kind": "text", "entries": 4441,
     "minimum": null, "maximum": null, "mean": null, "std": null,
     "distinct_values": 2,
     "most_frequent": [{"value": "Compton", "count": 2401}]}
  ]
}
```

The format is JSON because a report is meant to feed whatever comes next as
much as to be read: one run compared against another, a table of process
counts, a check that a file holds what a simulation was supposed to produce.

Three details are worth knowing:

- **events are counted by run and event identifier together**, and `event_key`
  names the branches that were used. A selection without `runID` counts by the
  event identifier alone and says so. See [Event Identifiers](events.md);
- **values that are not a number are counted, not averaged.** JSON has no way
  to write `NaN`, so they appear as `not_a_number`, and a branch holding
  nothing else reports `null` for its range;
- **the physics part is filled in from the branches that were extracted.** A
  selection without `edep` reports `null` for the deposited energy rather than
  failing.

The part describing branches is not specific to hits, so the other trees of a
GATE file will be summarised the same way once they are supported.
