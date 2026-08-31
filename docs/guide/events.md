# Event Identifiers

An entry of the "Hits" tree is one energy deposit. Several of them belong to
one track, several tracks to one event, and several events to one run. Which
entries belong together is written in the identifier branches, and the package
passes them on exactly as GATE wrote them.

## An Event Is A Run And An Event Together

GATE numbers events within a run. In a file holding three runs, event 5 exists
three times, and the three are different decays:

```python
from pathlib import Path

import numpy as np

from opengate_gate_tree import read_hits_trees

data = read_hits_trees(Path("simulation.root"))

by_event = len(np.unique(data["eventID"]))
by_pair = len(np.unique(np.stack([data["runID"], data["eventID"]], axis=1), axis=0))
print(by_event, by_pair)
```

On the file the package is tested against, that prints far fewer events by
identifier alone than by the pair. **An event is the pair of `runID` and
`eventID`**, and every selection, grouping or count has to use both.

The reverse case matters as much. When a simulation writes one tree per
sensitive detector, one decay is recorded in both of them under the same run
and the same event identifier. Counting by the pair keeps it as one event,
which is what it is.

The same holds one level down: GATE numbers tracks within an event, so every
event has a track 1, and a track is named by the run, the event and the track
identifier together.

The report a run can write names the branches it counted by, so a summary of a
selection that left `runID` out says so instead of quietly counting too few
events:

```json
"hits": {"event_key": ["runID", "eventID"], "events": 458, "runs": 3,
         "track_key": ["runID", "eventID", "trackID"], "tracks": 915}
```

## Why Identifiers Are Never Renumbered

Giving every event a number of its own would make `eventID` unique, and the
package deliberately does not do it.

The reason is what the identifiers are for. A simulation being debugged is run
twice with the same seed, and the two outputs are compared to see what a change
did to one suspicious event. A number the package derives from the contents of
a file would differ between those two runs as soon as anything else about them
differs, and the comparison the work is being done for would stop working.

Two more things follow from the same decision:

- an identifier ties a row back to the GATE output it came from, and to the
  logs of the simulation that produced it;
- the "Singles" and "Coincidences" trees carry identifiers from the same space,
  so joining them with the hits works without translating anything.

## Telling Rows Apart After A Merge

Reading a split file as one dataset adds a column naming the tree each row came
from, because the identifiers alone do not always answer that question. See
[Merging Trees](merging.md).
