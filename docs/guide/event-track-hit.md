# Events, Tracks And Hits

A row of the "Hits" tree is one hit. An analysis usually asks about something
larger: an event and the gammas in it, the second interaction of a gamma, the
events in which both annihilation photons scattered once and were absorbed.
`EventTrackHit` reads the tree that way - as events, the tracks recorded in
them, and the hits every track left - without copying it.

```python
from pathlib import Path

from opengate_gate_tree import EventTrackHit, GammaType, GateTree, Process, read_tree

representation = EventTrackHit(read_tree(Path("simulation.root"), GateTree.HITS))

annihilation = representation.with_gamma_type(GammaType.ANNIHILATION).with_track_count(2)
first_interactions = annihilation.first_hits().hits_frame(["edep", "processName"])
```

The notebook
[`tutorial-18-events-tracks-and-hits.ipynb`](https://github.com/MateuszBala/opengate-gate-tree/blob/main/tutorials/notebooks/tutorial-18-events-tracks-and-hits.ipynb)
goes through everything on this page on a real GATE output, with the answers
kept in it.

## What The Representation Holds

Building a representation reads four branches - `runID`, `eventID`, `trackID`
and `time` - and groups the rows once. It keeps a reference to the columns and
an index of which rows make up which hit, track and event; nothing else is
read until it is asked for. What is known from the start is what the grouping
says: how many tracks every event has and how many hits every track has.

```python
representation.event_count, representation.track_count, representation.hit_count
representation.track_counts()     # tracks of every event
representation.hit_counts()       # hits of every track
```

The data can be the columns a reader returns, or a pandas frame of them -
including a frame already narrowed by the filters of the `gate` namespace. The
rows of such a frame keep their labels, so a hit always leads back to the row
it was read from.

### How it is stored

The index holds the row of every hit, in order, and where the hits of every
track and the tracks of every event begin - the layout of a sparse matrix:

```
hit_rows       row of the data of every hit
track_offsets  hits of track k:  hit_rows[track_offsets[k]:track_offsets[k + 1]]
event_offsets  tracks of event e: event_offsets[e]:event_offsets[e + 1]
track_rows     row every track reads its own values from
event_rows     row every event reads its own values from
```

The hits of a track and the tracks of an event are neighbours, so a question
asked of every track at once - how many hits it has, whether the first is a
Compton scattering - is one vectorised reduction over consecutive segments.
No question loops over events.

## The Order Everything Is Counted In

| Level | Order | Why |
| --- | --- | --- |
| events | by `runID`, then `eventID` | an event is the pair of both |
| tracks in an event | by `trackID` | track 1 is the first primary particle the source emitted |
| hits in a track | by `time`; rows of the same time in the order of the data | the first hit is the first interaction |

A time that is not a number comes after every other hit of its track.

GATE writes the hits of a track one after another, in the order they happened,
but the tracks of an event **from the last one to the first**: an event of a
PositroniumSource with a prompt gamma is written as tracks 4, 3, 2, 1. The
representation puts them back in the order of their identifiers. For that
source, track 1 is the prompt gamma.

Positions an analysis speaks of - the first track, the second hit - are counted
**from 1**, the way they are said. The `events` sequence is indexed from 0, as
any Python sequence is; the two are different things and both are named so.

### Hits merged from several trees

A file split into one tree per sensitive detector records one track in both
trees, so after merging its rows are no longer neighbours. The representation
notices that and sorts every row instead of only the blocks of rows a track
occupies. The result is the same index either way.

## Fields

A field names a value together with the level it describes. Its value is the
name of the branch it is read from, so every table built from fields keeps the
column names GATE wrote:

| Level | Fields |
| --- | --- |
| event | `RUN_ID`, `EVENT_ID`, `SOURCE_TYPE`, `TRACK_COUNT` |
| track | `TRACK_ID`, `PARENT_ID`, `PDG_ENCODING`, `GAMMA_TYPE`, `DECAY_TYPE`, `DECAY_INDEX`, `SOURCE_POS_X/Y/Z`, `SOURCE_ENERGY`, `HIT_COUNT`, `TRACK_NUMBER` |
| hit | `EDEP`, `POS_X/Y/Z`, `LOCAL_POS_X/Y/Z`, `MOM_DIR_X/Y/Z`, `TIME`, `TRACK_LOCAL_TIME`, `STEP_LENGTH`, `TRACK_LENGTH`, `PROCESS_NAME`, `N_PHANTOM_COMPTON`, `N_CRYSTAL_COMPTON`, `N_PHANTOM_RAYLEIGH`, `N_CRYSTAL_RAYLEIGH`, `N_INTERACTIONS`, `VOLUME_ID`, `HIT_NUMBER` |

`TRACK_COUNT`, `HIT_COUNT`, `TRACK_NUMBER` and `HIT_NUMBER` are given by the
grouping rather than read from a branch. Any branch a field does not name is
read by its name, as a hit field.

A track or an event field is read from the first row of its track or event.
GATE writes the same value on every row of them; `check_consistency()` says
where data assembled by other means does not:

```python
report = representation.check_consistency()
report.is_consistent, report.inconsistent
```

### What the interaction counters count

`nCrystalCompton`, `nPhantomCompton` and `nInteractions` **include the hit
they are written on**: the first Compton scattering of a gamma already carries
`nCrystalCompton = 1`.

## Reading The Data

Every value is read on request, in one of three shapes.

### Arrays

`values(field)` gives one value per element of the level of the field, in the
order of the index. Asked at a lower level, a value is spread over it: an event
field read at the hit level gives every hit the value of its event. A hit field
asked at the track level is refused - which of the hits would it be?

```python
from opengate_gate_tree import ElementLevel, EventTrackHitField

gamma_types = representation.values(EventTrackHitField.GAMMA_TYPE)
event_of_every_hit = representation.values("eventID", ElementLevel.HIT)
```

### Tables

```python
events = representation.events_frame()      # runID, eventID, trackCount
tracks = representation.tracks_frame()      # eventIndex, trackNumber, runID, eventID, trackID, hitCount
hits = representation.hits_frame()          # eventIndex, trackIndex, trackNumber, hitNumber, every branch
```

The hits table is labelled with the rows of the data, so `hits.index` leads
back to the tree. Its columns are named after the branches, so the `gate`
namespace works on it: `hits.gate.position()` reads the positions of the hits.

### Objects

The events are also a sequence of `Event` objects, each holding its tracks,
each holding its hits:

```python
event = representation.events[0]
event.run_id, event.event_id, event.source_type
track = event.tracks[0]
track.track_id, track.gamma_type, track.source_position
hit = track.hits[0]
hit.edep, hit.position, hit.process_name, hit.row
```

The objects are views made when they are asked for: an `Event` is the
representation and a position, and reads a value when it is asked for one.
Building a representation of ten million hits creates no object at all. They
are there for looking at one event the way it is thought of; questions about
every event at once belong to the selectors and the tables.

## Selecting

Every question comes in two, as the [filters](filtering.md) of the pandas
view do. A method named `is_*` answers with a mask of the level it asks about,
exactly as there. The other half answers with a new representation of what was
kept, sharing the columns of the one it was made on, and is named after what it
keeps rather than after the kind of question: `with_*` for the elements that
carry something, `nth_*`, `first_hits` and `last_hits` for a position, and
`select_*` for a mask of your own. Where the pandas view says
`select_by_gamma_type`, the representation says `with_gamma_type`.

| Asks about | Mask | Selector |
| --- | --- | --- |
| events by the model of the source | `is_source_type` | `with_source_type` |
| events by their number of tracks | `is_track_count` | `with_track_count` |
| tracks by the kind of gamma | `is_gamma_type` | `with_gamma_type` |
| tracks by the decay channel | `is_decay_type` | `with_decay_type` |
| tracks by the particle | `is_particle` | `with_particle` |
| tracks by their number of hits | `is_hit_count` | `with_hit_count` |
| the track at a position of its event | `is_nth_track` | `nth_track` |
| hits by the process | `is_process` | `with_process` |
| the hit at a position of its track | `is_nth_hit` | `nth_hit`, `first_hits`, `last_hits` |
| any field, by a value, values or a range | `is_value` | `with_value` |
| a mask of your own | - | `select_events`, `select_tracks`, `select_hits` |

A count takes `eq`, or any combination of `ne`, `ge`, `le`, `gt` and `lt`.
A position counts from 1, and from -1 backwards. `Process` names the processes
of the reference outputs - `COMPTON`, `PHOTOELECTRIC` and `RAYLEIGH` - and is a
`StrEnum`, so a member is the string GATE writes; any other name is taken as a
string.

A numpy mask is read by position, in the order of the elements. A pandas
column is read by its labels: a column of the events or tracks table selects
the elements of its rows, and a column of hits - from `hits_frame()` or from
the frame the representation was built from - selects the hits of its rows,
whatever order those rows are in. A column that does not label every element
is refused.

```python
events = representation.events_frame()
three_tracks = representation.select_events(events["trackCount"] == 3)
large = representation.select_hits(frame["edep"] > 0.4)     # frame: the source
```

### What is left after a selection

A selection that leaves out every hit of a track leaves out the track, and one
that leaves out every track of an event leaves out the event. `keep_empty=True`
keeps them: a track without hits is still the track it was, with its
identifier and its kind of gamma, because every track keeps the row it reads
its own values from. What one selection kept empty, the next ones keep too -
each selection removes only what it empties itself - so an event left without
tracks can still be counted as one with `with_track_count(0)`.

### The order of selections is part of the question

A count asked for after a selection counts what the selection kept. These are
two different questions about the oPs decays with a prompt gamma:

```python
# events with three annihilation gammas, the prompt one beside them
representation.with_gamma_type(GammaType.ANNIHILATION).with_track_count(3)
# events of exactly three gammas, of which the annihilation ones are kept
representation.with_track_count(3).with_gamma_type(GammaType.ANNIHILATION)
```

On the reference oPs scene, the first finds every full event and the second
none: every event there has four gammas. The same goes for positions - the
first hit after keeping the photoelectric hits is the first photoelectric hit.

### Questions about a part of every track

A condition on one hit deciding about its whole track - "the first hit is a
Compton scattering" - is a mask of hits reduced to the tracks. The reductions
answer such questions for every track or every event at once:

```python
import numpy as np

from opengate_gate_tree import position_in_track, reduce_hits_to_tracks

every_hit = np.ones(representation.hit_count, dtype=bool)
first = position_in_track(representation.index, every_hit) == 1
starts_with_compton = reduce_hits_to_tracks(
    representation.index, first & representation.is_process(Process.COMPTON), "any"
)
representation.select_tracks(starts_with_compton)
```

`reduce_hits_to_tracks` and `reduce_tracks_to_events` answer `"any"`, `"all"`,
`"count"`, `"sum"`, `"min"`, `"max"`, `"first"` and `"last"`;
`broadcast_tracks_to_hits` and `broadcast_events_to_tracks` go the other way.
Sums of floating point values are computed in `float64` and of whole numbers in
`int64`. `"first"` and `"last"` also take a row of components per hit, such as
`volumeID`. An empty track has no smallest, largest, first or last hit, and
those four reductions refuse it.

## Writing And Reading Back

A representation is written as a flat tree of its hits, in its own order, with
five columns saying where every hit stands:

| Column | Type | Meaning |
| --- | --- | --- |
| `eventIndex` | int64 | the event of the hit, from 0 |
| `trackNumber` | int32 | the track in its event, from 1 |
| `hitNumber` | int32 | the hit in its track, from 1 |
| `trackCount` | int32 | tracks of the event |
| `hitCount` | int32 | hits of the track |

```python
from opengate_gate_tree import (
    OutputFileFormat,
    build_event_track_hit_file_path,
    read_event_track_hit,
    write_event_track_hit,
)

path = build_event_track_hit_file_path(Path("out"), "scene", OutputFileFormat.ROOT)
write_event_track_hit(annihilation, path, OutputFileFormat.ROOT)   # out/scene.eth.root
again = read_event_track_hit(path)
```

Every output format takes it, and a reader that knows nothing of this package
still sees which rows make one track and one event. The identifiers GATE wrote
are kept as they were; only the order of the rows and the added columns say
something new.

Reading a file back groups its rows again. A written file stands in the order
of its events, tracks and times, so a file whose rows would be put in another
order - an identifier or a time changed by hand - is refused, and so is one
whose five columns disagree with the grouping, such as a file cut in the middle
of a track. Nothing checks the other values: an energy changed by hand is read
as it stands. A track left empty by a selection has no row to be written in,
so writing refuses it too.

CSV keeps no types and no arrays. Reading one back gives every branch the type
the "Hits" tree schema gives it and folds `volumeID_0` to `volumeID_9` back
into `volumeID`, so a ROOT, HDF5 or CSV file reads back the same.

The representation is part of the library only; the command line does not
write it.

## How Fast It Is

`scripts/benchmark_event_track_hit.py` measures the representation on
synthetic trees in the proportions of the B1 reference output. On an Intel
Core i7-1355U, Python 3.14 and numpy 2.5, with ten million hits in about 3.2
million events:

| Step | As GATE writes the rows | Rows shuffled |
| --- | --- | --- |
| grouping | 0.46 s | 4.3 s |
| `with_track_count(2).with_hit_count(2).first_hits()` | 0.67 s | 0.58 s |
| hits table of four fields | 0.45 s | 0.63 s |
| writing `.eth.hdf5` | 3.5 s | 4.1 s |

The index keeps 22 bytes per hit, and holds at most about 50 bytes per hit
while it is built - against roughly 200 bytes per hit of the "Hits" tree it
groups. One million hits take a tenth of these times.

## Not Detected: Events Cut Short

A file cut to a number of entries can end in the middle of an event: the last
event of such a file has fewer tracks than it had in the simulation. Nothing in
the rows says so, and the representation does not guess. The fixtures of the
package are cut this way, and their last event is counted as it is.
