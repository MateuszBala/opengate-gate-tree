# Analysis Of Events, Tracks And Hits

An analysis asks for the events in which something happened to their tracks
and hits, and for some of their values once they are found. `EventTrackHitAnalysis`
writes such a question as a chain of operations, keeps it as a description,
and answers it only when it is given data:

```python
from pathlib import Path

from opengate_gate_tree import (
    DecayType,
    EventTrackHitAnalysis,
    EventTrackHitField,
    GammaType,
    GateTree,
    Process,
    SourceType,
    read_tree,
)

analysis = (
    EventTrackHitAnalysis()
    .tracks(2)
    .gamma_type(GammaType.ANNIHILATION)
    .decay_type(DecayType.STANDARD)
    .source_type(SourceType.PARA_POSITRONIUM)
    .hits(eq=2)
    .hit(1)
    .process(Process.COMPTON)
    .hit(2)
    .process(Process.PHOTOELECTRIC)
    .tracks(2)
)

data = read_tree(Path("simulation.root"), GateTree.HITS)
table = analysis.extract(data, [EventTrackHitField.EDEP, EventTrackHitField.TIME])
```

The analysis keeps the para-positronium decays whose two annihilation gammas
each scattered once and were then absorbed; the table has one row per such
event, with the energy and the time of all four hits. It works on the
representation of [Events, Tracks And Hits](event-track-hit.md), and follows
its order: tracks by their identifier, hits by their time.

Four notebooks go through this page on a real GATE output, with the answers
kept in them:
[an analysis as a description](https://github.com/MateuszBala/opengate-gate-tree/blob/main/tutorials/notebooks/tutorial-19-analysis-as-a-description.ipynb),
[the order of operations and positions](https://github.com/MateuszBala/opengate-gate-tree/blob/main/tutorials/notebooks/tutorial-20-order-of-operations-and-positions.ipynb),
[combining analyses](https://github.com/MateuszBala/opengate-gate-tree/blob/main/tutorials/notebooks/tutorial-21-combining-analyses.ipynb) and
[one row per event](https://github.com/MateuszBala/opengate-gate-tree/blob/main/tutorials/notebooks/tutorial-22-one-row-per-event.ipynb).

## A Description, Not A Computation

Building the chain reads nothing. `apply(source)` performs it and returns the
events, tracks and hits it kept, as an `EventTrackHit`; `extract(source,
variables)` performs it and returns the table. The source is a representation
or the data to build one from - the columns a reader returns, or a pandas
frame of them.

An analysis never changes: every call returns a new one. A part shared by two
analyses stays the same in both, so it can be written once:

```python
pairs = EventTrackHitAnalysis().tracks(2).gamma_type(GammaType.ANNIHILATION)
scattered = pairs.hits(ge=2)
absorbed_at_once = pairs.hits(1).process(Process.PHOTOELECTRIC)
```

An analysis prints as the chain it was written as, says its steps in words
with `describe()`, and survives `pickle`. Two analyses asking the same
question compare equal however they were written - `tracks(2)` and
`tracks(eq=2)`, a field by member or by name - but equality does not reorder
a combination: `a & b` and `b & a` keep the same, and are two objects.

## Where An Analysis Stands

An analysis stands at one level at a time - the events, the tracks or the hits
- and a condition is asked of the elements of that level. It starts at the
events.

| Call | Moves to | And first |
| --- | --- | --- |
| `events()` | the events | - |
| `tracks()` | the tracks of every event | - |
| `tracks(2)`, `tracks(ge=2)` | the tracks of every event | keeps the events with that number of tracks |
| `hits()` | the hits of every track | - |
| `hits(2)`, `hits(ge=2, lt=4)` | the hits of every track | keeps the tracks with that number of hits |
| `track(1)`, `track(-1)` | the track at that position of every event | - |
| `hit(1)`, `hit(-1)` | the hit at that position of every track | - |

A field of a level above the one the analysis stands at is asked of that
level: `tracks().source_type(...)` keeps the events of a source model. A field
of a level below has to be moved to first - a track has no single energy
deposit - and the analysis says so where the call is written.

Positions do not nest. `track(1).hit(1)` would read as the first hit of the
first track, but would stand at the first hit of every track, so moving to
the hits right after `track(i)` is refused. A count is allowed there:
`track(1).hits(2)` keeps the events whose first track has two hits, and then
stands at the hits of every track of those events.

## Conditions Are Filters

A condition keeps the elements that pass it and leaves out the rest:

| Condition | Asked of |
| --- | --- |
| `source_type(*types)` | events |
| `gamma_type(*types)`, `decay_type(*types)`, `particle(*pdg_codes)` | tracks |
| `process(*processes)`, `edep(**comparisons)`, `time(**comparisons)` | hits |
| `where(field, **comparisons)` | the level of the field |

The comparisons are `eq`, `ne`, `ge`, `le`, `gt` and `lt` against one value,
`isin` against several and `between` against a pair of ends, both of which
belong to the range unless `inclusive` says otherwise, as with the selectors
of the representation; a value has to pass all of them. Energies are in MeV
and times in s, as GATE writes them. A bound that would select by mistake - a
string given to `isin`, a range given backwards, a number that is not a
number, a count no number passes - is refused where it is written.

`where` also tests the counts and positions - `hitCount`, `trackCount`,
`hitNumber`, `trackNumber` - and counts them among what is kept so far, as
`hits(n)` and `hit(j)` do.

A track whose hits were all left out is left out with them, and an event whose
tracks were all left out is left out too. An analysis keeps no empty track or
event, also none that the representation it was given held empty.

### Positions decide about the parent

Standing at a position, a condition decides about the element holding it. A
track whose first hit is not a Compton scattering is left out with all its
hits, and so is a track that has no first hit:

```python
EventTrackHitAnalysis().hit(1).process(Process.COMPTON)
```

Positions count from 1 among the elements kept so far, and from -1 backwards:
after `hits().process(Process.COMPTON)`, `hit(2)` is the second Compton
scattering of a track, not its second hit.

## The Order Of Operations Is Part Of The Question

Every operation acts on what the ones before it kept. A count counts what is
left:

```python
# events with two annihilation gammas, whatever else they left
EventTrackHitAnalysis().tracks().gamma_type(GammaType.ANNIHILATION).tracks(2)
# events of two gammas, both annihilation gammas
EventTrackHitAnalysis().tracks(2).gamma_type(GammaType.ANNIHILATION).tracks(2)
```

That is also why the example at the top ends with `tracks(2)`. Without it, an
event in which only one of the two gammas scattered and was absorbed would
stay - with one track left. Conditions do not remember what was asked before
them; a count at the end asks for both gammas.

## Combining Analyses

```python
both = first & second        # what both keep
either = first | second      # what either keeps
after = first.then(second)   # second, performed on what first kept
```

`&` and `|` perform each analysis on its own, from the events, and combine
what they kept, level by level: events, tracks and hits. Both are associative
and commutative, so `a & b & c` means the same however it is grouped. After a
combination, the analysis stands at the events.

`then` is different: it performs the second analysis on what the first kept,
so a count or a position in the second counts what the first kept. The second
analysis starts at the events, as it does on its own: when the first ends
elsewhere, `then` puts `events()` between them, and the printed chain shows
it. `&` and `then` agree whenever no count or position of one side follows a
filter of the other, which is the usual case, and part ways otherwise:

```python
annihilation = EventTrackHitAnalysis().tracks().gamma_type(GammaType.ANNIHILATION)
three = EventTrackHitAnalysis().tracks(3)

annihilation & three       # events of three gammas, their annihilation gammas kept
annihilation.then(three)   # events of three annihilation gammas
```

Only analyses of the same kind combine; anything else is a `TypeError`.

## The Table

`extract(source, variables)` returns one row per event that passed. It starts
with `runID` and `eventID`, then gives every variable, in the order of the
list:

| Variable describes | Columns |
| --- | --- |
| events | one: `sourceType` |
| tracks | one per track: `t1_gammaType`, `t2_gammaType`… |
| hits | one per hit of every track: `t1h1_edep`, `t1h2_edep`, `t2h1_edep`… |

The columns run variable, track, hit: all the deposits, then all the times.
Their names end with the branch they are read from, so the columns of one hit
are read as a vector by the `gate` namespace, and every column keeps the type
of its branch:

```python
first_scattering = table.gate.vector("t1h1_posX", "t1h1_posY", "t1h1_posZ")
```

A variable of several components per hit, such as `volumeID`, gets a column per
component: `t1h1_volumeID_0`…. The variables are a list or a tuple, since their
order is that of the columns.

### How wide the table is

Events seldom keep the same number of tracks, nor tracks the same number of
hits, and a table has one width. It is the smallest number of tracks an event
of the result kept, and the smallest number of hits a track entering the
table kept, so the table has no empty cell. What does not fit is left out, and
the log says how much:

```text
A table of 2 track(s) per event and 1 hit(s) per track left out 81 track(s) of 62 event(s) ...
```

An analysis that ends with fixed counts, as the example at the top does, loses
nothing. `max_tracks` and `max_hits` narrow the table further; a width the
result does not allow is refused, since it would leave cells empty. When no
event passed, there is no width to take: `max_tracks` and `max_hits` give the
width of the empty table, so its columns are there whether anything passed or
not.

## Checked Against Analyses Made Outside The Package

An analysis written with this package is checked against the same analysis
made independently of it, on the same input, by the
[analysis consistency tests](../TESTING_CONVENTIONS.md#analysis-consistency-tests).
The two have to keep the same events and agree on every value.
