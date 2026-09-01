# Filtering And Selecting

An extracted tree becomes a question only once rows are picked out of it: the
hits of one event, the ones inside the detector, the ones a prompt gamma left.
The package names those questions, on the pandas view of the data.

```python
from pathlib import Path

from opengate_gate_tree import GammaType, read_hits_trees

frame = read_hits_trees(Path("simulation.root")).to_dataframe()

in_the_scanner = frame.gate.in_cylinder((0, 0), radius=500.0, inner_radius=409.0)
prompt = in_the_scanner["gammaType"].gate.is_gamma_type(GammaType.PROMPT)
energies = in_the_scanner["edep"][prompt]
```

## What Is Added, And What Is Not

Everything here works on a `pandas.Series` or a `pandas.DataFrame`, and pandas
already answers most questions about a column. Comparing, `isin`, combining
masks with `&` and `|` — none of that is restated:

```python
frame[frame["PDGEncoding"] == 22]        # gammas
frame[frame["trackID"].isin([1, 2])]     # two tracks
```

A filter earns its place by naming something the **data** means: a closed
range, a shape in the geometry of the scanner, the identity of an event, the
meaning of a code GATE wrote. Anything a line of pandas already says is left to
pandas.

The range is the one place where a call of pandas is wrapped rather than left
alone: `is_in_range` is `Series.between` under another name. It is here because
it comes in a pair with `in_range`, like every other filter, and because the
shapes are built out of it — a box is three ranges, and the ends of a cylinder
are a fourth.

## Every Filter Comes In Two

The same question can be asked in two ways, and both are needed, so every
filter exists twice under two names:

| Name | Answers with | What it is for |
| --- | --- | --- |
| `is_*`, `has_*` | a boolean column | combining conditions, indexing another column |
| `in_*`, `by_*`, `with_*`, `select_by_*` | what was asked about, narrowed | chaining one selection onto the next |

The mask half is one word wherever it appears. The other half is named after
how the question reads — `in_sphere`, `by_run`, `with_decay_metadata`,
`select_by_process` — so the four prefixes are worth knowing:

| Prefix | Asks about | Example |
| --- | --- | --- |
| `in_` | a range or a shape | `in_cylinder` |
| `by_` | an identifier | `by_event` |
| `with_` | metadata a row carries | `with_decay_metadata` |
| `select_by_` | the meaning of a code | `select_by_gamma_type` |

```python
inside = frame.gate.is_in_sphere((0, 0, 0), 500.0)     # a mask
rows = frame.gate.in_sphere((0, 0, 0), 500.0)          # the rows
```

The mask is the more useful of the two whenever a question has several parts.
Build the conditions, combine them, and cut the rows once:

```python
energetic = frame["edep"].gate.is_in_range(0.2, 0.511)
first_slice = frame["time"].gate.is_in_range(0.0, 0.02)
selected = frame[energetic & first_slice]
```

Numbers are read in the units of the file, which are the ones GATE wrote:
millimetres, seconds and MeV. Nothing is converted anywhere in the package.

A mask keeps the length and the index of what it was built from, so masks of
one frame always combine. Rows returned by a selection keep the index they had,
which is what lets a hit be traced back to the entry it came from.

## Ranges

```python
frame["edep"].gate.is_in_range(0.2, 0.511)                      # a mask
frame["edep"].gate.in_range(0.2, 0.511)                         # the values
frame["edep"].gate.in_range(0.2, 0.511, inclusive="neither")    # ends left out
```

Both ends belong to the range unless `inclusive` says otherwise. The vocabulary
is the one of `pandas.Series.between` — `"both"`, `"neither"`, `"left"`,
`"right"` — so there is no second convention to remember, and neither is there
a second behaviour: a missing value falls outside the range, and a column that
cannot be compared against the ends at all, such as text against numbers,
raises the `TypeError` pandas raises rather than answering.

## Shapes

A hit has a position, and a question about a detector is a question about a
shape. Three shapes are described the same way: by where they sit and how big
they are.

| Filter | Described by | Reads |
| --- | --- | --- |
| `is_in_box` / `in_box` | centre, side lengths | three columns |
| `is_in_sphere` / `in_sphere` | centre, radius | three columns |
| `is_in_cylinder` / `in_cylinder` | centre of the axis, radius, ends | three columns |

Each side of a box reaches half its length either way from the centre, which is
how a volume is described in a GATE macro as well:

```python
frame.gate.in_box((0, 0, 0), (3.2, 20.0, 3.2))   # one crystal at the origin
frame.gate.in_box((0, 0, 0), 100)                # a cube of side 100
```

A sphere and a cylinder take a radius. A cylinder takes the ends of its axis
too, and an inner radius that turns it into a ring, which is how a layer of a
scanner is usually asked for:

```python
frame.gate.in_cylinder(
    (0, 0),                       # where the axis crosses the xy plane
    radius=500.0,                 # outer radius, in the units of the file
    z_range=(-530.0, 530.0),      # the ends, unbounded when omitted
    inner_radius=409.0,           # what makes it a ring
)
```

The cylinder runs along the **third** of the columns it reads. Another axis is
a matter of naming the columns in another order rather than of another
parameter:

```python
frame.gate.in_cylinder((0, 0), 500.0, columns=("posX", "posZ", "posY"))
```

### Note

Surfaces belong to their shape: a hit sitting exactly on a face, on a sphere or
on the wall of a ring counts as inside. The other convention would drop hits on
a boundary, and a simulation puts them there — a crystal is where energy is
deposited, and its surface is where a gamma enters it.

### Which Columns A Shape Reads

A "Hits" tree carries three positions per entry: where the hit happened, where
it happened inside its volume, and where the gamma was born. The shapes read
the first of them unless told otherwise:

| Columns | What they hold |
| --- | --- |
| `posX`, `posY`, `posZ` | where the hit happened, in the frame of the world |
| `localPosX`, `localPosY`, `localPosZ` | the same, in the frame of the volume |
| `sourcePosX`, `sourcePosY`, `sourcePosZ` | where the gamma was born |

```python
from opengate_gate_tree import POSITION_COLUMNS

POSITION_COLUMNS      # ('posX', 'posY', 'posZ'), the default
frame.gate.in_sphere((0, 0, 0), 25.0, columns=("sourcePosX", "sourcePosY", "sourcePosZ"))
```

A shape always reads three columns. The centre gives one value per column it is
centred in, in the same order — three for a box and a sphere, two for a
cylinder, whose third column is its axis. Anything else raises `ValueError`,
and so does a radius, a side or a coordinate that is not a finite number: a
shape built from `nan` would answer "no rows", which is the answer hardest to
tell from a real one. A column the frame does not hold raises `KeyError`.

## Runs And Events

```python
frame.gate.by_run(2)                # the rows of a run
frame.gate.by_event(2, 5)           # the rows of one event of that run
frame.gate.is_from_event(2, 5)      # the same, as a mask
```

An event is named by **both** identifiers. GATE numbers events within a run, so
a file holding three runs holds an event 5 in each of them, and the three are
different decays. That is why there is no filter taking the event identifier
alone: `frame["eventID"] == 5` is one comparison, and it should look like the
guess it is. See [Event Identifiers](events.md).

## What A Gamma Was

The branches a `PositroniumSource` writes hold integers, and the selectors take
the members that name them:

```python
from opengate_gate_tree import DecayType, GammaType, SourceType

frame["sourceType"].gate.is_source_type(SourceType.ORTHO_POSITRONIUM)
frame["decayType"].gate.select_by_decay_type(DecayType.DEEXCITATION)
frame["gammaType"].gate.is_gamma_type(GammaType.PROMPT, GammaType.ANNIHILATION)
```

Several members can be given at once, and the answer covers all of them. Giving
none raises `ValueError`: an empty selection is a call that means nothing.

Rows carrying the decay metadata of such a source are told from the rest by
`decayIndex`, which the frame answers about directly:

```python
frame.gate.with_decay_metadata()      # the rows
frame.gate.has_decay_metadata()       # the mask
```

The answer is narrower than "written by a `PositroniumSource`": such a source
numbers every gamma it emits, including the ones from a direct annihilation
component. What a gamma itself was is said by `sourceType`. See
[PositroniumSource Data](positronium.md).

Both read `decayIndex` as the whole numbers GATE writes there, and raise
`ValueError` when the column holds something else. A frame does not have to
come straight out of `to_dataframe()`: a round trip through CSV, or a
concatenation that introduced a missing value, turns the column into floating
point numbers, where the value standing for "no metadata" can no longer be
compared for.

### Warning

Each selector takes the members of its own class. The classes share their
numbers — a source type of 2 is a para-positronium and a gamma type of 2 is an
annihilation gamma — so a member of the wrong class would select the right rows
for the wrong reason, or the wrong rows outright. Passing one raises
`ValueError` naming what was passed.

### Note

These four selectors are asked about **one column**, so `select_by_*` answers
with the values of that column — a `decayType` column holding nothing but
`DEEXCITATION`, which is what to count or to run `value_counts()` on. Rows are
selected through the mask, which is why the chains on this page use `is_*` and
index the frame with it:

```python
deexcitation = frame["decayType"].gate.is_decay_type(DecayType.DEEXCITATION)
rows = frame[deexcitation]
```

## The Process That Made A Hit

`processName` is text, so its selector takes names as GATE writes them:

```python
frame["processName"].gate.is_process("Compton")
frame["processName"].gate.select_by_process("Compton", "PhotoElectric")
```

The names are not checked against a list. GATE builds differ in the physics
lists they were compiled with, and a name this package had never heard of would
be refused for no reason.

## The `gate` Namespace

Importing `opengate_gate_tree` registers a `gate` accessor on `pandas.Series`
and on `pandas.DataFrame`. Which of the two carries a filter follows from what
the filter needs to know: a range is a question about one column, a shape about
three of them at once, and the identity of an event about two.

```python
import opengate_gate_tree   # registering the namespace is what the import does

frame.gate.by_run(0)                     # frame in, frame out
frame["edep"].gate.in_range(0.2, 0.4)    # column in, column out
```

Because each answers with what it was given, selections chain:

```python
energies = (
    frame.gate.by_event(1, 5)
    .gate.in_sphere((0, 0, 0), 500.0)["edep"]
    .gate.in_range(0.0, 0.511)
)
```

Every method calls the function of the same name and adds nothing, so the same
work reads either way. The functions are exported from the package root, and
are what to reach for when a filter is passed around rather than called:

```python
from opengate_gate_tree import in_range, in_sphere

in_sphere(frame, (0, 0, 0), 500.0)
in_range(frame["edep"], 0.2, 0.4)
```

### Note

`gate` is registered on classes the whole process shares. Should the name
already be taken, pandas says so with a warning, which the package does not
silence: it reports a real collision in somebody's code.

## Filtering And What Follows

A selection is a `pandas.DataFrame`, and everything downstream of it takes one:

```python
from opengate_gate_tree import GateTree, OutputFileFormat, TreeData, write_tree

selected = frame.gate.in_cylinder((0, 0), 500.0, inner_radius=409.0)
data = TreeData.from_dataframe(GateTree.HITS, selected)
write_tree(data, Path("out/in-the-ring.hdf5"), OutputFileFormat.HDF5)
```

A selection is also where a computation starts: the vectors of the rows that
were kept are read from the frame the filter answered with, and they carry its
index. See [Vectors And Angles](vectors.md).

```python
selected = frame.gate.in_cylinder((0, 0), 500.0, inner_radius=409.0)
angles = selected.gate.position().angle_to(selected.gate.momentum_direction())
```

`from_dataframe` drops the index, so a written file holds the rows that were
selected and nothing about which entries they had been. Identifiers are what
carries that, and they are written as GATE wrote them.

### Warning

A statistics report computed after a selection describes the selection, not the
file. That is usually the point — but `runs` and `events` counted on rows that
a filter has already removed say how much of the data survived it, not how much
the simulation produced.
