# PositroniumSource Data

A `PositroniumSource` writes four branches into the "Hits" tree saying where
each gamma came from: which model emitted it, through which decay channel,
what kind of gamma it is, and which channel of a configured mixture it belongs
to. GATE stores all four as integers.

The package gives those integers names, so that an analysis reads as what it
means:

```python
from pathlib import Path

from opengate_gate_tree import GammaType, GateTree, read_tree

data = read_tree(Path("simulation.root"), GateTree.HITS, ["gammaType", "edep"])
prompt_energies = data["edep"][data["gammaType"] == GammaType.PROMPT]
```

## The Three Classes

| Branch | Class | Meaning of the values |
| --- | --- | --- |
| `sourceType` | `SourceType` | which model emitted the gamma |
| `decayType` | `DecayType` | which decay channel it came through |
| `gammaType` | `GammaType` | what kind of gamma it is |

The classes are named after the branches, because a branch name is what you
work with when reading a file. They come from the enums of
`GateEmittedGammaInformation.hh`, and each class names its counterpart there:
`SourceType` is `SourceKind`, `DecayType` is `DecayModel`, `GammaType` is
`GammaKind`.

### `SourceType`

| Value | Member | GATE | What it means |
| --- | --- | --- | --- |
| 0 | `NOT_DEFINED` | `NotDefined` | the gamma carries no source information |
| 1 | `SINGLE_GAMMA_EMITTER` | `SingleGammaEmitter` | emitted alone, by the single gamma model |
| 2 | `PARA_POSITRONIUM` | `ParaPositronium` | from a pPs decay, two gammas |
| 3 | `ORTHO_POSITRONIUM` | `OrthoPositronium` | from an oPs decay, three gammas |
| 4 | `DIRECT_ANNIHILATION` | `DirectAnnihilation` | from a positron annihilating without forming positronium |

### `DecayType`

| Value | Member | GATE | What it means |
| --- | --- | --- | --- |
| 0 | `NONE` | `None` | no decay, or none recorded |
| 1 | `STANDARD` | `Standard` | the standard channel: pPs into two gammas, oPs into three |
| 2 | `DEEXCITATION` | `Deexcitation` | a channel with a prompt gamma from deexcitation |

### `GammaType`

| Value | Member | GATE | What it means |
| --- | --- | --- | --- |
| 0 | `UNKNOWN` | `Unknown` | the gamma carries no information about itself |
| 1 | `SINGLE` | `Single` | emitted alone, by the single gamma model |
| 2 | `ANNIHILATION` | `Annihilation` | a product of annihilation |
| 3 | `PROMPT` | `Prompt` | emitted during deexcitation |

## Comparing A Column Against A Member

The members **are** the integers GATE wrote, so a column compares against them
as it was read, with nothing in between:

```python
prompt = data["gammaType"] == GammaType.PROMPT          # NumPy column
frame = data.to_dataframe()
prompt_rows = frame[frame["gammaType"] == GammaType.PROMPT]   # pandas
```

```{warning}
An enum written by hand for the same purpose will not work, and will not say
so. A plain `enum.Enum` compared against a column yields a mask of `False`
rather than an error, because its members are not integers. That is why the
classes here derive from `enum.IntEnum`, and why they are worth importing
instead of retyping.
```

## Selecting By What A Gamma Was

A comparison covers one member; the selectors take as many as the question
needs, and each one takes the members of its own class:

```python
frame = data.to_dataframe()

annihilation = frame["gammaType"].gate.is_gamma_type(GammaType.ANNIHILATION)
positronium = frame["sourceType"].gate.select_by_source_type(
    SourceType.PARA_POSITRONIUM, SourceType.ORTHO_POSITRONIUM
)
from_positronium_rows = frame.gate.with_decay_metadata()
```

Passing a member of another class raises `ValueError`. The classes share their
numbers, so it would otherwise select by a value meaning something else. See
[Filtering And Selecting](filtering.md).

## Reading Values As Names

For a table meant for people, or a summary, the values can be read as names:

```python
from opengate_gate_tree import decode_positronium_column, decode_positronium_value

decode_positronium_value("gammaType", 3)                   # GammaType.PROMPT
meanings = decode_positronium_column("gammaType", data["gammaType"])
```

The two differ in what they do with a value the package does not know:

| Call | A value outside the class |
| --- | --- |
| `decode_positronium_value` | raises `ValueError`, naming the values GATE writes there |
| `decode_positronium_column` | leaves `None` in its place and reports it in the log, with how often it occurs |

A question about one value has one answer or none; reading an unknown value as
the one standing for "not defined" would report something the file does not
say. A column is different: a GATE build can write one value more, and an
analysis of the rows that are understood is still worth having.

## Which Rows Carry Decay Metadata

The fourth branch, `decayIndex`, holds the channel of the mixture a gamma came
from. Channel numbers depend on the order the fractions were configured in, so
they have no fixed meaning and no class of their own. One value does have one:

```python
from pathlib import Path

from opengate_gate_tree import (
    DECAY_INDEX_BRANCH,
    NO_POSITRONIUM_METADATA,
    GateTree,
    has_positronium_metadata,
    read_tree,
)

data = read_tree(Path("simulation.root"), GateTree.HITS, [DECAY_INDEX_BRANCH, "sourceType"])
carries_decay_metadata = has_positronium_metadata(data[DECAY_INDEX_BRANCH])
```

`NO_POSITRONIUM_METADATA`, the value `-1`, is what GATE writes when a row
carries no decay metadata at all: for a gamma from another kind of source, and
for a particle the metadata never reached. The function is named after what the
value guarantees rather than after what it usually means.

```{note}
A `PositroniumSource` writes that metadata for every gamma it emits, including
the ones from a direct annihilation component, since it numbers those like the
rest. So the mask holds `True` for gammas that formed no positronium. What a
gamma itself was is said by `sourceType`.
```

The deprecated `ExtendedVSource` never fills `decayIndex` at all, so a file
written by it holds `-1` everywhere.

`positronium_enum` answers which class describes a branch, and `None` for a
branch none of them does:

```python
from opengate_gate_tree import positronium_enum

positronium_enum("gammaType")     # GammaType
positronium_enum(DECAY_INDEX_BRANCH)   # None
```

## In A Report

A run with `--statistics`, and `compute_statistics` from code, name the values
of the three branches:

```text
- gammaType / int32: min 0, max 3, mean 1.9, std 0.78867, 3 distinct,
  most frequent ANNIHILATION (367), PROMPT (72), UNKNOWN (61)
```

A value the package cannot name is reported as the number it is: the report
says what the file holds. These three branches report every value that has a
name, so a number nothing names cannot be crowded out by the members that do;
the numbers without names are capped. How many values the column really held
is the count beside them — `3 distinct` in the line above, `distinct_values`
in the saved report. `decayIndex` keeps its
numbers, for the reason above.

## In A Written File

An output file holds the integers GATE wrote. The names are a reading of the
data, not the data, and a file holding names could not be read back: the
branch would become text where every structure of the tree describes a whole
number. Read the values as names in your own code, and keep the file as it is.
