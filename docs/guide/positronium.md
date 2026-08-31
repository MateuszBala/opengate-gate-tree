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

## Reading Values As Names

For a table meant for people, or a summary, the values can be read as names:

```python
from opengate_gate_tree import decode_column, decode_value

decode_value("gammaType", 3)                   # GammaType.PROMPT
meanings = decode_column("gammaType", data["gammaType"])
```

The two differ in what they do with a value the package does not know:

| Call | A value outside the class |
| --- | --- |
| `decode_value` | raises `ValueError`, naming the values GATE writes there |
| `decode_column` | leaves `None` in its place and reports it in the log, with how often it occurs |

A question about one value has one answer or none; reading an unknown value as
the one standing for "not defined" would report something the file does not
say. A column is different: a GATE build can write one value more, and an
analysis of the rows that are understood is still worth having.

## Which Rows A PositroniumSource Wrote

The fourth branch, `decayIndex`, holds the channel of the mixture a gamma came
from. Channel numbers depend on the order the fractions were configured in, so
they have no fixed meaning and no class of their own. One value does have one:

```python
from opengate_gate_tree import DECAY_INDEX_BRANCH, is_positronium_source, read_tree

data = read_tree(path, GateTree.HITS, [DECAY_INDEX_BRANCH, "sourceType"])
written_by_positronium = is_positronium_source(data[DECAY_INDEX_BRANCH])
```

`-1` is what GATE writes for a gamma emitted by another kind of source.

```{note}
The answer is about the source, not about the physics. A `PositroniumSource`
configured with a direct annihilation channel writes those gammas too, and
numbers them like the rest, so the mask holds `True` for them. What the gamma
itself was is said by `sourceType`.
```

The deprecated `ExtendedVSource` never fills `decayIndex` at all, so a file
written by it holds `-1` everywhere.

## In A Report

A run with `--statistics`, and `compute_statistics` from code, name the values
of the three branches:

```text
- gammaType / int32: min 0, max 3, mean 1.9, std 0.79, 3 distinct,
  most frequent ANNIHILATION (367), PROMPT (72), UNKNOWN (61)
```

A value the package cannot name is reported as the number it is: the report
says what the file holds. `decayIndex` keeps its numbers, for the reason above.

## In A Written File

An output file holds the integers GATE wrote. The names are a reading of the
data, not the data, and a file holding names could not be read back: the
branch would become text where every structure of the tree describes a whole
number. Read the values as names in your own code, and keep the file as it is.
