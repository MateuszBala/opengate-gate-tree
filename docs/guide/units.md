# Units

GATE writes every quantity in the units Geant4 works in: energy in MeV, length
in mm, time in s. An analysis rarely wants all three — a spectrum is read in
keV, a scanner is described in cm or m, and a time resolution is quoted in ns —
so the package converts between them and does nothing else to the numbers.

```python
from pathlib import Path

from opengate_gate_tree import GateTree, read_tree

data = read_tree(Path("simulation.root"), GateTree.HITS)
frame = data.to_dataframe()

energies = frame["edep"].gate.MeV_to_keV()
times = frame["time"].gate.s_to_ns()
```

## What GATE Writes

| Quantity | Unit | Branches |
| --- | --- | --- |
| energy | MeV | `edep`, `sourceEnergy` |
| length | mm | `posX`…, `localPosX`…, `sourcePosX`…, `stepLength`, `trackLength` |
| time | s | `time`, `trackLocalTime` |

`GATE_UNITS` says the same thing in code, which is where a conversion of a
branch starts:

```python
from opengate_gate_tree import GATE_UNITS

GATE_UNITS["energy"]     # 'MeV'
```

Angles are not written by GATE at all. Every angle in this package is computed,
and computed in radians, as in NumPy.

## The Four Families

Each family is closed: every unit reaches every other one, in both directions.

| Family | Units | Conversions |
| --- | --- | --- |
| energy | MeV, keV | `MeV_to_keV`, `keV_to_MeV` |
| length | mm, cm, m | `mm_to_cm`, `cm_to_mm`, `mm_to_m`, `m_to_mm`, `cm_to_m`, `m_to_cm` |
| time | s, ms, ns | `s_to_ms`, `ms_to_s`, `s_to_ns`, `ns_to_s`, `ms_to_ns`, `ns_to_ms` |
| angle | rad, deg | `rad_to_deg`, `deg_to_rad` |

The angle pair wraps `numpy.rad2deg` and `numpy.deg2rad`. It is here so that
the four families are found in one place, not because NumPy needed help.

## Why The Names Look Like That

A unit symbol keeps its capital letters: `MeV_to_keV`, not `mev_to_kev`. `MeV`
and `meV` are different units — megaelectronvolts and millielectronvolts — so
lowercasing the name would name a different conversion. This is the one place
in the package where a function name is not plain `snake_case`, and
[the coding conventions](../CODING_CONVENTIONS.md) record it as such.

## What A Conversion Answers With

The kind it was given, and `float64`:

```python
MeV_to_keV(0.511)               # 511.0, a float
MeV_to_keV(frame["edep"])       # a Series, with its index and its name
MeV_to_keV(numpy.array(...))    # an array of the same shape
```

A missing value stays missing, and nothing else about the column changes:
the index, the name and the number of rows are the ones it came in with, which
is what lets a conversion sit in the middle of a chain.

### Note

The computation is in `float64` whatever came in. A GATE file holds `float32`
columns, and sixty seconds read in nanoseconds is 6·10¹⁰ — a number a `float32`
mantissa places on a grid four microseconds wide. Converting in the dtype the
file happens to use would quietly cost that.

## In A Chain

Both halves of a chain are available on the `gate` namespace, so a selection
and a conversion read as one line:

```python
energies = (
    frame.gate.in_cylinder((0, 0), radius=500.0, inner_radius=409.0)["edep"]
    .gate.in_range(0.0, 0.511)
    .gate.MeV_to_keV()
)
```

The functions are exported from the package root as well, which is what to
reach for when a conversion is passed around rather than called:

```python
from opengate_gate_tree import mm_to_cm, s_to_ns
```

## Converting An Angle

Angles come out of the geometry in radians. Degrees are a reading of them, and
usually the last step before a plot:

```python
angles = frame.gate.position().angle_to(frame.gate.momentum_direction())
degrees = angles.gate.rad_to_deg()
```

See [Vectors And Angles](vectors.md) for what those angles are.

## What Is Not Here

There is no unit system: nothing carries its unit around, and no conversion is
applied for you. A column of a GATE file is in the unit GATE wrote it in, a
converted column is in the unit you asked for, and keeping track of which is
which is the analysis's business. Files written by this package hold the
numbers as they were read, so a conversion never reaches one by accident.

Units GATE does not use are not offered either: no eV or GeV, no ps. They would
be conversions nobody in this workflow asks for.
