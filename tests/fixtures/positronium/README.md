# PositroniumSource fixtures

One simulation output per way a `PositroniumSource` can be configured, so that the enum
representations of the `sourceType`, `decayType` and `gammaType` branches are tested against values
a simulation really wrote.

## Files

| File | Scene | `sourceType` | `decayType` | `gammaType` | `decayIndex` |
| --- | --- | --- | --- | --- | --- |
| `pps.root` | pPs, two gammas | 2 | 1 | 2 | 0 |
| `ops.root` | oPs, three gammas | 3 | 1 | 2 | 0 |
| `pps-prompt.root` | pPs with a prompt gamma | 2 | 2 | 2, 3 | 0 |
| `ops-prompt.root` | oPs with a prompt gamma | 3 | 2 | 2, 3 | 0 |
| `pps-direct.root` | half pPs, half direct annihilation | 2, 4 | 1 | 2 | 0, 1 |
| `ops-direct.root` | half oPs, half direct annihilation | 3, 4 | 1 | 2 | 0, 1 |
| `back-to-back.root` | back-to-back 511 keV, no positronium | 0 | 0 | 0 | -1 |
| `all-variants.root` | all seven channels, one seventh each | 0, 2, 3, 4 | 0, 1, 2 | 0, 2, 3 | -1, 0, 1 |

Every file holds the "Hits" tree of the A1 structure: 40 branches, no system. The branch list is
the one of `../hits-variants/a1-no-system.root`, so the files also exercise the variant detection
and validation of 0.3.0.

## Provenance

Each file keeps the first 500 entries of the simulation output, which held between 159 076 and
321 636 hits. Only the "Hits" tree was carried over; the `pet_data` and `OpticalData` trees and the
histograms next to them are covered by `../data-only-hits-tree-small-size.root`.

**Cutting to 500 entries loses no variant.** The set of distinct values of all four branches is the
same in the first 500 entries as in the whole file, for every one of the eight files; that was
checked before the files were cut, and the layout table above is compared against the files by
`tests/unit/test_positronium_fixtures.py`. No stratified sampling was needed.

Branch names, branch order, uproot interpretations and values were compared against the source
files programmatically: they match.

## What no file holds

Two values of the enums do not occur in any scene:

- `sourceType = 1` (a single gamma emitter),
- `gammaType = 1` (a single gamma).

Both are written by the `sg` model of the source, which none of these simulations uses. Their
meaning comes from `GateEmittedGammaInformation.hh`, so the package does not guess it, and
`test_the_values_no_scene_holds_are_read_all_the_same` in `tests/unit/test_positronium.py` reads
them from an array built in Python, since no file can provide one.

## Regenerating

With the simulation output in `<source-dir>`:

```python
import uproot

ENTRIES = 500

origin = uproot.open("<source-dir>/pps.root")["Hits"]
with uproot.recreate("pps.root") as out:
    types, data = {}, {}
    for name, branch in origin.items():
        column = branch.array(library="np", entry_stop=ENTRIES)
        if column.dtype == object:                       # text branch
            types[name], data[name] = str, [str(value) for value in column]
        elif column.ndim == 2:                           # fixed-width array branch
            types[name], data[name] = (column.dtype, column.shape[1:]), column
        else:
            types[name], data[name] = column.dtype, column
    out.mktree("Hits", types)
    out["Hits"].extend(data)
```

Check afterwards that the distinct values of the four branches still match the table above.
