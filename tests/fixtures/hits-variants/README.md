# "Hits" tree variant fixtures

One file per structure of the GATE "Hits" tree, so that branch schemas, variant detection and
validation are tested against output a simulation really produced, not against lists retyped from
documentation.

## Files

| File | Variant | Tree names | Entries | Branches |
| --- | --- | --- | --- | --- |
| `a1-no-system.root` | A1 — no system | `Hits` | 500 | 40 |
| `a2-system.root` | A2 — with a system | `Hits` | 500 | 46 |
| `a3-system-septal.root` | A3 — with a system, septal penetration counting | `Hits` | 500 | 47 |
| `a4-no-system-cc.root` | A4 — no system, Compton camera output | `Hits` | 500 | 30 |
| `a5-system-cc.root` | A5 — with a system, Compton camera output | `Hits` | 500 | 36 |
| `b1-tree-common-output.root` | B1 — GateToTree common output | `tree` | 4441 | 54 |
| `name-multi-run.root` | A1, one tree per run | `Hits`, `Hits_run1`, `Hits_run2` | 500 each | 40 |
| `name-multi-sd.root` | A1, one tree per sensitive detector | `Hits_DET_INNER`, `Hits_DET_OUTER` | 500 each | 40 |

`a3-system-septal.root` uses the `SPECThead` identifier naming scheme; every other file with a
system uses `cylindricalPET`.

## Provenance

The simulations were run on a GATE build carrying the patches of
`fix/multi-photon-analysis-interactions-counting`, with the `multianalysis`
(`GateMultiPhotonAnalysis`) module enabled. A stock GATE build may therefore write a slightly
different set of branches; the package treats unknown extra branches as a warning rather than an
error for exactly that reason.

Only the "Hits" trees were carried over. The `pet_data` and `OpticalData` trees and the two `TH1D`
histograms that GATE writes next to them are covered by
`tests/fixtures/data-only-hits-tree-small-size.root`, so repeating them here would only add weight.

Every file except `b1-tree-common-output.root` was rewritten with `uproot.mktree()` and `extend()`,
keeping the first 500 entries. Branch names, branch order, uproot interpretations and values were
compared against the source files programmatically: they match.

## Why B1 is not trimmed

`b1-tree-common-output.root` is a byte copy of the simulation output, with all 4441 entries.

The GateToTree layout splits `volumeID` into ten scalar branches named `volumeID[0]` … `volumeID[9]`,
and the uproot TTree writer reads the brackets in a branch name as an array dimension. Writing that
layout produces a file whose `volumeID[1]` branch has shape `(N, 1)` and whose `volumeID[2]` …
`volumeID[9]` branches cannot be read back at all:

```
ValueError: basket 0 in tree /tree;1:volumeID[2] has the wrong number of entries
            (expected 500, obtained 250) when interpreted as AsDtype("('>i4', (2,))")
```

Trimming would therefore corrupt the very structure this fixture exists to describe, so the original
file is used instead. The same limitation applies to writing such data back to ROOT from the package.

## What is not here

There is no fixture for the per-collection GateToTree output (B2, B3) or for the Compton camera
actor output (C), because no reference file holds them: GATE wrote those files with zero keys — not
zero entries, but no objects at all. The package recognises both layouts and reports them as
unsupported instead of guessing a schema that nothing can confirm.

## Regenerating

With the simulation output in `<source-dir>`:

```python
import uproot

ENTRIES = 500

origin = uproot.open("<source-dir>/a1-no-system.root")
with uproot.recreate("a1-no-system.root") as out:
    tree = origin["Hits"]
    types, data = {}, {}
    for name, branch in tree.items():
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

Copy `b1-tree-common-output.root` unchanged instead of running this on it.
