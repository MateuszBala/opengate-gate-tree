# The "Hits" Tree

What the "Hits" tree holds depends on the simulation that wrote it. A system
adds identifier branches, the Compton camera output replaces the second half of
the tree, septal penetration counting adds a counter, and the `GateToTree`
module writes the same data in another order under another tree name.

The package recognises which of those structures a file holds, checks the tree
against it, and reports what is wrong when they disagree. Nothing has to be
told to it: reading a tree is enough.

## Supported Structures

| Label | Structure | Written by | Branches | Written when |
| --- | --- | --- | --- | --- |
| `A1` | No system | GateToRoot | 40 | hits are attached to a sensitive detector outside a system |
| `A2` | System | GateToRoot | 46 | hits are attached to a sensitive detector inside a system |
| `A3` | System with septal penetration | GateToRoot | 47 | as above, with `/gate/output/analysis/recordSeptalPenetration true` |
| `A4` | No system, Compton camera output | GateToRoot | 30 | no system, with `/gate/output/root/CCoutput true` |
| `A5` | System, Compton camera output | GateToRoot | 36 | a system, with `/gate/output/root/CCoutput true` |
| `B1` | GateToTree common output | GateToTree | 54 | `/gate/output/tree/hitsCommonOutput/enable`, which writes its own file |

The labels come from the reference material the schemas were written from, and
appear in error messages and reports, so that a file can be talked about
without describing it every time.

Every structure was measured on the output of a simulation. The branch lists at
the end of this page are the same ones the package validates against; a test
compares them, so the page cannot drift away from the code.

## How A Structure Is Recognised

Recognition looks at a few branches that say what the simulation did, rather
than comparing the whole branch list against a schema:

| Branch | What it means |
| --- | --- |
| `layerName` | the Compton camera output, which this version does not support |
| `volumeID[0]` | the `GateToTree` output, which splits `volumeID` into ten branches |
| `volumeID` | the classic ROOT output |
| `sourceEnergy` with `postStepProcess` | the Compton camera output of the classic writer |
| `septalNb` | septal penetration counting |
| `gantryID`, `headID`, … | hits attached to a system |

Recognising first and checking afterwards is what makes the reports useful. A
file from a GATE build that adds or drops a branch is still recognised as the
structure it is, so the package can say *which* branch is missing instead of
answering "unknown structure" to everything that is not an exact match.

```python
from pathlib import Path

from opengate_gate_tree import RootFile, describe_hits_tree

with RootFile(Path("simulation.root")) as root_file:
    detection = root_file.detect_hits_tree()
    print(describe_hits_tree(detection))
```

```text
Detected Hits tree variant: System (A2)
Tree name in the file: Hits
System identifier scheme: cylindricalPET / OPET
Branches (46):
  - PDGEncoding / int32
  - trackLocalTime / float64
  ...
```

## System Identifier Branches

When hits are attached to a system, GATE writes one branch per level of its
hierarchy. Their names depend on the type of the system:

| Scheme | depth 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| `cylindricalPET`, `OPET` | `gantryID` | `rsectorID` | `moduleID` | `submoduleID` | `crystalID` | `layerID` |
| `CPET` | `gantryID` | `sectorID` | `cassetteID` | `moduleID` | `crystalID` | `layerID` |
| `SPECThead`, `OpticalSystem` | `headID` | `crystalID` | `pixelID` | `unused3ID` | `unused4ID` | `unused5ID` |
| `ecat`, `ecatAccel` | `gantryID` | `blockID` | `crystalID` | `unused3ID` | `unused4ID` | `unused5ID` |
| `CTscanner` | `gantryID` | `moduleID` | `clusterID` | `pixelID` | `unused4ID` | `unused5ID` |
| `scanner` | `baseID` | `level1ID` | `level2ID` | `level3ID` | `level4ID` | `level5ID` |

Two system types can share one set of names, so what the package reports is the
naming scheme, not the system itself. That is why a report says
`cylindricalPET / OPET`: the file does not say which of the two was used.

The classic ROOT output always writes six identifier branches, filling the
levels a system does not reach with names such as `unused4ID`. The `GateToTree`
output writes one per level, so its block is a prefix of the scheme.

A tree carrying identifier branches whose scheme cannot be told apart is
refused rather than read as a system-less structure. `gantryID` alone starts
four of the schemes, and reading such a file as "no system" would settle a
question that stayed open.

For the same reason, the `GateToTree` output is supported for a simulation
using a system of at least two levels. Its identifier block is what says which
scheme the file follows, so a block of one level is ambiguous and a block of
none — a `GateToTree` output of a simulation attaching hits outside any system
— cannot be told from a file that lost those branches. No reference file holds
such an output, so it is refused rather than guessed at.

## What Validation Checks

| Difference | Result |
| --- | --- |
| a branch of the structure is missing | error |
| a branch is stored with another type | error |
| the tree holds a branch beyond the structure | warning in the log |

The asymmetry is deliberate. The reference files come from a GATE build
carrying patches, and adding a branch is an ordinary thing for a build to do.
Refusing such a file would turn the package away from the simulations it exists
for, while a missing branch means the data is not what it was taken for.

The check runs on the whole tree before any data is loaded, so asking for two
branches of a broken tree still reports the tree as broken. It also runs on
branch types rather than on values, so a branch that could never be loaded is
still looked at.

A file the package cannot recognise is still readable, with the check turned
off:

```python
from opengate_gate_tree import GateTree, read_tree

data = read_tree(Path("simulation.root"), GateTree.HITS, validate=False)
```

On the command line that is `--skip-hits-validation`.

## Structures That Are Recognised But Not Supported

Two layouts are recognised by name and refused:

- the per-collection `GateToTree` output with the Compton camera columns
  (`/gate/output/tree/hits/enable` together with
  `/gate/output/tree/enableCCoutput`);
- the output of the Compton camera actor, and the file format
  `GateCCHitFileReader` reads.

Neither has a reference file holding any data, so their branch lists could only
be copied from documentation and never confirmed against a simulation. Naming
them in the error is more useful than a schema nothing can vouch for:

```text
UnknownHitsVariantError: Tree 'Hits' matches the Compton camera hits layout
(GateComptonCameraActor or GateCCHitTree), which this version does not support.
Supported structures: A1 (No system), A2 (System), ...
```

## Branch Lists

The lists below are what each structure holds, in the order GATE writes them,
with the type of every branch. Structures that use a system are shown with the
`cylindricalPET` naming scheme; another system changes those six names and
nothing else.

### A1 — No system (40 branches)

```text
PDGEncoding / int32
trackID / int32
parentID / int32
trackLocalTime / float64
time / float64
edep / float32
stepLength / float32
trackLength / float32
posX / float32
posY / float32
posZ / float32
localPosX / float32
localPosY / float32
localPosZ / float32
sourcePosX / float32
sourcePosY / float32
sourcePosZ / float32
sourceID / int32
eventID / int32
runID / int32
volumeID / int32[10]
processName / text
momDirX / float32
momDirY / float32
momDirZ / float32
photonID / int32
nPhantomCompton / int32
nCrystalCompton / int32
nPhantomRayleigh / int32
nCrystalRayleigh / int32
nInteractions / int32
primaryID / int32
axialPos / float32
rotationAngle / float32
comptVolName / text
RayleighVolName / text
sourceType / int32
decayType / int32
gammaType / int32
decayIndex / int32
```

### A2 — System (46 branches)

```text
PDGEncoding / int32
trackID / int32
parentID / int32
trackLocalTime / float64
time / float64
edep / float32
stepLength / float32
trackLength / float32
posX / float32
posY / float32
posZ / float32
localPosX / float32
localPosY / float32
localPosZ / float32
sourcePosX / float32
sourcePosY / float32
sourcePosZ / float32
sourceID / int32
eventID / int32
runID / int32
volumeID / int32[10]
processName / text
gantryID / int32
rsectorID / int32
moduleID / int32
submoduleID / int32
crystalID / int32
layerID / int32
momDirX / float32
momDirY / float32
momDirZ / float32
photonID / int32
nPhantomCompton / int32
nCrystalCompton / int32
nPhantomRayleigh / int32
nCrystalRayleigh / int32
nInteractions / int32
primaryID / int32
axialPos / float32
rotationAngle / float32
comptVolName / text
RayleighVolName / text
sourceType / int32
decayType / int32
gammaType / int32
decayIndex / int32
```

### A3 — System with septal penetration (47 branches)

```text
PDGEncoding / int32
trackID / int32
parentID / int32
trackLocalTime / float64
time / float64
edep / float32
stepLength / float32
trackLength / float32
posX / float32
posY / float32
posZ / float32
localPosX / float32
localPosY / float32
localPosZ / float32
sourcePosX / float32
sourcePosY / float32
sourcePosZ / float32
sourceID / int32
eventID / int32
runID / int32
volumeID / int32[10]
processName / text
gantryID / int32
rsectorID / int32
moduleID / int32
submoduleID / int32
crystalID / int32
layerID / int32
momDirX / float32
momDirY / float32
momDirZ / float32
photonID / int32
nPhantomCompton / int32
nCrystalCompton / int32
nPhantomRayleigh / int32
nCrystalRayleigh / int32
nInteractions / int32
primaryID / int32
axialPos / float32
rotationAngle / float32
comptVolName / text
RayleighVolName / text
septalNb / int32
sourceType / int32
decayType / int32
gammaType / int32
decayIndex / int32
```

### A4 — No system, Compton camera output (30 branches)

```text
PDGEncoding / int32
trackID / int32
parentID / int32
trackLocalTime / float64
time / float64
edep / float32
stepLength / float32
trackLength / float32
posX / float32
posY / float32
posZ / float32
localPosX / float32
localPosY / float32
localPosZ / float32
sourcePosX / float32
sourcePosY / float32
sourcePosZ / float32
sourceID / int32
eventID / int32
runID / int32
volumeID / int32[10]
processName / text
sourceEnergy / float32
sourcePDG / int32
nCrystalConv / int32
nCrystalCompt / int32
nCrystalRayl / int32
energyFinal / float32
energyIniT / float32
postStepProcess / text
```

### A5 — System, Compton camera output (36 branches)

```text
PDGEncoding / int32
trackID / int32
parentID / int32
trackLocalTime / float64
time / float64
edep / float32
stepLength / float32
trackLength / float32
posX / float32
posY / float32
posZ / float32
localPosX / float32
localPosY / float32
localPosZ / float32
sourcePosX / float32
sourcePosY / float32
sourcePosZ / float32
sourceID / int32
eventID / int32
runID / int32
volumeID / int32[10]
processName / text
gantryID / int32
rsectorID / int32
moduleID / int32
submoduleID / int32
crystalID / int32
layerID / int32
sourceEnergy / float32
sourcePDG / int32
nCrystalConv / int32
nCrystalCompt / int32
nCrystalRayl / int32
energyFinal / float32
energyIniT / float32
postStepProcess / text
```

### B1 — GateToTree common output (54 branches)

```text
PDGEncoding / int32
trackID / int32
parentID / int32
trackLocalTime / float64
time / float64
runID / int32
eventID / int32
sourceID / int32
primaryID / int32
posX / float32
posY / float32
posZ / float32
localPosX / float32
localPosY / float32
localPosZ / float32
momDirX / float32
momDirY / float32
momDirZ / float32
edep / float32
stepLength / float32
trackLength / float32
rotationAngle / float32
axialPos / float32
processName / text
comptVolName / text
RayleighVolName / text
volumeID[0] / int32
volumeID[1] / int32
volumeID[2] / int32
volumeID[3] / int32
volumeID[4] / int32
volumeID[5] / int32
volumeID[6] / int32
volumeID[7] / int32
volumeID[8] / int32
volumeID[9] / int32
sourcePosX / float32
sourcePosY / float32
sourcePosZ / float32
nPhantomCompton / int32
nCrystalCompton / int32
nPhantomRayleigh / int32
nCrystalRayleigh / int32
gantryID / int32
rsectorID / int32
moduleID / int32
submoduleID / int32
crystalID / int32
layerID / int32
photonID / int32
sourceType / int32
decayType / int32
gammaType / int32
decayIndex / int32
```
