# Data

What the notebooks read: ROOT files small enough to carry in the repository,
and the pictures the notebooks show.

## input-examples

ROOT files holding GATE output, **500 kB or less each**. Anything larger belongs in
`../simulations/` as the macros that produce it, so that a reader generates the
file themselves and can see what was simulated.

Every file gets a row here, with the simulation it came from and the notebook
that reads it.

The files carried so far are byte copies of fixtures the tests read, under
`tests/fixtures/`. Each of them keeps the first 500 entries of a GATE output -
of every tree, where hits were split over several - and only the trees of
hits: the other trees and the histograms GATE writes beside them were left
out, and the trees were rewritten with uproot. The README next to each fixture
gives the details. Nothing more than that is known about the simulations, so
anything said below about a detector is read off the data.

| File | Size | What it holds | Written by | Read by |
| --- | --- | --- | --- | --- |
| `positronium-source-all-variants.root` | 79 kB | 500 hits of a "Hits" tree, from a scene with a `PositroniumSource` configured with every component at once beside a back-to-back source, so the file holds annihilation, prompt and unknown gammas | byte copy of `tests/fixtures/positronium/all-variants.root`; every hit lies 200-264 mm from the centre, as in a spherical shell detector with an inner radius of 200 mm | `tutorial-01-tools-plotting.ipynb`, `tutorial-02-reading-writing-root-file.ipynb`, `tutorial-04-command-line.ipynb`, `tutorial-07-statistics-report.ipynb`, `tutorial-08-positronium-source-values.ipynb`, `tutorial-09-selecting-rows.ipynb`, `tutorial-12-units.ipynb`, `tutorial-13-vectors-from-columns.ipynb`, `tutorial-15-angle-ranges.ipynb`, `tutorial-16-rows-without-an-answer.ipynb`, `tutorial-18-events-tracks-and-hits.ipynb`, `tutorial-19-analysis-as-a-description.ipynb` |
| `hits-a1-no-system.root` | 77 kB | 500 hits of a "Hits" tree of the A1 structure - no system, 40 branches - in one tree `Hits` | byte copy of `tests/fixtures/hits-variants/a1-no-system.root` | `tutorial-04-command-line.ipynb`, `tutorial-05-hits-tree-variants.ipynb`, `tutorial-14-compton-scattering.ipynb`, `tutorial-16-rows-without-an-answer.ipynb` |
| `hits-a2-cylindrical-pet.root` | 86 kB | 500 hits of a "Hits" tree of the A2 structure - a `cylindricalPET` system, 46 branches - with every hit between 410 and 431 mm from the axis | byte copy of `tests/fixtures/hits-variants/a2-system.root`, from a scanner described as a cylindricalPET system | `tutorial-03-representations-and-formats.ipynb`, `tutorial-04-command-line.ipynb`, `tutorial-05-hits-tree-variants.ipynb`, `tutorial-10-shapes-and-detector-geometry.ipynb`, `tutorial-17-whole-analysis-in-one-chain.ipynb` |
| `hits-a4-compton-camera.root` | 69 kB | 500 hits of a "Hits" tree of the A4 structure - no system, Compton camera output, 30 branches | byte copy of `tests/fixtures/hits-variants/a4-no-system-cc.root` | `tutorial-05-hits-tree-variants.ipynb` |
| `hits-one-tree-per-run.root` | 192 kB | hits of the A1 structure split into one tree per run - `Hits`, `Hits_run1`, `Hits_run2`, 500 entries each - with `runID` 0, 1 and 2 | byte copy of `tests/fixtures/hits-variants/name-multi-run.root`, from three runs | `tutorial-04-command-line.ipynb`, `tutorial-06-split-hits-trees.ipynb`, `tutorial-07-statistics-report.ipynb`, `tutorial-11-events-and-tracks.ipynb` |
| `hits-one-tree-per-detector.root` | 137 kB | hits of the A1 structure split into one tree per sensitive detector - `Hits_DET_INNER`, `Hits_DET_OUTER`, 500 entries each | byte copy of `tests/fixtures/hits-variants/name-multi-sd.root`, from two sensitive detectors | `tutorial-04-command-line.ipynb`, `tutorial-06-split-hits-trees.ipynb`, `tutorial-18-events-tracks-and-hits.ipynb` |

## imgs

Pictures a notebook or a documentation page shows: a drawing of a detector, a
sketch of a source, a theoretical curve. They are here rather than beside the
notebook so that a page can show the same picture without a second copy.

| File | What it shows | Made by | Shown in |
| --- | --- | --- | --- |
| _none yet_ | | | |

## Adding A File

1. check the size: 500 kB is the limit under `input-examples/`, and a
   pre-commit hook enforces it;
2. add the row above, filling in every column - a file nobody can trace back to
   the simulation that wrote it is a file nobody can trust;
3. if it came from a simulation in `../simulations/`, name that folder in the
   "Written by" column.
