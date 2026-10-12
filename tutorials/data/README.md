# Data

What the notebooks read: ROOT files small enough to carry in the repository,
and the pictures the notebooks show.

## input-examples

ROOT files written by GATE, **500 kB or less each**. Anything larger belongs in
`../simulations/` as the macros that produce it, so that a reader generates the
file themselves and can see what was simulated.

Every file gets a row here, with the simulation it came from and the notebook
that reads it.

| File | Size | What it holds | Written by | Read by |
| --- | --- | --- | --- | --- |
| `positronium-source-all-variants.root` | 80 kB | 500 hits of a "Hits" tree, from a scene with a `PositroniumSource` configured with every component at once beside a back-to-back source, so the file holds annihilation, prompt and unknown gammas | a GATE simulation of a spherical shell detector with an inner radius of 200 mm | `tutorial-01-tools-plotting.ipynb`, `tutorial-02-reading-writing-root-file.ipynb`, `tutorial-04-command-line.ipynb`, `tutorial-07-statistics-report.ipynb` |
| `hits-a1-no-system.root` | 77 kB | 500 hits of a "Hits" tree of the A1 structure - no system, 40 branches - in one tree `Hits` | a GATE simulation; the first 500 entries of `tests/fixtures/hits-variants/a1-no-system.root`, whose README gives the provenance | `tutorial-04-command-line.ipynb`, `tutorial-05-hits-tree-variants.ipynb` |
| `hits-a2-cylindrical-pet.root` | 86 kB | 500 hits of a "Hits" tree of the A2 structure - a `cylindricalPET` system, 46 branches - with every hit between 410 and 431 mm from the axis | a GATE simulation of a cylindricalPET scanner; the first 500 entries of `tests/fixtures/hits-variants/a2-system.root`, whose README gives the provenance | `tutorial-03-representations-and-formats.ipynb`, `tutorial-04-command-line.ipynb`, `tutorial-05-hits-tree-variants.ipynb` |
| `hits-a4-compton-camera.root` | 69 kB | 500 hits of a "Hits" tree of the A4 structure - no system, Compton camera output, 30 branches | a GATE simulation; the first 500 entries of `tests/fixtures/hits-variants/a4-no-system-cc.root`, whose README gives the provenance | `tutorial-05-hits-tree-variants.ipynb` |
| `hits-one-tree-per-run.root` | 192 kB | hits of the A1 structure split into one tree per run - `Hits`, `Hits_run1`, `Hits_run2`, 500 entries each - with `runID` 0, 1 and 2 | a GATE simulation of three runs; the first 500 entries of each tree of `tests/fixtures/hits-variants/name-multi-run.root`, whose README gives the provenance | `tutorial-04-command-line.ipynb`, `tutorial-06-split-hits-trees.ipynb`, `tutorial-07-statistics-report.ipynb` |
| `hits-one-tree-per-detector.root` | 137 kB | hits of the A1 structure split into one tree per sensitive detector - `Hits_DET_INNER`, `Hits_DET_OUTER`, 500 entries each | a GATE simulation with two sensitive detectors; the first 500 entries of each tree of `tests/fixtures/hits-variants/name-multi-sd.root`, whose README gives the provenance | `tutorial-04-command-line.ipynb`, `tutorial-06-split-hits-trees.ipynb` |

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
