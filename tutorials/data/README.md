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
| _none yet_ | | | | |

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
