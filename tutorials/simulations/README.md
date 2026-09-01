# Simulations

Some tutorials need more data than a repository should carry. The macros that
produce it live here, so that a reader can generate the file themselves - and,
more usefully, can read what was simulated and why the analysis that follows
makes sense.

## What A Simulation Folder Holds

```text
sim-NN-<name>/
├── README.md            # what it simulates, why, and how to run it
├── run_simulation.mac   # the entry point GATE is given
├── detector.mac         # the geometry
├── source.mac           # the source
├── physics.mac          # the physics list
├── output.mac           # what is written, and where
├── application.mac      # the runs and the time slices
├── random.mac           # the seed
└── GateMaterials.db     # and any other file the macros need
```

`NN` is the next number in order, so that a tutorial can name its simulation
without ambiguity. The names of the macros follow the ones GATE examples use;
what matters is that `run_simulation.mac` is the one file a reader has to pass
to GATE.

The ROOT file a simulation writes is **not** carried in the repository, and
neither is its log. Anything else a simulation leaves behind goes to
`.tmp/sim-NN-<name>/`.

## What A Simulation README Says

Copy this into `sim-NN-<name>/README.md` and fill it in:

```markdown
# sim-NN-<name>

## What This Simulates

What the scene is, and what question it was built to answer.

## Geometry

The detector: shape, sizes, materials, and where they come from in the macros.
A picture in `../../data/imgs/` helps here.

## Source

What is emitted, from where, with what energy and what activity.

## Physics And Time

The physics list, the time slices, the number of runs, the seed.

## Output

Which trees are written, under what name, and where the file lands.

## How To Run It

    Gate run_simulation.mac

with the version of GATE it was written for, and roughly how long it takes.

## What Reads This

The tutorial or tutorials that use the file.
```

## The Version Of GATE

Say in the README which version of GATE the macros were written for. The
package supports the C++ line from 9.4.2 onwards, and a macro that needs
something newer has to say so.
