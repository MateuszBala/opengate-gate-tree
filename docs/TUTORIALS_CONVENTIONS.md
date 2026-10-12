# Tutorial Conventions

Tutorials are worked examples: a notebook reads real GATE output, does
something with it, and shows the answer. They are written for people to read,
and they are the last check that a feature works — a tutorial that stops
running is a bug report about the package, the data, or itself.

## Table Of Contents

- [Where Tutorials Live](#where-tutorials-live)
- [What A Notebook Contains](#what-a-notebook-contains)
- [When To Add One](#when-to-add-one)
- [The Cycle](#the-cycle)
- [Rules Of Thumb](#rules-of-thumb)

## Where Tutorials Live

Everything is under `tutorials/` in the repository:

```text
tutorials/
├── data/
│   ├── input-examples/     # ROOT files of 500 kB or less, carried in the repository
│   ├── imgs/               # pictures a notebook or a page shows
│   └── README.md           # every file above, described
├── notebooks/
│   ├── .tmp/               # what a notebook writes; never carried
│   ├── tools/              # helpers the notebooks import instead of repeating
│   ├── template.ipynb      # the shape every tutorial takes
│   ├── tutorial-NN-<name>.ipynb
│   └── README.md           # the index of tutorials
├── simulations/
│   ├── .tmp/               # what a simulation writes besides its ROOT file
│   ├── sim-NN-<name>/      # the macros of one simulation, and what it is for
│   └── README.md
├── .gitignore
├── Makefile
└── README.md
```

What each part is for:

- **`data/input-examples/`** holds ROOT files small enough to carry — **500 kB
  or less**. A pre-commit hook refuses anything larger under `tutorials/data/`.
- **`data/imgs/`** holds the pictures a notebook shows: a drawing of a
  detector, a sketch of a source, a theoretical curve. They live here rather
  than beside one notebook so that a documentation page can show the same
  picture without a second copy.
- **`data/README.md`** describes every file in `data/`, with the simulation it
  came from and the notebook that reads it. A file nobody can trace back to
  what produced it is a file nobody can trust.
- **`notebooks/`** holds the notebooks themselves, one per subject, named
  `tutorial-NN-<name>.ipynb` where `NN` is the next number in order.
- **`notebooks/.tmp/`** is the only place a notebook may write, in its own
  directory `tutorial-NN-<name>/`. Nothing there is carried in the repository.
- **`notebooks/tools/`** holds the helpers that keep the notebooks short:
  where the files are, what version is installed, what the plots look like.
- **`simulations/sim-NN-<name>/`** holds the macros of one simulation, its
  supporting files, and a README describing it in full. The ROOT file it writes
  is **not** carried: a reader generates it, and the macros tell them what was
  simulated and why the analysis that follows makes sense.
- **`simulations/.tmp/`** takes anything else a simulation leaves behind, under
  `sim-NN-<name>/`.
- **`Makefile`** builds the environment the notebooks run in, registers a
  Jupyter kernel, starts JupyterLab and runs every notebook start to finish.

## What A Notebook Contains

A tutorial explains itself. Someone who opens it on GitHub, with no environment
and no context, should be able to read it through and know what it did.

The sections, in order — `template.ipynb` carries all of them:

```markdown
# Title Of The Notebook

**Version**: N

**Last modified**: YYYY-MM-DD

**Compatible with opengate-gate-tree version:** X.Y.Z

**Author**: Name Surname

## About

What this notebook is about. What question it answers, not what functions it
calls.

## What will you learn from this notebook

A list of the things a reader will be able to do afterwards.

## Simulation

Optional; only when the notebook works on data a simulation produced. What was
simulated and why, briefly, with a picture of the detector, the source or the
phantom where one helps. It ends by pointing at
`tutorials/simulations/sim-NN-<name>/README.md`, which holds the full
description — this section is the short version, and a reader should not have
to spend time on it.

## Theory

Optional. The definitions and formulas the analysis rests on, written as
mathematics rather than as prose.

## Tutorial

A line of text, then the code that does what the line says, then a word about
the answer where the answer needs one. Repeated until the subject is covered.

## Summary

One to three sentences: what was covered, and what the reader can now do.

## Bibliography

- the sections of the package documentation the notebook used, linked one by one
- the articles and publications behind the physics
- where any additional data or parameters came from

## Changelog

- v1: written.
- v2: what changed, and why.
```

The changelog is filled in whenever the version is raised, and says why the
change was needed — not only what moved.

## When To Add One

A pull request that adds a feature a user will reach for carries the tutorial
that shows it. The tutorial is part of the change, not a follow-up: it is what
proves the feature is usable, and it is written while the reasoning behind the
feature is still fresh.

The one exception is the material written before this convention existed: those
tutorials arrive together, in one pull request covering the features already
released.

## The Cycle

Tutorials are written after the code is reviewed and before the branch is
pushed:

1. build the feature, stage by stage;
2. finish the last stage before the review;
3. have the branch reviewed;
4. apply what the review warrants;
5. write the tutorial;
6. if the tutorial finds a defect or a surprise, fix it and run the notebook
   again — without committing in between, so the two stay one change;
7. commit what writing the tutorial uncovered;
8. commit the tutorial;
9. push;
10. open the pull request;
11. then, for each round of review: apply what it warrants, run the notebooks
    again, and if a regression appears, fix it and run them again before
    committing.

Step 11 is where `make verify` earns its place. Run it from `tutorials/` before
every push. It never runs in continuous integration: executing a notebook needs
data that CI does not have, and a green build that skipped the notebooks would
claim more than it knows.

## Rules Of Thumb

- **Write in English.** An author who drafts in another language has it
  translated before it is published.
- **Keep the outputs in the notebook.** Somebody reading it on GitHub has no
  environment, and a notebook without outputs shows them nothing.
- **Write only under `.tmp/`.** `tools.paths.output_dir()` is the way there,
  and it makes the directory on first use.
- **Add a row to the index** in `tutorials/notebooks/README.md`. A tutorial
  nobody can find is a tutorial nobody reads.
- **Describe what you add** — a file in `data/README.md`, a simulation in its
  own README.
- **Run every notebook after a change to the package**, even one that needed no
  new test. A release must not go out with a tutorial that no longer runs.
- **Say which version you ran against.** The header states it, and the notebook
  prints what is actually installed beside it, so a reader can see whether the
  two agree.
