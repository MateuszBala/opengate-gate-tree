# ROADMAP

This file presents the project implementation plan.

## Roadmap status

Each stage can have only one of the following statuses:

- planned: the stage is planned
- completed: the stage has been completed


## Version 0.1.0 (status: completed)

Initialize the project structure and add the foundational code required to build the package.

### Definition of Done (DoD)

- the project structure has been added and is complete
- a minimal codebase required to build the package has been added

## Version 0.2.0 (status: completed)

Basic support for trees from a GATE output file:

- loading a ROOT file
- validating consistency between the ROOT file and its trees
- tree extraction
- branch extraction
- exporting output files to ROOT format
- exporting output files to HDF5 format
- exporting output files to CSV format

### Definition of Done (DoD)

- support and validation for ROOT files have been implemented
- tree and branch extraction works correctly
- a tree representation based on NumPy arrays has been added
- a tree representation based on pandas.DataFrame has been added
- all three data export formats work correctly
- the package can be used both as an application and as a library
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.3.0 (status: completed)

Basic support for the "Hits" tree:

- branch definitions depending on the simulation type
- tree validation
- statistics generation
- merging "Hits" trees that a single file stores under several names

### Definition of Done (DoD)

- all versions of the "Hits" tree structure confirmed by reference output files are fully supported: GateToRoot without a system, with a system, with septal penetration counting, both Compton camera output variants, and the common GateToTree output
- structures written by GateToTree per hits collection with the Compton camera columns, and by the Compton camera actor, are recognised and reported as unsupported; they are not covered by test fixtures, because no reference file holds them. Without those columns the per-collection output holds the branches of the common output and is read as that structure
- all supported versions of the "Hits" tree structure have dedicated validation
- a file that stores the "Hits" tree under several names, one per run or per sensitive detector, can be read as a single dataset
- identifiers written by GATE are preserved exactly as they are; an event is identified by the pair of `runID` and `eventID`, which stays valid across simulation runs started with the same seed
- output files are named using the format `<title>.hits.<file-format>`
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features, including the full branch list of every supported version

## Version 0.4.0 (status: completed)

Add support for representing PositroniumSource data in the "Hits" tree:

- casting the `gammaType`, `sourceType` and `decayType` branches to enum representations
- telling data written by a PositroniumSource apart from data written by another source

### Definition of Done (DoD)

- each of the three branches has its own representation as an enum class, whose values are the integers GATE writes, so that a column can be compared against them as it was read
- a value outside an enum is reported rather than silently read as another one
- the `decayIndex` branch tells a row written by a PositroniumSource from one written by another source
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.5.0 (status: completed)

Add a set of basic filters and selectors for data from the "Hits" tree:

- filters and selectors for every branch
- filters and selectors for the PositroniumSource branches, built on their enum representations

### Definition of Done (DoD)

- each branch provides a set of filters and selectors in a pandas-style API: `Series.functionA.functionB... -> Series`
- the three PositroniumSource branches provide filters and selectors of their own
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

### What this stage delivered

The chained API is registered on `pandas.Series` and `pandas.DataFrame`, so it
answers about every branch: a range on any column, the shapes on any triple of
position columns, the run and event a row belongs to, and the branches whose
values stand for something. A branch that a comparison already answers about -
`trackID`, `parentID`, `PDGEncoding` - keeps that comparison; the package adds
a name only where the data means something pandas has no word for.

Selecting by the detector identifier hierarchy, which is what `volumeID`
holds, is deliberately not here: it belongs with the geometry of version 1.1.0,
where a detector is described rather than guessed at from identifiers.

## Version 0.6.0 (status: completed)

Add physical unit converters and vector functionality:
- reconstruction of polarization from hit-position vectors
- decomposition of a vector into perpendicular and parallel components
- calculation of spherical vector components
- calculation of the angle between vectors
- calculation of the angle between planes

### Definition of Done (DoD)

- unit converters are available for energy, length, angle, and time
- dataframe operations can be performed on vectors as if they were vector objects
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.8.0 (status: completed)

There is no version 0.7.0. This stage was planned as 0.7.0, but its release
was published as 0.8.0, and PyPI never takes a version number a second time,
so the stage keeps the number it was released under and the stages after it
move on by one.

Add a representation of the "Hits" tree using the Event, Track, and Hit concepts:

- introduction of Event, Track, and Hit classes
- a new output file structure
- a set of filters and selectors

### Definition of Done (DoD)

- the "Hits" tree can be represented as a list of Event objects
- a flexible set of filters and selectors is available for Event, Track, and Hit classes
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.9.0 (status: planned)

Add a lazy Event-Track-Hit analysis:

- adding the ability to define an analysis as an object by applying operations to Event-Track-Hit elements
- combining multiple analyses into one
- performing the analysis only when data is provided for analysis

### Definition of Done (DoD)

- an advanced analysis can be defined as a chain of operations within a single object
- multiple analyses can be combined into one
- analyses performed using an `Analysis` object produce the same results as reference analyses performed using methods outside the package

## Version 0.10.0 (status: planned)

Add lazy Event-Track-Hit data smearing:

- adding the ability to define data smearing as a chain of operations within a single object
- combining multiple smearing operations into one
- performing smearing only when data is provided for smearing

### Definition of Done (DoD)

- advanced data smearing can be defined as a chain of operations within a single object
- multiple smearing operations can be combined into one
- smearing performed using a `Smearer` object produces the same results as reference analyses with smearing performed using methods outside the package
