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

## Version 0.3.0 (status: planned)

Basic support for the "Hits" tree:

- branch definitions depending on the simulation type
- tree validation
- statistics generation

### Definition of Done (DoD)

- all versions of the "Hits" tree structure are fully supported
- all versions of the "Hits" tree structure have dedicated validation
- output files are named using the format `<title>.hits.<file-format>`
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.4.0 (status: planned)

Add a set of basic filters and selectors for data from the "Hits" tree.

### Definition of Done (DoD)

- each branch provides a set of filters and selectors in a pandas-style API: `Series.functionA.functionB... -> Series`
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.5.0 (status: planned)

Add support for representing PositroniuSource data in the "Hits" tree:

- casting `gammaType`, `sourceType`, and `decayType` branches to enum representations
- filters and selectors for PositroniuSource data

### Definition of Done (DoD)

- each of the three branches has its own representation as an enum class
- each of the three branches provides a set of filters and selectors in a pandas-style API: `Series.functionA.functionB... -> Series`
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.6.0 (status: planned)

Add a representation of the "Hits" tree using the Event, Track, and Hit concepts:

- introduction of Event, Track, and Hit classes
- a new output file structure
- a set of filters and selectors

### Definition of Done (DoD)

- the "Hits" tree can be represented as a list of Event objects
- a flexible set of filters and selectors is available for Event, Track, and Hit classes
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.7.0 (status: planned)

Add a set of smearing functions for the "Hits" tree and the Event-Track-Hit representation:

- position smearing
- time smearing
- energy smearing

### Definition of Done (DoD)

- a broad set of methods for smearing time, position, and energy is provided
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.8.0 (status: planned)

Add functionality for merging hits from the "Hits" tree and the Event-Track-Hit representation.

### Definition of Done (DoD)

- hit merging in tree-based representation follows the GATE convention for the digitizer adder
- hit merging in Event-Track-Hit representation follows the GATE convention for the digitizer adder
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.9.0 (status: planned)

Optimization of "Hits" tree handling.

### Definition of Done (DoD)

- reduced CPU time and RAM usage
- faster data processing
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.10.0 (status: planned)

Basic support for the "Singles" tree:

- branch definitions
- tree validation
- statistics generation

### Definition of Done (DoD)

- all versions of the "Singles" tree structure are fully supported
- all versions of the "Singles" tree structure have dedicated validation
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.11.0 (status: planned)

Add a set of filters and selectors for data from the "Singles" tree.

### Definition of Done (DoD)

- each branch provides a set of filters and selectors in a pandas-style API: `Series.functionA.functionB... -> Series`
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.12.0 (status: planned)

Add a representation of the "Singles" tree using the SinglesEvent and Single concepts:

- introduction of SinglesEvent and Single classes
- a new output file structure
- a set of filters and selectors

### Definition of Done (DoD)

- the "Singles" tree can be represented as a list of SinglesEvent objects
- a flexible set of filters and selectors is available for SinglesEvent and Single classes
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.13.0 (status: planned)

Optimization of "Singles" tree handling.

### Definition of Done (DoD)

- reduced CPU time and RAM usage
- faster data processing
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.14.0 (status: planned)

Basic support for the "Coincidences" tree:

- branch definitions
- tree validation
- statistics generation

### Definition of Done (DoD)

- all versions of the "Coincidences" tree structure are fully supported
- all versions of the "Coincidences" tree structure have dedicated validation
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.15.0 (status: planned)

Add a set of filters and selectors for data from the "Coincidences" tree.

### Definition of Done (DoD)

- each branch provides a set of filters and selectors in a pandas-style API: `Series.functionA.functionB... -> Series`
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 0.16.0 (status: planned)

Optimization of "Coincidences" tree handling.

### Definition of Done (DoD)

- reduced CPU time and RAM usage
- faster data processing
- all tests pass
- user documentation (ReadTheDocs) clearly and comprehensively covers the added representations and features

## Version 1.0.0 (status: planned)

The package is fully ready to support all branches from GATE output files.

### Definition of Done (DoD)

- all package features work correctly
- all tests pass
- user documentation (ReadTheDocs) is complete

## Version 1.1.0 (status: planned)

Add detector geometry support:

- defining the detector geometry used in a simulation through a Detector class
- visualizing detector geometry
- filtering and selecting "Hits" tree content by detector position
- filtering Hit-class objects by detector position

### Definition of Done (DoD)

- flexible definition and visualization of detector geometry
- a broad set of methods for geometry-based data filtering
- all tests pass
- user documentation (ReadTheDocs) is complete

## Version 1.2.0 (status: planned)

Add conversion of GATE macros that define a detector into Detector definitions:

- macro loading
- macro validation
- macro conversion into a Detector structure

### Definition of Done (DoD)

- full support for converting macros into package representations
- all tests pass
- user documentation (ReadTheDocs) is complete