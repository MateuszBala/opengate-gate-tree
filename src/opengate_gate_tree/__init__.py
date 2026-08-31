"""Processing of GATE output file trees.

The package can be used as a command line tool and as a library. Everything
the command line does is reachable from here.

Reading a tree and writing it in another format:

.. code-block:: python

    from pathlib import Path

    from opengate_gate_tree import GateTree, OutputFileFormat, read_tree, write_tree

    data = read_tree(Path("simulation.root"), GateTree.HITS, ["eventID", "edep"])
    frame = data.to_dataframe()
    write_tree(data, Path("out/hits.hdf5"), OutputFileFormat.HDF5)

Use :class:`RootFile` directly when several trees are read from one file, so
that the file is opened once.

The structure of the "Hits" tree depends on the simulation that wrote it.
:class:`HitsTreeVariant` names the structures the package supports,
:func:`detect_hits_variant` recognises the one a tree has, and
:func:`expected_branches` states which branches each of them holds. Reading a
"Hits" tree recognises and checks its structure unless ``validate=False`` says
otherwise. Hits stored under another name, one tree per run or per sensitive
detector, are found by their structure, and the tree to read can be named.
:func:`read_hits_trees` reads them all as one dataset, recording which tree
each row came from.

:func:`compute_statistics` summarises what was extracted, and
:func:`write_statistics` saves that summary next to the data.

The branches a ``PositroniumSource`` writes carry integers whose meaning is
described by :class:`SourceType`, :class:`DecayType` and :class:`GammaType`.
Their members are those integers, so a column compares against them as it was
read. :func:`decode_positronium_value` and :func:`decode_positronium_column`
read those values as names, :func:`positronium_enum` says which class describes
a branch, and :func:`has_positronium_metadata` reads the ``decayIndex`` branch to
tell rows carrying the decay metadata of such a source from the rest.

Failures while reading or writing files are reported through a subclass of
:class:`GateTreeError`, so a single ``except`` clause covers them. Malformed
arguments, such as an empty branch name or inconsistent columns, raise
``ValueError`` instead, and are not caught by that clause.

The package does not configure logging on import. Applications decide that for
themselves; :func:`opengate_gate_tree.logging_setup.configure_logging` is
available for the ones that want the defaults used by the command line.

Public attributes:

__version__ : str
    Package version consistent with ``pyproject.toml``.
"""

# The version is defined before the re-exports below: the HDF5 writer records
# it in the files it writes, and importing it from a package that is still
# being initialised only works once the name exists.
__version__ = "0.3.0"

import logging  # noqa: E402

from opengate_gate_tree.errors import (  # noqa: E402
    AmbiguousTreeError,
    BranchNotFoundError,
    ExportError,
    GateTreeError,
    HitsTreeValidationError,
    RootFileError,
    TreeMergeError,
    TreeNotFoundError,
    UnknownHitsVariantError,
    UnsupportedBranchTypeError,
)
from opengate_gate_tree.io.fileformat import (  # noqa: E402
    OutputFileFormat,
    parse_output_file_format,
)
from opengate_gate_tree.io.naming import (  # noqa: E402
    build_output_file_name,
    build_output_file_path,
    build_statistics_file_path,
)
from opengate_gate_tree.io.reader import read_hits_trees, read_tree  # noqa: E402
from opengate_gate_tree.io.rootfile import RootFile  # noqa: E402
from opengate_gate_tree.io.statistics import write_statistics  # noqa: E402
from opengate_gate_tree.io.writers import write_tree  # noqa: E402
from opengate_gate_tree.logging_setup import LOGGER_NAME  # noqa: E402
from opengate_gate_tree.tree.gatetree import GateTree, parse_gate_tree  # noqa: E402
from opengate_gate_tree.tree.hits.detection import (  # noqa: E402
    HitsTreeDetection,
    describe_hits_tree,
    detect_hits_variant,
    summarise_hits_tree,
)
from opengate_gate_tree.tree.hits.positronium import (  # noqa: E402
    DECAY_INDEX_BRANCH,
    NO_POSITRONIUM_METADATA,
    POSITRONIUM_BRANCHES,
    DecayType,
    GammaType,
    SourceType,
    decode_positronium_column,
    decode_positronium_value,
    has_positronium_metadata,
    positronium_enum,
)
from opengate_gate_tree.tree.hits.schema import (  # noqa: E402
    BranchKind,
    BranchSpec,
    expected_branches,
    supported_variants,
    variant_reference,
)
from opengate_gate_tree.tree.hits.validation import validate_hits_tree  # noqa: E402
from opengate_gate_tree.tree.hits.variant import (  # noqa: E402
    GateSystemType,
    HitsTreeVariant,
)
from opengate_gate_tree.tree.merge import SOURCE_TREE_BRANCH, merge_tree_data  # noqa: E402
from opengate_gate_tree.tree.statistics import (  # noqa: E402
    BranchStatistics,
    HitsSummary,
    TreeStatistics,
    compute_statistics,
    format_statistics,
    statistics_to_dict,
)
from opengate_gate_tree.tree.treedata import TreeData  # noqa: E402

# Keep the package quiet when the application using it has not configured
# logging. Without a handler, logging falls back to writing warnings straight
# to stderr, ignoring the configuration of the surrounding application.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

__all__ = [
    "AmbiguousTreeError",
    "BranchKind",
    "BranchNotFoundError",
    "BranchSpec",
    "BranchStatistics",
    "DECAY_INDEX_BRANCH",
    "DecayType",
    "ExportError",
    "GammaType",
    "GateSystemType",
    "GateTree",
    "GateTreeError",
    "HitsSummary",
    "HitsTreeDetection",
    "HitsTreeValidationError",
    "HitsTreeVariant",
    "NO_POSITRONIUM_METADATA",
    "OutputFileFormat",
    "POSITRONIUM_BRANCHES",
    "RootFile",
    "RootFileError",
    "SOURCE_TREE_BRANCH",
    "SourceType",
    "TreeData",
    "TreeMergeError",
    "TreeNotFoundError",
    "TreeStatistics",
    "UnknownHitsVariantError",
    "UnsupportedBranchTypeError",
    "__version__",
    "build_output_file_name",
    "build_output_file_path",
    "build_statistics_file_path",
    "compute_statistics",
    "decode_positronium_column",
    "decode_positronium_value",
    "describe_hits_tree",
    "detect_hits_variant",
    "expected_branches",
    "format_statistics",
    "has_positronium_metadata",
    "merge_tree_data",
    "parse_gate_tree",
    "parse_output_file_format",
    "positronium_enum",
    "read_hits_trees",
    "read_tree",
    "statistics_to_dict",
    "summarise_hits_tree",
    "supported_variants",
    "validate_hits_tree",
    "variant_reference",
    "write_statistics",
    "write_tree",
]
