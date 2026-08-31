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
__version__ = "0.2.1"

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
from opengate_gate_tree.io.reader import read_hits_trees, read_tree  # noqa: E402
from opengate_gate_tree.io.rootfile import RootFile  # noqa: E402
from opengate_gate_tree.io.writers import write_tree  # noqa: E402
from opengate_gate_tree.logging_setup import LOGGER_NAME  # noqa: E402
from opengate_gate_tree.tree.gatetree import GateTree, parse_gate_tree  # noqa: E402
from opengate_gate_tree.tree.hits.detection import (  # noqa: E402
    HitsTreeDetection,
    describe_hits_tree,
    detect_hits_variant,
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
    "ExportError",
    "GateSystemType",
    "GateTree",
    "GateTreeError",
    "HitsTreeDetection",
    "HitsTreeValidationError",
    "HitsTreeVariant",
    "OutputFileFormat",
    "RootFile",
    "SOURCE_TREE_BRANCH",
    "RootFileError",
    "TreeData",
    "TreeMergeError",
    "TreeNotFoundError",
    "UnknownHitsVariantError",
    "UnsupportedBranchTypeError",
    "__version__",
    "describe_hits_tree",
    "detect_hits_variant",
    "expected_branches",
    "merge_tree_data",
    "parse_gate_tree",
    "parse_output_file_format",
    "read_hits_trees",
    "read_tree",
    "supported_variants",
    "validate_hits_tree",
    "variant_reference",
    "write_tree",
]
