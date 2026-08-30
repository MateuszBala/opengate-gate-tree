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
__version__ = "0.1.1"

import logging  # noqa: E402

from opengate_gate_tree.errors import (  # noqa: E402
    BranchNotFoundError,
    ExportError,
    GateTreeError,
    RootFileError,
    TreeNotFoundError,
    UnsupportedBranchTypeError,
)
from opengate_gate_tree.io.fileformat import (  # noqa: E402
    OutputFileFormat,
    parse_output_file_format,
)
from opengate_gate_tree.io.reader import read_tree  # noqa: E402
from opengate_gate_tree.io.rootfile import RootFile  # noqa: E402
from opengate_gate_tree.io.writers import write_tree  # noqa: E402
from opengate_gate_tree.logging_setup import LOGGER_NAME  # noqa: E402
from opengate_gate_tree.tree.gatetree import GateTree, parse_gate_tree  # noqa: E402
from opengate_gate_tree.tree.treedata import TreeData  # noqa: E402

# Keep the package quiet when the application using it has not configured
# logging. Without a handler, logging falls back to writing warnings straight
# to stderr, ignoring the configuration of the surrounding application.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

__all__ = [
    "BranchNotFoundError",
    "ExportError",
    "GateTree",
    "GateTreeError",
    "OutputFileFormat",
    "RootFile",
    "RootFileError",
    "TreeData",
    "TreeNotFoundError",
    "UnsupportedBranchTypeError",
    "__version__",
    "parse_gate_tree",
    "parse_output_file_format",
    "read_tree",
    "write_tree",
]
