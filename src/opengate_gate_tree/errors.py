"""Exception hierarchy raised by the package.

All errors reported by the library derive from :class:`GateTreeError`, so code
using the package as a library can catch every package-specific failure with a
single ``except`` clause.

The hierarchy deliberately does not derive from ``ValueError``. A file that
cannot be opened is not an invalid value, and merging both cases would make it
impossible to tell argument validation apart from input/output failures.

Public objects
--------------
GateTreeError
    Base class for every error raised by the package.
RootFileError
    The ROOT file cannot be opened or is not a valid ROOT file.
TreeNotFoundError
    The requested tree is not present in the ROOT file.
BranchNotFoundError
    One or more requested branches are not present in the tree.
UnsupportedBranchTypeError
    A branch uses a type that the package does not support.
ExportError
    The output file cannot be written.
"""


class GateTreeError(Exception):
    """Base class for every error raised by the package."""


class RootFileError(GateTreeError):
    """Raised when a ROOT file cannot be opened or is not a valid ROOT file."""


class TreeNotFoundError(GateTreeError):
    """Raised when the requested tree is not present in the ROOT file."""


class BranchNotFoundError(GateTreeError):
    """Raised when requested branches are not present in the tree."""


class UnsupportedBranchTypeError(GateTreeError):
    """Raised when a branch uses a type that the package does not support."""


class ExportError(GateTreeError):
    """Raised when the output file cannot be written."""
