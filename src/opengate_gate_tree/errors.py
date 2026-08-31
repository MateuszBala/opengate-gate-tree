"""Exception hierarchy raised by the package.

All errors reported by the library derive from :class:`GateTreeError`, so code
using the package as a library can catch every package-specific failure with a
single ``except`` clause.

The hierarchy deliberately does not derive from ``ValueError``. A file that
cannot be opened is not an invalid value, and merging both cases would make it
impossible to tell argument validation apart from input/output failures.

Public objects:

GateTreeError
    Base class for every error raised by the package.
RootFileError
    The ROOT file cannot be opened or is not a valid ROOT file.
TreeNotFoundError
    The requested tree is not present in the ROOT file.
AmbiguousTreeError
    Several trees in the file could be the requested one.
BranchNotFoundError
    One or more requested branches are not present in the tree.
UnsupportedBranchTypeError
    A branch uses a type that the package does not support.
UnknownHitsVariantError
    The structure of a "Hits" tree could not be recognised, or was recognised
    as one the package does not support.
HitsTreeValidationError
    A "Hits" tree does not match the structure it was recognised as.
TreeMergeError
    Trees cannot be placed one after another into a single dataset.
ExportError
    The output file cannot be written.
"""


class GateTreeError(Exception):
    """Base class for every error raised by the package."""


class RootFileError(GateTreeError):
    """Raised when a ROOT file cannot be opened or is not a valid ROOT file."""


class TreeNotFoundError(GateTreeError):
    """Raised when the requested tree is not present in the ROOT file."""


class AmbiguousTreeError(GateTreeError):
    """Raised when several trees in a file could be the requested one.

    A file can hold the hits of one run per tree, or of one sensitive
    detector per tree. Picking one of them would decide something the caller
    never asked the package to decide, so they are all reported instead.
    """


class BranchNotFoundError(GateTreeError):
    """Raised when requested branches are not present in the tree."""


class UnsupportedBranchTypeError(GateTreeError):
    """Raised when a branch uses a type that the package does not support."""


class UnknownHitsVariantError(GateTreeError):
    """Raised when the structure of a "Hits" tree is not a supported one.

    Covers both a tree whose branches match none of the known structures and
    one recognised as a structure the package does not support, such as the
    output of the Compton camera actor.
    """


class HitsTreeValidationError(GateTreeError):
    """Raised when a "Hits" tree does not match its recognised structure.

    Covers a branch the structure describes but the tree does not hold, and a
    branch stored with a type other than the expected one. Branches beyond the
    structure are reported as a warning instead, because GATE builds routinely
    add them.
    """


class TreeMergeError(GateTreeError):
    """Raised when trees cannot be merged into a single dataset.

    Trees stored under several names hold the same structure when they come
    from one simulation. Different branches, or a branch stored with another
    type, mean they do not, and concatenating them would produce a dataset
    that describes nothing.
    """


class ExportError(GateTreeError):
    """Raised when the output file cannot be written."""
