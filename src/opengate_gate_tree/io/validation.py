"""Consistency checks for GATE ROOT files.

The module holds the checks performed before and while a tree is extracted:
the input path, the presence of the requested tree and branches, and whether
the branch types are supported.

The functions take plain names rather than a :class:`~opengate_gate_tree.io.rootfile.RootFile`
so that the module stays independent of the reader and can be tested on its own.

Public functions
----------------
validate_root_file_path(path: Path) -> None
    Check that the path can point to a readable ROOT file.
find_tree_name(available: Sequence[str], tree: GateTree) -> str | None
    Look up the key of a tree without raising.
resolve_tree_name(available: Sequence[str], tree: GateTree, source: Path | None) -> str
    Look up the key of a tree, reporting the available trees when absent.
validate_branches_present(available: Sequence[str], requested: Sequence[str]) -> None
    Check that every requested branch exists in the tree.
validate_branch_interpretations(interpretations: Mapping[str, Any]) -> None
    Check that every branch uses a supported type.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from uproot.interpretation.numerical import AsDtype
from uproot.interpretation.strings import AsStrings

from opengate_gate_tree.errors import (
    BranchNotFoundError,
    RootFileError,
    TreeNotFoundError,
    UnsupportedBranchTypeError,
)
from opengate_gate_tree.tree.gatetree import GateTree

# File extension expected for GATE output files.
ROOT_FILE_SUFFIX: Final[str] = ".root"


def validate_root_file_path(path: Path) -> None:
    """Check that the path can point to a readable ROOT file.

    Parameters
    ----------
    path : Path
        Path to check.

    Raises
    ------
    RootFileError
        If the extension is wrong, the path does not exist, or it is not a
        regular file.
    """
    if path.suffix != ROOT_FILE_SUFFIX:
        raise RootFileError(
            f"Input file must have a '{ROOT_FILE_SUFFIX}' extension, got '{path.suffix}': {path}"
        )
    if not path.exists():
        raise RootFileError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise RootFileError(f"Input path is not a regular file: {path}")


def find_tree_name(available: Sequence[str], tree: GateTree) -> str | None:
    """Return the key under which a tree is stored, or ``None`` when absent.

    Names are compared exactly first, then without regard to case.

    Parameters
    ----------
    available : Sequence[str]
        Names of the trees present in the file.
    tree : GateTree
        Requested tree.

    Returns
    -------
    str | None
        Matching key, or ``None`` if the tree is not present.
    """
    if tree.value in available:
        return tree.value
    for name in available:
        if name.lower() == tree.value.lower():
            return name
    return None


def resolve_tree_name(
    available: Sequence[str],
    tree: GateTree,
    source: Path | None = None,
) -> str:
    """Return the key under which the requested tree is stored.

    Parameters
    ----------
    available : Sequence[str]
        Names of the trees present in the file.
    tree : GateTree
        Requested tree.
    source : Path | None
        File the names came from, named in the error message when given.

    Returns
    -------
    str
        Matching key.

    Raises
    ------
    TreeNotFoundError
        If no tree of the requested name is present.
    """
    resolved = find_tree_name(available, tree)
    if resolved is None:
        location = f" in file: {source}" if source is not None else " in the file"
        raise TreeNotFoundError(
            f"Tree '{tree.value}' is not present{location}. "
            f"Trees available in the file: {list(available)}."
        )
    return resolved


def validate_branches_present(available: Sequence[str], requested: Sequence[str]) -> None:
    """Check that every requested branch exists in the tree.

    Parameters
    ----------
    available : Sequence[str]
        Names of the branches present in the tree.
    requested : Sequence[str]
        Names of the requested branches.

    Raises
    ------
    BranchNotFoundError
        If any requested branch is missing.
    """
    known = set(available)
    missing = [name for name in requested if name not in known]
    if missing:
        raise BranchNotFoundError(
            f"Branches not found in the tree: {missing}. "
            f"Branches available in the tree: {list(available)}."
        )


def validate_branch_interpretations(interpretations: Mapping[str, Any]) -> None:
    """Check that every branch uses a type supported by the package.

    Scalar branches, fixed-width array branches and text branches are
    supported. Branches whose length varies per entry are not, because they
    have no representation in the supported output formats.

    The check runs on the uproot interpretation rather than on loaded data:
    text branches and branches of varying length both load as object arrays,
    so they cannot be told apart afterwards.

    Parameters
    ----------
    interpretations : Mapping[str, Any]
        Branch name to uproot interpretation mapping.

    Raises
    ------
    UnsupportedBranchTypeError
        If any branch uses an unsupported type.
    """
    unsupported = {
        name: str(interpretation)
        for name, interpretation in interpretations.items()
        if not isinstance(interpretation, AsDtype | AsStrings)
    }
    if unsupported:
        raise UnsupportedBranchTypeError(
            f"Branches use types that are not supported: {unsupported}. "
            f"Only scalar, fixed-width array and text branches can be extracted."
        )
