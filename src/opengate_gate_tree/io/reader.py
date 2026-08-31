"""Extraction of GATE trees into the package representation.

The module provides :func:`read_tree`, the entry point used by the command
line interface and by code using the package as a library. It opens a GATE
ROOT file, extracts one tree and closes the file again.

Use :class:`~opengate_gate_tree.io.rootfile.RootFile` directly when several
trees are read from the same file, so that the file is opened only once.

Public functions:

read_tree(path, tree, branches, tree_name, validate) -> TreeData
    Read one tree from a GATE ROOT file.
read_hits_trees(path, branches, tree_names, validate, add_source_branch) -> TreeData
    Read the hits of a GATE ROOT file as a single dataset.
"""

from collections.abc import Sequence
from pathlib import Path

from opengate_gate_tree.io.rootfile import RootFile
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.treedata import TreeData


def read_tree(
    path: Path,
    tree: GateTree,
    branches: Sequence[str] | None = None,
    tree_name: str | None = None,
    validate: bool = True,
) -> TreeData:
    """Read one tree from a GATE ROOT file.

    Parameters
    ----------
    path : Path
        Path to the GATE ROOT file.
    tree : GateTree
        Tree to read.
    branches : Sequence[str] | None
        Branches to read. When omitted or empty, every branch of the tree is
        read. Repeated names are read once, at the position of their first
        occurrence.
    tree_name : str | None
        Name of the tree in the file, when it differs from the standard one or
        when a file holds several trees of hits.
    validate : bool
        Whether to recognise the structure of the "Hits" tree and check the
        tree against it before reading. Turn it off to extract branches from a
        file whose structure the package does not know.

    Returns
    -------
    TreeData
        Columns of the requested branches.

    Raises
    ------
    RootFileError
        If the path is not a readable ROOT file.
    TreeNotFoundError
        If the tree is not present in the file.
    AmbiguousTreeError
        If several trees hold hits and none of them was named.
    UnknownHitsVariantError
        If the structure of the "Hits" tree is not a supported one.
    HitsTreeValidationError
        If the "Hits" tree does not match the structure it was recognised as.
    BranchNotFoundError
        If any requested branch is not present in the tree.
    UnsupportedBranchTypeError
        If any requested branch uses an unsupported type.
    ValueError
        If any requested branch name is empty.
    """
    with RootFile(path) as root_file:
        return root_file.read(tree, branches, tree_name=tree_name, validate=validate)


def read_hits_trees(
    path: Path,
    branches: Sequence[str] | None = None,
    tree_names: Sequence[str] | None = None,
    validate: bool = True,
    add_source_branch: bool = True,
) -> TreeData:
    """Read the hits of a GATE ROOT file as a single dataset.

    A file splitting its hits into one tree per run, or one per sensitive
    detector, is read as one dataset with the trees placed one after another.
    Identifiers are left as GATE wrote them, so an event is still told apart
    by its run and its event identifier together.

    Parameters
    ----------
    path : Path
        Path to the GATE ROOT file.
    branches : Sequence[str] | None
        Branches to read from every tree. When omitted or empty, every branch
        is read.
    tree_names : Sequence[str] | None
        Trees to read, in the order their rows should follow. When omitted,
        every tree of the file holding hits is read, in file order.
    validate : bool
        Whether to recognise the structure of each tree and check it.
    add_source_branch : bool
        Whether to record which tree each row came from, in a column named
        ``sourceTreeName``.

    Returns
    -------
    TreeData
        Rows of every tree that was read, one tree after another.

    Raises
    ------
    RootFileError
        If the path is not a readable ROOT file.
    TreeNotFoundError
        If the file holds no hits, or a named tree is not present.
    UnknownHitsVariantError
        If the structure of a tree is not a supported one.
    HitsTreeValidationError
        If a tree does not match the structure it was recognised as.
    TreeMergeError
        If the trees do not hold the same structure.
    BranchNotFoundError
        If any requested branch is not present in a tree.
    """
    with RootFile(path) as root_file:
        return root_file.read_hits(
            branches,
            tree_names,
            validate=validate,
            add_source_branch=add_source_branch,
        )
