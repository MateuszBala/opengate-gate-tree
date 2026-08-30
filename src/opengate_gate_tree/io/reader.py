"""Extraction of GATE trees into the package representation.

The module provides :func:`read_tree`, the entry point used by the command
line interface and by code using the package as a library. It opens a GATE
ROOT file, extracts one tree and closes the file again.

Use :class:`~opengate_gate_tree.io.rootfile.RootFile` directly when several
trees are read from the same file, so that the file is opened only once.

Public functions
----------------
read_tree(path: Path, tree: GateTree, branches: Sequence[str] | None) -> TreeData
    Read one tree from a GATE ROOT file.
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
    BranchNotFoundError
        If any requested branch is not present in the tree.
    UnsupportedBranchTypeError
        If any requested branch uses an unsupported type.
    ValueError
        If any requested branch name is empty.
    """
    with RootFile(path) as root_file:
        return root_file.read(tree, branches)
