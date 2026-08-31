"""Merging trees a file stores under several names into one dataset.

GATE can split the hits of a simulation across several trees: one per run, or
one per sensitive detector. Each of them is a whole tree of the same
structure, so a merge is a concatenation of columns, in the order the trees
appear in the file.

Nothing else happens to the data. Rows are not sorted, identifiers are not
renumbered, and rows sharing a run and an event are not collapsed:

- sorting would restore no original order, because GATE writes hits in the
  order of the tracks within an event rather than by time;
- identifiers are what ties a row back to the simulation that produced it, and
  to the "Singles" and "Coincidences" trees written next to the hits;
- a run and an event repeating across trees is one event recorded in two
  detectors, which is the very thing a merge is performed to see.

The name of the tree each row came from is recorded in an added column. In a
file split per sensitive detector, the runs and the events of the two trees
are the same, so without it nothing in the data says which detector recorded
a deposit.

Public objects:

SOURCE_TREE_BRANCH
    Name of the column recording where a row came from.
merge_tree_data(parts, source_names, add_source_branch) -> TreeData
    Merge trees of one structure into a single dataset.
"""

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from opengate_gate_tree.errors import TreeMergeError
from opengate_gate_tree.tree.treedata import TreeData

# Name of the column recording which tree a row came from.
SOURCE_TREE_BRANCH: Final[str] = "sourceTreeName"


def merge_tree_data(
    parts: Sequence[TreeData],
    source_names: Sequence[str] | None = None,
    add_source_branch: bool = True,
) -> TreeData:
    """Merge trees of one structure into a single dataset.

    Parameters
    ----------
    parts : Sequence[TreeData]
        Trees to merge, in the order their rows should follow.
    source_names : Sequence[str] | None
        Name of the tree each part came from. Required while the source
        column is recorded, one name per part.
    add_source_branch : bool
        Whether to record where each row came from, in a column named
        ``sourceTreeName``. Turn it off when the result has to match the
        branches of the structure exactly, and when merging data that already
        carries the column, such as a merged dataset written out and read
        back.

    Returns
    -------
    TreeData
        The parts, one after another.

    Raises
    ------
    ValueError
        If no part was given, or the names do not account for every part.
    TreeMergeError
        If the parts describe different trees, hold different branches, store
        a branch with a different type or width, or already carry the source
        column while it is being recorded.
    """
    if not parts:
        raise ValueError("At least one tree is needed to merge.")
    if add_source_branch:
        if source_names is None:
            raise ValueError(
                "Names of the source trees are required to record where the rows came from."
            )
        if len(source_names) != len(parts):
            raise ValueError(
                f"Each tree needs a name: {len(parts)} tree(s) were given "
                f"and {len(source_names)} name(s)."
            )

    _validate_parts(parts, add_source_branch)

    columns: dict[str, npt.NDArray[Any]] = {
        name: _concatenate([part[name] for part in parts]) for name in parts[0].branch_names
    }
    if add_source_branch and source_names is not None:
        columns[SOURCE_TREE_BRANCH] = _source_column(parts, source_names)

    return TreeData(parts[0].tree, columns)


def _validate_parts(parts: Sequence[TreeData], add_source_branch: bool) -> None:
    """Check that the parts can be placed one after another."""
    first = parts[0]
    if add_source_branch and SOURCE_TREE_BRANCH in first.branch_names:
        raise TreeMergeError(
            f"The trees already hold a '{SOURCE_TREE_BRANCH}' branch, so recording where their "
            f"rows came from would overwrite it."
        )

    for index, part in enumerate(parts[1:], start=1):
        if part.tree is not first.tree:
            raise TreeMergeError(
                f"Trees of different kinds cannot be merged: '{first.tree.value}' and "
                f"'{part.tree.value}'."
            )
        if part.branch_names != first.branch_names:
            raise TreeMergeError(_branch_difference_message(index, first, part))
        _validate_column_types(index, first, part)


def _validate_column_types(index: int, first: TreeData, part: TreeData) -> None:
    """Check that both parts store every branch the same way."""
    for name in first.branch_names:
        expected, actual = first[name], part[name]
        if expected.dtype != actual.dtype or expected.shape[1:] != actual.shape[1:]:
            raise TreeMergeError(
                f"Branch '{name}' is stored differently in tree {index + 1}: "
                f"{_column_type(expected)} and {_column_type(actual)}."
            )


def _branch_difference_message(index: int, first: TreeData, part: TreeData) -> str:
    """Return the message describing how two parts disagree on branches."""
    missing = [name for name in first.branch_names if name not in set(part.branch_names)]
    extra = [name for name in part.branch_names if name not in set(first.branch_names)]
    if missing or extra:
        return (
            f"Tree {index + 1} holds other branches than the first one. "
            f"Missing from it: {missing}. Held only by it: {extra}."
        )
    return (
        f"Tree {index + 1} holds the same branches in another order: "
        f"{list(part.branch_names)} against {list(first.branch_names)}."
    )


def _column_type(column: npt.NDArray[Any]) -> str:
    """Return how a column is stored, for use in a message."""
    if column.ndim == 1:
        return str(column.dtype)
    return f"{column.dtype}{list(column.shape[1:])}"


def _concatenate(columns: Sequence[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """Return the columns placed one after another."""
    return np.concatenate(columns)


def _source_column(
    parts: Sequence[TreeData],
    source_names: Sequence[str],
) -> npt.NDArray[Any]:
    """Return the column naming the tree each row came from."""
    values: list[str] = []
    for part, name in zip(parts, source_names, strict=True):
        values.extend([name] * part.entry_count)
    return np.array(values, dtype=object)
