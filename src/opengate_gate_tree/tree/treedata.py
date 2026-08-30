"""NumPy-backed representation of a GATE tree.

The module defines :class:`TreeData`, the in-memory representation shared by
every reader and writer in the package. Data is stored as NumPy arrays, which
is the common denominator of the supported output formats.

Two kinds of branches are supported:

- scalar branches, stored as one-dimensional arrays
- fixed-width array branches, stored as two-dimensional arrays of shape
  ``(entries, width)``; GATE uses them for branches such as ``volumeID``

Branches of varying length per entry are not supported and are rejected by the
readers before a :class:`TreeData` instance is built.

Arrays are referenced, not copied, so building a :class:`TreeData` from an
already loaded tree does not duplicate its memory. The mapping of columns is
read-only, but the arrays themselves stay writable.

A :class:`pandas.DataFrame` view is available through
:meth:`TreeData.to_dataframe`. Because a data frame holds scalar cells, a
fixed-width array branch is expanded there into one column per component,
named ``<branch>_<index>``. The expansion is one way: reading such a frame
back with :meth:`TreeData.from_dataframe` keeps the expanded columns separate.

Public objects
--------------
TreeData
    Immutable set of branch columns extracted from a single GATE tree.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from opengate_gate_tree.errors import BranchNotFoundError
from opengate_gate_tree.tree.gatetree import GateTree

# Number of dimensions of a scalar branch column.
SCALAR_BRANCH_NDIM: Final[int] = 1

# Number of dimensions of a fixed-width array branch column.
ARRAY_BRANCH_NDIM: Final[int] = 2


@dataclass(frozen=True, eq=False, repr=False)
class TreeData:
    """Branch columns extracted from a single GATE tree.

    Equality is intentionally not defined: comparing NumPy arrays with ``==``
    yields arrays rather than booleans, so a generated ``__eq__`` would raise
    instead of answering. Compare the fields explicitly when needed.

    Attributes
    ----------
    tree : GateTree
        Tree the columns were extracted from.
    columns : Mapping[str, numpy.ndarray]
        Branch name to column mapping. Exposed as a read-only mapping; the
        insertion order is the branch order of the data.
    """

    tree: GateTree
    columns: Mapping[str, npt.NDArray[Any]]

    def __post_init__(self) -> None:
        """Validate the columns and store them as a read-only mapping."""
        owned_columns = dict(self.columns)
        _validate_columns(owned_columns)
        object.__setattr__(self, "columns", MappingProxyType(owned_columns))

    @property
    def branch_names(self) -> tuple[str, ...]:
        """Branch names in their original order."""
        return tuple(self.columns)

    @property
    def entry_count(self) -> int:
        """Number of entries stored in every branch."""
        for column in self.columns.values():
            return int(column.shape[0])
        return 0

    @property
    def dtypes(self) -> Mapping[str, np.dtype[Any]]:
        """Branch name to NumPy data type mapping."""
        return MappingProxyType({name: column.dtype for name, column in self.columns.items()})

    @property
    def array_branches(self) -> Mapping[str, int]:
        """Fixed-width array branches mapped to their width."""
        return MappingProxyType(
            {
                name: int(column.shape[1])
                for name, column in self.columns.items()
                if column.ndim == ARRAY_BRANCH_NDIM
            }
        )

    def __len__(self) -> int:
        """Return the number of entries."""
        return self.entry_count

    def __getitem__(self, name: str) -> npt.NDArray[Any]:
        """Return the column of a single branch.

        Parameters
        ----------
        name : str
            Branch name.

        Returns
        -------
        numpy.ndarray
            Column of the requested branch.

        Raises
        ------
        BranchNotFoundError
            If the branch is not present.
        """
        try:
            return self.columns[name]
        except KeyError as err:
            raise BranchNotFoundError(_missing_branches_message([name], self.branch_names)) from err

    def select(self, names: Sequence[str]) -> "TreeData":
        """Return a new instance holding only the requested branches.

        Repeated names are kept once, at the position of their first
        occurrence.

        Parameters
        ----------
        names : Sequence[str]
            Branch names to keep, in the requested order.

        Returns
        -------
        TreeData
            New instance sharing the selected columns.

        Raises
        ------
        BranchNotFoundError
            If any requested branch is not present.
        """
        missing = [name for name in names if name not in self.columns]
        if missing:
            raise BranchNotFoundError(_missing_branches_message(missing, self.branch_names))
        return TreeData(self.tree, {name: self.columns[name] for name in names})

    def to_dataframe(self) -> pd.DataFrame:
        """Return the data as a pandas data frame.

        Scalar branches become one column each. Fixed-width array branches are
        expanded into one column per component, named ``<branch>_<index>`` with
        indices counted from zero, placed where the original branch was.

        Returns
        -------
        pandas.DataFrame
            Data frame holding every branch as a scalar column.

        Raises
        ------
        ValueError
            If expanding an array branch would collide with another column.
        """
        frame_columns: dict[str, npt.NDArray[Any]] = {}

        for name, column in self.columns.items():
            if column.ndim == ARRAY_BRANCH_NDIM:
                for index in range(column.shape[1]):
                    _add_frame_column(
                        frame_columns,
                        _expanded_branch_name(name, index),
                        column[:, index],
                        name,
                    )
            else:
                _add_frame_column(frame_columns, name, column, name)

        return pd.DataFrame(frame_columns)

    @classmethod
    def from_dataframe(cls, tree: GateTree, frame: pd.DataFrame) -> "TreeData":
        """Build an instance from a pandas data frame.

        Every column becomes a scalar branch and the index is dropped. Columns
        produced by expanding an array branch stay separate; they are not
        folded back into a two-dimensional branch.

        Parameters
        ----------
        tree : GateTree
            Tree the data belongs to.
        frame : pandas.DataFrame
            Data frame to convert.

        Returns
        -------
        TreeData
            Instance holding one branch per data frame column.

        Raises
        ------
        ValueError
            If the columns are a ``MultiIndex`` or contain repeated labels.
        """
        if isinstance(frame.columns, pd.MultiIndex):
            raise ValueError(
                "Data frames with MultiIndex columns are not supported; "
                "flatten the columns before converting."
            )

        repeated = [str(name) for name in frame.columns[frame.columns.duplicated()]]
        if repeated:
            raise ValueError(
                f"Data frame columns must be unique, but these are repeated: {repeated}."
            )

        columns = {str(name): frame[name].to_numpy() for name in frame.columns}
        return cls(tree, columns)

    def __repr__(self) -> str:
        """Return a compact representation that does not render the arrays."""
        return (
            f"{type(self).__name__}(tree={self.tree.value!r}, "
            f"entries={self.entry_count}, branches={len(self.columns)})"
        )


def _validate_columns(columns: Mapping[str, npt.NDArray[Any]]) -> None:
    """Check that the columns form a consistent tree.

    Raises
    ------
    ValueError
        If a branch name is empty, a column is not a NumPy array, a column has
        an unsupported number of dimensions, or the columns disagree on the
        number of entries.
    """
    entry_count: int | None = None

    for name, column in columns.items():
        if not name.strip():
            raise ValueError("Branch names must not be empty.")
        if not isinstance(column, np.ndarray):
            raise ValueError(f"Branch '{name}' must be a NumPy array, got {type(column).__name__}.")
        if column.ndim not in (SCALAR_BRANCH_NDIM, ARRAY_BRANCH_NDIM):
            raise ValueError(
                f"Branch '{name}' has {column.ndim} dimensions; only scalar branches "
                f"({SCALAR_BRANCH_NDIM}) and fixed-width array branches "
                f"({ARRAY_BRANCH_NDIM}) are supported."
            )

        if entry_count is None:
            entry_count = int(column.shape[0])
        elif int(column.shape[0]) != entry_count:
            raise ValueError(
                f"Branch '{name}' has {column.shape[0]} entries but {entry_count} were "
                f"expected; all branches must hold the same number of entries."
            )


def _missing_branches_message(missing: Sequence[str], available: Sequence[str]) -> str:
    """Build an error message listing missing and available branch names."""
    return f"Branches not found: {list(missing)}. Available branches: {list(available)}."


def _expanded_branch_name(name: str, index: int) -> str:
    """Return the data frame column name for one component of an array branch."""
    return f"{name}_{index}"


def _add_frame_column(
    frame_columns: dict[str, npt.NDArray[Any]],
    column_name: str,
    column: npt.NDArray[Any],
    branch_name: str,
) -> None:
    """Add one data frame column, rejecting names claimed by another branch.

    Raises
    ------
    ValueError
        If the column name is already taken.
    """
    if column_name in frame_columns:
        raise ValueError(
            f"Branch '{branch_name}' cannot be represented as column "
            f"'{column_name}' because another branch already uses that name."
        )
    frame_columns[column_name] = column
