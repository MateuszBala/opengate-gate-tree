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
