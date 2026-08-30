"""Shared test fixtures.

Two sources of test data are used together:

- a real GATE 9.x file, trimmed to 2000 entries, which guarantees that the
  readers work against branch names, types and file layout produced by GATE
- files generated with ``uproot``, which cover cases the real file does not
  contain, such as branches of varying length or a missing tree
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import numpy.typing as npt
import pytest
import uproot

# Directory holding the binary test fixtures.
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Real GATE 9.x output file, trimmed to 2000 entries of the "Hits" tree.
GATE_HITS_FIXTURE = FIXTURES_DIR / "data-only-hits-tree-small-size.root"

TreeColumns = Mapping[str, npt.NDArray[Any]]


@dataclass(frozen=True)
class GateHitsLayout:
    """Known properties of the "Hits" tree in the real GATE fixture file."""

    entries: int
    branch_count: int
    volume_id_width: int
    tree_names: tuple[str, ...]


@pytest.fixture(scope="session")
def gate_hits_file() -> Path:
    """Return the path of the real GATE fixture file."""
    return GATE_HITS_FIXTURE


@pytest.fixture(scope="session")
def gate_hits_layout() -> GateHitsLayout:
    """Return the properties the real GATE fixture file is expected to have."""
    return GateHitsLayout(
        entries=2000,
        branch_count=40,
        volume_id_width=10,
        tree_names=("pet_data", "Hits", "OpticalData"),
    )


@pytest.fixture
def make_gate_root_file(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a ROOT file with the given trees.

    Columns may be one-dimensional arrays, two-dimensional arrays of a fixed
    width, or object arrays of strings. A tree without entries is written with
    its branches declared but no data appended, because uproot cannot extend a
    tree that holds a text branch with an empty batch.
    """

    def write_root_file(
        trees: Mapping[str, TreeColumns],
        file_name: str = "generated.root",
    ) -> Path:
        path = tmp_path / file_name
        with uproot.recreate(path) as out:
            for tree_name, columns in trees.items():
                branch_types, branch_data = _branch_specification(columns)
                out.mktree(tree_name, branch_types)
                if _entry_count(columns):
                    out[tree_name].extend(branch_data)
        return path

    return write_root_file


@pytest.fixture
def make_jagged_root_file(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a ROOT file with a branch of varying length."""

    def write_jagged_file(
        tree_name: str = "Hits",
        file_name: str = "jagged.root",
    ) -> Path:
        path = tmp_path / file_name
        with uproot.recreate(path) as out:
            out.mktree(tree_name, {"eventID": np.int32, "hitTimes": "var * float64"})
            out[tree_name].extend(
                {
                    "eventID": np.arange(3, dtype=np.int32),
                    "hitTimes": ak.Array([[1.0], [2.0, 3.0], []]),
                }
            )
        return path

    return write_jagged_file


def _branch_specification(
    columns: TreeColumns,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split columns into uproot branch types and the matching data."""
    branch_types: dict[str, Any] = {}
    branch_data: dict[str, Any] = {}

    for name, column in columns.items():
        if column.dtype == object:
            branch_types[name] = str
            branch_data[name] = [str(value) for value in column]
        elif column.ndim == 2:
            branch_types[name] = (column.dtype, column.shape[1:])
            branch_data[name] = column
        else:
            branch_types[name] = column.dtype
            branch_data[name] = column

    return branch_types, branch_data


def _entry_count(columns: TreeColumns) -> int:
    """Return the number of entries described by the columns."""
    for column in columns.values():
        return int(column.shape[0])
    return 0
