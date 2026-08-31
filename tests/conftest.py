"""Shared test fixtures.

Two sources of test data are used together:

- a real GATE 9.x file, trimmed to 2000 entries, which guarantees that the
  readers work against branch names, types and file layout produced by GATE
- files generated with ``uproot``, which cover cases the real file does not
  contain, such as branches of varying length or a missing tree

The files in ``fixtures/hits-variants`` add one output per structure the "Hits"
tree can have. Their provenance and the reason one of them is not trimmed are
described in the README next to them.
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
from uproot.interpretation.strings import AsStrings

# Directory holding the binary test fixtures.
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Real GATE 9.x output file, trimmed to 2000 entries of the "Hits" tree.
GATE_HITS_FIXTURE = FIXTURES_DIR / "data-only-hits-tree-small-size.root"

# One simulation output per structure the "Hits" tree can have.
HITS_VARIANTS_DIR = FIXTURES_DIR / "hits-variants"

TreeColumns = Mapping[str, npt.NDArray[Any]]


@dataclass(frozen=True)
class HitsVariantLayout:
    """Measured properties of one "Hits" tree variant fixture.

    Attributes
    ----------
    key : str
        Short name the tests refer to the fixture by.
    file_name : str
        Name of the file in ``fixtures/hits-variants``.
    reference : str
        Label the reference material gives the structure, such as ``"A1"``.
    tree_names : tuple[str, ...]
        Trees stored in the file, in file order.
    entries : int
        Number of entries of every tree in the file.
    branch_count : int
        Number of branches of every tree in the file.
    """

    key: str
    file_name: str
    reference: str
    tree_names: tuple[str, ...]
    entries: int
    branch_count: int

    @property
    def path(self) -> Path:
        """Path of the fixture file."""
        return HITS_VARIANTS_DIR / self.file_name


# Properties of the variant fixtures, measured on the files they were cut from.
# "b1-tree-common-output.root" keeps every entry of the simulation, because the
# GateToTree layout cannot be rewritten by uproot; see the README next to it.
HITS_VARIANT_LAYOUTS: tuple[HitsVariantLayout, ...] = (
    HitsVariantLayout("a1", "a1-no-system.root", "A1", ("Hits",), 500, 40),
    HitsVariantLayout("a2", "a2-system.root", "A2", ("Hits",), 500, 46),
    HitsVariantLayout("a3", "a3-system-septal.root", "A3", ("Hits",), 500, 47),
    HitsVariantLayout("a4", "a4-no-system-cc.root", "A4", ("Hits",), 500, 30),
    HitsVariantLayout("a5", "a5-system-cc.root", "A5", ("Hits",), 500, 36),
    HitsVariantLayout("b1", "b1-tree-common-output.root", "B1", ("tree",), 4441, 54),
    HitsVariantLayout(
        "multi-run",
        "name-multi-run.root",
        "A1",
        ("Hits", "Hits_run1", "Hits_run2"),
        500,
        40,
    ),
    HitsVariantLayout(
        "multi-sd",
        "name-multi-sd.root",
        "A1",
        ("Hits_DET_INNER", "Hits_DET_OUTER"),
        500,
        40,
    ),
)


def branch_type_name(interpretation: Any) -> str:
    """Return a short name for the type an uproot interpretation reads.

    Parameters
    ----------
    interpretation : Any
        Interpretation of a branch, as reported by uproot.

    Returns
    -------
    str
        One of ``"int32"``, ``"float32"``, ``"float64"``, ``"text"`` or
        ``"<type>[<width>]"`` for a fixed-width array branch.
    """
    if isinstance(interpretation, AsStrings):
        return "text"
    dtype = interpretation.to_dtype
    scalar = dtype.base.newbyteorder("=")
    if dtype.shape:
        return f"{scalar.name}[{dtype.shape[0]}]"
    return str(scalar.name)


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


@pytest.fixture(scope="session")
def hits_variant_layouts() -> Mapping[str, HitsVariantLayout]:
    """Return the variant fixtures, keyed by their short name."""
    return {layout.key: layout for layout in HITS_VARIANT_LAYOUTS}


@pytest.fixture(scope="session")
def hits_variant_files() -> Mapping[str, Path]:
    """Return the paths of the variant fixtures, keyed by their short name."""
    return {layout.key: layout.path for layout in HITS_VARIANT_LAYOUTS}


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
