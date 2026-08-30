"""Unit tests for the NumPy-backed tree representation."""

from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from opengate_gate_tree.errors import BranchNotFoundError
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.treedata import TreeData

# Width of the volumeID branch produced by GATE.
VOLUME_ID_WIDTH = 10


def make_columns(entries: int = 3) -> dict[str, npt.NDArray[Any]]:
    """Build a column mapping shaped like a GATE "Hits" tree."""
    return {
        "eventID": np.arange(entries, dtype=np.int32),
        "edep": np.linspace(0.1, 1.0, entries).astype(np.float32),
        "volumeID": np.tile(
            np.arange(VOLUME_ID_WIDTH, dtype=np.int32),
            (entries, 1),
        ),
    }


def test_tree_data_exposes_branch_names_in_original_order() -> None:
    """Branch order should follow the order of the provided columns."""
    # ARRANGE
    columns = make_columns()

    # ACT
    data = TreeData(GateTree.HITS, columns)

    # ASSERT
    assert data.branch_names == ("eventID", "edep", "volumeID")


def test_tree_data_reports_entry_count_and_dtypes() -> None:
    """Entry count and data types should describe the stored columns."""
    # ARRANGE
    columns = make_columns(entries=5)

    # ACT
    data = TreeData(GateTree.HITS, columns)

    # ASSERT
    assert data.entry_count == 5
    assert len(data) == 5
    assert data.dtypes["eventID"] == np.dtype(np.int32)
    assert data.dtypes["edep"] == np.dtype(np.float32)


def test_tree_data_accepts_fixed_width_array_branch() -> None:
    """A two-dimensional branch should count as one branch of a given width."""
    # ARRANGE
    columns = make_columns(entries=4)

    # ACT
    data = TreeData(GateTree.HITS, columns)

    # ASSERT
    assert data.entry_count == 4
    assert data.array_branches == {"volumeID": VOLUME_ID_WIDTH}
    assert data["volumeID"].shape == (4, VOLUME_ID_WIDTH)


def test_tree_data_reports_no_array_branches_for_scalar_columns() -> None:
    """Scalar branches should not appear among array branches."""
    # ARRANGE
    columns = {"eventID": np.arange(3, dtype=np.int32)}

    # ACT
    data = TreeData(GateTree.HITS, columns)

    # ASSERT
    assert data.array_branches == {}


def test_tree_data_accepts_empty_column_mapping() -> None:
    """A tree without branches should be representable."""
    # ARRANGE
    columns: dict[str, npt.NDArray[Any]] = {}

    # ACT
    data = TreeData(GateTree.HITS, columns)

    # ASSERT
    assert data.entry_count == 0
    assert data.branch_names == ()


def test_tree_data_accepts_branches_without_entries() -> None:
    """A tree with branches but no entries should be representable."""
    # ARRANGE
    columns = {
        "eventID": np.array([], dtype=np.int32),
        "edep": np.array([], dtype=np.float32),
    }

    # ACT
    data = TreeData(GateTree.COINCIDENCES, columns)

    # ASSERT
    assert data.entry_count == 0
    assert data.branch_names == ("eventID", "edep")


def test_tree_data_rejects_columns_with_different_entry_counts() -> None:
    """Columns disagreeing on the number of entries should be rejected."""
    # ARRANGE
    columns = {
        "eventID": np.arange(3, dtype=np.int32),
        "edep": np.arange(2, dtype=np.float32),
    }

    # ACT & ASSERT
    with pytest.raises(ValueError, match="all branches must hold the same number of entries"):
        TreeData(GateTree.HITS, columns)


def test_tree_data_rejects_three_dimensional_column() -> None:
    """Only scalar and fixed-width array branches should be accepted."""
    # ARRANGE
    columns = {"volumeID": np.zeros((3, 2, 2), dtype=np.int32)}

    # ACT & ASSERT
    with pytest.raises(ValueError, match="has 3 dimensions"):
        TreeData(GateTree.HITS, columns)


@pytest.mark.parametrize("branch_name", ["", "   "])
def test_tree_data_rejects_empty_branch_name(branch_name: str) -> None:
    """Branch names must carry a value."""
    # ARRANGE
    columns = {branch_name: np.arange(3, dtype=np.int32)}

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Branch names must not be empty"):
        TreeData(GateTree.HITS, columns)


def test_tree_data_rejects_column_that_is_not_an_array() -> None:
    """Columns provided as plain sequences should be rejected with a clear message."""
    # ARRANGE
    columns = {"eventID": [1, 2, 3]}

    # ACT & ASSERT
    with pytest.raises(ValueError, match="must be a NumPy array, got list"):
        TreeData(GateTree.HITS, columns)  # type: ignore[arg-type]


def test_getitem_returns_the_stored_column() -> None:
    """Indexing should return the column of the requested branch."""
    # ARRANGE
    columns = make_columns()
    data = TreeData(GateTree.HITS, columns)

    # ACT
    column = data["eventID"]

    # ASSERT
    assert np.array_equal(column, columns["eventID"])


def test_getitem_raises_for_unknown_branch() -> None:
    """An unknown branch should report both the request and the alternatives."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT & ASSERT
    with pytest.raises(BranchNotFoundError) as error_info:
        data["missing"]

    assert "missing" in str(error_info.value)
    assert "eventID" in str(error_info.value)


def test_select_keeps_the_requested_order() -> None:
    """Selection order should follow the caller, not the stored order."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT
    selected = data.select(["edep", "eventID"])

    # ASSERT
    assert selected.branch_names == ("edep", "eventID")
    assert selected.tree == GateTree.HITS


def test_select_returns_a_new_instance_and_leaves_the_source_intact() -> None:
    """Selection should not modify the instance it was called on."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT
    selected = data.select(["eventID"])

    # ASSERT
    assert selected is not data
    assert data.branch_names == ("eventID", "edep", "volumeID")


def test_select_keeps_repeated_names_once() -> None:
    """A repeated branch name should be kept at its first position."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT
    selected = data.select(["edep", "eventID", "edep"])

    # ASSERT
    assert selected.branch_names == ("edep", "eventID")


def test_select_preserves_array_branch_shape() -> None:
    """Selecting an array branch should not flatten it."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns(entries=4))

    # ACT
    selected = data.select(["volumeID"])

    # ASSERT
    assert selected["volumeID"].shape == (4, VOLUME_ID_WIDTH)
    assert selected.array_branches == {"volumeID": VOLUME_ID_WIDTH}


def test_select_raises_for_unknown_branch() -> None:
    """Selection should report every missing branch at once."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT & ASSERT
    with pytest.raises(BranchNotFoundError) as error_info:
        data.select(["eventID", "missing", "absent"])

    assert "missing" in str(error_info.value)
    assert "absent" in str(error_info.value)


def test_columns_are_exposed_as_a_read_only_mapping() -> None:
    """Stored columns should not be replaceable through the public mapping."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT & ASSERT
    assert isinstance(data.columns, MappingProxyType)
    with pytest.raises(TypeError):
        data.columns["eventID"] = np.arange(3, dtype=np.int32)  # type: ignore[index]


def test_tree_data_is_frozen() -> None:
    """Attributes should not be reassignable."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT & ASSERT
    with pytest.raises(AttributeError):
        data.tree = GateTree.SINGLES  # type: ignore[misc]


def test_tree_data_does_not_track_later_changes_to_the_source_mapping() -> None:
    """The instance should own its column mapping."""
    # ARRANGE
    columns = make_columns()
    data = TreeData(GateTree.HITS, columns)

    # ACT
    columns["extra"] = np.arange(3, dtype=np.int32)

    # ASSERT
    assert data.branch_names == ("eventID", "edep", "volumeID")


def test_repr_stays_compact_for_large_trees() -> None:
    """The representation should summarise the data instead of rendering it."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns(entries=10_000))

    # ACT
    rendered = repr(data)

    # ASSERT
    assert rendered == "TreeData(tree='Hits', entries=10000, branches=3)"
