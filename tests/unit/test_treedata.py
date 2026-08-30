"""Unit tests for the NumPy-backed tree representation."""

from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
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


def test_to_dataframe_keeps_scalar_branches_and_their_order() -> None:
    """Scalar branches should map to columns of the same name and order."""
    # ARRANGE
    columns = {
        "eventID": np.arange(3, dtype=np.int32),
        "edep": np.linspace(0.1, 1.0, 3).astype(np.float32),
    }
    data = TreeData(GateTree.HITS, columns)

    # ACT
    frame = data.to_dataframe()

    # ASSERT
    assert list(frame.columns) == ["eventID", "edep"]
    assert len(frame) == 3
    assert np.array_equal(frame["eventID"].to_numpy(), columns["eventID"])


def test_to_dataframe_expands_array_branch_in_place() -> None:
    """An array branch should become one column per component, where it stood."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns(entries=4))

    # ACT
    frame = data.to_dataframe()

    # ASSERT
    expected_columns = [
        "eventID",
        "edep",
        *[f"volumeID_{index}" for index in range(VOLUME_ID_WIDTH)],
    ]
    assert list(frame.columns) == expected_columns
    assert np.array_equal(frame["volumeID_3"].to_numpy(), np.full(4, 3, dtype=np.int32))


def test_to_dataframe_returns_a_plain_dataframe() -> None:
    """The frame must be a plain pandas object so pandas accessors keep working."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns())

    # ACT
    frame = data.to_dataframe()

    # ASSERT
    assert type(frame) is pd.DataFrame


def test_to_dataframe_rejects_collision_between_expansion_and_scalar_branch() -> None:
    """An expanded name already used by another branch should be reported."""
    # ARRANGE
    columns = {
        "volumeID_0": np.arange(3, dtype=np.int32),
        "volumeID": np.tile(np.arange(2, dtype=np.int32), (3, 1)),
    }
    data = TreeData(GateTree.HITS, columns)

    # ACT & ASSERT
    with pytest.raises(ValueError, match="another branch already uses that name"):
        data.to_dataframe()


def test_to_dataframe_handles_a_tree_without_entries() -> None:
    """A tree with branches but no entries should give an empty frame."""
    # ARRANGE
    columns = {"eventID": np.array([], dtype=np.int32)}
    data = TreeData(GateTree.SINGLES, columns)

    # ACT
    frame = data.to_dataframe()

    # ASSERT
    assert list(frame.columns) == ["eventID"]
    assert len(frame) == 0


def test_from_dataframe_builds_branches_from_columns() -> None:
    """Every column should become a branch of the same name."""
    # ARRANGE
    frame = pd.DataFrame(
        {
            "eventID": np.arange(3, dtype=np.int32),
            "edep": np.linspace(0.1, 1.0, 3).astype(np.float32),
        }
    )

    # ACT
    data = TreeData.from_dataframe(GateTree.HITS, frame)

    # ASSERT
    assert data.tree == GateTree.HITS
    assert data.branch_names == ("eventID", "edep")
    assert data.entry_count == 3


def test_from_dataframe_drops_the_index() -> None:
    """Index values should not leak into the branch data."""
    # ARRANGE
    frame = pd.DataFrame({"eventID": np.arange(3, dtype=np.int32)}, index=[10, 11, 12])

    # ACT
    data = TreeData.from_dataframe(GateTree.HITS, frame)

    # ASSERT
    assert np.array_equal(data["eventID"], np.arange(3, dtype=np.int32))


def test_from_dataframe_rejects_multiindex_columns() -> None:
    """Nested column labels have no branch equivalent."""
    # ARRANGE
    frame = pd.DataFrame(
        np.arange(6).reshape(3, 2),
        columns=pd.MultiIndex.from_tuples([("hits", "eventID"), ("hits", "edep")]),
    )

    # ACT & ASSERT
    with pytest.raises(ValueError, match="MultiIndex columns are not supported"):
        TreeData.from_dataframe(GateTree.HITS, frame)


def test_from_dataframe_rejects_labels_that_collide_once_converted() -> None:
    """Labels distinct to pandas can still name the same branch."""
    # ARRANGE
    frame = pd.DataFrame({1: [1, 2], "1": [3, 4]})

    # ACT & ASSERT
    with pytest.raises(ValueError, match="repeated"):
        TreeData.from_dataframe(GateTree.HITS, frame)


def test_from_dataframe_rejects_repeated_column_labels() -> None:
    """Repeated labels would silently produce a two-dimensional branch."""
    # ARRANGE
    frame = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["eventID", "eventID"])

    # ACT & ASSERT
    with pytest.raises(ValueError, match="repeated"):
        TreeData.from_dataframe(GateTree.HITS, frame)


def test_dataframe_round_trip_preserves_scalar_values_and_types() -> None:
    """Numeric branches should survive a conversion to a frame and back."""
    # ARRANGE
    columns = {
        "eventID": np.arange(5, dtype=np.int32),
        "edep": np.linspace(0.1, 1.0, 5).astype(np.float32),
    }
    data = TreeData(GateTree.HITS, columns)

    # ACT
    restored = TreeData.from_dataframe(GateTree.HITS, data.to_dataframe())

    # ASSERT
    assert restored.branch_names == data.branch_names
    for name in data.branch_names:
        assert np.array_equal(restored[name], data[name])
        assert restored.dtypes[name] == data.dtypes[name]


def test_dataframe_round_trip_preserves_text_values() -> None:
    """Text branches should survive the round trip and stay text afterwards."""
    # ARRANGE
    process_names = np.array(["Compton", "PhotoElectric", "NULL"], dtype=object)
    data = TreeData(GateTree.HITS, {"processName": process_names})

    # ACT
    restored = TreeData.from_dataframe(GateTree.HITS, data.to_dataframe())

    # ASSERT
    assert np.array_equal(restored["processName"], process_names)
    assert restored.dtypes["processName"].kind == "O"


def test_dataframe_round_trip_keeps_array_branch_expanded() -> None:
    """Expansion is one way: components stay separate branches after the round trip."""
    # ARRANGE
    data = TreeData(GateTree.HITS, make_columns(entries=4))

    # ACT
    restored = TreeData.from_dataframe(GateTree.HITS, data.to_dataframe())

    # ASSERT
    assert "volumeID" not in restored.branch_names
    assert restored.array_branches == {}
    assert restored.branch_names[2:] == tuple(
        f"volumeID_{index}" for index in range(VOLUME_ID_WIDTH)
    )
