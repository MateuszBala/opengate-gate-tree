"""Unit tests for merging trees stored under several names."""

import numpy as np
import pytest

from opengate_gate_tree.errors import TreeMergeError
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.merge import SOURCE_TREE_BRANCH, merge_tree_data
from opengate_gate_tree.tree.treedata import TreeData

# Names given to the parts of a merge in most of the tests.
TREE_NAMES = ("Hits_DET_INNER", "Hits_DET_OUTER")


def hits_part(
    event_ids: list[int],
    run_id: int = 0,
    process: str = "Compton",
) -> TreeData:
    """Return a small tree of hits with the given events."""
    entries = len(event_ids)
    return TreeData(
        GateTree.HITS,
        {
            "runID": np.full(entries, run_id, dtype=np.int32),
            "eventID": np.array(event_ids, dtype=np.int32),
            "edep": np.arange(entries, dtype=np.float32),
            "volumeID": np.arange(entries * 10, dtype=np.int32).reshape(entries, 10),
            "processName": np.array([process] * entries, dtype=object),
        },
    )


def test_rows_follow_the_order_of_the_parts() -> None:
    """The result reads as the trees read, one after another."""
    # ARRANGE
    parts = [hits_part([0, 1]), hits_part([7, 8], run_id=1)]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES)

    # ASSERT
    assert merged.entry_count == 4
    assert merged["eventID"].tolist() == [0, 1, 7, 8]
    assert merged["runID"].tolist() == [0, 0, 1, 1]


def test_every_row_records_the_tree_it_came_from() -> None:
    """Which detector recorded a deposit is not in the data otherwise."""
    # ARRANGE
    parts = [hits_part([0, 1]), hits_part([0, 1])]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES)

    # ASSERT
    assert list(merged[SOURCE_TREE_BRANCH]) == [
        "Hits_DET_INNER",
        "Hits_DET_INNER",
        "Hits_DET_OUTER",
        "Hits_DET_OUTER",
    ]


def test_the_source_column_can_be_left_out() -> None:
    """A result feeding a tool expecting the structure exactly needs no extras."""
    # ARRANGE
    parts = [hits_part([0]), hits_part([1])]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES, add_source_branch=False)

    # ASSERT
    assert merged.branch_names == parts[0].branch_names


def test_identifiers_repeat_across_parts_untouched() -> None:
    """One event in two detectors is one event, and stays numbered as one."""
    # ARRANGE
    # Both trees hold run 0, event 4: the same decay seen twice.
    parts = [hits_part([4, 5]), hits_part([4, 9])]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES)

    # ASSERT
    assert merged["eventID"].tolist() == [4, 5, 4, 9]
    assert merged["runID"].tolist() == [0, 0, 0, 0]


def test_rows_are_not_reordered() -> None:
    """GATE writes hits per track, not by time, so there is no order to restore."""
    # ARRANGE
    parts = [hits_part([9, 2, 7]), hits_part([1])]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES)

    # ASSERT
    assert merged["eventID"].tolist() == [9, 2, 7, 1]


def test_array_branches_keep_their_width() -> None:
    """The detector hierarchy is one value per row, and stays that way."""
    # ARRANGE
    parts = [hits_part([0, 1]), hits_part([2])]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES)

    # ASSERT
    assert merged["volumeID"].shape == (3, 10)


def test_text_branches_keep_their_values() -> None:
    """Text columns are concatenated like any other."""
    # ARRANGE
    parts = [hits_part([0], process="Compton"), hits_part([1], process="Rayleigh")]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES)

    # ASSERT
    assert list(merged["processName"]) == ["Compton", "Rayleigh"]


def test_a_part_without_entries_contributes_nothing() -> None:
    """A run that recorded no hits is still a run of the simulation."""
    # ARRANGE
    parts = [hits_part([0, 1]), hits_part([])]

    # ACT
    merged = merge_tree_data(parts, TREE_NAMES)

    # ASSERT
    assert merged.entry_count == 2
    assert list(merged[SOURCE_TREE_BRANCH]) == ["Hits_DET_INNER", "Hits_DET_INNER"]


def test_a_single_part_is_merged_with_itself_alone() -> None:
    """Reading a file that happens to hold one tree needs no special case."""
    # ARRANGE
    part = hits_part([0, 1])

    # ACT
    merged = merge_tree_data([part], ["Hits"])

    # ASSERT
    assert merged.entry_count == 2
    assert set(merged[SOURCE_TREE_BRANCH]) == {"Hits"}


def test_parts_holding_other_branches_are_refused() -> None:
    """Concatenating trees of different structures describes nothing."""
    # ARRANGE
    first = hits_part([0])
    second = first.select(["runID", "eventID"])

    # ACT
    with pytest.raises(TreeMergeError) as raised:
        merge_tree_data([first, second], TREE_NAMES)

    # ASSERT
    assert "Missing from it" in str(raised.value)
    assert "edep" in str(raised.value)


def test_parts_holding_the_branches_in_another_order_are_refused() -> None:
    """Branch order is part of what the package promises about a dataset."""
    # ARRANGE
    first = hits_part([0])
    second = first.select(["eventID", "runID", "edep", "volumeID", "processName"])

    # ACT
    with pytest.raises(TreeMergeError) as raised:
        merge_tree_data([first, second], TREE_NAMES)

    # ASSERT
    assert "another order" in str(raised.value)


def test_parts_storing_a_branch_differently_are_refused() -> None:
    """Concatenating an integer with a float would rewrite both of them."""
    # ARRANGE
    first = hits_part([0])
    columns = dict(first.columns)
    columns["eventID"] = columns["eventID"].astype(np.int64)
    second = TreeData(GateTree.HITS, columns)

    # ACT
    with pytest.raises(TreeMergeError) as raised:
        merge_tree_data([first, second], TREE_NAMES)

    # ASSERT
    assert "'eventID' is stored differently" in str(raised.value)


def test_parts_storing_an_array_of_another_width_are_refused() -> None:
    """A hierarchy of another depth is another detector, not more rows."""
    # ARRANGE
    first = hits_part([0])
    columns = dict(first.columns)
    columns["volumeID"] = np.arange(4, dtype=np.int32).reshape(1, 4)
    second = TreeData(GateTree.HITS, columns)

    # ACT
    with pytest.raises(TreeMergeError) as raised:
        merge_tree_data([first, second], TREE_NAMES)

    # ASSERT
    assert "volumeID" in str(raised.value)


def test_parts_describing_different_trees_are_refused() -> None:
    """Hits and singles are different measurements of the same simulation."""
    # ARRANGE
    first = hits_part([0])
    second = TreeData(GateTree.SINGLES, dict(first.columns))

    # ACT
    with pytest.raises(TreeMergeError) as raised:
        merge_tree_data([first, second], TREE_NAMES)

    # ASSERT
    assert "different kinds" in str(raised.value)


def test_parts_already_recording_their_source_are_refused() -> None:
    """Merging a merged dataset would overwrite where its rows came from."""
    # ARRANGE
    merged = merge_tree_data([hits_part([0]), hits_part([1])], TREE_NAMES)

    # ACT
    with pytest.raises(TreeMergeError) as raised:
        merge_tree_data([merged, merged], TREE_NAMES)

    # ASSERT
    assert SOURCE_TREE_BRANCH in str(raised.value)


def test_merging_nothing_is_refused() -> None:
    """There is no dataset to build out of no trees."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="At least one tree"):
        merge_tree_data([])


def test_names_are_required_while_the_source_is_recorded() -> None:
    """Recording where a row came from needs somewhere to take the name from."""
    # ARRANGE
    parts = [hits_part([0])]

    # ACT / ASSERT
    with pytest.raises(ValueError, match="Names of the source trees"):
        merge_tree_data(parts)


def test_every_part_needs_a_name() -> None:
    """A name missing for one tree would shift the names of the others."""
    # ARRANGE
    parts = [hits_part([0]), hits_part([1])]

    # ACT / ASSERT
    with pytest.raises(ValueError, match="Each tree needs a name"):
        merge_tree_data(parts, ["Hits"])


def test_names_are_not_needed_without_the_source_column() -> None:
    """Nothing needs a name when nothing records it."""
    # ARRANGE
    parts = [hits_part([0]), hits_part([1])]

    # ACT
    merged = merge_tree_data(parts, add_source_branch=False)

    # ASSERT
    assert merged.entry_count == 2


def test_data_already_recording_its_source_merges_without_the_column() -> None:
    """A merged dataset read back can be merged again, without the column."""
    # ARRANGE
    merged = merge_tree_data([hits_part([0]), hits_part([1])], TREE_NAMES)

    # ACT
    again = merge_tree_data([merged, merged], add_source_branch=False)

    # ASSERT
    assert again.entry_count == 4
    assert list(again[SOURCE_TREE_BRANCH]) == ["Hits_DET_INNER", "Hits_DET_OUTER"] * 2
