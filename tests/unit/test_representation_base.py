"""Unit tests for the base class of every representation of a tree."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.errors import RepresentationError
from opengate_gate_tree.representation.base import DataRepresentation
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.treedata import TreeData


class RowRepresentation(DataRepresentation):
    """Representation whose elements are the rows themselves."""

    TREE: ClassVar[GateTree] = GateTree.HITS

    def __len__(self) -> int:
        return len(self.data)

    def to_tree_data(self) -> TreeData:
        return self.data


def hits_data() -> TreeData:
    """Return three entries of a "Hits" tree."""
    return TreeData(
        GateTree.HITS,
        {"eventID": np.array([0, 0, 1], dtype=np.int32), "edep": np.array([0.1, 0.2, 0.3])},
    )


def test_the_base_class_cannot_be_used_on_its_own() -> None:
    """Without elements to count, there is nothing it represents."""
    # ARRANGE
    data = hits_data()

    # ACT
    with pytest.raises(TypeError):
        DataRepresentation(data)  # type: ignore[abstract]

    # ASSERT
    # The constructor refusing is the assertion.


def test_a_subclass_that_counts_no_elements_cannot_be_used() -> None:
    """Counting its elements is what every representation has to say."""

    # ARRANGE
    class Uncounted(DataRepresentation):
        TREE: ClassVar[GateTree] = GateTree.HITS

        def to_tree_data(self) -> TreeData:
            return self.data

    # ACT
    with pytest.raises(TypeError):
        Uncounted(hits_data())  # type: ignore[abstract]

    # ASSERT
    # The constructor refusing is the assertion.


def test_columns_are_kept_as_they_were_given() -> None:
    """A representation is a way of reading the data, not a copy of it."""
    # ARRANGE
    data = hits_data()

    # ACT
    representation = RowRepresentation(data)

    # ASSERT
    assert representation.data is data
    assert len(representation) == 3
    assert representation.to_tree_data() is data


def test_rows_of_columns_are_labelled_by_their_position() -> None:
    """Columns carry no labels, so the position of an entry names it."""
    # ARRANGE
    data = hits_data()

    # ACT
    representation = RowRepresentation(data)

    # ASSERT
    assert representation.row_labels.tolist() == [0, 1, 2]


def test_a_frame_keeps_the_labels_its_rows_were_left_with() -> None:
    """After a selection, the labels are the way back to the rows of the tree."""
    # ARRANGE
    frame = hits_data().to_dataframe()
    selected = frame[frame["edep"] > 0.15]

    # ACT
    representation = RowRepresentation(selected)

    # ASSERT
    assert representation.row_labels.tolist() == [1, 2]
    assert representation.data.tree is GateTree.HITS
    assert representation.data["edep"].tolist() == [0.2, 0.3]


def test_the_columns_of_a_frame_are_not_copied() -> None:
    """Building a representation must not double the memory of the data."""
    # ARRANGE
    frame = pd.DataFrame({"eventID": np.array([0, 1, 2], dtype=np.int32)})

    # ACT
    representation = RowRepresentation(frame)

    # ASSERT
    assert np.shares_memory(representation.data["eventID"], frame["eventID"].to_numpy())


def test_data_of_another_tree_is_refused() -> None:
    """Singles are not hits, whatever their columns are called."""
    # ARRANGE
    singles = TreeData(GateTree.SINGLES, {"eventID": np.array([0], dtype=np.int32)})

    # ACT
    with pytest.raises(RepresentationError) as raised:
        RowRepresentation(singles)

    # ASSERT
    assert "'Hits'" in str(raised.value)
    assert "'Singles'" in str(raised.value)
