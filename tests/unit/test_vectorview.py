"""Unit tests for reading three columns of a frame as vectors."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.geometry.vectorview import VectorView
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree

# The triples a "Hits" tree carries, by the columns they are written in.
SOURCE_COLUMNS = ("sourcePosX", "sourcePosY", "sourcePosZ")
DIRECTION_COLUMNS = ("momDirX", "momDirY", "momDirZ")

# The lead shell of the a1 scene, from its detector.mac: setRmin 0.2 m.
A1_SHELL_INNER_RADIUS = 200.0


@pytest.fixture(scope="module")
def hits(hits_variant_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return a scene whose geometry the macro states."""
    return read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()


def test_three_columns_are_read_as_the_vectors_they_are(hits: pd.DataFrame) -> None:
    """A position is one thing written in three columns."""
    # ARRANGE
    # No additional setup required.

    # ACT
    position = VectorView.from_frame(hits)

    # ASSERT
    assert len(position) == len(hits)
    assert position.index.equals(hits.index)
    assert position.names == ("posX", "posY", "posZ")
    assert position.array.shape == (len(hits), 3)
    assert position.array.dtype == np.float64


def test_the_lengths_are_the_ones_the_detector_macro_allows(hits: pd.DataFrame) -> None:
    """Every hit of this scene happened in a shell whose inner radius is known."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    lengths = position.norm()

    # ASSERT
    assert isinstance(lengths, pd.Series)
    assert lengths.index.equals(hits.index)
    assert lengths.min() >= A1_SHELL_INNER_RADIUS


def test_a_direction_is_built_by_subtracting_two_positions(hits: pd.DataFrame) -> None:
    """Where a gamma went is where it arrived minus where it started."""
    # ARRANGE
    position = VectorView.from_frame(hits)
    source = VectorView.from_frame(hits, SOURCE_COLUMNS)

    # ACT
    incoming = (position - source).unit()

    # ASSERT
    assert incoming.norm().to_numpy() == pytest.approx(np.ones(len(hits)))
    assert incoming.names == ("x", "y", "z")


def test_adding_and_subtracting_undo_each_other(hits: pd.DataFrame) -> None:
    """The algebra is the point of the object, so it has to be the algebra."""
    # ARRANGE
    position = VectorView.from_frame(hits)
    source = VectorView.from_frame(hits, SOURCE_COLUMNS)

    # ACT
    unchanged = (position + source) - source

    # ASSERT
    assert unchanged.array == pytest.approx(position.array)


def test_scaling_by_a_number_and_by_a_column(hits: pd.DataFrame) -> None:
    """A vector scaled is the same vector, so it keeps the names it had."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    twice = 2.0 * position
    weighted = position * hits["edep"]

    # ASSERT
    assert twice.array == pytest.approx((position + position).array)
    assert twice.names == position.names
    assert weighted.array[:, 0] == pytest.approx(position.array[:, 0] * hits["edep"].to_numpy())


def test_a_vector_pointing_the_other_way(hits: pd.DataFrame) -> None:
    """Negation is what tells a direction from the one it came from."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    reversed_view = -position

    # ASSERT
    assert reversed_view.array == pytest.approx(-position.array)
    assert reversed_view.norm().to_numpy() == pytest.approx(position.norm().to_numpy())


def test_the_scalar_and_vector_products_come_back_in_the_right_shapes(
    hits: pd.DataFrame,
) -> None:
    """A number per row is a column; a vector per row is another view."""
    # ARRANGE
    position = VectorView.from_frame(hits)
    direction = VectorView.from_frame(hits, DIRECTION_COLUMNS)

    # ACT
    projection = position.dot(direction)
    normal = position.cross(direction)

    # ASSERT
    assert isinstance(projection, pd.Series)
    assert projection.index.equals(hits.index)
    assert isinstance(normal, VectorView)
    assert normal.array.shape == (len(hits), 3)
    assert normal.dot(position).to_numpy() == pytest.approx(np.zeros(len(hits)), abs=1e-9)


def test_the_components_are_columns_of_their_own(hits: pd.DataFrame) -> None:
    """Reading one component back is how a result reaches a plot."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    first = position.x

    # ASSERT
    assert isinstance(first, pd.Series)
    assert first.name == "posX"
    assert list(first) == pytest.approx(list(hits["posX"]))
    assert list(position.y) == pytest.approx(list(hits["posY"]))
    assert list(position.z) == pytest.approx(list(hits["posZ"]))


def test_vectors_go_back_into_a_frame(hits: pd.DataFrame) -> None:
    """A computed direction is a result, and a result belongs in a frame."""
    # ARRANGE
    position = VectorView.from_frame(hits)
    source = VectorView.from_frame(hits, SOURCE_COLUMNS)

    # ACT
    written = (position - source).unit().to_frame("incomingDir")

    # ASSERT
    assert list(written.columns) == ["incomingDirX", "incomingDirY", "incomingDirZ"]
    assert written.index.equals(hits.index)
    assert list(position.to_frame().columns) == ["posX", "posY", "posZ"]


def test_what_a_view_hands_out_belongs_to_the_caller(hits: pd.DataFrame) -> None:
    """A frame or a column taken out of a view is the caller's to write to.

    Handing out the storage the view holds would make the two change together,
    which is what pandas did with an array before version 3 unless told
    otherwise, and the read-only array would be a promise with a way around it.
    """
    # ARRANGE
    position = VectorView.from_frame(hits)
    before = float(position.array[0, 0])

    # ACT
    exported = position.to_frame("dir")
    component = position.x
    exported.iloc[0, 0] = -999.0
    component.iloc[0] = -999.0

    # ASSERT
    assert float(position.array[0, 0]) == before
    assert not np.shares_memory(exported["dirX"].to_numpy(), position.array)
    assert not np.shares_memory(component.to_numpy(), position.array)


def test_a_view_of_selected_rows_keeps_those_rows(hits: pd.DataFrame) -> None:
    """Filtering and reading vectors have to compose, in that order."""
    # ARRANGE
    selected = hits.gate.in_sphere((0, 0, 0), 210.0)

    # ACT
    position = VectorView.from_frame(selected)

    # ASSERT
    assert 0 < len(position) < len(hits)
    assert position.index.equals(selected.index)
    assert list(position.index) != list(range(len(position)))
    assert position.norm().max() <= 210.0


def test_two_views_of_different_rows_are_not_combined(hits: pd.DataFrame) -> None:
    """Zipping them by position is the mistake pandas alignment prevents."""
    # ARRANGE
    everything = VectorView.from_frame(hits)
    some = VectorView.from_frame(hits.iloc[:10])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="describe the same rows"):
        everything.dot(some)


def test_two_views_of_as_many_other_rows_are_not_combined(hits: pd.DataFrame) -> None:
    """The case the guard exists for: as many rows, and none of them the same.

    A length check would let this through, and the vectors of one selection
    would be read against the vectors of another, row by row, as if they
    described the same hits.
    """
    # ARRANGE
    first = VectorView.from_frame(hits.iloc[:10])
    second = VectorView.from_frame(hits.iloc[10:20])

    # ACT / ASSERT
    assert len(first) == len(second)
    with pytest.raises(ValueError, match="different index values"):
        first.dot(second)
    with pytest.raises(ValueError, match="different index values"):
        first.cross(second)


def test_a_factor_of_as_many_other_rows_is_not_applied(hits: pd.DataFrame) -> None:
    """The same case for a weight given per row."""
    # ARRANGE
    position = VectorView.from_frame(hits.iloc[:10])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="the same rows"):
        position * hits["edep"].iloc[10:20]


def test_the_array_a_view_hands_back_cannot_be_written_to(hits: pd.DataFrame) -> None:
    """A view owns nothing, so it must not hand out something writable.

    Whether the underlying array is a copy depends on the dtype of the frame,
    which is not something to leave a caller guessing about.
    """
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    values = position.array

    # ASSERT
    assert not values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        values[0, 0] = 1.0


def test_a_factor_of_other_rows_is_not_applied(hits: pd.DataFrame) -> None:
    """A weight per row belongs to the rows it weighs."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT / ASSERT
    with pytest.raises(ValueError, match="the same rows"):
        position * hits["edep"].iloc[:10]


def test_a_position_is_three_columns_and_not_two(hits: pd.DataFrame) -> None:
    """Two columns describe a plane, and this is not one."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="3 columns"):
        VectorView.from_frame(hits, ("posX", "posY"))  # type: ignore[arg-type]


def test_a_column_the_frame_does_not_hold_is_named(hits: pd.DataFrame) -> None:
    """The name of the missing column is the whole of the mistake."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(KeyError, match="momentum"):
        VectorView.from_frame(hits, ("posX", "posY", "momentum"))


def test_a_view_holds_one_vector_per_row() -> None:
    """The index says which row a vector describes, so there is one of each."""
    # ARRANGE
    vectors = np.zeros((3, 3))

    # ACT / ASSERT
    with pytest.raises(ValueError, match="one vector per row"):
        VectorView(vectors, pd.Index([0, 1]))


def test_an_empty_frame_answers_an_empty_view(hits: pd.DataFrame) -> None:
    """A selection that removed every row is still a frame to read vectors from."""
    # ARRANGE
    empty = hits.iloc[:0]

    # ACT
    position = VectorView.from_frame(empty)

    # ASSERT
    assert len(position) == 0
    assert position.norm().empty
    assert position.to_frame().empty


def test_a_view_says_what_it_holds(hits: pd.DataFrame) -> None:
    """What a debugger prints has to say which triple this is."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    description = repr(position)

    # ASSERT
    assert description == "VectorView(500 vectors, names=('posX', 'posY', 'posZ'))"
