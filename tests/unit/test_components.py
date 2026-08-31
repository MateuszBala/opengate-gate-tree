"""Unit tests for taking a vector apart along an axis and in spherical components."""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.geometry.components import (
    parallel_component,
    perpendicular_component,
    spherical_components,
)
from opengate_gate_tree.geometry.vectors import as_vectors, dot, norm
from opengate_gate_tree.geometry.vectorview import VectorView
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree

Z_AXIS = [0.0, 0.0, 1.0]
X_AXIS = [1.0, 0.0, 0.0]

# A half turn and a quarter of one, which is where the spherical conventions
# are decided.
HALF_TURN = np.pi
QUARTER_TURN = np.pi / 2


@pytest.fixture(scope="module")
def hits(hits_variant_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return a scene to take apart."""
    return read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()


@pytest.fixture(scope="module")
def positions(hits: pd.DataFrame) -> np.ndarray:
    """Return the positions of that scene."""
    return as_vectors(hits[["posX", "posY", "posZ"]])


def test_a_vector_splits_along_an_axis() -> None:
    """The part along z is the z component, and the rest is what is left."""
    # ARRANGE
    vectors = [[1.0, 2.0, 3.0]]

    # ACT
    along = parallel_component(vectors, Z_AXIS)
    across = perpendicular_component(vectors, Z_AXIS)

    # ASSERT
    assert along.tolist() == [[0.0, 0.0, 3.0]]
    assert across.tolist() == [[1.0, 2.0, 0.0]]


def test_the_two_parts_add_back_up(positions: np.ndarray) -> None:
    """That is what makes them a decomposition rather than two computations."""
    # ARRANGE
    axis = [1.0, 1.0, 0.0]

    # ACT
    along = parallel_component(positions, axis)
    across = perpendicular_component(positions, axis)

    # ASSERT
    assert (along + across) == pytest.approx(positions)


def test_the_part_across_an_axis_is_across_it(positions: np.ndarray) -> None:
    """Perpendicular has to mean perpendicular, on real data too."""
    # ARRANGE
    axis = np.array([0.3, -0.5, 0.8])

    # ACT
    across = perpendicular_component(positions, axis)

    # ASSERT
    projection = across @ axis / np.linalg.norm(axis)
    assert projection == pytest.approx(np.zeros(len(positions)), abs=1e-9)


def test_only_the_direction_of_the_axis_matters(positions: np.ndarray) -> None:
    """An axis is a direction, so its length must not reach the answer."""
    # ARRANGE
    # No additional setup required.

    # ACT
    unit_axis = parallel_component(positions, Z_AXIS)
    long_axis = parallel_component(positions, [0.0, 0.0, 137.0])

    # ASSERT
    assert unit_axis == pytest.approx(long_axis)


def test_one_axis_per_row_is_allowed(positions: np.ndarray) -> None:
    """A direction that differs per hit is the case a polarization makes."""
    # ARRANGE
    axes = np.tile(Z_AXIS, (len(positions), 1))

    # ACT
    per_row = parallel_component(positions, axes)

    # ASSERT
    assert per_row == pytest.approx(parallel_component(positions, Z_AXIS))


def test_an_axis_of_another_length_is_refused(positions: np.ndarray) -> None:
    """An axis is one direction, or one for each row, and nothing between."""
    # ARRANGE
    axes = np.tile(Z_AXIS, (3, 1))

    # ACT / ASSERT
    with pytest.raises(ValueError, match="one direction for the whole column"):
        parallel_component(positions, axes)


def test_an_axis_of_no_direction_answers_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """There is no such thing as the part along nothing."""
    # ARRANGE
    vectors = [[1.0, 2.0, 3.0]]

    # ACT
    with caplog.at_level(logging.WARNING):
        along = parallel_component(vectors, [0.0, 0.0, 0.0])

    # ASSERT
    assert np.isnan(along).all()
    assert "1 of 1 axis values have no length" in caplog.text


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ([0.0, 0.0, 1.0], (1.0, 0.0, 0.0)),
        ([0.0, 0.0, -1.0], (1.0, HALF_TURN, 0.0)),
        ([1.0, 0.0, 0.0], (1.0, QUARTER_TURN, 0.0)),
        ([0.0, 1.0, 0.0], (1.0, QUARTER_TURN, QUARTER_TURN)),
        ([-1.0, 0.0, 0.0], (1.0, QUARTER_TURN, HALF_TURN)),
        ([0.0, -1.0, 0.0], (1.0, QUARTER_TURN, 3 * QUARTER_TURN)),
        ([0.0, 0.0, 5.0], (5.0, 0.0, 0.0)),
    ],
    ids=["up", "down", "x", "y", "minus-x", "minus-y", "five-up"],
)
def test_the_spherical_convention_is_the_physical_one(
    vector: list[float],
    expected: tuple[float, float, float],
) -> None:
    """The polar angle counts from z and the azimuth turns from x towards y.

    The other convention in circulation swaps the two names, so every axis is
    written out here: this is what the package means by each of them.
    """
    # ARRANGE
    # No additional setup required.

    # ACT
    radius, polar, azimuth = spherical_components([vector])

    # ASSERT
    assert (radius[0], polar[0], azimuth[0]) == pytest.approx(expected)


def test_the_azimuth_covers_a_whole_turn(positions: np.ndarray) -> None:
    """A half turn either side of zero would read as two halves in a histogram."""
    # ARRANGE
    # No additional setup required.

    # ACT
    _, polar, azimuth = spherical_components(positions)

    # ASSERT
    assert azimuth.min() >= 0.0
    assert azimuth.max() < 2 * np.pi
    assert azimuth.max() > HALF_TURN
    assert polar.min() >= 0.0
    assert polar.max() <= HALF_TURN


def test_spherical_components_come_back_from_where_they_went(positions: np.ndarray) -> None:
    """A description of the same vector has to describe the same vector."""
    # ARRANGE
    radius, polar, azimuth = spherical_components(positions)

    # ACT
    rebuilt = np.stack(
        [
            radius * np.sin(polar) * np.cos(azimuth),
            radius * np.sin(polar) * np.sin(azimuth),
            radius * np.cos(polar),
        ],
        axis=1,
    )

    # ASSERT
    assert rebuilt == pytest.approx(positions, abs=1e-9)


def test_a_vector_of_no_length_has_a_radius_and_no_angles() -> None:
    """Its length is nothing, which is a fact; its direction is not."""
    # ARRANGE
    # No additional setup required.

    # ACT
    radius, polar, azimuth = spherical_components([[0.0, 0.0, 0.0]])

    # ASSERT
    assert radius.tolist() == [0.0]
    assert np.isnan(polar[0])
    assert np.isnan(azimuth[0])


def test_the_view_takes_itself_apart(hits: pd.DataFrame) -> None:
    """The same decomposition, reached the way an analysis reaches it."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    along = position.parallel_to(Z_AXIS)
    across = position.perpendicular_to(Z_AXIS)

    # ASSERT
    assert isinstance(along, VectorView)
    assert along.index.equals(hits.index)
    assert along.array == pytest.approx(parallel_component(position.array, Z_AXIS))
    assert (along + across).array == pytest.approx(position.array)
    assert list(along.z) == pytest.approx(list(hits["posZ"]))


def test_the_axis_can_be_another_view(hits: pd.DataFrame) -> None:
    """A direction that differs per hit is read from the tree like any other."""
    # ARRANGE
    position = VectorView.from_frame(hits)
    direction = VectorView.from_frame(hits, ("momDirX", "momDirY", "momDirZ"))

    # ACT
    along = position.parallel_to(direction)

    # ASSERT
    assert along.array == pytest.approx(parallel_component(position.array, direction.array))
    assert norm(along.array) == pytest.approx(np.abs(dot(position.array, direction.array)))


def test_an_axis_of_other_rows_is_refused(hits: pd.DataFrame) -> None:
    """The rule for two views holds when one of them is the axis."""
    # ARRANGE
    position = VectorView.from_frame(hits)
    other_rows = VectorView.from_frame(hits.iloc[:10])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="describe the same rows"):
        position.parallel_to(other_rows)


def test_the_view_answers_spherical_components_as_a_frame(hits: pd.DataFrame) -> None:
    """Three numbers per row are a frame, and the names say which is which."""
    # ARRANGE
    position = VectorView.from_frame(hits)

    # ACT
    spherical = position.spherical()

    # ASSERT
    assert list(spherical.columns) == ["radius", "polar", "azimuth"]
    assert spherical.index.equals(hits.index)
    assert spherical["radius"].to_numpy() == pytest.approx(position.norm().to_numpy())
    assert spherical["azimuth"].between(0.0, 2 * np.pi).all()


def test_an_empty_column_answers_empty() -> None:
    """Every operation survives a selection that removed every row."""
    # ARRANGE
    empty = np.zeros((0, 3))

    # ACT
    along = parallel_component(empty, Z_AXIS)
    radius, polar, azimuth = spherical_components(empty)

    # ASSERT
    assert along.shape == (0, 3)
    assert (radius.shape, polar.shape, azimuth.shape) == ((0,), (0,), (0,))


def test_what_does_not_hold_vectors_is_refused_here_too() -> None:
    """The shape contract holds wherever vectors are read."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        spherical_components([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        parallel_component([[1.0, 2.0, 3.0]], [[1.0, 2.0]])
