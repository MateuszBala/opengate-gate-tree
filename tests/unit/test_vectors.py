"""Unit tests for the vectorised operations on columns of vectors."""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.geometry.vectors import (
    MIN_NORM,
    as_vectors,
    clip_cosine,
    cross,
    dot,
    ensure_vectors,
    norm,
    normalize,
)
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree

# The axes, which is what every question about direction is checked against.
X_AXIS = np.array([[1.0, 0.0, 0.0]])
Y_AXIS = np.array([[0.0, 1.0, 0.0]])
Z_AXIS = np.array([[0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def positions(hits_variant_files: Mapping[str, Path]) -> np.ndarray:
    """Return the positions of a file, as the vectors they are."""
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    return as_vectors(frame[["posX", "posY", "posZ"]])


def test_a_length_is_the_length(positions: np.ndarray) -> None:
    """The one identity every other one is built on."""
    # ARRANGE
    # No additional setup required.

    # ACT
    lengths = norm([[3.0, 4.0, 0.0]])

    # ASSERT
    assert lengths.tolist() == [5.0]
    assert norm(positions).shape == (len(positions),)


def test_a_unit_vector_has_unit_length(positions: np.ndarray) -> None:
    """Normalising is what every angle in this package starts with."""
    # ARRANGE
    # No additional setup required.

    # ACT
    unit = normalize(positions)

    # ASSERT
    assert normalize([[3.0, 4.0, 0.0]]).tolist() == [[0.6, 0.8, 0.0]]
    assert norm(unit) == pytest.approx(np.ones(len(positions)))


def test_normalising_keeps_the_direction(positions: np.ndarray) -> None:
    """Only the length changes, which is what makes it safe to do first."""
    # ARRANGE
    unit = normalize(positions)

    # ACT
    along = dot(unit, positions) / norm(positions)

    # ASSERT
    assert along == pytest.approx(np.ones(len(positions)))


def test_a_vector_of_no_length_has_no_direction(caplog: pytest.LogCaptureFixture) -> None:
    """It is read as nothing at all, and the column is still answered.

    Refusing the whole column would end an analysis of half a million hits
    over one degenerate row, so the row is reported instead.
    """
    # ARRANGE
    vectors = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [MIN_NORM / 2, 0.0, 0.0]]

    # ACT
    with caplog.at_level(logging.WARNING):
        unit = normalize(vectors, name="direction")

    # ASSERT
    assert np.isnan(unit[0]).all()
    assert unit[1].tolist() == [0.6, 0.8, 0.0]
    assert np.isnan(unit[2]).all()
    assert "2 of 3 direction values have no length to divide out" in caplog.text


def test_a_column_of_directions_is_reported_once(caplog: pytest.LogCaptureFixture) -> None:
    """A column that holds no degenerate row says nothing at all."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with caplog.at_level(logging.WARNING):
        normalize([[1.0, 0.0, 0.0]])

    # ASSERT
    assert caplog.text == ""


def test_the_scalar_product_measures_agreement() -> None:
    """Perpendicular is zero, parallel is one, opposite is minus one."""
    # ARRANGE
    # No additional setup required.

    # ACT
    perpendicular = dot(X_AXIS, Y_AXIS)
    parallel = dot(X_AXIS, X_AXIS)
    opposite = dot(X_AXIS, -X_AXIS)

    # ASSERT
    assert perpendicular.tolist() == [0.0]
    assert parallel.tolist() == [1.0]
    assert opposite.tolist() == [-1.0]


def test_the_vector_product_leaves_the_plane_of_its_arguments() -> None:
    """That is what makes it the normal of the plane those two span."""
    # ARRANGE
    # No additional setup required.

    # ACT
    product = cross(X_AXIS, Y_AXIS)

    # ASSERT
    assert product.tolist() == [[0.0, 0.0, 1.0]]
    assert dot(product, X_AXIS).tolist() == [0.0]
    assert dot(product, Y_AXIS).tolist() == [0.0]


def test_the_vector_product_changes_sign_with_its_arguments() -> None:
    """The plane is the same either way round; only the side of it differs."""
    # ARRANGE
    # No additional setup required.

    # ACT
    forward = cross(X_AXIS, Y_AXIS)
    backward = cross(Y_AXIS, X_AXIS)

    # ASSERT
    assert forward.tolist() == (-backward).tolist()


def test_parallel_vectors_span_no_plane() -> None:
    """The product is zero, which is what later makes their plane undefined."""
    # ARRANGE
    # No additional setup required.

    # ACT
    product = cross(X_AXIS, 2.0 * X_AXIS)

    # ASSERT
    assert product.tolist() == [[0.0, 0.0, 0.0]]
    assert norm(product).tolist() == [0.0]


def test_a_cosine_is_pulled_back_into_the_domain_of_arccos() -> None:
    """Without this, an angle taken from such a cosine would be nothing.

    The values have to be outside the domain to begin with: 1e-16 is below the
    spacing of float64 at one, so `1.0 + 1e-16` is `1.0` and would prove
    nothing. 1e-15 is above it.
    """
    # ARRANGE
    just_outside = np.array([1.0 + 1e-15, -1.0 - 1e-15, 0.5])

    # ACT
    clipped = clip_cosine(just_outside)

    # ASSERT
    assert just_outside[0] > 1.0
    assert just_outside[1] < -1.0
    assert clipped.tolist() == [1.0, -1.0, 0.5]
    assert not np.isnan(np.arccos(clipped)).any()


def test_a_vector_too_long_to_measure_has_no_direction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its length overflows, and dividing by that would answer a vector of zeros.

    Of the ways a row can fail, that is the only one that would be neither an
    exception, nor nan, nor a line in the log.
    """
    # ARRANGE
    enormous = [[1e200, 1e200, 1e200]]

    # ACT
    with caplog.at_level(logging.WARNING):
        unit = normalize(enormous)

    # ASSERT
    assert np.isnan(unit).all()
    assert "1 of 1 vector values have no length to divide out" in caplog.text


@pytest.mark.parametrize(
    "values",
    [[1.0, 2.0, 3.0], [[1.0, 2.0]], [[1.0, 2.0, 3.0, 4.0]], np.zeros((2, 3, 1))],
    ids=["one-vector-flat", "two-components", "four-components", "three-dimensional"],
)
def test_what_does_not_hold_vectors_is_refused(values: object) -> None:
    """A single vector read as three columns of one would answer plausibly."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        as_vectors(values)  # type: ignore[arg-type]


def test_the_message_names_the_shape_that_was_given() -> None:
    """The shape is the whole of the mistake, so the message carries it."""
    # ARRANGE
    array = np.zeros((4, 2))

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"got shape \(4, 2\)"):
        ensure_vectors(array)


def test_two_columns_of_different_length_are_refused() -> None:
    """They are compared row by row, so there has to be a row for each."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="as many rows"):
        dot(X_AXIS, np.vstack([X_AXIS, Y_AXIS]))


def test_vectors_are_read_from_the_columns_of_a_frame() -> None:
    """Three columns of a tree are the usual way in."""
    # ARRANGE
    frame = pd.DataFrame({"posX": [1.0, 4.0], "posY": [2.0, 5.0], "posZ": [3.0, 6.0]})

    # ACT
    array = as_vectors(frame)

    # ASSERT
    assert array.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_a_column_of_a_file_is_read_in_float64(positions: np.ndarray) -> None:
    """GATE writes float32, and an angle needs the digits float64 keeps."""
    # ARRANGE
    single = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    # ACT
    array = as_vectors(single)

    # ASSERT
    assert array.dtype == np.float64
    assert positions.dtype == np.float64


def test_an_empty_column_answers_empty() -> None:
    """A selection that removed every row is still a column of vectors."""
    # ARRANGE
    empty = np.zeros((0, 3))

    # ACT
    lengths = norm(empty)
    unit = normalize(empty)

    # ASSERT
    assert lengths.shape == (0,)
    assert unit.shape == (0, 3)
