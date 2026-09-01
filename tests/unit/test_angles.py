"""Unit tests for the angles between directions and between planes."""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.geometry.angles import (
    angle_between,
    angle_between_normals,
    angle_between_planes,
    plane_normal,
)
from opengate_gate_tree.geometry.vectors import as_vectors, dot
from opengate_gate_tree.geometry.vectorview import VectorView
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.units import rad_to_deg

X_AXIS = np.array([[1.0, 0.0, 0.0]])
Y_AXIS = np.array([[0.0, 1.0, 0.0]])
Z_AXIS = np.array([[0.0, 0.0, 1.0]])

QUARTER_TURN = np.pi / 2
HALF_TURN = np.pi

# What the identifiers of one track are.
TRACK_KEY = ["runID", "eventID", "trackID"]

# The columns a scattering is read from.
POSITION = ["posX", "posY", "posZ"]
SOURCE = ["sourcePosX", "sourcePosY", "sourcePosZ"]
DIRECTION = ["momDirX", "momDirY", "momDirZ"]


@pytest.fixture(scope="module")
def scattered(hits_variant_files: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the first two hits of every track that has two.

    GATE numbers tracks within an event, so a track is named by the run, the
    event and the track identifier together. The two frames are indexed by
    that triple, which is what pairs them.
    """
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    order = frame.groupby(TRACK_KEY, sort=False).cumcount()
    first = frame[order == 0].set_index(TRACK_KEY)
    second = frame[order == 1].set_index(TRACK_KEY)
    paired = first.index.intersection(second.index)
    return first.loc[paired], second.loc[paired]


def test_the_angles_between_the_axes() -> None:
    """Perpendicular, parallel and opposite, which fixes the whole range."""
    # ARRANGE
    # No additional setup required.

    # ACT
    perpendicular = angle_between(X_AXIS, Y_AXIS)
    parallel = angle_between(X_AXIS, X_AXIS)
    opposite = angle_between(X_AXIS, -X_AXIS)

    # ASSERT
    assert perpendicular == pytest.approx([QUARTER_TURN])
    assert parallel == pytest.approx([0.0])
    assert opposite == pytest.approx([HALF_TURN])


def test_the_angle_between_a_direction_and_itself_is_nothing() -> None:
    """The scalar product of a unit vector with itself overshoots one.

    Without pulling the cosine back into the domain of arccos, the first thing
    anybody checks would answer nan.
    """
    # ARRANGE
    awkward = as_vectors([[0.5773502691896258, 0.5773502691896258, 0.5773502691896258]])

    # ACT
    angle = angle_between(awkward, awkward)

    # ASSERT
    assert not np.isnan(angle).any()
    assert angle == pytest.approx([0.0], abs=1e-8)


def test_an_angle_does_not_depend_on_length_or_on_order() -> None:
    """It is a question about directions, and it is symmetric."""
    # ARRANGE
    long_one = 137.0 * Y_AXIS

    # ACT
    scaled = angle_between(X_AXIS, long_one)
    reversed_order = angle_between(long_one, X_AXIS)

    # ASSERT
    assert scaled == pytest.approx([QUARTER_TURN])
    assert reversed_order == pytest.approx(scaled)


def test_a_direction_of_no_length_has_no_angle(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing points nowhere, so there is nothing to measure against."""
    # ARRANGE
    nothing = np.zeros((1, 3))

    # ACT
    with caplog.at_level(logging.WARNING):
        angle = angle_between(X_AXIS, nothing)

    # ASSERT
    assert np.isnan(angle).all()
    assert "have no length" in caplog.text


def test_the_normal_leaves_the_plane_it_describes() -> None:
    """A plane through the origin is what one direction is perpendicular to."""
    # ARRANGE
    # No additional setup required.

    # ACT
    normal = plane_normal(X_AXIS, Y_AXIS)

    # ASSERT
    assert normal == pytest.approx(Z_AXIS)
    assert dot(normal, X_AXIS) == pytest.approx([0.0])
    assert dot(normal, Y_AXIS) == pytest.approx([0.0])


def test_parallel_directions_span_no_plane(caplog: pytest.LogCaptureFixture) -> None:
    """Two directions that agree leave a whole turn of planes to choose from."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with caplog.at_level(logging.WARNING):
        normal = plane_normal(X_AXIS, 2.0 * X_AXIS)

    # ASSERT
    assert np.isnan(normal).all()
    assert "1 of 1 plane normal values have no length to divide out" in caplog.text


def test_the_angle_between_two_planes() -> None:
    """The xy plane and the xz plane stand at a right angle to each other."""
    # ARRANGE
    # No additional setup required.

    # ACT
    across = angle_between_planes(X_AXIS, Y_AXIS, X_AXIS, Z_AXIS)
    same = angle_between_planes(X_AXIS, Y_AXIS, X_AXIS, Y_AXIS)

    # ASSERT
    assert across == pytest.approx([QUARTER_TURN])
    assert same == pytest.approx([0.0], abs=1e-8)


def test_a_plane_read_backwards_answers_from_the_other_side() -> None:
    """A normal points to one side, and which side follows from the order.

    Written down because it is the mistake to make: the two pairs have to be
    given in the same order, before and then after.
    """
    # ARRANGE
    # No additional setup required.

    # ACT
    in_order = angle_between_planes(X_AXIS, Y_AXIS, X_AXIS, Z_AXIS)
    second_pair_reversed = angle_between_planes(X_AXIS, Y_AXIS, Z_AXIS, X_AXIS)

    # ASSERT
    assert in_order == pytest.approx([QUARTER_TURN])
    assert second_pair_reversed == pytest.approx([HALF_TURN - QUARTER_TURN])


def test_planes_are_compared_by_their_normals_alone() -> None:
    """The pair of functions has to agree, since one is written in the other."""
    # ARRANGE
    first = plane_normal(X_AXIS, Y_AXIS)
    second = plane_normal(X_AXIS, Z_AXIS)

    # ACT
    from_normals = angle_between_normals(first, second)

    # ASSERT
    assert from_normals == pytest.approx(angle_between_planes(X_AXIS, Y_AXIS, X_AXIS, Z_AXIS))


def test_the_direction_after_a_hit_points_at_the_next_one(
    scattered: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """An identity the simulation itself guarantees, checked on its output.

    GATE writes the momentum direction of a hit as it is after the
    interaction there, so for a track with a second hit it points straight at
    it. The angle between the two is zero to within what float32 positions
    allow: the step is a difference of coordinates around 200 mm, so the
    cancellation leaves about four digits.
    """
    # ARRANGE
    first, second = scattered
    step = second[POSITION].to_numpy() - first[POSITION].to_numpy()
    outgoing = first[DIRECTION].to_numpy()

    # ACT
    angles = angle_between(outgoing, step)

    # ASSERT
    assert len(angles) == 151
    assert rad_to_deg(float(np.max(angles))) < 0.01


def test_the_scattering_angles_of_a_file_are_physical(
    scattered: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Between the incoming and the outgoing direction of the first hit."""
    # ARRANGE
    first, _ = scattered
    incoming = first[POSITION].to_numpy() - first[SOURCE].to_numpy()
    outgoing = first[DIRECTION].to_numpy()

    # ACT
    angles = angle_between(incoming, outgoing)

    # ASSERT
    assert angles.min() > 0.0
    assert angles.max() < HALF_TURN
    assert not np.isnan(angles).any()


def test_the_view_measures_the_angle_the_way_an_analysis_asks_for_it(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Column in, column out, with the rows it was asked about."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    position = VectorView.from_frame(frame)
    direction = VectorView.from_frame(frame, ("momDirX", "momDirY", "momDirZ"))

    # ACT
    angles = position.angle_to(direction)

    # ASSERT
    assert isinstance(angles, pd.Series)
    assert angles.index.equals(frame.index)
    assert angles.name == "angle"
    assert angles.between(0.0, HALF_TURN).all()
    assert angles.to_numpy() == pytest.approx(angle_between(position.array, direction.array))


def test_an_angle_to_other_rows_is_refused(hits_variant_files: Mapping[str, Path]) -> None:
    """The rule for two views holds here as everywhere else."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    position = VectorView.from_frame(frame)
    fewer = VectorView.from_frame(frame.iloc[:5])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="describe the same rows"):
        position.angle_to(fewer)


def test_every_angle_lies_where_arccos_answers(hits_variant_files: Mapping[str, Path]) -> None:
    """The range is part of the contract, so it is checked on a whole file."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a2"], GateTree.HITS).to_dataframe()
    position = as_vectors(frame[POSITION])
    direction = as_vectors(frame[DIRECTION])

    # ACT
    angles = angle_between(position, direction)

    # ASSERT
    assert angles.min() >= 0.0
    assert angles.max() <= HALF_TURN
    assert rad_to_deg(angles).max() <= 180.0


@pytest.mark.parametrize("angle", [1e-9, 1e-7, 1e-5, 1e-3], ids=["1e-9", "1e-7", "1e-5", "1e-3"])
def test_a_small_angle_is_measured_and_not_rounded_away(angle: float) -> None:
    """The angles this package measures live where a cosine carries nothing.

    A cosine near one holds the angle in its last bits, so `arccos` of the
    scalar product answers 0 for an angle of 1e-9 rad even in float64. The
    computation goes through `atan2` of the two products instead, which is the
    same angle and keeps it. The identity this package checks on real data -
    the angle between a momentum direction and the step it points along - is
    exactly such an angle.
    """
    # ARRANGE
    turned = np.array([[np.cos(angle), np.sin(angle), 0.0]])

    # ACT
    measured = angle_between(X_AXIS, turned)

    # ASSERT
    assert measured == pytest.approx([angle], rel=1e-6)


@pytest.mark.parametrize("gap", [1e-9, 1e-7], ids=["1e-9", "1e-7"])
def test_an_angle_just_short_of_a_half_turn_is_measured_too(gap: float) -> None:
    """The other end of the range, where the cosine is just as flat."""
    # ARRANGE
    angle = HALF_TURN - gap
    turned = np.array([[np.cos(angle), np.sin(angle), 0.0]])

    # ACT
    measured = angle_between(X_AXIS, turned)

    # ASSERT
    assert measured == pytest.approx([angle], abs=1e-12)
