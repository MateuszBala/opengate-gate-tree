"""Unit tests for the polarization estimated from a scattering."""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from opengate_gate_tree.geometry.angles import angle_between, plane_normal
from opengate_gate_tree.geometry.momentum import momentum_direction_from_positions
from opengate_gate_tree.geometry.polarization import polarization_direction
from opengate_gate_tree.geometry.vectors import dot, norm
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree

X_AXIS = np.array([[1.0, 0.0, 0.0]])
Y_AXIS = np.array([[0.0, 1.0, 0.0]])
Z_AXIS = np.array([[0.0, 0.0, 1.0]])

QUARTER_TURN = np.pi / 2

TRACK_KEY = ["runID", "eventID", "trackID"]
POSITION = ["posX", "posY", "posZ"]
SOURCE = ["sourcePosX", "sourcePosY", "sourcePosZ"]


@pytest.fixture(scope="module")
def scattering(hits_variant_files: Mapping[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Return the directions before and after the first scattering of a track.

    Built the way an analysis builds them: the incoming direction from where
    the gamma was born to where it first interacted, the outgoing one from
    there to where it interacted next. Only tracks with a second hit have both.
    """
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    order = frame.groupby(TRACK_KEY, sort=False).cumcount()
    first = frame[order == 0].set_index(TRACK_KEY)
    second = frame[order == 1].set_index(TRACK_KEY)
    paired = first.index.intersection(second.index)
    first, second = first.loc[paired], second.loc[paired]
    incoming = momentum_direction_from_positions(
        first[SOURCE].to_numpy(), first[POSITION].to_numpy()
    )
    outgoing = momentum_direction_from_positions(
        first[POSITION].to_numpy(), second[POSITION].to_numpy()
    )
    return incoming, outgoing


def test_a_photon_that_turned_left_was_polarized_upwards() -> None:
    """The estimate is the normal of the plane the photon turned in."""
    # ARRANGE
    # No additional setup required.

    # ACT
    polarization = polarization_direction(X_AXIS, Y_AXIS)

    # ASSERT
    assert polarization == pytest.approx(Z_AXIS)


def test_the_estimate_is_a_direction_perpendicular_to_the_scattering() -> None:
    """It is a unit vector, and it leaves the plane of the two directions."""
    # ARRANGE
    before = np.array([[1.0, 2.0, 3.0]])
    after = np.array([[-1.0, 0.5, 2.0]])

    # ACT
    polarization = polarization_direction(before, after)

    # ASSERT
    assert norm(polarization) == pytest.approx([1.0])
    assert dot(polarization, before) == pytest.approx([0.0], abs=1e-12)
    assert dot(polarization, after) == pytest.approx([0.0], abs=1e-12)


def test_the_lengths_of_the_directions_do_not_reach_the_answer() -> None:
    """They are directions; only where they point matters."""
    # ARRANGE
    # No additional setup required.

    # ACT
    from_units = polarization_direction(X_AXIS, Y_AXIS)
    from_long_ones = polarization_direction(137.0 * X_AXIS, 0.001 * Y_AXIS)

    # ASSERT
    assert from_units == pytest.approx(from_long_ones)


def test_reading_the_scattering_backwards_reverses_the_sense() -> None:
    """Which is why the sense carries nothing, and the line does.

    The estimate is a line in space: a polarization along z and one along -z
    describe the same thing. An analysis that summed these vectors would find
    them cancelling, so this is written down.
    """
    # ARRANGE
    # No additional setup required.

    # ACT
    forwards = polarization_direction(X_AXIS, Y_AXIS)
    backwards = polarization_direction(Y_AXIS, X_AXIS)

    # ASSERT
    assert forwards == pytest.approx(-backwards)
    assert angle_between(forwards, backwards) == pytest.approx([np.pi])


def test_a_photon_that_carried_straight_on_says_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No plane, no normal, no estimate - and the rows are reported."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with caplog.at_level(logging.WARNING):
        polarization = polarization_direction(X_AXIS, 2.0 * X_AXIS)

    # ASSERT
    assert np.isnan(polarization).all()
    assert "1 of 1 plane normal values have no length" in caplog.text


def test_the_estimate_is_the_normal_of_the_scattering_plane() -> None:
    """Named for what it estimates, computed as what it is."""
    # ARRANGE
    before = np.array([[0.3, -0.5, 0.8]])
    after = np.array([[1.0, 0.2, -0.4]])

    # ACT
    polarization = polarization_direction(before, after)

    # ASSERT
    assert polarization == pytest.approx(plane_normal(before, after))


def test_the_estimate_holds_for_every_scattering_in_a_file(
    scattering: tuple[np.ndarray, np.ndarray],
) -> None:
    """On real hits: a unit vector, square to both directions, for every pair.

    The tolerance is what float32 positions leave: a direction is built from
    a difference of coordinates around 200 mm, so the cancellation costs about
    three digits.
    """
    # ARRANGE
    incoming, outgoing = scattering

    # ACT
    polarization = polarization_direction(incoming, outgoing)

    # ASSERT
    assert len(polarization) == 151
    assert not np.isnan(polarization).any()
    assert norm(polarization) == pytest.approx(np.ones(len(polarization)))
    assert angle_between(polarization, incoming) == pytest.approx(
        np.full(len(polarization), QUARTER_TURN), abs=1e-4
    )
    assert angle_between(polarization, outgoing) == pytest.approx(
        np.full(len(polarization), QUARTER_TURN), abs=1e-4
    )


def test_the_estimates_of_a_file_point_in_every_direction(
    scattering: tuple[np.ndarray, np.ndarray],
) -> None:
    """A source that is not polarized scatters into every plane.

    If the reconstruction had an axis of its own - a bug fixing the plane -
    the estimates would cluster instead.
    """
    # ARRANGE
    incoming, outgoing = scattering

    # ACT
    polarization = polarization_direction(incoming, outgoing)

    # ASSERT
    against_z = np.abs(dot(polarization, np.tile(Z_AXIS, (len(polarization), 1))))
    assert against_z.min() < 0.2
    assert against_z.max() > 0.8


def test_the_two_directions_describe_the_same_rows() -> None:
    """One photon scattered once, so there is a pair of directions per row."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="as many rows"):
        polarization_direction(np.zeros((2, 3)), np.zeros((1, 3)))


def test_what_does_not_hold_directions_is_refused() -> None:
    """The shape contract holds here as everywhere else."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        polarization_direction([1.0, 0.0, 0.0], [[0.0, 1.0, 0.0]])
