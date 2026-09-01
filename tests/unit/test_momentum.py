"""Unit tests for rebuilding a momentum direction from positions."""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.geometry.angles import angle_between
from opengate_gate_tree.geometry.momentum import momentum_direction_from_positions
from opengate_gate_tree.geometry.vectors import norm
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.units import rad_to_deg

TRACK_KEY = ["runID", "eventID", "trackID"]
POSITION = ["posX", "posY", "posZ"]
SOURCE = ["sourcePosX", "sourcePosY", "sourcePosZ"]
DIRECTION = ["momDirX", "momDirY", "momDirZ"]


@pytest.fixture(scope="module")
def hit_pairs(hits_variant_files: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the first two hits of every track of the a1 scene that has two."""
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    order = frame.groupby(TRACK_KEY, sort=False).cumcount()
    first = frame[order == 0].set_index(TRACK_KEY)
    second = frame[order == 1].set_index(TRACK_KEY)
    paired = first.index.intersection(second.index)
    return first.loc[paired], second.loc[paired]


def test_a_direction_is_the_step_between_two_places() -> None:
    """Which is what a detector leaves to work with."""
    # ARRANGE
    start = [[0.0, 0.0, 0.0]]
    end = [[0.0, 5.0, 0.0]]

    # ACT
    direction = momentum_direction_from_positions(start, end)

    # ASSERT
    assert direction.tolist() == [[0.0, 1.0, 0.0]]


def test_the_answer_is_a_direction_and_not_a_distance() -> None:
    """How far apart the two places are must not reach the answer."""
    # ARRANGE
    near = momentum_direction_from_positions([[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]])

    # ACT
    far = momentum_direction_from_positions([[1.0, 1.0, 1.0]], [[401.0, 1.0, 1.0]])

    # ASSERT
    assert near == pytest.approx(far)
    assert norm(near) == pytest.approx([1.0])


def test_the_direction_back_is_the_direction_reversed() -> None:
    """A step has a sense, unlike the polarization built out of two of them."""
    # ARRANGE
    start = [[0.0, 0.0, 0.0]]
    end = [[1.0, 2.0, 3.0]]

    # ACT
    there = momentum_direction_from_positions(start, end)
    back = momentum_direction_from_positions(end, start)

    # ASSERT
    assert there == pytest.approx(-back)


def test_a_particle_that_went_nowhere_has_no_direction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two hits at the same place leave nothing to reconstruct."""
    # ARRANGE
    place = [[1.0, 2.0, 3.0]]

    # ACT
    with caplog.at_level(logging.WARNING):
        direction = momentum_direction_from_positions(place, place)

    # ASSERT
    assert np.isnan(direction).all()
    assert "1 of 1 momentum direction values have no length" in caplog.text


def test_the_rebuilt_direction_is_the_one_gate_wrote(
    hit_pairs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The check that says the reconstruction is the right one.

    GATE writes the momentum direction of a hit as it is after the interaction
    there, so between two hits of one track the rebuilt direction is that
    branch. It agrees to within what float32 positions leave: a step is a
    difference of coordinates around 200 mm, so about four digits survive.
    """
    # ARRANGE
    first, second = hit_pairs

    # ACT
    rebuilt = momentum_direction_from_positions(
        first[POSITION].to_numpy(), second[POSITION].to_numpy()
    )

    # ASSERT
    assert len(rebuilt) == 151
    written = first[DIRECTION].to_numpy()
    assert rad_to_deg(angle_between(rebuilt, written)).max() < 0.01


def test_the_direction_a_photon_arrived_with_is_not_in_the_file(
    hit_pairs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Which is the reason this function exists.

    The momDir branch of the first hit is the direction after that
    interaction, so it is not the direction the photon arrived with. The two
    differ by the scattering angle, and on this file that is far from nothing.
    """
    # ARRANGE
    first, _ = hit_pairs

    # ACT
    incoming = momentum_direction_from_positions(
        first[SOURCE].to_numpy(), first[POSITION].to_numpy()
    )

    # ASSERT
    scattering = angle_between(incoming, first[DIRECTION].to_numpy())
    assert rad_to_deg(scattering).mean() > 30.0
    assert norm(incoming) == pytest.approx(np.ones(len(incoming)))


def test_the_two_places_describe_the_same_rows() -> None:
    """A direction runs from one place to another, one pair per row."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="as many rows each"):
        momentum_direction_from_positions(np.zeros((2, 3)), np.zeros((1, 3)))


def test_what_does_not_hold_positions_is_refused() -> None:
    """The shape contract holds here as everywhere else."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        momentum_direction_from_positions([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0]])


def test_an_empty_column_answers_empty() -> None:
    """A selection that removed every row is still a column of positions."""
    # ARRANGE
    empty = np.zeros((0, 3))

    # ACT
    direction = momentum_direction_from_positions(empty, empty)

    # ASSERT
    assert direction.shape == (0, 3)
