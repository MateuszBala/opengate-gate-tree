"""Unit tests for reading an angle into a chosen range."""

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.geometry.anglerange import (
    PI,
    TWO_PI,
    transform_to_angle_range,
    wrap_to_signed_pi,
    wrap_to_two_pi,
)

QUARTER = np.pi / 2

# The angles are written as multiples of pi, and so are the answers, so that
# the table below reads as the arithmetic it stands for rather than as a list
# of decimals. Every answer was worked out by hand, not by running the
# function: a table computed the way the code computes it would agree with a
# wrong code.
ANGLES_IN_PI: list[float] = [
    # whole and half turns, where a range boundary is decided
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -0.5,
    0.0,
    0.5,
    1.5,
    2.0,
    3.0,
    4.0,
    # angles that are not multiples of a quarter turn, in both signs, where
    # the narrower ranges do something other than answering zero
    1 / 3,
    -1 / 3,
    1 / 4,
    -1 / 4,
    1 / 6,
    -1 / 6,
]

ANGLE_IDS = [
    "-4pi",
    "-3pi",
    "-2pi",
    "-3pi/2",
    "-pi/2",
    "0",
    "pi/2",
    "3pi/2",
    "2pi",
    "3pi",
    "4pi",
    "pi/3",
    "-pi/3",
    "pi/4",
    "-pi/4",
    "pi/6",
    "-pi/6",
]

# One row per range, one entry per angle above, in multiples of pi.
#
#   [0, pi/2)      identifies angles a quarter turn apart, so every multiple of
#                  a quarter turn is read as zero
#   [0, pi)        identifies a direction with its reverse, which is what an
#                  undirected line means
#   [0, 2pi)       the same direction, read as a positive turn
#   [-pi, pi)      the same direction, read as the shorter turn either way
#   [-pi/2, pi/2)  an undirected line, read as the shorter turn either way
EXPECTED_IN_PI: dict[str, list[float]] = {
    "[0, pi/2)": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1 / 3,
        1 / 6,
        1 / 4,
        1 / 4,
        1 / 6,
        1 / 3,
    ],
    "[0, pi)": [
        0.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1 / 3,
        2 / 3,
        1 / 4,
        3 / 4,
        1 / 6,
        5 / 6,
    ],
    "[0, 2pi)": [
        0.0,
        1.0,
        0.0,
        0.5,
        1.5,
        0.0,
        0.5,
        1.5,
        0.0,
        1.0,
        0.0,
        1 / 3,
        5 / 3,
        1 / 4,
        7 / 4,
        1 / 6,
        11 / 6,
    ],
    "[-pi, pi)": [
        0.0,
        -1.0,
        0.0,
        0.5,
        -0.5,
        0.0,
        0.5,
        -0.5,
        0.0,
        -1.0,
        0.0,
        1 / 3,
        -1 / 3,
        1 / 4,
        -1 / 4,
        1 / 6,
        -1 / 6,
    ],
    "[-pi/2, pi/2)": [
        0.0,
        0.0,
        0.0,
        -0.5,
        -0.5,
        0.0,
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        1 / 3,
        -1 / 3,
        1 / 4,
        -1 / 4,
        1 / 6,
        -1 / 6,
    ],
}

RANGES_IN_PI: dict[str, tuple[float, float]] = {
    "[0, pi/2)": (0.0, 0.5),
    "[0, pi)": (0.0, 1.0),
    "[0, 2pi)": (0.0, 2.0),
    "[-pi, pi)": (-1.0, 1.0),
    "[-pi/2, pi/2)": (-0.5, 0.5),
}

CASES: list[tuple[str, float, float, float, float]] = [
    (label, low, high, angle, expected)
    for label, (low, high) in RANGES_IN_PI.items()
    for angle, expected in zip(ANGLES_IN_PI, EXPECTED_IN_PI[label], strict=True)
]

CASE_IDS = [f"{label}-{name}" for label in RANGES_IN_PI for name in ANGLE_IDS]


@pytest.mark.parametrize(("label", "low", "high", "angle", "expected"), CASES, ids=CASE_IDS)
def test_an_angle_is_read_into_the_range_it_is_asked_about(
    label: str,
    low: float,
    high: float,
    angle: float,
    expected: float,
) -> None:
    """Every angle of the table, into every range of it.

    Eighty-five cases, because the answer depends on both: the same direction
    is 3*pi/2 in one range, -pi/2 in another and 0 in a third, and which of
    them is wanted is the question, not the data.

    The angles that are not multiples of a quarter turn are what tells the
    narrow ranges apart from a function that answers zero: pi/3 read into
    [0, pi/2) is pi/3, while -pi/3 is pi/6.
    """
    # ARRANGE
    angle_min, angle_max = low * np.pi, high * np.pi

    # ACT
    read = transform_to_angle_range([angle * np.pi], angle_min, angle_max)

    # ASSERT
    assert read[0] == pytest.approx(expected * np.pi, abs=1e-12)
    assert angle_min <= read[0] < angle_max, label


@pytest.mark.parametrize(
    ("label", "low", "high"),
    [(label, low, high) for label, (low, high) in RANGES_IN_PI.items()],
    ids=list(RANGES_IN_PI),
)
def test_a_whole_column_is_read_at_once(label: str, low: float, high: float) -> None:
    """The table again, as one call - which is how a column is handled."""
    # ARRANGE
    angles = np.array(ANGLES_IN_PI) * np.pi
    angle_min, angle_max = low * np.pi, high * np.pi

    # ACT
    read = transform_to_angle_range(angles, angle_min, angle_max)

    # ASSERT
    assert read == pytest.approx(np.array(EXPECTED_IN_PI[label]) * np.pi, abs=1e-12)
    assert ((read >= angle_min) & (read < angle_max)).all()


@pytest.mark.parametrize(
    ("label", "low", "high"),
    [(label, low, high) for label, (low, high) in RANGES_IN_PI.items()],
    ids=list(RANGES_IN_PI),
)
def test_an_angle_only_moves_by_whole_widths(label: str, low: float, high: float) -> None:
    """What the transformation may change, and what it may not.

    An angle read into a range is the same angle when the range is a whole
    turn wide, and the same undirected quantity when it is narrower. Either
    way the difference has to be a whole number of widths, which is the part
    the table of expected values cannot state on its own.
    """
    # ARRANGE
    angles = np.array(ANGLES_IN_PI) * np.pi
    angle_min, angle_max = low * np.pi, high * np.pi
    width = angle_max - angle_min

    # ACT
    read = transform_to_angle_range(angles, angle_min, angle_max)

    # ASSERT
    turns = (angles - read) / width
    assert turns == pytest.approx(np.round(turns), abs=1e-9)


def test_an_angle_already_in_the_range_stays_where_it_is() -> None:
    """The transformation is not a rounding, and leaves what fits alone."""
    # ARRANGE
    inside = np.array([0.1, 1.0, 3.0, 6.2])

    # ACT
    read = transform_to_angle_range(inside, 0.0, TWO_PI)

    # ASSERT
    assert read == pytest.approx(inside)


def test_the_upper_end_belongs_to_the_next_turn() -> None:
    """The range is half-open, which is what a histogram needs."""
    # ARRANGE
    # No additional setup required.

    # ACT
    at_the_top = transform_to_angle_range([TWO_PI], 0.0, TWO_PI)
    a_hair_below_the_bottom = transform_to_angle_range([-1e-17], 0.0, TWO_PI)

    # ASSERT
    assert at_the_top[0] == 0.0
    assert a_hair_below_the_bottom[0] == 0.0


@pytest.mark.parametrize(
    ("angle_min", "angle_max", "match"),
    [
        (1.0, 1.0, "empty"),
        (2.0, 1.0, "empty"),
        (float("nan"), 1.0, "finite"),
        (0.0, float("inf"), "finite"),
    ],
    ids=["one-point", "reversed", "nan", "unbounded"],
)
def test_a_range_that_holds_no_angle_is_refused(
    angle_min: float, angle_max: float, match: str
) -> None:
    """A range nothing can be read into is a question that means nothing."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match=match):
        transform_to_angle_range([0.0], angle_min, angle_max)


def test_a_column_of_a_frame_is_read_as_well() -> None:
    """Angles usually arrive as a column, computed from the data."""
    # ARRANGE
    azimuths = pd.Series([-np.pi / 2, np.pi / 2], index=[7, 9])

    # ACT
    read = transform_to_angle_range(azimuths, 0.0, TWO_PI)

    # ASSERT
    assert read == pytest.approx([3 * np.pi / 2, np.pi / 2])


def test_an_empty_column_answers_empty() -> None:
    """A selection that removed every row is still a column of angles."""
    # ARRANGE
    # No additional setup required.

    # ACT
    read = transform_to_angle_range(np.zeros(0), 0.0, TWO_PI)

    # ASSERT
    assert read.shape == (0,)


def test_the_two_named_ranges_are_the_general_one() -> None:
    """They exist because they are asked for by name, not because they differ."""
    # ARRANGE
    angles = np.array(ANGLES_IN_PI) * np.pi

    # ACT
    full_turn = wrap_to_two_pi(angles)
    signed = wrap_to_signed_pi(angles)

    # ASSERT
    assert full_turn == pytest.approx(transform_to_angle_range(angles, 0.0, TWO_PI))
    assert signed == pytest.approx(transform_to_angle_range(angles, -PI, PI))
    assert ((full_turn >= 0.0) & (full_turn < TWO_PI)).all()
    assert ((signed >= -PI) & (signed < PI)).all()
