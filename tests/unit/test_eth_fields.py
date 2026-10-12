"""Unit tests for the fields an event, a track and a hit carry."""

import pytest

from opengate_gate_tree.representation.fields import (
    DERIVED_FIELDS,
    FIELD_LEVELS,
    ElementLevel,
    EventTrackHitField,
)
from opengate_gate_tree.tree.hits.schema import VARIANT_BRANCHES


def test_every_field_describes_one_level() -> None:
    """A field is read from an event, a track or a hit, never from two."""
    # ARRANGE
    fields = list(EventTrackHitField)

    # ACT
    levels = {field: field.level for field in fields}

    # ASSERT
    assert set(FIELD_LEVELS) == set(fields)
    assert set(levels.values()) == set(ElementLevel)


@pytest.mark.parametrize(
    ("field", "level"),
    [
        (EventTrackHitField.RUN_ID, ElementLevel.EVENT),
        (EventTrackHitField.SOURCE_TYPE, ElementLevel.EVENT),
        (EventTrackHitField.GAMMA_TYPE, ElementLevel.TRACK),
        (EventTrackHitField.SOURCE_POS_X, ElementLevel.TRACK),
        (EventTrackHitField.EDEP, ElementLevel.HIT),
        (EventTrackHitField.PROCESS_NAME, ElementLevel.HIT),
    ],
)
def test_fields_describe_the_level_their_values_belong_to(
    field: EventTrackHitField, level: ElementLevel
) -> None:
    """The level is where GATE writes the same value on every row."""
    # ARRANGE
    # No additional setup required.

    # ACT
    described = field.level

    # ASSERT
    assert described is level


def test_every_field_not_derived_is_a_branch_gate_writes() -> None:
    """A field is named as GATE named the column, so a table keeps the name."""
    # ARRANGE
    written = {name for branches in VARIANT_BRANCHES.values() for name in branches}

    # ACT
    read = {field.value for field in EventTrackHitField if not field.is_derived}

    # ASSERT
    assert read <= written


def test_derived_fields_are_the_counts_and_the_positions() -> None:
    """What the grouping says rather than what a branch holds."""
    # ARRANGE
    # No additional setup required.

    # ACT
    derived = {field for field in EventTrackHitField if field.is_derived}

    # ASSERT
    assert (
        derived
        == DERIVED_FIELDS
        == {
            EventTrackHitField.TRACK_COUNT,
            EventTrackHitField.HIT_COUNT,
            EventTrackHitField.TRACK_NUMBER,
            EventTrackHitField.HIT_NUMBER,
        }
    )
    assert EventTrackHitField.TRACK_COUNT.level is ElementLevel.EVENT
    assert EventTrackHitField.HIT_NUMBER.level is ElementLevel.HIT
