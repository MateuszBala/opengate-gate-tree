"""Unit tests for the questions asked of every track or every event at once."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.index import (
    EventTrackHitIndex,
    build_event_track_hit_index,
)
from opengate_gate_tree.representation.reduce import (
    REDUCTIONS,
    broadcast_events_to_tracks,
    broadcast_tracks_to_hits,
    position_in_event,
    position_in_track,
    reduce_hits_to_tracks,
    reduce_tracks_to_events,
)
from opengate_gate_tree.tree.gatetree import GateTree

TRACK_KEY = ["runID", "eventID", "trackID"]


def small_index() -> EventTrackHitIndex:
    """Return two events: one of a track of three hits and an empty track, one of one hit.

    Tracks: [h0 h1 h2] [ ] | [h3]. The empty track is what a selection
    keeping empty tracks leaves.
    """
    return EventTrackHitIndex(
        hit_rows=np.arange(4, dtype=np.int64),
        track_offsets=np.array([0, 3, 3, 4], dtype=np.int64),
        event_offsets=np.array([0, 2, 3], dtype=np.int64),
    )


@pytest.fixture(scope="module")
def a1_frame(hits_variant_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return the "Hits" tree of the a1 scene."""
    return read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()


@pytest.fixture(scope="module")
def a1_index(a1_frame: pd.DataFrame) -> EventTrackHitIndex:
    """Return the index of the a1 scene."""
    return build_event_track_hit_index(
        a1_frame["runID"], a1_frame["eventID"], a1_frame["trackID"], a1_frame["time"]
    )


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("any", [True, False, False]),
        ("all", [False, True, False]),
        ("count", [2, 0, 0]),
        ("sum", [2, 0, 0]),
    ],
)
def test_a_mask_of_hits_is_summarised_for_every_track(how: str, expected: list[object]) -> None:
    """An empty track has some of nothing, all of nothing, and none of it."""
    # ARRANGE
    mask = np.array([True, False, True, False])

    # ACT
    answer = reduce_hits_to_tracks(small_index(), mask, how)  # type: ignore[arg-type]

    # ASSERT
    assert answer.tolist() == expected


def test_float32_deposits_are_summed_in_float64(
    a1_frame: pd.DataFrame, a1_index: EventTrackHitIndex
) -> None:
    """Summing many single precision values must not lose what they add up to."""
    # ARRANGE
    edep = a1_frame["edep"].to_numpy()[a1_index.hit_rows]
    expected = a1_frame.astype({"edep": np.float64}).groupby(TRACK_KEY)["edep"].sum()

    # ACT
    answer = reduce_hits_to_tracks(a1_index, edep, "sum")

    # ASSERT
    assert answer.dtype == np.float64
    np.testing.assert_allclose(answer, expected.to_numpy(), rtol=1e-12)


def test_whole_numbers_are_summed_as_whole_numbers() -> None:
    """A count of interactions stays a count."""
    # ARRANGE
    values = np.array([1, 2, 3, 4], dtype=np.int32)

    # ACT
    answer = reduce_hits_to_tracks(small_index(), values, "sum")

    # ASSERT
    assert answer.dtype == np.int64
    assert answer.tolist() == [6, 0, 4]


def test_every_track_of_the_a1_scene_ends_in_a_photoelectric_absorption(
    a1_frame: pd.DataFrame, a1_index: EventTrackHitIndex
) -> None:
    """The scene's gammas are stopped in the detector, so the last hit absorbs."""
    # ARRANGE
    processes = a1_frame["processName"].to_numpy()[a1_index.hit_rows]

    # ACT
    first = reduce_hits_to_tracks(a1_index, processes, "first")
    last = reduce_hits_to_tracks(a1_index, processes, "last")

    # ASSERT
    expected_first = a1_frame.groupby(TRACK_KEY)["processName"].first()
    assert first.tolist() == expected_first.tolist()
    assert set(last.tolist()) == {"PhotoElectric"}


@pytest.mark.parametrize("how", ["first", "last"])
def test_an_empty_track_has_no_first_or_last_hit(how: str) -> None:
    """There is no value to answer with, and no value that would mean none."""
    # ARRANGE
    values = np.array([1.0, 2.0, 3.0, 4.0])

    # ACT
    with pytest.raises(ValueError) as raised:
        reduce_hits_to_tracks(small_index(), values, how)  # type: ignore[arg-type]

    # ASSERT
    assert "1 segments are empty" in str(raised.value)


def test_tracks_are_summarised_for_every_event(a1_index: EventTrackHitIndex) -> None:
    """How many tracks of every event have more than one hit."""
    # ARRANGE
    scattered = a1_index.hits_per_track() > 1

    # ACT
    answer = reduce_tracks_to_events(a1_index, scattered, "count")

    # ASSERT
    assert int(answer.sum()) == 110 + 33 + 8
    assert set(answer.tolist()) <= {0, 1, 2}


def test_an_unknown_reduction_is_refused() -> None:
    """The name of a reduction is what it answers; a misspelt one answers nothing."""
    # ARRANGE
    mask = np.zeros(4, dtype=bool)

    # ACT
    with pytest.raises(ValueError) as raised:
        reduce_hits_to_tracks(small_index(), mask, "mean")  # type: ignore[arg-type]

    # ASSERT
    assert "'mean'" in str(raised.value)
    assert str(list(REDUCTIONS)) in str(raised.value)


@pytest.mark.parametrize("how", ["any", "all", "count"])
def test_a_mask_reduction_refuses_values_that_are_not_a_mask(how: str) -> None:
    """Counting energies as truths would answer a question nobody asked."""
    # ARRANGE
    values = np.array([0.1, 0.0, 0.3, 0.0])

    # ACT
    with pytest.raises(ValueError) as raised:
        reduce_hits_to_tracks(small_index(), values, how)  # type: ignore[arg-type]

    # ASSERT
    assert "mask of the hits" in str(raised.value)


def test_values_that_are_not_one_per_hit_are_refused() -> None:
    """A value per track is not a value per hit."""
    # ARRANGE
    values = np.zeros(3, dtype=bool)

    # ACT
    with pytest.raises(ValueError) as raised:
        reduce_hits_to_tracks(small_index(), values, "any")

    # ASSERT
    assert "one value per hit, 4 in all" in str(raised.value)


def test_a_track_gives_its_value_to_every_hit() -> None:
    """The empty track gives it to nothing."""
    # ARRANGE
    values = np.array([10, 20, 30])

    # ACT
    answer = broadcast_tracks_to_hits(small_index(), values)

    # ASSERT
    assert answer.tolist() == [10, 10, 10, 30]


def test_an_event_gives_its_value_to_every_track() -> None:
    """Including a track without hits."""
    # ARRANGE
    values = np.array(["a", "b"])

    # ACT
    answer = broadcast_events_to_tracks(small_index(), values)

    # ASSERT
    assert answer.tolist() == ["a", "a", "b"]


def test_broadcasting_needs_one_value_per_parent() -> None:
    """A value per hit cannot be spread over hits."""
    # ARRANGE
    values = np.zeros(4)

    # ACT
    with pytest.raises(ValueError) as raised:
        broadcast_tracks_to_hits(small_index(), values)

    # ASSERT
    assert "one value per track, 3 in all" in str(raised.value)


def test_kept_hits_are_numbered_from_one_within_their_track() -> None:
    """The second kept hit is the second, whatever was left out before it."""
    # ARRANGE
    mask = np.array([False, True, True, True])

    # ACT
    forward = position_in_track(small_index(), mask)
    backward = position_in_track(small_index(), mask, from_end=True)

    # ASSERT
    assert forward.tolist() == [0, 1, 2, 1]
    assert backward.tolist() == [0, 2, 1, 1]


def test_positions_of_the_a1_scene_agree_with_counting_within_groups(
    a1_frame: pd.DataFrame, a1_index: EventTrackHitIndex
) -> None:
    """With every hit kept, a position is the order of a hit in its track."""
    # ARRANGE
    every_hit = np.ones(a1_index.hit_count, dtype=bool)
    ordered = a1_frame.iloc[a1_index.hit_rows]
    expected = ordered.groupby(TRACK_KEY, sort=False).cumcount() + 1

    # ACT
    positions = position_in_track(a1_index, every_hit)

    # ASSERT
    assert positions.tolist() == expected.tolist()


def test_kept_tracks_are_numbered_from_one_within_their_event() -> None:
    """The empty track is a track: what it holds does not decide its position."""
    # ARRANGE
    mask = np.array([True, True, True])

    # ACT
    forward = position_in_event(small_index(), mask)
    backward = position_in_event(small_index(), mask, from_end=True)

    # ASSERT
    assert forward.tolist() == [1, 2, 1]
    assert backward.tolist() == [2, 1, 1]


def test_a_position_needs_a_mask() -> None:
    """Positions count what is kept, so they need to know what that is."""
    # ARRANGE
    values = np.array([1, 0, 1])

    # ACT
    with pytest.raises(ValueError) as raised:
        position_in_event(small_index(), values)

    # ASSERT
    assert "mask of the tracks" in str(raised.value)


def test_nothing_to_reduce_answers_with_nothing() -> None:
    """An index of no rows has no tracks to answer for."""
    # ARRANGE
    empty = np.empty(0, dtype=np.int32)
    index = build_event_track_hit_index(empty, empty, empty, np.empty(0))

    # ACT
    answer = reduce_hits_to_tracks(index, np.empty(0, dtype=bool), "count")

    # ASSERT
    assert answer.tolist() == []
