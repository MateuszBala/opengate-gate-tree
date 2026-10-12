"""Unit tests for grouping the rows of a "Hits" tree into hits, tracks and events.

The numbers the tests compare against were measured on the fixtures with
pandas, and every index is also compared with what ``groupby`` answers on the
same file - a second way of counting that shares no code with the index.
"""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_hits_trees, read_tree
from opengate_gate_tree.representation.index import (
    EventTrackHitIndex,
    build_event_track_hit_index,
    segment_of,
)
from opengate_gate_tree.tree.gatetree import GateTree

EVENT_KEY = ["runID", "eventID"]
TRACK_KEY = ["runID", "eventID", "trackID"]


def frame_of(path: Path) -> pd.DataFrame:
    """Return the identifier and time columns of the "Hits" tree of a file."""
    return read_hits_trees(path, ["runID", "eventID", "trackID", "time"]).to_dataframe()


def index_of(frame: pd.DataFrame, force_sort: bool = False) -> EventTrackHitIndex:
    """Return the index of a frame of identifier and time columns."""
    return build_event_track_hit_index(
        frame["runID"], frame["eventID"], frame["trackID"], frame["time"], force_sort
    )


@pytest.fixture(scope="module")
def fixture_frames(
    hits_variant_files: Mapping[str, Path], positronium_files: Mapping[str, Path]
) -> Mapping[str, pd.DataFrame]:
    """Return the identifier and time columns of the fixtures the tests read."""
    paths = {
        "a1": hits_variant_files["a1"],
        "a4": hits_variant_files["a4"],
        "b1": hits_variant_files["b1"],
        "multi-run": hits_variant_files["multi-run"],
        "multi-sd": hits_variant_files["multi-sd"],
        "all-variants": positronium_files["all-variants"],
        "ops-prompt": positronium_files["ops-prompt"],
    }
    return {key: frame_of(path) for key, path in paths.items()}


@pytest.mark.parametrize(
    ("key", "tracks_per_event", "hits_per_track"),
    [
        ("a1", {2: 150}, {1: 149, 2: 110, 3: 33, 4: 8}),
        ("all-variants", {2: 53, 3: 43, 4: 19}, {1: 181, 2: 86, 3: 31, 4: 11, 5: 2}),
        # The single event with one track is the last one, cut when the
        # fixture was trimmed to 500 entries.
        ("ops-prompt", {1: 1, 4: 79}, {1: 190, 2: 86, 3: 28, 4: 11, 5: 2}),
    ],
)
def test_tracks_and_hits_are_counted_as_measured(
    fixture_frames: Mapping[str, pd.DataFrame],
    key: str,
    tracks_per_event: dict[int, int],
    hits_per_track: dict[int, int],
) -> None:
    """The number of tracks of every event and of hits of every track."""
    # ARRANGE
    frame = fixture_frames[key]

    # ACT
    index = index_of(frame)

    # ASSERT
    assert Counter(index.tracks_per_event().tolist()) == tracks_per_event
    assert Counter(index.hits_per_track().tolist()) == hits_per_track
    assert index.hit_count == len(frame)


@pytest.mark.parametrize("key", ["a1", "a4", "b1", "multi-run", "multi-sd", "all-variants"])
def test_the_index_agrees_with_grouping_the_rows(
    fixture_frames: Mapping[str, pd.DataFrame], key: str
) -> None:
    """Counting by groupby, which shares no code with the index, says the same."""
    # ARRANGE
    frame = fixture_frames[key]
    expected_tracks = frame.groupby(EVENT_KEY)["trackID"].nunique()
    expected_hits = frame.groupby(TRACK_KEY).size()
    expected_rows = frame.sort_values(["runID", "eventID", "trackID", "time"], kind="stable").index

    # ACT
    index = index_of(frame)

    # ASSERT
    assert index.tracks_per_event().tolist() == expected_tracks.tolist()
    assert index.hits_per_track().tolist() == expected_hits.tolist()
    assert index.hit_rows.tolist() == expected_rows.tolist()


@pytest.mark.parametrize("key", ["a1", "b1", "multi-run", "multi-sd", "ops-prompt"])
def test_sorting_every_row_gives_the_same_index(
    fixture_frames: Mapping[str, pd.DataFrame], key: str
) -> None:
    """Sorting the blocks of rows is a shortcut, not another answer."""
    # ARRANGE
    frame = fixture_frames[key]

    # ACT
    by_blocks = index_of(frame)
    by_rows = index_of(frame, force_sort=True)

    # ASSERT
    assert np.array_equal(by_blocks.hit_rows, by_rows.hit_rows)
    assert np.array_equal(by_blocks.track_offsets, by_rows.track_offsets)
    assert np.array_equal(by_blocks.event_offsets, by_rows.event_offsets)
    assert np.array_equal(by_blocks.track_rows, by_rows.track_rows)
    assert np.array_equal(by_blocks.event_rows, by_rows.event_rows)


def test_hits_merged_from_two_detectors_are_grouped_as_one_event(
    fixture_frames: Mapping[str, pd.DataFrame],
) -> None:
    """A track recorded by both detectors has rows in both trees."""
    # ARRANGE
    frame = fixture_frames["multi-sd"]

    # ACT
    index = index_of(frame)

    # ASSERT
    assert (index.event_count, index.track_count, index.hit_count) == (305, 537, 1000)


def test_tracks_follow_their_identifier_and_hits_their_time(
    fixture_frames: Mapping[str, pd.DataFrame],
) -> None:
    """GATE writes the tracks of an event from the last one to the first."""
    # ARRANGE
    frame = fixture_frames["ops-prompt"]

    # ACT
    index = index_of(frame)

    # ASSERT
    track_ids = frame["trackID"].to_numpy()[index.track_rows]
    first_event = track_ids[index.event_offsets[0] : index.event_offsets[1]]
    assert first_event.tolist() == [1, 2, 3, 4]
    times = frame["time"].to_numpy()[index.hit_rows]
    within_track = np.diff(times)[np.diff(index.track_of_hit()) == 0]
    assert np.all(within_track >= 0)


def test_events_follow_their_run_and_their_identifier(
    fixture_frames: Mapping[str, pd.DataFrame],
) -> None:
    """Event 5 of the second run comes after every event of the first."""
    # ARRANGE
    frame = fixture_frames["multi-run"]

    # ACT
    index = index_of(frame)

    # ASSERT
    first_rows = index.event_rows
    runs = frame["runID"].to_numpy()[first_rows]
    events = frame["eventID"].to_numpy()[first_rows]
    keys = list(zip(runs, events, strict=True))
    assert keys == sorted(keys)
    assert len(set(keys)) == index.event_count


def test_rows_in_any_order_give_the_same_grouping(
    fixture_frames: Mapping[str, pd.DataFrame],
) -> None:
    """The index depends on what the rows say, not on where they stand."""
    # ARRANGE
    frame = fixture_frames["a1"]
    shuffled = frame.sample(frac=1.0, random_state=7)

    # ACT
    original = index_of(frame)
    reordered = index_of(shuffled)

    # ASSERT
    assert np.array_equal(original.track_offsets, reordered.track_offsets)
    assert np.array_equal(original.event_offsets, reordered.event_offsets)
    assert shuffled.index.to_numpy()[reordered.hit_rows].tolist() == original.hit_rows.tolist()


def test_hits_written_at_the_same_time_keep_their_order() -> None:
    """Rows a time cannot tell apart stay in the order of the data."""
    # ARRANGE
    ones = np.ones(4, dtype=np.int32)

    # ACT
    index = build_event_track_hit_index(ones, ones, [1, 2, 1, 1], [0.0, 0.0, 0.0, 0.0])

    # ASSERT
    assert index.hit_rows.tolist() == [0, 2, 3, 1]
    assert index.hits_per_track().tolist() == [3, 1]


def test_a_track_whose_time_runs_back_is_put_in_order() -> None:
    """One block of rows is not enough when its time decreases."""
    # ARRANGE
    ones = np.ones(3, dtype=np.int32)

    # ACT
    index = build_event_track_hit_index(ones, ones, ones, [3.0, 1.0, 2.0])

    # ASSERT
    assert index.hit_rows.tolist() == [1, 2, 0]


def test_no_rows_make_no_events() -> None:
    """An empty selection is still something to index."""
    # ARRANGE
    empty = np.empty(0, dtype=np.int32)

    # ACT
    index = build_event_track_hit_index(empty, empty, empty, np.empty(0))

    # ASSERT
    assert (index.event_count, index.track_count, index.hit_count) == (0, 0, 0)
    assert index.tracks_per_event().tolist() == []


def test_columns_of_different_lengths_are_refused() -> None:
    """Every row needs its identifiers and its time."""
    # ARRANGE
    ones = np.ones(3, dtype=np.int32)

    # ACT
    with pytest.raises(ValueError) as raised:
        build_event_track_hit_index(ones, ones, ones, [0.0, 1.0])

    # ASSERT
    assert "[2, 3]" in str(raised.value)


def test_a_column_that_is_not_flat_is_refused() -> None:
    """A two-dimensional column holds more than one value per row."""
    # ARRANGE
    ones = np.ones(3, dtype=np.int32)

    # ACT
    with pytest.raises(ValueError) as raised:
        build_event_track_hit_index(np.ones((3, 2)), ones, ones, ones)

    # ASSERT
    assert "'run_ids'" in str(raised.value)


def test_the_parents_of_tracks_and_hits_are_named() -> None:
    """The way up from a hit to its track, and from a track to its event."""
    # ARRANGE
    index = EventTrackHitIndex(
        hit_rows=np.arange(4, dtype=np.int64),
        track_offsets=np.array([0, 1, 3, 4], dtype=np.int64),
        event_offsets=np.array([0, 2, 3], dtype=np.int64),
        track_rows=np.array([0, 1, 3], dtype=np.int64),
        event_rows=np.array([0, 3], dtype=np.int64),
    )

    # ACT
    tracks = index.track_of_hit()
    events = index.event_of_track()

    # ASSERT
    assert tracks.tolist() == [0, 1, 1, 2]
    assert events.tolist() == [0, 0, 1]
    assert repr(index) == "EventTrackHitIndex(events=2, tracks=3, hits=4)"


def test_one_row_per_track_and_per_event_is_required() -> None:
    """Every track and every event reads its values from a row."""
    # ARRANGE
    offsets = np.array([0, 1], dtype=np.int64)

    # ACT
    with pytest.raises(ValueError) as raised:
        EventTrackHitIndex(np.zeros(1, dtype=np.int64), offsets, offsets, offsets, offsets[:1])

    # ASSERT
    assert "one row per track and per event (1 and 1), got 2 and 1" in str(raised.value)


@pytest.mark.parametrize(
    ("track_offsets", "message"),
    [
        (np.array([], dtype=np.int64), "non-empty"),
        (np.array([[0, 2]], dtype=np.int64), "non-empty"),
        (np.array([1, 2], dtype=np.int64), "from 0 to 2"),
        (np.array([0, 1], dtype=np.int64), "from 0 to 2"),
        (np.array([0, 2, 1, 2], dtype=np.int64), "must not decrease"),
    ],
)
def test_offsets_that_do_not_divide_the_hits_are_refused(
    track_offsets: np.ndarray, message: str
) -> None:
    """An index that does not cover every hit exactly once means nothing."""
    # ARRANGE
    hit_rows = np.arange(2, dtype=np.int64)

    # ACT
    with pytest.raises(ValueError) as raised:
        EventTrackHitIndex(
            hit_rows, track_offsets, np.array([0, 1], dtype=np.int64), hit_rows[:1], hit_rows[:1]
        )

    # ASSERT
    assert message in str(raised.value)


def test_a_frame_left_by_a_selection_is_indexed_by_position(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Rows are positions in the columns; labels are the representation's to keep."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    selected = frame[frame["processName"] == "Compton"]

    # ACT
    index = index_of(selected)

    # ASSERT
    assert index.hit_count == 200
    assert int(index.hit_rows.max()) == 199


def two_events() -> EventTrackHitIndex:
    """Return event 0 of tracks [h0 h1] [h2], and event 1 of track [h3 h4]."""
    return build_event_track_hit_index(
        [0] * 5, [0, 0, 0, 1, 1], [1, 1, 2, 1, 1], [0.0, 1.0, 0.0, 0.0, 1.0]
    )


def test_selecting_hits_keeps_their_order_and_their_tracks() -> None:
    """The second hit of every track, and nothing of the track that has one."""
    # ARRANGE
    index = two_events()

    # ACT
    selected = index.select(hit_mask=np.array([False, True, False, False, True]))

    # ASSERT
    assert selected.hit_rows.tolist() == [1, 4]
    assert selected.hits_per_track().tolist() == [1, 1]
    assert selected.tracks_per_event().tolist() == [1, 1]
    assert selected.track_rows.tolist() == [0, 3]
    assert selected.event_rows.tolist() == [0, 3]


def test_a_track_left_without_hits_can_be_kept() -> None:
    """It is still the track it was, and reads its values from the same row."""
    # ARRANGE
    index = two_events()

    # ACT
    selected = index.select(hit_mask=np.array([True, True, False, False, False]), keep_empty=True)

    # ASSERT
    assert selected.hits_per_track().tolist() == [2, 0, 0]
    assert selected.track_rows.tolist() == [0, 2, 3]
    assert selected.event_count == 2


def test_leaving_out_an_event_leaves_out_its_tracks_and_hits() -> None:
    """A level keeps only what the levels above it keep."""
    # ARRANGE
    index = two_events()

    # ACT
    selected = index.select(event_mask=np.array([False, True]), keep_empty=True)

    # ASSERT
    assert selected.hit_rows.tolist() == [3, 4]
    assert selected.event_rows.tolist() == [3]


def test_an_event_left_without_tracks_is_removed_unless_kept() -> None:
    """Removing its only track removes the event, unless it is kept empty."""
    # ARRANGE
    index = two_events()
    mask = np.array([True, True, False])

    # ACT
    removed = index.select(track_mask=mask)
    kept = index.select(track_mask=mask, keep_empty=True)

    # ASSERT
    assert removed.event_count == 1
    assert kept.tracks_per_event().tolist() == [2, 0]
    assert kept.event_rows.tolist() == [0, 3]


@pytest.mark.parametrize(
    "mask",
    [np.array([1, 0]), np.array([True]), np.array([[True, False]])],
)
def test_a_selection_needs_one_boolean_per_element(mask: np.ndarray) -> None:
    """A mask of the wrong kind or length selects nothing that can be told."""
    # ARRANGE
    index = two_events()

    # ACT
    with pytest.raises(ValueError) as raised:
        index.select(event_mask=mask)

    # ASSERT
    assert "one boolean per event, 2 in all" in str(raised.value)


def test_selecting_nothing_leaves_nothing(fixture_frames: Mapping[str, pd.DataFrame]) -> None:
    """An empty index is still an index."""
    # ARRANGE
    index = index_of(fixture_frames["a1"])

    # ACT
    selected = index.select(event_mask=np.zeros(index.event_count, dtype=bool))

    # ASSERT
    assert (selected.event_count, selected.track_count, selected.hit_count) == (0, 0, 0)


@pytest.mark.parametrize("force_sort", [False, True])
def test_a_time_that_is_not_a_number_comes_last_in_its_track(force_sort: bool) -> None:
    """Both ways of grouping put it after every other hit of its track."""
    # ARRANGE
    ones = np.ones(3, dtype=np.int32)

    # ACT
    index = build_event_track_hit_index(ones, ones, ones, [1.0, np.nan, 0.5], force_sort)

    # ASSERT
    assert index.hit_rows.tolist() == [2, 0, 1]


def test_what_an_earlier_selection_kept_empty_stays_empty() -> None:
    """A later selection removes only what it empties itself."""
    # ARRANGE
    index = two_events()
    kept = index.select(hit_mask=np.array([True, False, False, False, False]), keep_empty=True)

    # ACT
    everything = kept.select(event_mask=np.ones(kept.event_count, dtype=bool))
    first_track = kept.select(track_mask=np.array([True, False, True]))

    # ASSERT
    assert everything.hits_per_track().tolist() == [1, 0, 0]
    assert everything.event_count == 2
    assert first_track.tracks_per_event().tolist() == [1, 1]


def test_the_segment_of_a_position_skips_empty_segments() -> None:
    """A hit belongs to the track holding it, not to an empty one before it."""
    # ARRANGE
    offsets = np.array([0, 2, 2, 3], dtype=np.int64)

    # ACT
    segments = [segment_of(offsets, position) for position in range(3)]

    # ASSERT
    assert segments == [0, 0, 2]
