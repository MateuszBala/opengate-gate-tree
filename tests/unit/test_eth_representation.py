"""Unit tests for reading a "Hits" tree as events, tracks and hits."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.errors import BranchNotFoundError, RepresentationError
from opengate_gate_tree.io.reader import read_hits_trees, read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import ElementLevel, EventTrackHitField
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import GammaType
from opengate_gate_tree.tree.treedata import TreeData

TRACK_KEY = ["runID", "eventID", "trackID"]
Field = EventTrackHitField


@pytest.fixture(scope="module")
def a1(hits_variant_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the a1 scene as events, tracks and hits."""
    return EventTrackHit(read_tree(hits_variant_files["a1"], GateTree.HITS))


@pytest.fixture(scope="module")
def ops_prompt(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the oPs scene with a prompt gamma as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))


def test_the_counts_are_known_from_the_start(a1: EventTrackHit) -> None:
    """How many events, tracks and hits there are is what the grouping says."""
    # ARRANGE
    # No additional setup required.

    # ACT
    counts = (len(a1), a1.event_count, a1.track_count, a1.hit_count)

    # ASSERT
    assert counts == (150, 150, 300, 500)
    assert set(a1.track_counts().tolist()) == {2}
    assert a1.hit_counts().sum() == 500
    assert repr(a1) == "EventTrackHit(events=150, tracks=300, hits=500)"


def test_a_frame_left_by_a_selection_can_be_represented(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A selection made on the pandas view keeps its labels in the representation."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    compton = frame[frame["processName"] == "Compton"]

    # ACT
    representation = EventTrackHit(compton)

    # ASSERT
    assert representation.hit_count == 200
    hits = representation.hits_frame([Field.EDEP])
    assert set(hits.index) == set(compton.index)
    assert np.array_equal(hits["edep"].to_numpy(), compton.loc[hits.index, "edep"].to_numpy())


def test_data_missing_an_identifier_is_refused(hits_variant_files: Mapping[str, Path]) -> None:
    """Without trackID, nothing says which hits belong together."""
    # ARRANGE
    data = read_tree(hits_variant_files["a1"], GateTree.HITS, ["runID", "eventID", "time"])

    # ACT
    with pytest.raises(RepresentationError) as raised:
        EventTrackHit(data)

    # ASSERT
    assert "['trackID']" in str(raised.value)


def test_a_track_field_is_read_once_per_track(ops_prompt: EventTrackHit) -> None:
    """In the full events, track 1 is the prompt gamma and the others annihilation gammas."""
    # ARRANGE
    full = np.repeat(ops_prompt.track_counts() == 4, ops_prompt.track_counts())

    # ACT
    gamma_types = ops_prompt.values(Field.GAMMA_TYPE)[full]
    numbers = ops_prompt.values(Field.TRACK_NUMBER)[full]

    # ASSERT
    assert len(gamma_types) == 4 * 79
    assert np.all((gamma_types == GammaType.PROMPT) == (numbers == 1))


def test_an_event_field_is_spread_over_the_levels_below(ops_prompt: EventTrackHit) -> None:
    """Every track, and every hit, gets the value of its event."""
    # ARRANGE
    events = ops_prompt.values(Field.EVENT_ID)

    # ACT
    per_track = ops_prompt.values(Field.EVENT_ID, ElementLevel.TRACK)
    per_hit = ops_prompt.values(Field.EVENT_ID, ElementLevel.HIT)

    # ASSERT
    assert np.array_equal(per_track, np.repeat(events, ops_prompt.track_counts()))
    assert np.array_equal(per_hit, ops_prompt.data["eventID"][ops_prompt.index.hit_rows])


def test_a_hit_field_has_no_single_value_per_track(a1: EventTrackHit) -> None:
    """Which of the hits would it be? A reduction has to say."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        a1.values(Field.EDEP, ElementLevel.TRACK)

    # ASSERT
    assert "'edep' describes every hit" in str(raised.value)


def test_a_branch_named_by_a_string_is_read_as_a_hit_field(a1: EventTrackHit) -> None:
    """Every branch can be read, whether or not a field names it."""
    # ARRANGE
    # No additional setup required.

    # ACT
    by_name = a1.values("photonID")
    by_field_name = a1.values("gammaType")

    # ASSERT
    assert len(by_name) == a1.hit_count
    assert len(by_field_name) == a1.track_count


def test_derived_fields_count_and_number_the_elements(a1: EventTrackHit) -> None:
    """Counts per parent, and positions from 1 within it."""
    # ARRANGE
    # No additional setup required.

    # ACT
    hit_numbers = a1.values(Field.HIT_NUMBER)
    hit_counts = a1.values(Field.HIT_COUNT, ElementLevel.HIT)

    # ASSERT
    assert hit_numbers.min() == 1
    assert np.all(hit_numbers <= hit_counts)
    assert np.array_equal(a1.values(Field.TRACK_COUNT), a1.track_counts())
    assert a1.values(Field.TRACK_NUMBER).tolist()[:4] == [1, 2, 1, 2]


def test_a_field_the_variant_does_not_write_is_reported(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The Compton camera output has no gammaType branch."""
    # ARRANGE
    representation = EventTrackHit(read_tree(hits_variant_files["a4"], GateTree.HITS))

    # ACT
    with pytest.raises(BranchNotFoundError) as raised:
        representation.values(Field.GAMMA_TYPE)

    # ASSERT
    assert "gammaType" in str(raised.value)


def test_the_events_table_has_a_row_per_event(a1: EventTrackHit) -> None:
    """By default, the identifiers of an event and its number of tracks."""
    # ARRANGE
    # No additional setup required.

    # ACT
    events = a1.events_frame()

    # ASSERT
    assert list(events.columns) == ["runID", "eventID", "trackCount"]
    assert events.index.name == "eventIndex"
    assert len(events) == 150


def test_the_events_table_refuses_a_track_field(a1: EventTrackHit) -> None:
    """An event has several tracks, and so several values of it."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        a1.events_frame([Field.TRACK_ID])

    # ASSERT
    assert "'trackID' describes every track" in str(raised.value)


def test_the_tracks_table_agrees_with_grouping_the_rows(
    a1: EventTrackHit, hits_variant_files: Mapping[str, Path]
) -> None:
    """The first hit of every track, and the number of its hits."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()
    grouped = frame.groupby(TRACK_KEY)

    # ACT
    tracks = a1.tracks_frame([*TRACK_KEY, Field.HIT_COUNT, Field.SOURCE_POS_X])

    # ASSERT
    assert list(tracks.columns) == ["eventIndex", *TRACK_KEY, "hitCount", "sourcePosX"]
    assert tracks["hitCount"].tolist() == grouped.size().tolist()
    assert tracks["sourcePosX"].tolist() == grouped["sourcePosX"].first().tolist()
    assert tracks.index.name == "trackIndex"


def test_the_hits_table_holds_every_branch_and_where_each_hit_stands(a1: EventTrackHit) -> None:
    """The gate namespace works on it, because the columns keep their branch names."""
    # ARRANGE
    # No additional setup required.

    # ACT
    hits = a1.hits_frame()

    # ASSERT
    assert list(hits.columns[:4]) == ["eventIndex", "trackIndex", "trackNumber", "hitNumber"]
    assert "volumeID_9" in hits.columns
    assert len(hits.gate.position()) == 500
    assert hits.index.tolist() == a1.index.hit_rows.tolist()


def test_a_field_already_in_a_table_is_not_added_twice(a1: EventTrackHit) -> None:
    """Asking for the position of a track keeps one column of it."""
    # ARRANGE
    # No additional setup required.

    # ACT
    hits = a1.hits_frame([Field.TRACK_NUMBER, Field.EDEP])

    # ASSERT
    assert list(hits.columns) == ["eventIndex", "trackIndex", "trackNumber", "hitNumber", "edep"]


def test_fields_agree_within_every_track_and_event_of_the_fixtures(
    positronium_files: Mapping[str, Path],
) -> None:
    """GATE writes the values of a track on every row of it."""
    # ARRANGE
    representation = EventTrackHit(read_tree(positronium_files["all-variants"], GateTree.HITS))

    # ACT
    report = representation.check_consistency()

    # ASSERT
    assert report.is_consistent
    assert Field.GAMMA_TYPE in report.checked
    assert Field.SOURCE_TYPE in report.checked
    assert Field.SOURCE_ENERGY not in report.checked


def test_a_track_whose_rows_disagree_is_found(positronium_files: Mapping[str, Path]) -> None:
    """One row of another gamma type makes its track inconsistent, and only it."""
    # ARRANGE
    frame = read_tree(positronium_files["ops-prompt"], GateTree.HITS).to_dataframe()
    representation = EventTrackHit(frame)
    track = int(np.flatnonzero(representation.hit_counts() > 1)[0])
    second_hit = representation.index.hit_rows[representation.index.track_offsets[track] + 1]
    altered = frame.copy()
    altered.loc[altered.index[second_hit], ["gammaType", "sourceType"]] = 0

    # ACT
    report = EventTrackHit(altered).check_consistency([Field.GAMMA_TYPE, "sourceType"])

    # ASSERT
    assert not report.is_consistent
    assert report.inconsistent == {Field.GAMMA_TYPE: 1, Field.SOURCE_TYPE: 1}
    assert report.examples[Field.GAMMA_TYPE] == track


def test_floating_point_fields_that_are_not_numbers_agree() -> None:
    """A source position nobody recorded is the same nothing on every row."""
    # ARRANGE
    data = TreeData(
        GateTree.HITS,
        {
            "runID": np.zeros(2, dtype=np.int32),
            "eventID": np.zeros(2, dtype=np.int32),
            "trackID": np.ones(2, dtype=np.int32),
            "time": np.array([0.0, 1.0]),
            "sourcePosX": np.array([np.nan, np.nan]),
        },
    )

    # ACT
    report = EventTrackHit(data).check_consistency([Field.SOURCE_POS_X])

    # ASSERT
    assert report.is_consistent


@pytest.mark.parametrize(
    ("field", "message"),
    [(Field.EDEP, "describes hits"), ("photonID", "describes hits"), (Field.HIT_COUNT, "grouping")],
)
def test_only_fields_read_from_a_branch_of_tracks_or_events_are_checked(
    a1: EventTrackHit, field: EventTrackHitField | str, message: str
) -> None:
    """A hit field differs from hit to hit, and a count cannot disagree with itself."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        a1.check_consistency([field])

    # ASSERT
    assert message in str(raised.value)


def test_an_array_branch_is_read_with_every_component(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """volumeID is ten numbers per hit, and stays ten numbers."""
    # ARRANGE
    representation = EventTrackHit(read_hits_trees(hits_variant_files["multi-sd"]))

    # ACT
    volume = representation.values(Field.VOLUME_ID)
    first = representation.events[0].tracks[0].hits[0].value(Field.VOLUME_ID)

    # ASSERT
    assert volume.shape == (1000, 10)
    assert first.tolist() == volume[0].tolist()


def test_an_empty_frame_is_represented_as_no_events() -> None:
    """A selection that kept nothing is still a selection."""
    # ARRANGE
    frame = pd.DataFrame(
        {name: np.empty(0, dtype=np.int32) for name in ("runID", "eventID", "trackID")}
        | {"time": np.empty(0)}
    )

    # ACT
    representation = EventTrackHit(frame)

    # ASSERT
    assert len(representation) == 0
    assert representation.events_frame().empty
    assert representation.hits_frame().empty
