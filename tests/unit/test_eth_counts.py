"""Unit tests for selecting by how many tracks and hits there are, and by position."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import EventTrackHitField
from opengate_gate_tree.representation.reduce import position_in_track, reduce_hits_to_tracks
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process

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


def test_tracks_are_selected_by_their_number_of_hits(a1: EventTrackHit) -> None:
    """110 tracks of the a1 scene have exactly two hits."""
    # ARRANGE
    # No additional setup required.

    # ACT
    two = a1.with_hit_count(2)
    scattered = a1.with_hit_count(ge=2)
    between = a1.with_hit_count(gt=1, lt=4)

    # ASSERT
    assert two.track_count == 110
    assert scattered.track_count == 110 + 33 + 8
    assert between.track_count == 110 + 33
    assert set(two.hit_counts().tolist()) == {2}


def test_events_are_selected_by_their_number_of_tracks(ops_prompt: EventTrackHit) -> None:
    """The full events of the oPs scene, and the one cut short."""
    # ARRANGE
    # No additional setup required.

    # ACT
    full = ops_prompt.with_track_count(eq=4)
    cut = ops_prompt.with_track_count(ne=4)

    # ASSERT
    assert full.event_count == 79
    assert cut.event_count == 1


def test_a_count_after_a_selection_counts_what_it_kept(ops_prompt: EventTrackHit) -> None:
    """Three annihilation gammas in an event, or an event of three gammas: two questions."""
    # ARRANGE
    # No additional setup required.

    # ACT
    three_annihilation = ops_prompt.with_gamma_type(GammaType.ANNIHILATION).with_track_count(3)
    three_tracks = ops_prompt.with_track_count(3).with_gamma_type(GammaType.ANNIHILATION)

    # ASSERT
    assert three_annihilation.event_count == 79
    assert three_tracks.event_count == 0


@pytest.mark.parametrize(
    ("asked", "message"),
    [({}, "needs eq, ne, ge, le, gt or lt"), ({"eq": 2, "ge": 1}, "also be compared with ['ge']")],
)
def test_a_count_is_asked_for_in_one_way(
    a1: EventTrackHit, asked: dict[str, int], message: str
) -> None:
    """Equal to two and at least one is either redundant or a mistake."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        a1.is_hit_count(**asked)

    # ASSERT
    assert message in str(raised.value)


def test_the_hit_at_a_position_of_every_track(a1: EventTrackHit) -> None:
    """Every track has a first hit; only the longer ones have a third."""
    # ARRANGE
    # No additional setup required.

    # ACT
    first = a1.first_hits()
    third = a1.nth_hit(3)
    last = a1.last_hits()

    # ASSERT
    assert first.hit_count == first.track_count == 300
    assert third.hit_count == 33 + 8
    assert set(last.values(Field.PROCESS_NAME).tolist()) == {"PhotoElectric"}
    assert np.all(last.values(Field.HIT_NUMBER) == 1)


def test_a_position_beyond_a_track_selects_nothing_of_it(a1: EventTrackHit) -> None:
    """No track of the scene has five hits."""
    # ARRANGE
    # No additional setup required.

    # ACT
    fifth = a1.nth_hit(5)
    kept = a1.nth_hit(5, keep_empty=True)

    # ASSERT
    assert fifth.event_count == 0
    assert kept.track_count == 300
    assert kept.hit_count == 0


def test_the_last_hit_is_counted_from_the_end(a1: EventTrackHit) -> None:
    """For a track of one hit, the first and the last are the same hit."""
    # ARRANGE
    hits = a1.hits_frame([Field.HIT_NUMBER, Field.HIT_COUNT])

    # ACT
    last = a1.is_nth_hit(-1)
    second_to_last = a1.is_nth_hit(-2)

    # ASSERT
    assert np.array_equal(last, (hits["hitNumber"] == hits["hitCount"]).to_numpy())
    assert np.array_equal(second_to_last, (hits["hitNumber"] == hits["hitCount"] - 1).to_numpy())


def test_the_track_at_a_position_of_every_event(ops_prompt: EventTrackHit) -> None:
    """Track 1 of every full event is its prompt gamma."""
    # ARRANGE
    full = ops_prompt.with_track_count(4)

    # ACT
    first = full.nth_track(1)
    last = full.nth_track(-1)

    # ASSERT
    assert set(first.values(Field.GAMMA_TYPE).tolist()) == {GammaType.PROMPT}
    assert set(last.values(Field.TRACK_ID).tolist()) == {4}


def test_position_zero_is_refused(a1: EventTrackHit) -> None:
    """Positions are counted the way "the first hit" is said."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        a1.nth_track(0)

    # ASSERT
    assert "count from 1" in str(raised.value)


def test_a_position_counts_what_a_selection_kept(a1: EventTrackHit) -> None:
    """After keeping the photoelectric hits, the first hit is the first of those."""
    # ARRANGE
    photoelectric = a1.with_process(Process.PHOTOELECTRIC)

    # ACT
    first = photoelectric.first_hits()

    # ASSERT
    assert set(first.values(Field.PROCESS_NAME).tolist()) == {"PhotoElectric"}
    assert first.hit_count == photoelectric.track_count


def test_the_first_example_of_the_analysis_concept(positronium_files: Mapping[str, Path]) -> None:
    """Two annihilation gammas of pPs, each a Compton scattering then an absorption.

    The same question counted with pandas, sharing no code with the
    representation, finds 18 events in the pPs fixture.
    """
    # ARRANGE
    frame = read_tree(positronium_files["pps"], GateTree.HITS).to_dataframe()
    representation = EventTrackHit(frame)
    frame["position"] = frame.groupby(TRACK_KEY).cumcount() + 1
    tracks = frame.groupby(TRACK_KEY).agg(
        hits=("position", "size"),
        first=("processName", "first"),
        second=("processName", lambda names: names.iloc[1] if len(names) > 1 else None),
        gamma=("gammaType", "first"),
    )
    passing = (
        (tracks["hits"] == 2)
        & (tracks["first"] == "Compton")
        & (tracks["second"] == "PhotoElectric")
        & (tracks["gamma"] == GammaType.ANNIHILATION)
    )
    per_event = passing.groupby(level=["runID", "eventID"]).agg(["size", "sum"])
    expected = int(((per_event["size"] == 2) & (per_event["sum"] == 2)).sum())

    # ACT
    selected = (
        representation.with_track_count(2)
        .with_gamma_type(GammaType.ANNIHILATION)
        .with_decay_type(DecayType.STANDARD)
        .with_source_type(SourceType.PARA_POSITRONIUM)
        .with_hit_count(2)
    )
    for position, process in ((1, Process.COMPTON), (2, Process.PHOTOELECTRIC)):
        every_hit = np.ones(selected.hit_count, dtype=bool)
        at = position_in_track(selected.index, every_hit) == position
        track_passes = reduce_hits_to_tracks(
            selected.index, at & selected.is_process(process), "any"
        )
        selected = selected.select_tracks(track_passes)
    selected = selected.with_track_count(2)

    # ASSERT
    assert expected == 18
    assert selected.event_count == expected
    assert isinstance(selected.events_frame(), pd.DataFrame)
