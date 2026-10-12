"""Unit tests for an event, a track and a hit as objects."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation import eventtrackhit
from opengate_gate_tree.representation.elements import Event, Hit, Track
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import EventTrackHitField
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType


@pytest.fixture(scope="module")
def ops_prompt(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the oPs scene with a prompt gamma as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))


def test_the_events_are_a_sequence_of_objects(ops_prompt: EventTrackHit) -> None:
    """The "Hits" tree as a list of Event objects, made when asked for."""
    # ARRANGE
    events = ops_prompt.events

    # ACT
    first = events[0]
    last = events[-1]
    some = events[2:5]

    # ASSERT
    assert len(events) == ops_prompt.event_count == 80
    assert isinstance(first, Event)
    assert last == Event(ops_prompt, 79)
    assert [event.index for event in some] == [2, 3, 4]
    assert sum(1 for _ in events) == 80
    assert repr(events) == "ElementSequence(elements=80)"


def test_a_position_outside_the_events_is_refused(ops_prompt: EventTrackHit) -> None:
    """As a list would refuse it."""
    # ARRANGE
    events = ops_prompt.events

    # ACT
    with pytest.raises(IndexError) as raised:
        events[80]

    # ASSERT
    assert "outside 80" in str(raised.value)


def test_no_object_is_made_until_one_is_asked_for(
    positronium_files: Mapping[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the representation of a file creates no event."""
    # ARRANGE
    made: list[int] = []

    class CountedEvent(Event):
        def __init__(self, representation: EventTrackHit, index: int) -> None:
            made.append(index)
            super().__init__(representation, index)

    monkeypatch.setattr(eventtrackhit, "Event", CountedEvent)

    # ACT
    representation = EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))
    sequence = representation.events
    before = list(made)
    sequence[3]

    # ASSERT
    assert before == []
    assert made == [3]


def test_an_event_carries_its_identifiers_and_its_tracks(ops_prompt: EventTrackHit) -> None:
    """The first event of the scene: an oPs decay with a prompt gamma."""
    # ARRANGE
    event = ops_prompt.events[0]

    # ACT
    tracks = event.tracks

    # ASSERT
    assert (event.run_id, event.event_id, event.track_count) == (0, 0, 4)
    assert event.source_type is SourceType.ORTHO_POSITRONIUM
    assert [track.track_id for track in tracks] == [1, 2, 3, 4]
    assert repr(event) == "Event(runID=0, eventID=0, tracks=4)"


def test_a_track_carries_what_its_gamma_was(ops_prompt: EventTrackHit) -> None:
    """Track 1 is the prompt gamma, the others come from the annihilation."""
    # ARRANGE
    tracks = ops_prompt.events[0].tracks

    # ACT
    prompt, annihilation = tracks[0], tracks[1]

    # ASSERT
    assert prompt.gamma_type is GammaType.PROMPT
    assert annihilation.gamma_type is GammaType.ANNIHILATION
    assert prompt.decay_type is DecayType.DEEXCITATION
    assert prompt.decay_index == 0
    assert (prompt.parent_id, prompt.pdg_encoding) == (0, 22)
    assert (prompt.number, annihilation.number) == (1, 2)
    assert prompt.event == ops_prompt.events[0]
    assert prompt.source_position.shape == (3,)
    assert prompt.hit_count == len(prompt.hits)
    assert repr(prompt) == f"Track(trackID=1, hits={prompt.hit_count})"


def test_a_hit_is_the_row_it_was_read_from(ops_prompt: EventTrackHit) -> None:
    """Every value of a hit is the value of its row of the data."""
    # ARRANGE
    hit = ops_prompt.events[0].tracks[0].hits[0]
    data = ops_prompt.data

    # ACT
    row = hit.row

    # ASSERT
    assert isinstance(hit, Hit)
    assert hit.edep == pytest.approx(float(data["edep"][row]))
    assert hit.time == data["time"][row]
    assert hit.local_time == data["trackLocalTime"][row]
    assert hit.process_name == data["processName"][row]
    assert hit.position.tolist() == [float(data[name][row]) for name in ("posX", "posY", "posZ")]
    assert np.linalg.norm(hit.momentum_direction) == pytest.approx(1.0, rel=1e-6)
    assert hit.number == 1
    assert hit.track == ops_prompt.events[0].tracks[0]
    assert repr(hit) == f"Hit(row={row!r}, number=1)"


def test_hits_follow_their_time_within_a_track(ops_prompt: EventTrackHit) -> None:
    """The first hit is the first interaction of the gamma."""
    # ARRANGE
    long_track = next(
        Track(ops_prompt, position)
        for position in range(ops_prompt.track_count)
        if ops_prompt.hit_counts()[position] >= 3
    )

    # ACT
    times = [hit.time for hit in long_track.hits]

    # ASSERT
    assert times == sorted(times)
    assert [hit.number for hit in long_track.hits] == list(range(1, len(times) + 1))


def test_an_element_reads_the_fields_of_the_levels_above_it(ops_prompt: EventTrackHit) -> None:
    """A hit knows its track and event; an event knows nothing of a single hit."""
    # ARRANGE
    event = ops_prompt.events[5]
    track = event.tracks[1]
    hit = track.hits[0]

    # ACT
    from_hit = (hit.value("eventID"), hit.value("trackID"), hit.value("trackCount"))
    from_track = (track.value("eventID"), track.value(EventTrackHitField.TRACK_COUNT))

    # ASSERT
    assert from_hit == (event.event_id, track.track_id, 4)
    assert from_track == (event.event_id, 4)
    assert hit.value(EventTrackHitField.HIT_COUNT) == track.hit_count
    with pytest.raises(ValueError):
        event.value(EventTrackHitField.TRACK_ID)
