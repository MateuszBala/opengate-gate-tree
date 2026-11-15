"""Unit tests for conditions on the track or the hit at a position.

Every expected number is counted again with pandas on the same file.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from opengate_gate_tree.analysis.eventtrackhit import EventTrackHitAnalysis
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process

Analysis = EventTrackHitAnalysis
EVENT_KEY = ["runID", "eventID"]
TRACK_KEY = ["runID", "eventID", "trackID"]


def numbered(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the hits in the order of their tracks and times, numbered within the track."""
    ordered = frame.sort_values([*TRACK_KEY, "time"], kind="stable").copy()
    ordered["position"] = ordered.groupby(TRACK_KEY).cumcount() + 1
    return ordered


@pytest.fixture(scope="module")
def a1_frame(hits_variant_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return the "Hits" tree of the a1 scene."""
    return read_tree(hits_variant_files["a1"], GateTree.HITS).to_dataframe()


@pytest.fixture(scope="module")
def a1(a1_frame: pd.DataFrame) -> EventTrackHit:
    """Return the a1 scene as events, tracks and hits."""
    return EventTrackHit(a1_frame)


@pytest.fixture(scope="module")
def ops_prompt(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the oPs scene with a prompt gamma as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))


def test_the_first_hit_decides_about_its_track(a1_frame: pd.DataFrame, a1: EventTrackHit) -> None:
    """Tracks whose first interaction was a Compton scattering, with all their hits."""
    # ARRANGE
    hits = numbered(a1_frame)
    first = hits[hits["position"] == 1]
    expected = int((first["processName"] == "Compton").sum())

    # ACT
    result = Analysis().hit(1).process(Process.COMPTON).apply(a1)

    # ASSERT
    assert result.track_count == expected
    assert result.hit_count > result.track_count
    assert set(result.first_hits().values("processName").tolist()) == {"Compton"}


def test_the_last_hit_is_counted_from_the_end(a1: EventTrackHit) -> None:
    """Every gamma of the a1 scene ends in a photoelectric absorption."""
    # ARRANGE
    # No additional setup required.

    # ACT
    absorbed = Analysis().hit(-1).process(Process.PHOTOELECTRIC).apply(a1)
    scattered_last = Analysis().hit(-1).process(Process.COMPTON).apply(a1)

    # ASSERT
    assert absorbed.track_count == a1.track_count
    assert scattered_last.event_count == 0


def test_a_track_without_the_position_is_left_out(a1: EventTrackHit) -> None:
    """No track of the scene has a fifth hit to pass anything."""
    # ARRANGE
    # No additional setup required.

    # ACT
    result = Analysis().hit(5).process(Process.COMPTON, Process.PHOTOELECTRIC).apply(a1)

    # ASSERT
    assert result.event_count == 0


def test_a_position_counts_what_was_kept_before_it(
    a1_frame: pd.DataFrame, a1: EventTrackHit
) -> None:
    """The second Compton scattering of a track, not its second hit."""
    # ARRANGE
    compton = a1_frame[a1_frame["processName"] == "Compton"]
    expected = int((compton.groupby(TRACK_KEY).size() >= 2).sum())

    # ACT
    result = Analysis().hits().process(Process.COMPTON).hit(2).where("edep", ge=0.0).apply(a1)

    # ASSERT
    assert result.track_count == expected


def test_the_first_track_decides_about_its_event(ops_prompt: EventTrackHit) -> None:
    """Track 1 is the prompt gamma in every full event; the cut one has an annihilation gamma."""
    # ARRANGE
    # No additional setup required.

    # ACT
    prompt_first = Analysis().track(1).gamma_type(GammaType.PROMPT).apply(ops_prompt)
    fifth = Analysis().track(5).gamma_type(GammaType.PROMPT).apply(ops_prompt)

    # ASSERT
    assert prompt_first.event_count == 79
    assert prompt_first.track_count == 4 * 79
    assert fifth.event_count == 0


def test_a_count_at_a_position_decides_about_the_parent(ops_prompt: EventTrackHit) -> None:
    """Events whose prompt gamma left exactly two hits."""
    # ARRANGE
    tracks = ops_prompt.tracks_frame(["trackNumber", "hitCount"])
    expected = int(((tracks["trackNumber"] == 1) & (tracks["hitCount"] == 2)).sum())

    # ACT
    result = Analysis().track(1).hits(2).apply(ops_prompt)

    # ASSERT
    assert result.event_count == expected
    assert result.track_count == 4 * expected


def test_a_field_above_a_position_is_asked_of_its_own_level(a1: EventTrackHit) -> None:
    """Standing at hit 1, a track field is asked of every track kept."""
    # ARRANGE
    # No additional setup required.

    # ACT
    result = Analysis().hit(1).where("trackID", eq=1).apply(a1)

    # ASSERT
    assert set(result.values("trackID").tolist()) == {1}
    assert result.track_count == 150


def example_one(closed: bool = True) -> EventTrackHitAnalysis:
    """Return the first example of the analysis concept, closed with a count or not."""
    analysis = (
        Analysis()
        .tracks(2)
        .gamma_type(GammaType.ANNIHILATION)
        .decay_type(DecayType.STANDARD)
        .source_type(SourceType.PARA_POSITRONIUM)
        .hits(eq=2)
        .hit(1)
        .process(Process.COMPTON)
        .hit(2)
        .process(Process.PHOTOELECTRIC)
    )
    return analysis.tracks(2) if closed else analysis


def test_the_first_example_of_the_concept_finds_what_pandas_finds(
    positronium_files: Mapping[str, Path],
) -> None:
    """Two annihilation gammas of pPs, each a Compton scattering then an absorption."""
    # ARRANGE
    frame = read_tree(positronium_files["pps"], GateTree.HITS).to_dataframe()
    hits = numbered(frame)
    tracks = hits.groupby(TRACK_KEY).agg(
        hits=("position", "size"),
        first=("processName", "first"),
        last=("processName", "last"),
        gamma=("gammaType", "first"),
        decay=("decayType", "first"),
        source=("sourceType", "first"),
    )
    passing = (
        (tracks["hits"] == 2)
        & (tracks["first"] == "Compton")
        & (tracks["last"] == "PhotoElectric")
        & (tracks["gamma"] == GammaType.ANNIHILATION)
        & (tracks["decay"] == DecayType.STANDARD)
        & (tracks["source"] == SourceType.PARA_POSITRONIUM)
    )
    per_event = passing.groupby(level=EVENT_KEY).agg(["size", "sum"])
    expected = int(((per_event["size"] == 2) & (per_event["sum"] == 2)).sum())

    # ACT
    result = example_one().apply(frame)

    # ASSERT
    assert expected == 18
    assert result.event_count == expected
    assert result.track_count == 2 * expected
    assert result.hit_count == 4 * expected


def test_without_the_closing_count_a_half_passing_event_stays(
    positronium_files: Mapping[str, Path],
) -> None:
    """Filters keep what passes; only the count at the end asks for both gammas."""
    # ARRANGE
    frame = read_tree(positronium_files["pps"], GateTree.HITS).to_dataframe()

    # ACT
    closed = example_one().apply(frame)
    unclosed = example_one(closed=False).apply(frame)

    # ASSERT
    assert unclosed.event_count > closed.event_count
    assert 1 in unclosed.track_counts()


def test_the_example_split_in_two_gives_the_same(positronium_files: Mapping[str, Path]) -> None:
    """A shared part, continued by a second analysis, is the same chain."""
    # ARRANGE
    frame = read_tree(positronium_files["pps"], GateTree.HITS).to_dataframe()
    two_tracks_from_source = (
        Analysis()
        .tracks(2)
        .gamma_type(GammaType.ANNIHILATION)
        .decay_type(DecayType.STANDARD)
        .source_type(SourceType.PARA_POSITRONIUM)
    )
    scatter_then_absorb = (
        Analysis()
        .hits(eq=2)
        .hit(1)
        .process(Process.COMPTON)
        .hit(2)
        .process(Process.PHOTOELECTRIC)
        .tracks(2)
    )

    # ACT
    result = two_tracks_from_source.then(scatter_then_absorb).apply(frame)

    # ASSERT
    assert result.event_count == example_one().apply(frame).event_count == 18


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Analysis().hit(0), "count from 1"),
        (lambda: Analysis().track(1.5), "a whole number, got 1.5"),  # type: ignore[arg-type]
        (lambda: Analysis().track(True), "a whole number, got True"),
    ],
)
def test_a_position_is_a_whole_number_other_than_zero(build: object, message: str) -> None:
    """Positions are counted the way "the first hit" is said."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        build()  # type: ignore[operator]

    # ASSERT
    assert message in str(raised.value)


def test_a_position_is_shown_as_written_and_described() -> None:
    """And says that a condition there decides about the parent."""
    # ARRANGE
    analysis = Analysis().track(1).events().hit(-1)

    # ACT
    shown = repr(analysis)
    described = analysis.describe().splitlines()

    # ASSERT
    assert shown == "EventTrackHitAnalysis().track(1).events().hit(-1)"
    assert described == [
        "1. Move to track 1 of every event; a condition there decides about the event.",
        "2. Move to the events.",
        "3. Move to hit 1 from the end of every track; a condition there decides about the track.",
    ]


def test_a_condition_at_a_position_is_described_as_deciding_about_the_parent() -> None:
    """Words that match what the condition does."""
    # ARRANGE
    analysis = Analysis().hit(1).process(Process.COMPTON).track(-1).hits(2)

    # ACT
    described = analysis.describe().splitlines()

    # ASSERT
    assert described[1] == "2. Keep the tracks whose hit 1 passes process(Process.COMPTON)."
    assert described[3] == (
        "4. Keep the events whose track 1 from the end has a number of hits passing eq=2, "
        "and move to their hits."
    )


@pytest.mark.parametrize("move", [lambda a: a.hits(), lambda a: a.hit(1)])
def test_hits_cannot_be_moved_to_from_one_track(move: object) -> None:
    """Positions do not nest: hit 1 after track 1 would be hit 1 of every track."""
    # ARRANGE
    analysis = Analysis().track(1)

    # ACT
    with pytest.raises(ValueError) as raised:
        move(analysis)  # type: ignore[operator]

    # ASSERT
    assert "after track(1) would stand at the hits of every track" in str(raised.value)


def test_a_count_of_hits_at_one_track_is_still_allowed(ops_prompt: EventTrackHit) -> None:
    """It decides about the event, and then stands at the hits of every track of it."""
    # ARRANGE
    analysis = Analysis().track(1).hits(2).process(Process.PHOTOELECTRIC)

    # ACT
    result = analysis.apply(ops_prompt)

    # ASSERT
    assert set(result.values("processName").tolist()) == {"PhotoElectric"}


def test_positions_counted_in_a_condition_are_the_kept_ones(
    a1_frame: pd.DataFrame, a1: EventTrackHit
) -> None:
    """where() on hitCount or hitNumber counts what the analysis kept, as hits(n) does."""
    # ARRANGE
    compton = Analysis().hits().process(Process.COMPTON)

    # ACT
    by_where = compton.tracks().where("hitCount", eq=1).apply(a1)
    by_count = compton.hits(eq=1).apply(a1)
    second_by_where = compton.where("hitNumber", eq=2).apply(a1)
    second_by_position = compton.hit(2).where("edep", ge=0.0).apply(a1)

    # ASSERT
    assert by_where.track_count == by_count.track_count
    assert by_where.track_count > 0
    assert second_by_where.hit_count == second_by_where.track_count
    assert second_by_where.track_count == second_by_position.track_count


def test_track_counts_and_numbers_in_a_condition_are_the_kept_ones(
    ops_prompt: EventTrackHit,
) -> None:
    """trackCount and trackNumber after keeping the annihilation gammas."""
    # ARRANGE
    annihilation = Analysis().tracks().gamma_type(GammaType.ANNIHILATION)

    # ACT
    three = annihilation.where("trackCount", eq=3).apply(ops_prompt)
    first = annihilation.where("trackNumber", eq=1).apply(ops_prompt)

    # ASSERT
    assert three.event_count == annihilation.tracks(3).apply(ops_prompt).event_count == 79
    # Track 2 in every full event; the cut last event kept only its track 4.
    assert sorted(first.values("trackID").tolist()) == [2] * 79 + [4]
