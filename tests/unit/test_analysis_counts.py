"""Unit tests for keeping events and tracks by how many tracks and hits they kept."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from opengate_gate_tree.analysis.eventtrackhit import EventTrackHitAnalysis
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import GammaType
from opengate_gate_tree.tree.hits.processes import Process

Analysis = EventTrackHitAnalysis
TRACK_KEY = ["runID", "eventID", "trackID"]


@pytest.fixture(scope="module")
def a1(hits_variant_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the a1 scene as events, tracks and hits."""
    return EventTrackHit(read_tree(hits_variant_files["a1"], GateTree.HITS))


@pytest.fixture(scope="module")
def ops_prompt(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the oPs scene with a prompt gamma as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))


@pytest.fixture(scope="module")
def all_variants_frame(positronium_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return the scene holding every channel of a PositroniumSource."""
    return read_tree(positronium_files["all-variants"], GateTree.HITS).to_dataframe()


def test_events_are_kept_by_their_number_of_tracks(ops_prompt: EventTrackHit) -> None:
    """Every full event of the oPs scene has four gammas; the cut last one has one."""
    # ARRANGE
    # No additional setup required.

    # ACT
    four = Analysis().tracks(4).apply(ops_prompt)
    not_four = Analysis().tracks(ne=4).apply(ops_prompt)

    # ASSERT
    assert four.event_count == 79
    assert not_four.event_count == 1
    assert Analysis().tracks(eq=4).apply(ops_prompt).event_count == 79


def test_tracks_are_kept_by_their_number_of_hits(a1: EventTrackHit) -> None:
    """110 tracks of the a1 scene have two hits, 41 more than two."""
    # ARRANGE
    # No additional setup required.

    # ACT
    two = Analysis().hits(2).apply(a1)
    more = Analysis().hits(gt=2).apply(a1)
    some = Analysis().hits(ge=2, le=3).apply(a1)

    # ASSERT
    assert two.track_count == 110
    assert more.track_count == 33 + 8
    assert some.track_count == 110 + 33
    assert set(two.hit_counts().tolist()) == {2}


def test_a_count_counts_what_was_kept_before_it(ops_prompt: EventTrackHit) -> None:
    """Three annihilation gammas, or three gammas of which the annihilation ones."""
    # ARRANGE
    annihilation_then_three = Analysis().tracks().gamma_type(GammaType.ANNIHILATION).tracks(3)
    three_then_annihilation = Analysis().tracks(3).gamma_type(GammaType.ANNIHILATION)

    # ACT
    first = annihilation_then_three.apply(ops_prompt)
    second = three_then_annihilation.apply(ops_prompt)

    # ASSERT
    assert first.event_count == 79
    assert second.event_count == 0


def test_the_order_of_a_count_and_a_condition_changes_the_answer(
    all_variants_frame: pd.DataFrame,
) -> None:
    """Counted again with pandas: two annihilation gammas, or two gammas both annihilation."""
    # ARRANGE
    representation = EventTrackHit(all_variants_frame)
    tracks = all_variants_frame.groupby(TRACK_KEY)["gammaType"].first().reset_index()
    per_event = tracks.groupby(["runID", "eventID"])["gammaType"].agg(
        total="size", annihilation=lambda kinds: int((kinds == GammaType.ANNIHILATION).sum())
    )
    two_annihilation = int((per_event["annihilation"] == 2).sum())
    two_both = int(((per_event["total"] == 2) & (per_event["annihilation"] == 2)).sum())

    # ACT
    first = Analysis().tracks().gamma_type(GammaType.ANNIHILATION).tracks(2).apply(representation)
    second = Analysis().tracks(2).gamma_type(GammaType.ANNIHILATION).apply(representation)
    both = Analysis().tracks(2).gamma_type(GammaType.ANNIHILATION).tracks(2).apply(representation)

    # ASSERT
    assert (first.event_count, two_annihilation) == (46, 46)
    assert second.event_count == 35
    assert both.event_count == two_both == 35


def test_a_count_of_hits_from_the_events_counts_every_track(a1: EventTrackHit) -> None:
    """The analysis does not need to stand at the tracks to count their hits."""
    # ARRANGE
    # No additional setup required.

    # ACT
    from_events = Analysis().hits(2).apply(a1)
    from_tracks = Analysis().tracks().hits(2).apply(a1)

    # ASSERT
    assert from_events.track_count == from_tracks.track_count == 110


def test_a_count_moves_the_analysis_to_what_it_counted(a1: EventTrackHit) -> None:
    """After hits(2), a condition on hits is asked of the hits of those tracks."""
    # ARRANGE
    analysis = Analysis().hits(2).process(Process.COMPTON)

    # ACT
    result = analysis.apply(a1)

    # ASSERT
    assert set(result.values("processName").tolist()) == {"Compton"}
    assert set(result.hit_counts().tolist()) == {1}


def test_a_count_after_a_filter_leaves_out_what_it_empties(a1: EventTrackHit) -> None:
    """Events of two tracks each with two hits: tracks fail, then their events."""
    # ARRANGE
    analysis = Analysis().hits(2).tracks(2)

    # ACT
    result = analysis.apply(a1)

    # ASSERT
    assert set(result.hit_counts().tolist()) == {2}
    assert set(result.track_counts().tolist()) == {2}


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Analysis().tracks(2, eq=2), "say the same thing twice"),
        (lambda: Analysis().tracks(2, ge=1), "also be compared with ['ge']"),
        (lambda: Analysis().hits(eq=1, lt=3), "also be compared with ['lt']"),
    ],
)
def test_a_count_is_asked_for_in_one_way(build: object, message: str) -> None:
    """Equal to two and at least one is either redundant or a mistake."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        build()  # type: ignore[operator]

    # ASSERT
    assert message in str(raised.value)


def test_a_count_is_shown_as_written_and_described() -> None:
    """A positional count stays positional."""
    # ARRANGE
    analysis = Analysis().tracks(2).hits(ge=1, lt=4)

    # ACT
    shown = repr(analysis)
    described = analysis.describe().splitlines()

    # ASSERT
    assert shown == "EventTrackHitAnalysis().tracks(2).hits(ge=1, lt=4)"
    assert described == [
        "1. Keep the events whose number of tracks passes eq=2, and move to their tracks.",
        "2. Keep the tracks whose number of hits passes ge=1, lt=4, and move to their hits.",
    ]
