"""Unit tests for combining analyses of events, tracks and hits."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from opengate_gate_tree.analysis.eventtrackhit import EventTrackHitAnalysis
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process

Analysis = EventTrackHitAnalysis


@pytest.fixture(scope="module")
def representation(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the scene holding every channel of a PositroniumSource."""
    return EventTrackHit(read_tree(positronium_files["all-variants"], GateTree.HITS))


def hits_kept(analysis: EventTrackHitAnalysis, representation: EventTrackHit) -> set[int]:
    """Return the rows of the hits an analysis keeps."""
    return set(analysis.apply(representation).index.hit_rows.tolist())


def test_and_keeps_the_hits_both_keep(representation: EventTrackHit) -> None:
    """Prompt gammas of ortho-positronium, from two analyses."""
    # ARRANGE
    prompt = Analysis().tracks().gamma_type(GammaType.PROMPT)
    ortho = Analysis().source_type(SourceType.ORTHO_POSITRONIUM)

    # ACT
    both = hits_kept(prompt & ortho, representation)

    # ASSERT
    assert both == hits_kept(prompt, representation) & hits_kept(ortho, representation)
    assert both == hits_kept(ortho.tracks().gamma_type(GammaType.PROMPT), representation)


def test_or_keeps_the_hits_either_keeps(representation: EventTrackHit) -> None:
    """Prompt gammas, or Compton scatterings of any gamma."""
    # ARRANGE
    prompt = Analysis().tracks().gamma_type(GammaType.PROMPT)
    compton = Analysis().hits().process(Process.COMPTON)

    # ACT
    either = hits_kept(prompt | compton, representation)

    # ASSERT
    assert either == hits_kept(prompt, representation) | hits_kept(compton, representation)


def test_a_combination_leaves_out_what_it_empties(representation: EventTrackHit) -> None:
    """Two analyses keeping different tracks of one event keep none of it together."""
    # ARRANGE
    first = Analysis().tracks().where("trackID", eq=1)
    second = Analysis().tracks().where("trackID", eq=2)

    # ACT
    result = (first & second).apply(representation)

    # ASSERT
    assert result.event_count == 0


def test_after_a_combination_the_analysis_stands_at_the_events() -> None:
    """A track field has to be moved to again."""
    # ARRANGE
    combined = Analysis().tracks() & Analysis().hits()

    # ACT
    with pytest.raises(ValueError) as raised:
        combined.gamma_type(GammaType.PROMPT)

    # ASSERT
    assert "stands at the events" in str(raised.value)
    assert combined.describe().startswith("1. Keep what EventTrackHitAnalysis().tracks() and")


@pytest.fixture(scope="module")
def pps(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the pPs scene as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["pps"], GateTree.HITS))


def concept_parts() -> tuple[EventTrackHitAnalysis, EventTrackHitAnalysis, EventTrackHitAnalysis]:
    """Return the three analyses of the second example of the analysis concept."""
    two_tracks_from_source = (
        Analysis()
        .tracks(2)
        .gamma_type(GammaType.ANNIHILATION)
        .decay_type(DecayType.STANDARD)
        .source_type(SourceType.PARA_POSITRONIUM)
    )
    first_compton = Analysis().hits(eq=2).hit(1).process(Process.COMPTON)
    second_absorbed = Analysis().hit(2).process(Process.PHOTOELECTRIC)
    return two_tracks_from_source, first_compton, second_absorbed


def test_the_three_writings_of_the_second_example_agree(pps: EventTrackHit) -> None:
    """However the three analyses are grouped, they keep the same hits."""
    # ARRANGE
    first, second, third = concept_parts()

    # ACT
    flat = hits_kept(first & second & third, pps)
    left = hits_kept((first & second) & third, pps)
    right = hits_kept(first & (second & third), pps)
    swapped = hits_kept(third & first & second, pps)

    # ASSERT
    assert flat == left == right == swapped


def test_the_second_example_is_the_first_one_written_in_parts(pps: EventTrackHit) -> None:
    """None of the parts counts what another kept, so & and the chain agree here."""
    # ARRANGE
    first, second, third = concept_parts()
    chain = first.then(second).then(third)

    # ACT
    combined = (first & second & third).then(Analysis().tracks(2)).apply(pps)
    chained = chain.tracks(2).apply(pps)

    # ASSERT
    assert hits_kept(first & second & third, pps) == hits_kept(chain, pps)
    assert combined.event_count == chained.event_count == 18


def test_and_differs_from_then_when_a_count_follows_a_filter(
    positronium_files: Mapping[str, Path],
) -> None:
    """Two tracks at all, or two annihilation tracks: & counts before the other side filters."""
    # ARRANGE
    representation = EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))
    annihilation = Analysis().tracks().gamma_type(GammaType.ANNIHILATION)
    three = Analysis().tracks(3)

    # ACT
    together = (annihilation & three).apply(representation)
    after = annihilation.then(three).apply(representation)

    # ASSERT
    assert together.event_count == 0
    assert after.event_count == 79


def test_a_mixed_combination_keeps_what_its_grouping_says(pps: EventTrackHit) -> None:
    """(a & b) | c keeps the hits (a ∩ b) ∪ c keeps."""
    # ARRANGE
    first, second, third = concept_parts()

    # ACT
    mixed = hits_kept((first & second) | third, pps)
    swapped = hits_kept(third | (second & first), pps)

    # ASSERT
    expected = (hits_kept(first, pps) & hits_kept(second, pps)) | hits_kept(third, pps)
    assert mixed == swapped == expected


def test_a_combination_is_shown_with_brackets() -> None:
    """So that the grouping is read as it was written."""
    # ARRANGE
    first = Analysis().tracks(2)
    second = Analysis().hits(2)
    third = Analysis().hit(1).process(Process.COMPTON)

    # ACT
    shown = repr((first & second) | third)

    # ASSERT
    assert shown == (
        "((EventTrackHitAnalysis().tracks(2) & EventTrackHitAnalysis().hits(2)) | "
        "EventTrackHitAnalysis().hit(1).process(Process.COMPTON))"
    )


def test_then_starts_the_second_analysis_where_it_starts_on_its_own(
    representation: EventTrackHit,
) -> None:
    """After track(1), a count of hits keeps tracks, as it does alone."""
    # ARRANGE
    first = Analysis().track(1).gamma_type(GammaType.ANNIHILATION)
    second = Analysis().hits(2)

    # ACT
    chained = first.then(second)

    # ASSERT
    assert repr(chained).endswith(".events().hits(2)")
    expected = second.apply(first.apply(representation))
    assert hits_kept(chained, representation) == set(expected.index.hit_rows.tolist())


def test_a_combination_inside_a_chain_starts_from_the_events(
    representation: EventTrackHit,
) -> None:
    """Each side means what it means alone, wherever the chain stood."""
    # ARRANGE
    first = Analysis().track(1).gamma_type(GammaType.ANNIHILATION)
    combined = Analysis().hits(2) & Analysis().events()

    # ACT
    chained = first.then(combined).apply(representation)

    # ASSERT
    expected = combined.apply(first.apply(representation))
    assert set(chained.index.hit_rows.tolist()) == set(expected.index.hit_rows.tolist())


def test_tracks_and_events_given_empty_are_left_out(representation: EventTrackHit) -> None:
    """An event without a track passes no condition, and is not part of the result."""
    # ARRANGE
    emptied = representation.with_process(Process.RAYLEIGH, keep_empty=True)

    # ACT
    nothing_asked = Analysis().apply(emptied)
    tracks_asked = Analysis().tracks().where("trackID", eq=1).apply(emptied)

    # ASSERT
    assert emptied.event_count == representation.event_count
    assert nothing_asked.event_count == tracks_asked.event_count == 0
