"""Unit tests for selecting events, tracks and hits by what they carry.

Every expected number is counted again with pandas on the same file, so the
selectors are compared with a second way of answering the same question.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import EventTrackHitField
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process

EVENT_KEY = ["runID", "eventID"]
TRACK_KEY = ["runID", "eventID", "trackID"]
Field = EventTrackHitField


@pytest.fixture(scope="module")
def all_variants_frame(positronium_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return the scene holding every channel of a PositroniumSource."""
    return read_tree(positronium_files["all-variants"], GateTree.HITS).to_dataframe()


@pytest.fixture(scope="module")
def all_variants(all_variants_frame: pd.DataFrame) -> EventTrackHit:
    """Return that scene as events, tracks and hits."""
    return EventTrackHit(all_variants_frame)


@pytest.fixture(scope="module")
def ops_prompt(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the oPs scene with a prompt gamma as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))


def test_events_are_selected_by_the_model_that_emitted_them(
    all_variants: EventTrackHit, all_variants_frame: pd.DataFrame
) -> None:
    """Para-positronium decays, with every track and hit they left."""
    # ARRANGE
    pps = all_variants_frame[all_variants_frame["sourceType"] == SourceType.PARA_POSITRONIUM]

    # ACT
    selected = all_variants.with_source_type(SourceType.PARA_POSITRONIUM)

    # ASSERT
    assert selected.event_count == pps.groupby(EVENT_KEY).ngroups
    assert selected.track_count == pps.groupby(TRACK_KEY).ngroups
    assert selected.hit_count == len(pps)


def test_tracks_are_selected_by_the_kind_of_gamma(ops_prompt: EventTrackHit) -> None:
    """One prompt gamma in every full event; the cut last event has none."""
    # ARRANGE
    # No additional setup required.

    # ACT
    prompt = ops_prompt.with_gamma_type(GammaType.PROMPT)

    # ASSERT
    assert prompt.track_count == 79
    assert set(prompt.track_counts().tolist()) == {1}
    assert set(prompt.values(Field.TRACK_ID).tolist()) == {1}


def test_tracks_are_selected_by_their_decay_channel(
    all_variants: EventTrackHit, all_variants_frame: pd.DataFrame
) -> None:
    """Gammas of a decay with de-excitation, prompt and annihilation alike."""
    # ARRANGE
    expected = all_variants_frame[all_variants_frame["decayType"] == DecayType.DEEXCITATION]

    # ACT
    selected = all_variants.with_decay_type(DecayType.DEEXCITATION)

    # ASSERT
    assert selected.track_count == expected.groupby(TRACK_KEY).ngroups


def test_a_member_of_another_class_is_refused(all_variants: EventTrackHit) -> None:
    """The classes share their numbers, so another one would select something else."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        all_variants.is_gamma_type(SourceType.PARA_POSITRONIUM)  # type: ignore[arg-type]

    # ASSERT
    assert "GammaType takes its own members" in str(raised.value)


def test_tracks_are_selected_by_particle(all_variants: EventTrackHit) -> None:
    """Every track of the scene is a gamma, and none an electron."""
    # ARRANGE
    # No additional setup required.

    # ACT
    gammas = all_variants.with_particle(22)
    electrons = all_variants.with_particle(11)

    # ASSERT
    assert gammas.track_count == all_variants.track_count
    assert electrons.event_count == 0


def test_a_particle_has_to_be_named(all_variants: EventTrackHit) -> None:
    """Selecting by no particle at all would select nothing without saying so."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        all_variants.is_particle()

    # ASSERT
    assert "at least one PDG code" in str(raised.value)


def test_hits_are_selected_by_process_named_either_way(
    all_variants: EventTrackHit, all_variants_frame: pd.DataFrame
) -> None:
    """A member of Process is the string GATE writes."""
    # ARRANGE
    compton = all_variants_frame[all_variants_frame["processName"] == "Compton"]

    # ACT
    by_member = all_variants.with_process(Process.COMPTON)
    by_name = all_variants.with_process("Compton")

    # ASSERT
    assert by_member.hit_count == by_name.hit_count == len(compton)
    assert np.array_equal(by_member.index.hit_rows, by_name.index.hit_rows)
    assert by_member.track_count == compton.groupby(TRACK_KEY).ngroups


def test_tracks_and_events_left_empty_are_removed_unless_kept(all_variants: EventTrackHit) -> None:
    """A track without a Compton scattering has nothing left to say about one."""
    # ARRANGE
    # No additional setup required.

    # ACT
    removed = all_variants.with_process(Process.COMPTON)
    kept = all_variants.with_process(Process.COMPTON, keep_empty=True)

    # ASSERT
    assert removed.track_count < all_variants.track_count
    assert kept.track_count == all_variants.track_count
    assert kept.event_count == all_variants.event_count
    assert int((kept.hit_counts() == 0).sum()) == all_variants.track_count - removed.track_count


def test_a_track_left_empty_keeps_its_own_values(all_variants: EventTrackHit) -> None:
    """Its identifier and its kind of gamma do not depend on the hits it kept."""
    # ARRANGE
    nothing = np.zeros(all_variants.hit_count, dtype=bool)

    # ACT
    empty = all_variants.select_hits(nothing, keep_empty=True)

    # ASSERT
    tracks = empty.tracks_frame([Field.TRACK_ID, Field.GAMMA_TYPE, Field.HIT_COUNT])
    original = all_variants.tracks_frame([Field.TRACK_ID, Field.GAMMA_TYPE])
    assert tracks["trackID"].tolist() == original["trackID"].tolist()
    assert tracks["gammaType"].tolist() == original["gammaType"].tolist()
    assert set(tracks["hitCount"].tolist()) == {0}
    assert empty.events[0].tracks[0].track_id == all_variants.events[0].tracks[0].track_id


def test_a_selection_leaves_the_representation_it_was_made_on(
    all_variants: EventTrackHit,
) -> None:
    """And shares its columns rather than copying them."""
    # ARRANGE
    before = (all_variants.event_count, all_variants.track_count, all_variants.hit_count)

    # ACT
    selected = all_variants.with_gamma_type(GammaType.ANNIHILATION)

    # ASSERT
    assert (all_variants.event_count, all_variants.track_count, all_variants.hit_count) == before
    assert selected.data is all_variants.data
    assert selected.row_labels is all_variants.row_labels


def test_a_column_of_a_table_selects_by_position(all_variants: EventTrackHit) -> None:
    """A mask built on the events table selects the events of its rows."""
    # ARRANGE
    events = all_variants.events_frame()

    # ACT
    selected = all_variants.select_events(events["trackCount"] == 3)

    # ASSERT
    assert selected.event_count == 43
    assert set(selected.track_counts().tolist()) == {3}


@pytest.mark.parametrize(
    ("method", "length"),
    [("select_events", 115), ("select_tracks", 311), ("select_hits", 500)],
)
def test_a_mask_of_the_wrong_length_is_refused(
    all_variants: EventTrackHit, method: str, length: int
) -> None:
    """A mask of events is not a mask of tracks."""
    # ARRANGE
    mask = np.ones(length + 1, dtype=bool)

    # ACT
    with pytest.raises(ValueError) as raised:
        getattr(all_variants, method)(mask)

    # ASSERT
    assert f"{length} in all" in str(raised.value)


def test_a_value_is_tested_at_the_level_of_its_field(
    all_variants: EventTrackHit, all_variants_frame: pd.DataFrame
) -> None:
    """Deposits of 511 keV are photoelectric absorptions of a whole gamma."""
    # ARRANGE
    whole = all_variants_frame["edep"].between(0.510, 0.512)

    # ACT
    hits = all_variants.with_value(Field.EDEP, between=(0.510, 0.512))
    first_tracks = all_variants.with_value(Field.TRACK_ID, eq=1)
    some_events = all_variants.with_value(Field.EVENT_ID, isin=[0, 1, 2])

    # ASSERT
    assert hits.hit_count == int(whole.sum())
    assert set(hits.values(Field.PROCESS_NAME).tolist()) == {"PhotoElectric"}
    assert first_tracks.track_count == all_variants.event_count
    assert some_events.event_count == 3


def test_a_derived_field_can_be_tested_as_any_other(all_variants: EventTrackHit) -> None:
    """The tracks with exactly two hits."""
    # ARRANGE
    # No additional setup required.

    # ACT
    two_hits = all_variants.with_value(Field.HIT_COUNT, eq=2)

    # ASSERT
    assert two_hits.track_count == 86


@pytest.mark.parametrize(
    ("asked", "message"),
    [({}, "got none"), ({"eq": 1, "isin": [1]}, "['eq', 'isin']")],
)
def test_a_value_test_says_what_it_tests_for_once(
    all_variants: EventTrackHit, asked: dict[str, object], message: str
) -> None:
    """Two tests at once would leave open how they combine."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        all_variants.is_value(Field.TRACK_ID, **asked)  # type: ignore[arg-type]

    # ASSERT
    assert message in str(raised.value)


def test_selections_follow_one_another(all_variants: EventTrackHit) -> None:
    """Para-positronium annihilation gammas that scattered at least once."""
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = (
        all_variants.with_source_type(SourceType.PARA_POSITRONIUM)
        .with_gamma_type(GammaType.ANNIHILATION)
        .with_process(Process.COMPTON)
    )

    # ASSERT
    assert set(selected.values(Field.SOURCE_TYPE).tolist()) == {SourceType.PARA_POSITRONIUM}
    assert set(selected.values(Field.GAMMA_TYPE).tolist()) == {GammaType.ANNIHILATION}
    assert set(selected.values(Field.PROCESS_NAME).tolist()) == {"Compton"}
