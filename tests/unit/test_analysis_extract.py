"""Unit tests for the table of one row per event an analysis extracts."""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.analysis.eventtrackhit import EventTrackHitAnalysis
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import EventTrackHitField
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process

Analysis = EventTrackHitAnalysis
Field = EventTrackHitField

EXAMPLE_ONE = (
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
    .tracks(2)
)


@pytest.fixture(scope="module")
def pps(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the pPs scene as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["pps"], GateTree.HITS))


@pytest.fixture(scope="module")
def all_variants(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the scene holding every channel of a PositroniumSource."""
    return EventTrackHit(read_tree(positronium_files["all-variants"], GateTree.HITS))


def test_the_third_example_of_the_concept_has_the_columns_it_names(pps: EventTrackHit) -> None:
    """Variable after variable, track after track, hit after hit."""
    # ARRANGE
    variables = [Field.EDEP, Field.POS_X, Field.POS_Y, Field.POS_Z, Field.TIME]

    # ACT
    table = EXAMPLE_ONE.extract(pps, variables)

    # ASSERT
    elements = ["t1h1", "t1h2", "t2h1", "t2h2"]
    expected = ["runID", "eventID"] + [
        f"{element}_{name}"
        for name in ("edep", "posX", "posY", "posZ", "time")
        for element in elements
    ]
    assert list(table.columns) == expected
    assert len(table) == 18


def test_every_cell_is_the_value_of_its_hit(pps: EventTrackHit) -> None:
    """The cells of one event are the rows of its hits, in order."""
    # ARRANGE
    kept = EXAMPLE_ONE.apply(pps)
    hits = kept.hits_frame([Field.EDEP, Field.TIME])

    # ACT
    table = EXAMPLE_ONE.extract(kept, [Field.EDEP, Field.TIME])

    # ASSERT
    first_event = hits[hits["eventIndex"] == 0]
    assert [table.loc[0, f"t{t}h{h}_edep"] for t in (1, 2) for h in (1, 2)] == first_event[
        "edep"
    ].tolist()
    assert table.loc[0, "t2h2_time"] == first_event["time"].iloc[-1]


def test_a_gamma_absorbed_after_one_scattering_deposits_all_it_had(pps: EventTrackHit) -> None:
    """Each pair of deposits of a track adds up to 511 keV."""
    # ARRANGE
    # No additional setup required.

    # ACT
    table = EXAMPLE_ONE.extract(pps, [Field.EDEP])

    # ASSERT
    for track in (1, 2):
        total = table[f"t{track}h1_edep"].astype(float) + table[f"t{track}h2_edep"]
        np.testing.assert_allclose(total, 0.511, atol=1e-5)


def test_fields_of_events_and_tracks_have_columns_of_their_own(pps: EventTrackHit) -> None:
    """An event field is one column, a track field one per track, types kept."""
    # ARRANGE
    variables = [Field.SOURCE_TYPE, Field.TRACK_ID, Field.GAMMA_TYPE, Field.HIT_COUNT]

    # ACT
    table = EXAMPLE_ONE.extract(pps, variables)

    # ASSERT
    assert list(table.columns) == [
        "runID",
        "eventID",
        "sourceType",
        "t1_trackID",
        "t2_trackID",
        "t1_gammaType",
        "t2_gammaType",
        "t1_hitCount",
        "t2_hitCount",
    ]
    assert table["t1_trackID"].dtype == np.int32
    assert set(table["t1_trackID"]) == {1}
    assert set(table["t2_trackID"]) == {2}
    assert set(table["t1_hitCount"]) == {2}


def test_the_columns_of_one_hit_are_read_as_a_vector(pps: EventTrackHit) -> None:
    """The names of the branches stay, so the gate namespace finds the triples."""
    # ARRANGE
    table = EXAMPLE_ONE.extract(pps, [Field.POS_X, Field.POS_Y, Field.POS_Z])

    # ACT
    first = table.gate.vector("t1h1_posX", "t1h1_posY", "t1h1_posZ")
    second = table.gate.vector("t1h2_posX", "t1h2_posY", "t1h2_posZ")

    # ASSERT
    assert len((second - first).norm()) == 18


def test_a_table_is_as_wide_as_the_narrowest_event_allows(
    all_variants: EventTrackHit, caplog: pytest.LogCaptureFixture
) -> None:
    """Two tracks per event and one hit per track, and the log says what was left out."""
    # ARRANGE
    analysis = Analysis().tracks()

    # ACT
    with caplog.at_level(logging.WARNING, logger="opengate_gate_tree"):
        table = analysis.extract(all_variants, [Field.EDEP])

    # ASSERT
    assert list(table.columns) == ["runID", "eventID", "t1h1_edep", "t2h1_edep"]
    assert len(table) == 115
    assert not table.isna().any().any()
    left_tracks = int(np.clip(all_variants.track_counts() - 2, 0, None).sum())
    assert f"left out {left_tracks} track(s) of 62 event(s)" in caplog.text


def test_nothing_is_reported_when_everything_fits(
    pps: EventTrackHit, caplog: pytest.LogCaptureFixture
) -> None:
    """An analysis ending with fixed counts loses nothing."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with caplog.at_level(logging.WARNING, logger="opengate_gate_tree"):
        EXAMPLE_ONE.extract(pps, [Field.EDEP])

    # ASSERT
    assert caplog.text == ""


def test_a_narrower_table_can_be_asked_for_but_not_a_wider_one(pps: EventTrackHit) -> None:
    """A wider one would leave cells empty."""
    # ARRANGE
    # No additional setup required.

    # ACT
    narrow = EXAMPLE_ONE.extract(pps, [Field.EDEP], max_tracks=1, max_hits=1)
    with pytest.raises(ValueError) as raised:
        EXAMPLE_ONE.extract(pps, [Field.EDEP], max_hits=3)

    # ASSERT
    assert list(narrow.columns) == ["runID", "eventID", "t1h1_edep"]
    assert "3 hits per track would leave cells empty" in str(raised.value)
    with pytest.raises(ValueError):
        EXAMPLE_ONE.extract(pps, [Field.EDEP], max_tracks=-1)


def test_the_variables_have_to_be_in_order(pps: EventTrackHit) -> None:
    """A set has no order for the columns to follow."""
    # ARRANGE
    variables = {Field.EDEP, Field.TIME}

    # ACT
    with pytest.raises(ValueError) as raised:
        EXAMPLE_ONE.extract(pps, variables)  # type: ignore[arg-type]

    # ASSERT
    assert "as a list or a tuple" in str(raised.value)
    assert "got a set" in str(raised.value)


def test_the_event_key_comes_first_and_once(pps: EventTrackHit) -> None:
    """Asking for it again, or a variable twice, adds no column."""
    # ARRANGE
    variables = ["eventID", Field.EDEP, "edep", Field.RUN_ID]

    # ACT
    table = EXAMPLE_ONE.extract(pps, variables)

    # ASSERT
    assert list(table.columns[:2]) == ["runID", "eventID"]
    assert list(table.columns).count("t1h1_edep") == 1
    assert len(table.columns) == 2 + 4


def test_a_variable_of_several_components_has_a_column_per_component(
    positronium_files: Mapping[str, Path],
) -> None:
    """volumeID read from the columns of a file is ten numbers per hit."""
    # ARRANGE
    data = read_tree(positronium_files["pps"], GateTree.HITS)

    # ACT
    table = EXAMPLE_ONE.extract(data, [Field.VOLUME_ID])

    # ASSERT
    assert "t1h1_volumeID_0" in table.columns
    assert "t2h2_volumeID_9" in table.columns
    assert len(table.columns) == 2 + 4 * 10


def test_an_analysis_that_keeps_nothing_gives_an_empty_table(pps: EventTrackHit) -> None:
    """With the event key alone, as no event gives the table a width."""
    # ARRANGE
    analysis = Analysis().tracks(9)

    # ACT
    table = analysis.extract(pps, [Field.EDEP, Field.TRACK_ID])

    # ASSERT
    assert table.empty
    assert list(table.columns) == ["runID", "eventID"]
    assert isinstance(table, pd.DataFrame)


def test_an_empty_table_can_be_given_the_width_it_would_have(pps: EventTrackHit) -> None:
    """So that code reading its columns finds them whether anything passed or not."""
    # ARRANGE
    analysis = Analysis().tracks(9)

    # ACT
    table = analysis.extract(pps, [Field.EDEP, Field.TRACK_ID], max_tracks=2, max_hits=2)

    # ASSERT
    assert table.empty
    assert list(table.columns) == list(
        EXAMPLE_ONE.extract(pps, [Field.EDEP, Field.TRACK_ID]).columns
    )
    assert table["t1_trackID"].dtype == np.int32
    assert table["t1h1_edep"].dtype == np.float32


def test_a_branch_no_field_names_is_a_variable_of_hits(pps: EventTrackHit) -> None:
    """Any branch of the data can be tabulated by its name."""
    # ARRANGE
    # No additional setup required.

    # ACT
    table = EXAMPLE_ONE.extract(pps, ["photonID"])

    # ASSERT
    assert list(table.columns[2:]) == [
        "t1h1_photonID",
        "t1h2_photonID",
        "t2h1_photonID",
        "t2h2_photonID",
    ]
