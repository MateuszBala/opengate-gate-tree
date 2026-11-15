"""Unit tests for moving through events, tracks and hits and keeping what passes.

Every condition of an analysis is compared with the selector of the
representation that asks the same question, which was tested against pandas.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.analysis.eventtrackhit import EventTrackHitAnalysis
from opengate_gate_tree.analysis.operations import ValueTest
from opengate_gate_tree.errors import BranchNotFoundError
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation import eventtrackhit as representation_module
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import EventTrackHitField
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process

Field = EventTrackHitField
Analysis = EventTrackHitAnalysis


@pytest.fixture(scope="module")
def frame(positronium_files: Mapping[str, Path]) -> pd.DataFrame:
    """Return the scene holding every channel of a PositroniumSource."""
    return read_tree(positronium_files["all-variants"], GateTree.HITS).to_dataframe()


@pytest.fixture(scope="module")
def representation(frame: pd.DataFrame) -> EventTrackHit:
    """Return that scene as events, tracks and hits."""
    return EventTrackHit(frame)


def same(first: EventTrackHit, second: EventTrackHit) -> bool:
    """Return whether two representations keep the same hits, tracks and events."""
    return (
        np.array_equal(first.index.hit_rows, second.index.hit_rows)
        and np.array_equal(first.index.track_offsets, second.index.track_offsets)
        and np.array_equal(first.index.event_offsets, second.index.event_offsets)
        and np.array_equal(first.index.track_rows, second.index.track_rows)
    )


def test_an_analysis_of_nothing_keeps_everything(representation: EventTrackHit) -> None:
    """Without operations, the result is the representation it was given."""
    # ARRANGE
    analysis = Analysis()

    # ACT
    result = analysis.apply(representation)

    # ASSERT
    assert same(result, representation)
    assert result.data is representation.data


@pytest.mark.parametrize(
    ("analysis", "selector"),
    [
        (
            Analysis().source_type(SourceType.PARA_POSITRONIUM),
            lambda rep: rep.with_source_type(SourceType.PARA_POSITRONIUM),
        ),
        (
            Analysis().tracks().gamma_type(GammaType.PROMPT),
            lambda rep: rep.with_gamma_type(GammaType.PROMPT),
        ),
        (
            Analysis().tracks().decay_type(DecayType.DEEXCITATION, DecayType.NONE),
            lambda rep: rep.with_decay_type(DecayType.DEEXCITATION, DecayType.NONE),
        ),
        (Analysis().tracks().particle(22), lambda rep: rep.with_particle(22)),
        (
            Analysis().hits().process(Process.COMPTON),
            lambda rep: rep.with_process(Process.COMPTON),
        ),
        (Analysis().hits().process("Compton"), lambda rep: rep.with_process("Compton")),
        (
            Analysis().hits().edep(between=(0.510, 0.512)),
            lambda rep: rep.with_value(Field.EDEP, between=(0.510, 0.512)),
        ),
        (
            Analysis().hits().time(lt=2e-5),
            lambda rep: rep.select_hits(rep.values(Field.TIME) < 2e-5),
        ),
        (
            Analysis().tracks().where(Field.TRACK_ID, eq=1),
            lambda rep: rep.with_value(Field.TRACK_ID, eq=1),
        ),
        (
            Analysis().where("eventID", isin=[0, 1, 2]),
            lambda rep: rep.with_value(Field.EVENT_ID, isin=[0, 1, 2]),
        ),
        (
            Analysis().hits().where("photonID", eq=0),
            lambda rep: rep.with_value("photonID", eq=0),
        ),
    ],
)
def test_every_condition_keeps_what_the_selector_of_the_same_question_keeps(
    representation: EventTrackHit, analysis: EventTrackHitAnalysis, selector: object
) -> None:
    """The analysis and the representation answer the same question the same way."""
    # ARRANGE
    expected = selector(representation)  # type: ignore[operator]

    # ACT
    result = analysis.apply(representation)

    # ASSERT
    assert same(result, expected)


def test_a_field_of_a_level_above_is_asked_of_that_level(representation: EventTrackHit) -> None:
    """The source model of an event can be asked while standing at its tracks."""
    # ARRANGE
    analysis = (
        Analysis()
        .tracks()
        .gamma_type(GammaType.ANNIHILATION)
        .source_type(SourceType.PARA_POSITRONIUM)
    )

    # ACT
    result = analysis.apply(representation)

    # ASSERT
    expected = representation.with_gamma_type(GammaType.ANNIHILATION).with_source_type(
        SourceType.PARA_POSITRONIUM
    )
    assert same(result, expected)


def test_a_field_of_a_level_below_has_to_be_moved_to_first() -> None:
    """A track has no single energy deposit; the analysis says where to go."""
    # ARRANGE
    analysis = Analysis().tracks()

    # ACT
    with pytest.raises(ValueError) as raised:
        analysis.edep(gt=0.1)

    # ASSERT
    assert "'edep' describes hits, but the analysis stands at the tracks" in str(raised.value)
    assert "hits()" in str(raised.value)


def test_moving_back_to_the_events_stands_there_again() -> None:
    """After events(), a track field has to be moved to again."""
    # ARRANGE
    analysis = Analysis().tracks().gamma_type(GammaType.PROMPT).events()

    # ACT
    with pytest.raises(ValueError) as raised:
        analysis.gamma_type(GammaType.PROMPT)

    # ASSERT
    assert "stands at the events" in str(raised.value)


def test_a_filter_leaves_out_what_it_empties(representation: EventTrackHit) -> None:
    """Tracks with no Compton scattering, and events with none, are gone."""
    # ARRANGE
    analysis = Analysis().hits().process(Process.COMPTON)

    # ACT
    result = analysis.apply(representation)

    # ASSERT
    assert result.track_count == 131
    assert set(result.hit_counts().tolist()) >= {1}
    assert 0 not in result.hit_counts()
    assert 0 not in result.track_counts()


def test_the_data_is_read_only_when_the_analysis_is_performed(
    representation: EventTrackHit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And a field asked for twice is read once."""
    # ARRANGE
    reads: list[object] = []
    original = representation_module.EventTrackHit.values

    def counted(self: EventTrackHit, field: object, level: object = None) -> np.ndarray:
        reads.append(field)
        return original(self, field, level)  # type: ignore[arg-type]

    monkeypatch.setattr(representation_module.EventTrackHit, "values", counted)

    # ACT
    analysis = Analysis().hits().edep(gt=0.1).process(Process.COMPTON).edep(lt=0.4)
    before = list(reads)
    analysis.apply(representation)

    # ASSERT
    assert before == []
    assert reads.count(Field.EDEP) == 1
    assert reads.count(Field.PROCESS_NAME) == 1


def test_an_analysis_takes_columns_or_a_frame_as_well(
    positronium_files: Mapping[str, Path], representation: EventTrackHit
) -> None:
    """The representation is then built inside."""
    # ARRANGE
    data = read_tree(positronium_files["all-variants"], GateTree.HITS)
    analysis = Analysis().tracks().gamma_type(GammaType.PROMPT)

    # ACT
    from_columns = analysis.apply(data)
    from_frame = analysis.apply(data.to_dataframe())

    # ASSERT
    expected = analysis.apply(representation)
    assert same(from_columns, expected)
    assert same(from_frame, expected)


def test_a_field_the_data_does_not_hold_is_reported_when_performed(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The Compton camera output has no gammaType; nothing knows that before the data."""
    # ARRANGE
    analysis = Analysis().tracks().gamma_type(GammaType.PROMPT)
    data = read_tree(hits_variant_files["a4"], GateTree.HITS)

    # ACT
    with pytest.raises(BranchNotFoundError) as raised:
        analysis.apply(data)

    # ASSERT
    assert "gammaType" in str(raised.value)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Analysis().tracks().gamma_type(), "at least one member"),
        (
            lambda: Analysis().tracks().gamma_type(SourceType.PARA_POSITRONIUM),  # type: ignore[arg-type]
            "GammaType takes its own members",
        ),
        (lambda: Analysis().tracks().particle(), "at least one PDG code"),
        (lambda: Analysis().hits().process(), "at least one process"),
        (lambda: Analysis().hits().edep(), "at least one of"),
        (lambda: Analysis().hits().edep(above=0.1), "Unknown comparison(s) ['above']"),
        (lambda: Analysis().hits().edep(between=(0.1,)), "two ends of a range"),
    ],
)
def test_a_malformed_condition_is_refused_where_it_is_written(build: object, message: str) -> None:
    """Not later, when the analysis meets data."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        build()  # type: ignore[operator]

    # ASSERT
    assert message in str(raised.value)


def test_a_field_of_several_components_cannot_be_tested(
    positronium_files: Mapping[str, Path],
) -> None:
    """Ten numbers per hit cannot equal one."""
    # ARRANGE
    data = read_tree(positronium_files["all-variants"], GateTree.HITS)
    analysis = Analysis().hits().where(Field.VOLUME_ID, eq=1)

    # ACT
    with pytest.raises(ValueError) as raised:
        analysis.apply(data)

    # ASSERT
    assert "10 components per element" in str(raised.value)


def test_comparisons_combine_and_a_missing_value_passes_none() -> None:
    """ge and lt together are a half-open range; ne keeps what is not a number."""
    # ARRANGE
    values = np.array([0.1, np.nan, 0.5])

    # ACT
    half_open = ValueTest((("ge", 0.1), ("lt", 0.5))).passes(values)
    between = ValueTest((("between", (0.1, 0.5)),)).passes(values)
    not_equal = ValueTest((("ne", 0.1),)).passes(values)

    # ASSERT
    assert half_open.tolist() == [True, False, False]
    assert between.tolist() == [True, False, True]
    assert not_equal.tolist() == [False, True, True]


def test_an_analysis_is_shown_and_described() -> None:
    """As it was written, and in words."""
    # ARRANGE
    analysis = (
        Analysis()
        .source_type(SourceType.PARA_POSITRONIUM)
        .tracks()
        .where(Field.TRACK_ID, isin=[1, 2])
        .hits()
        .process(Process.COMPTON, "Rayl")
        .edep(ge=0.1)
    )

    # ACT
    shown = repr(analysis)
    described = analysis.describe()

    # ASSERT
    assert shown == (
        "EventTrackHitAnalysis().source_type(SourceType.PARA_POSITRONIUM).tracks()"
        ".where(EventTrackHitField.TRACK_ID, isin=(1, 2)).hits()"
        ".process(Process.COMPTON, 'Rayl').edep(ge=0.1)"
    )
    assert described.splitlines() == [
        "1. Keep the events that pass source_type(SourceType.PARA_POSITRONIUM).",
        "2. Move to the tracks.",
        "3. Keep the tracks that pass where(EventTrackHitField.TRACK_ID, isin=(1, 2)).",
        "4. Move to the hits.",
        "5. Keep the hits that pass process(Process.COMPTON, 'Rayl').",
        "6. Keep the hits that pass edep(ge=0.1).",
    ]
    assert Analysis().describe() == "Keep every event, track and hit."


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Analysis().tracks(ge=3, le=1), "No count passes ge=3, le=1"),
        (lambda: Analysis().tracks(lt=0), "No count passes lt=0"),
        (lambda: Analysis().hits(gt=2, lt=3), "No count passes gt=2, lt=3"),
        (lambda: Analysis().tracks(2.5), "count=2.5"),  # type: ignore[arg-type]
        (lambda: Analysis().tracks(-1), "count=-1"),
        (lambda: Analysis().hits(True), "count=True"),
        (lambda: Analysis().tracks("2"), "count='2'"),  # type: ignore[arg-type]
        (lambda: Analysis().hits().where("processName", isin="Compton"), "isin=['Compton']"),
        (lambda: Analysis().hits().where("processName", isin=[]), "at least one value"),
        (lambda: Analysis().hits().edep(between="ab"), "two ends of a range"),
        (lambda: Analysis().hits().edep(between=5), "two ends of a range"),  # type: ignore[arg-type]
        (lambda: Analysis().hits().edep(between=(0.5, 0.1)), "lower end first"),
        (lambda: Analysis().hits().edep(between=(0.1, float("nan"))), "two ends of a range"),
        (lambda: Analysis().hits().edep(gt=float("nan")), "compares against nothing"),
        (lambda: Analysis().hits().edep(ge=0.1, inclusive="left"), "give between too"),
        (lambda: Analysis().hits().edep(between=(0, 1), inclusive="open"), "inclusive is one of"),
    ],
)
def test_bounds_that_select_by_mistake_are_refused(build: object, message: str) -> None:
    """A typo becomes an error where it is written, not an empty result."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        build()  # type: ignore[operator]

    # ASSERT
    assert message in str(raised.value)


def test_which_ends_of_a_range_belong_to_it_is_said_as_in_the_representation(
    representation: EventTrackHit,
) -> None:
    """inclusive takes the sides pandas names, as with_value does."""
    # ARRANGE
    analysis = Analysis().tracks().where(Field.TRACK_ID, between=(1, 2), inclusive="left")

    # ACT
    result = analysis.apply(representation)

    # ASSERT
    expected = representation.with_value(Field.TRACK_ID, between=(1, 2), inclusive="left")
    assert same(result, expected)
    assert repr(analysis).endswith(
        "where(EventTrackHitField.TRACK_ID, between=(1, 2), inclusive='left')"
    )


def test_analyses_asking_the_same_question_are_equal_however_spelt() -> None:
    """A count given positionally or as eq, a field by name or by member."""
    # ARRANGE
    # No additional setup required.

    # ACT
    counts = (Analysis().tracks(2), Analysis().tracks(eq=2))
    fields = (Analysis().hits().edep(ge=0.1), Analysis().hits().where("edep", ge=0.1))

    # ASSERT
    assert counts[0] == counts[1]
    assert hash(counts[0]) == hash(counts[1])
    assert fields[0] == fields[1]
