"""Unit tests for combining analyses of events, tracks and hits."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from opengate_gate_tree.analysis.eventtrackhit import EventTrackHitAnalysis
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import GammaType, SourceType
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
