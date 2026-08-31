"""Unit tests for the selectors reading the branches of a PositroniumSource."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.filters import (
    is_decay_type,
    is_gamma_type,
    is_process,
    is_source_type,
    select_by_decay_type,
    select_by_gamma_type,
    select_by_process,
    select_by_source_type,
)
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType

# Energies the source macros of the scenes give their gammas: annihilation
# gammas start at 511 keV, and where a prompt gamma is asked for it starts at
# 1.3 MeV (setPromptPhotonEnergies). No annihilation gamma can deposit more
# than it was born with.
ANNIHILATION_ENERGY = 0.511


def hits(files: Mapping[str, Path], key: str) -> pd.DataFrame:
    """Return the hits of a scene as a frame."""
    return read_tree(files[key], GateTree.HITS).to_dataframe()


def test_a_kind_of_gamma_is_selected_by_what_it_is(
    positronium_files: Mapping[str, Path],
) -> None:
    """The rows kept have to be the prompt gammas, not merely as many of them.

    The source macro of this scene asks for a prompt gamma of 1.3 MeV on every
    decay, so a prompt gamma can leave more than 511 keV in the detector and an
    annihilation gamma cannot. That is what tells the two selections apart
    beyond their size.
    """
    # ARRANGE
    frame = hits(positronium_files, "ops-prompt")

    # ACT
    prompt = frame[is_gamma_type(frame["gammaType"], GammaType.PROMPT)]
    annihilation = frame[is_gamma_type(frame["gammaType"], GammaType.ANNIHILATION)]

    # ASSERT
    assert (prompt["edep"] > ANNIHILATION_ENERGY).any()
    assert (annihilation["edep"] <= ANNIHILATION_ENERGY).all()


def test_several_kinds_are_selected_at_once(positronium_files: Mapping[str, Path]) -> None:
    """Asking for annihilation or prompt should not mean building a mask by hand."""
    # ARRANGE
    frame = hits(positronium_files, "ops-prompt")
    kinds = frame["gammaType"]

    # ACT
    both = select_by_gamma_type(kinds, GammaType.ANNIHILATION, GammaType.PROMPT)

    # ASSERT
    assert len(both) == len(kinds)
    assert len(both) > len(select_by_gamma_type(kinds, GammaType.PROMPT))


def test_a_source_type_selects_the_component_its_macro_configured(
    positronium_files: Mapping[str, Path],
) -> None:
    """The macro of this scene mixes two components, half and half.

    setPositronInteractions names them in order - para-positronium first,
    direct annihilation second - and GATE numbers the components in that
    order, so the source type and the component index have to agree.
    """
    # ARRANGE
    frame = hits(positronium_files, "pps-direct")

    # ACT
    para = frame[is_source_type(frame["sourceType"], SourceType.PARA_POSITRONIUM)]
    direct = frame[is_source_type(frame["sourceType"], SourceType.DIRECT_ANNIHILATION)]

    # ASSERT
    assert set(para["decayIndex"]) == {0}
    assert set(direct["decayIndex"]) == {1}
    assert len(para) + len(direct) == len(frame)


def test_a_decay_channel_is_selected_by_what_it_is(
    positronium_files: Mapping[str, Path],
) -> None:
    """A prompt gamma comes from a deexcitation channel, and the branch says so."""
    # ARRANGE
    frame = hits(positronium_files, "ops-prompt")

    # ACT
    deexcitation = select_by_decay_type(frame["decayType"], DecayType.DEEXCITATION)
    standard = select_by_decay_type(frame["decayType"], DecayType.STANDARD)

    # ASSERT
    assert len(deexcitation) == len(frame)
    assert len(standard) == 0


def test_the_selection_of_a_scene_matches_what_it_holds(
    positronium_files: Mapping[str, Path],
) -> None:
    """A scene of one channel keeps every row, and any other keeps none."""
    # ARRANGE
    frame = hits(positronium_files, "pps")

    # ACT
    para = select_by_source_type(frame["sourceType"], SourceType.PARA_POSITRONIUM)
    ortho = select_by_source_type(frame["sourceType"], SourceType.ORTHO_POSITRONIUM)

    # ASSERT
    assert len(para) == len(frame)
    assert len(ortho) == 0


def test_a_member_of_another_class_is_refused(positronium_files: Mapping[str, Path]) -> None:
    """The classes share their numbers, so the wrong one selects by another meaning.

    A source type of 2 is a para-positronium and a gamma type of 2 is an
    annihilation gamma. Passing one where the other belongs would answer
    something, which is worse than refusing.
    """
    # ARRANGE
    frame = hits(positronium_files, "all-variants")

    # ACT / ASSERT
    with pytest.raises(ValueError, match="takes its own members"):
        is_gamma_type(frame["gammaType"], SourceType.PARA_POSITRONIUM)


@pytest.mark.parametrize("given", [2, "PROMPT", None], ids=["number", "text", "nothing"])
def test_a_value_that_is_not_a_member_is_refused(given: object) -> None:
    """The number behind a member is exactly what must not be passed instead of it."""
    # ARRANGE
    values = pd.Series([0, 2, 3])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="takes its own members"):
        is_gamma_type(values, given)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "select",
    [is_source_type, is_decay_type, is_gamma_type],
    ids=["source", "decay", "gamma"],
)
def test_selecting_by_nothing_is_refused(select: object) -> None:
    """A selection of no members is a call with no meaning."""
    # ARRANGE
    values = pd.Series([0, 1, 2])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="at least one member"):
        select(values)  # type: ignore[operator]


def test_a_process_is_selected_by_its_name(positronium_files: Mapping[str, Path]) -> None:
    """Which interaction a hit was is written as a name, and selected as one."""
    # ARRANGE
    frame = hits(positronium_files, "all-variants")
    processes = frame["processName"]
    present = sorted(set(processes))

    # ACT
    first = select_by_process(processes, present[0])
    every = select_by_process(processes, *present)

    # ASSERT
    assert set(first) == {present[0]}
    assert len(every) == len(processes)


def test_a_process_no_hit_underwent_selects_nothing(
    positronium_files: Mapping[str, Path],
) -> None:
    """Asking about a process the file does not hold is a question, not a mistake."""
    # ARRANGE
    frame = hits(positronium_files, "pps")

    # ACT
    selected = select_by_process(frame["processName"], "NoSuchProcess")

    # ASSERT
    assert len(selected) == 0


def test_selecting_by_no_process_is_refused() -> None:
    """A selection of no names is a call with no meaning either."""
    # ARRANGE
    values = pd.Series(["Compton", "PhotoElectric"])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="at least one name"):
        is_process(values)


def test_the_masks_cover_every_row(positronium_files: Mapping[str, Path]) -> None:
    """A mask is what combines with other conditions, so it keeps its shape."""
    # ARRANGE
    frame = hits(positronium_files, "all-variants")

    # ACT
    of_kind = is_gamma_type(frame["gammaType"], GammaType.PROMPT)
    of_process = is_process(frame["processName"], "Compton")

    # ASSERT
    assert len(of_kind) == len(frame)
    assert list(of_process.index) == list(frame.index)
    assert of_kind.dtype == bool
