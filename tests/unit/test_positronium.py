"""Unit tests for the enum representations of the PositroniumSource branches."""

from collections.abc import Mapping
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest
from conftest import POSITRONIUM_BRANCH_FIELDS, POSITRONIUM_LAYOUTS, PositroniumLayout

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import (
    DECAY_INDEX_BRANCH,
    NOT_A_POSITRONIUM_SOURCE,
    POSITRONIUM_BRANCHES,
    DecayType,
    GammaType,
    SourceType,
    is_positronium_source,
    positronium_enum,
)
from opengate_gate_tree.tree.hits.schema import expected_branches
from opengate_gate_tree.tree.hits.variant import HitsTreeVariant

# Values GateEmittedGammaInformation.hh gives each member.
GATE_VALUES: dict[type[IntEnum], dict[str, int]] = {
    SourceType: {
        "NOT_DEFINED": 0,
        "SINGLE_GAMMA_EMITTER": 1,
        "PARA_POSITRONIUM": 2,
        "ORTHO_POSITRONIUM": 3,
        "DIRECT_ANNIHILATION": 4,
    },
    DecayType: {"NONE": 0, "STANDARD": 1, "DEEXCITATION": 2},
    GammaType: {"UNKNOWN": 0, "SINGLE": 1, "ANNIHILATION": 2, "PROMPT": 3},
}


@pytest.mark.parametrize("enum_class", GATE_VALUES, ids=lambda enum_class: enum_class.__name__)
def test_members_carry_the_values_gate_writes(enum_class: type[IntEnum]) -> None:
    """The values are the contract with GATE, so they are stated twice on purpose."""
    # ARRANGE
    expected = GATE_VALUES[enum_class]

    # ACT
    members = {member.name: int(member) for member in enum_class}

    # ASSERT
    assert members == expected


@pytest.mark.parametrize("enum_class", GATE_VALUES, ids=lambda enum_class: enum_class.__name__)
def test_members_are_the_integers_themselves(enum_class: type[IntEnum]) -> None:
    """A member has to be usable wherever the integer it stands for is."""
    # ARRANGE
    member = next(iter(enum_class))

    # ACT / ASSERT
    assert isinstance(member, int)
    assert member == int(member)


def test_a_column_compares_against_a_member_as_it_was_read(
    positronium_files: Mapping[str, Path],
) -> None:
    """This is why the classes are integer enums: no conversion before a comparison.

    A plain enum would answer a mask of False here, without raising, so the
    test works on a file rather than on a constructed array.
    """
    # ARRANGE
    data = read_tree(positronium_files["pps-prompt"], GateTree.HITS, ["gammaType"])
    column = data["gammaType"]

    # ACT
    prompt = column == GammaType.PROMPT

    # ASSERT
    assert prompt.sum() == np.count_nonzero(column == 3)
    assert 0 < prompt.sum() < column.size


def test_a_data_frame_column_compares_against_a_member_too(
    positronium_files: Mapping[str, Path],
) -> None:
    """The pandas view has to behave the way the NumPy columns do."""
    # ARRANGE
    data = read_tree(positronium_files["ops"], GateTree.HITS, ["sourceType", "edep"])
    frame = data.to_dataframe()

    # ACT
    ortho = frame[frame["sourceType"] == SourceType.ORTHO_POSITRONIUM]

    # ASSERT
    assert len(ortho) == len(frame)


@pytest.mark.parametrize("layout", POSITRONIUM_LAYOUTS, ids=lambda layout: layout.key)
def test_every_scene_reads_as_the_members_it_holds(layout: PositroniumLayout) -> None:
    """Each configuration of the source has to arrive as the members describing it."""
    # ARRANGE
    branches = list(POSITRONIUM_BRANCHES)
    data = read_tree(layout.path, GateTree.HITS, branches)

    # ACT
    members = {
        branch: {POSITRONIUM_BRANCHES[branch](value) for value in set(data[branch].tolist())}
        for branch in branches
    }

    # ASSERT
    expected = {
        branch: {POSITRONIUM_BRANCHES[branch](value) for value in getattr(layout, field)}
        for branch, field in POSITRONIUM_BRANCH_FIELDS.items()
        if branch in POSITRONIUM_BRANCHES
    }
    assert members == expected


def test_a_prompt_scene_holds_annihilation_and_prompt_gammas(
    positronium_files: Mapping[str, Path],
) -> None:
    """A deexcitation channel is what puts two kinds of gamma in one file."""
    # ARRANGE
    data = read_tree(positronium_files["ops-prompt"], GateTree.HITS, ["gammaType", "decayType"])

    # ACT
    kinds = {GammaType(value) for value in set(data["gammaType"].tolist())}
    decays = {DecayType(value) for value in set(data["decayType"].tolist())}

    # ASSERT
    assert kinds == {GammaType.ANNIHILATION, GammaType.PROMPT}
    assert decays == {DecayType.DEEXCITATION}


def test_a_branch_without_a_class_is_reported_as_such() -> None:
    """Only three branches carry values a class describes."""
    # ARRANGE
    # decayIndex holds channel numbers, whose meaning depends on the source.

    # ACT
    described = {name: positronium_enum(name) for name in ("gammaType", DECAY_INDEX_BRANCH, "edep")}

    # ASSERT
    assert described == {"gammaType": GammaType, DECAY_INDEX_BRANCH: None, "edep": None}


def test_the_described_branches_are_branches_of_the_tree() -> None:
    """A class describing a branch no structure holds would describe nothing."""
    # ARRANGE
    hits_branches = {spec.name for spec in expected_branches(HitsTreeVariant.NO_SYSTEM)}

    # ACT
    described = set(POSITRONIUM_BRANCHES)

    # ASSERT
    assert described == {"sourceType", "decayType", "gammaType"}
    assert described <= hits_branches
    assert DECAY_INDEX_BRANCH in hits_branches


def test_rows_of_a_source_that_is_not_positronium_are_told_apart(
    positronium_files: Mapping[str, Path],
) -> None:
    """A back-to-back source writes no channel number, and that is the marker."""
    # ARRANGE
    data = read_tree(positronium_files["back-to-back"], GateTree.HITS, [DECAY_INDEX_BRANCH])

    # ACT
    from_positronium = is_positronium_source(data[DECAY_INDEX_BRANCH])

    # ASSERT
    assert not from_positronium.any()
    assert set(data[DECAY_INDEX_BRANCH].tolist()) == {NOT_A_POSITRONIUM_SOURCE}


def test_rows_of_a_positronium_source_are_recognised(
    positronium_files: Mapping[str, Path],
) -> None:
    """Every row of a positronium scene carries a channel number."""
    # ARRANGE
    data = read_tree(positronium_files["pps"], GateTree.HITS, [DECAY_INDEX_BRANCH])

    # ACT
    from_positronium = is_positronium_source(data[DECAY_INDEX_BRANCH])

    # ASSERT
    assert from_positronium.all()


def test_a_mixed_file_is_split_between_the_two(
    positronium_files: Mapping[str, Path],
) -> None:
    """One file can hold both, and the mask is what separates them."""
    # ARRANGE
    data = read_tree(positronium_files["all-variants"], GateTree.HITS, [DECAY_INDEX_BRANCH])
    column = data[DECAY_INDEX_BRANCH]

    # ACT
    from_positronium = is_positronium_source(column)

    # ASSERT
    assert from_positronium.sum() == np.count_nonzero(column >= 0)
    assert 0 < from_positronium.sum() < column.size


def test_the_mask_answers_for_an_empty_column() -> None:
    """An extraction can end up with no rows, and asking is still legitimate."""
    # ARRANGE
    column = np.array([], dtype=np.int32)

    # ACT
    from_positronium = is_positronium_source(column)

    # ASSERT
    assert from_positronium.dtype == np.bool_
    assert from_positronium.size == 0
