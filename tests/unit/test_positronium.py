"""Unit tests for the enum representations of the PositroniumSource branches."""

import logging
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
    decode_positronium_column,
    decode_positronium_value,
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
    """The pandas view has to behave the way the NumPy columns do.

    The file mixes two kinds of source on purpose: a selection keeping every
    row would pass just as well against a comparison that never filtered.
    """
    # ARRANGE
    data = read_tree(positronium_files["pps-direct"], GateTree.HITS, ["sourceType", "edep"])
    frame = data.to_dataframe()

    # ACT
    para = frame[frame["sourceType"] == SourceType.PARA_POSITRONIUM]

    # ASSERT
    assert 0 < len(para) < len(frame)
    assert set(para["sourceType"]) == {int(SourceType.PARA_POSITRONIUM)}


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


def test_one_value_is_read_as_what_it_means() -> None:
    """A single value has one meaning, taken from the class describing it."""
    # ARRANGE
    # No additional setup required.

    # ACT
    meaning = decode_positronium_value("gammaType", 3)

    # ASSERT
    assert meaning is GammaType.PROMPT


def test_a_value_the_package_does_not_know_is_refused() -> None:
    """Reading an unknown value as "not defined" would report what no file says."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        decode_positronium_value("gammaType", 7)

    # ASSERT
    assert "no meaning for the value 7" in str(raised.value)
    assert "2 (ANNIHILATION)" in str(raised.value)


def test_a_branch_the_package_does_not_describe_is_refused() -> None:
    """Only three branches carry values with a described meaning."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with pytest.raises(ValueError) as raised:
        decode_positronium_value("edep", 1)

    # ASSERT
    assert "does not hold values this package describes" in str(raised.value)
    assert "sourceType" in str(raised.value)


def test_a_column_is_read_row_by_row(positronium_files: Mapping[str, Path]) -> None:
    """Every row keeps its place, so the result lines up with the other columns."""
    # ARRANGE
    data = read_tree(positronium_files["all-variants"], GateTree.HITS, ["sourceType"])
    column = data["sourceType"]

    # ACT
    meanings = decode_positronium_column("sourceType", column)

    # ASSERT
    assert len(meanings) == len(column)
    assert all(int(meaning) == value for meaning, value in zip(meanings, column, strict=True))
    assert set(meanings) == {SourceType(value) for value in set(column.tolist())}


def test_a_column_with_an_unknown_value_is_read_as_far_as_it_can_be(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GATE build writing one value more must not cost the whole column."""
    # ARRANGE
    column = np.array([0, 2, 7, 3, 7], dtype=np.int32)

    # ACT
    with caplog.at_level(logging.WARNING):
        meanings = decode_positronium_column("gammaType", column)

    # ASSERT
    assert list(meanings) == [
        GammaType.UNKNOWN,
        GammaType.ANNIHILATION,
        None,
        GammaType.PROMPT,
        None,
    ]
    assert "7 (2 row(s))" in caplog.text


def test_a_column_the_package_understands_is_read_without_a_word(
    positronium_files: Mapping[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file GATE wrote as expected deserves no warning."""
    # ARRANGE
    data = read_tree(positronium_files["pps"], GateTree.HITS, ["decayType"])

    # ACT
    with caplog.at_level(logging.WARNING):
        decode_positronium_column("decayType", data["decayType"])

    # ASSERT
    assert caplog.text == ""


def test_an_empty_column_is_read_as_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """An extraction can end up with no rows, and reading them is not an error."""
    # ARRANGE
    column = np.array([], dtype=np.int32)

    # ACT
    with caplog.at_level(logging.WARNING):
        meanings = decode_positronium_column("gammaType", column)

    # ASSERT
    assert list(meanings) == []
    assert caplog.text == ""


def test_reading_a_column_of_a_branch_without_a_class_is_refused() -> None:
    """The refusal is the same whether one value is asked about, or a column."""
    # ARRANGE
    column = np.array([1, 2], dtype=np.int32)

    # ACT / ASSERT
    with pytest.raises(ValueError, match="does not hold values"):
        decode_positronium_column(DECAY_INDEX_BRANCH, column)


def test_a_column_given_as_a_list_is_read_row_by_row() -> None:
    """A column is a column, however the caller happens to hold it."""
    # ARRANGE
    values = [-1, 0, 1]

    # ACT
    from_positronium = is_positronium_source(values)

    # ASSERT
    assert list(from_positronium) == [False, True, True]


@pytest.mark.parametrize(
    ("column", "label"),
    [
        (np.zeros((3, 2), dtype=np.int32), "two-dimensional"),
        (np.array(["a", "b"]), "text"),
        (np.array([1.5, 2.5]), "floating point"),
    ],
    ids=["two-dimensional", "text", "floating-point"],
)
def test_asking_about_something_that_is_not_a_column_is_refused(
    column: np.ndarray,
    label: str,
) -> None:
    """A wrong argument must not answer with a mask that selects everything.

    Comparing a value that is not a column of whole numbers answers with a
    single truth value, and a selection made with it keeps every row while
    changing its shape - the quiet kind of wrong this package is built to
    avoid.
    """
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="one-dimensional column of whole numbers"):
        is_positronium_source(column)


def test_reading_a_column_of_another_type_is_refused() -> None:
    """Truncating a floating point column would read one thing as another."""
    # ARRANGE
    # Passing the wrong branch is how a column like this arrives here.
    column = np.array([2.9, 0.4])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="one-dimensional column of whole numbers"):
        decode_positronium_column("gammaType", column)


def test_reading_a_two_dimensional_column_is_refused() -> None:
    """The tree holds such a column - volumeID - so this is not hypothetical."""
    # ARRANGE
    column = np.zeros((3, 10), dtype=np.int32)

    # ACT / ASSERT
    with pytest.raises(ValueError, match="got a 2-dimensional column"):
        decode_positronium_column("sourceType", column)


def test_the_values_no_scene_holds_are_read_all_the_same() -> None:
    """A single gamma emitter and a single gamma come from a model no fixture uses.

    Their meaning comes from the GATE header rather than from a file, so the
    only thing that can exercise them is an array built here.
    """
    # ARRANGE
    column = np.array([1], dtype=np.int32)

    # ACT
    source = decode_positronium_column("sourceType", column)
    gamma = decode_positronium_column("gammaType", column)

    # ASSERT
    assert list(source) == [SourceType.SINGLE_GAMMA_EMITTER]
    assert list(gamma) == [GammaType.SINGLE]
    assert decode_positronium_value("sourceType", 1) is SourceType.SINGLE_GAMMA_EMITTER
    assert decode_positronium_value("gammaType", 1) is GammaType.SINGLE
