"""Unit tests for checking a "Hits" tree against its recognised structure."""

import logging
from collections.abc import Sequence

import pytest
import uproot
from conftest import GATE_HITS_FIXTURE, HITS_VARIANT_LAYOUTS, HitsVariantLayout, branch_type_name

from opengate_gate_tree.errors import HitsTreeValidationError, UnknownHitsVariantError
from opengate_gate_tree.tree.hits.detection import detect_hits_variant
from opengate_gate_tree.tree.hits.schema import expected_branches
from opengate_gate_tree.tree.hits.validation import validate_hits_tree
from opengate_gate_tree.tree.hits.variant import GateSystemType, HitsTreeVariant


def tree_layout(path: object, tree_name: str) -> tuple[list[str], dict[str, str]]:
    """Return the branch names and the types of a tree stored in a file."""
    with uproot.open(path) as opened:
        tree = opened[tree_name]
        names = [str(name) for name in tree.keys()]
        dtypes = {name: branch_type_name(tree[name].interpretation) for name in names}
    return names, dtypes


def schema_layout(
    variant: HitsTreeVariant,
    system: GateSystemType | None = None,
    system_depth: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return the branch names and types a structure describes."""
    specs = expected_branches(variant, system, system_depth)
    return [spec.name for spec in specs], {spec.name: spec.dtype for spec in specs}


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_reference_files_match_their_structure(layout: HitsVariantLayout) -> None:
    """Every file the schemas were written from should validate cleanly."""
    # ARRANGE
    names, dtypes = tree_layout(layout.path, layout.tree_names[0])

    # ACT
    detection = validate_hits_tree(names, dtypes)

    # ASSERT
    assert detection.branch_count == layout.branch_count


def test_the_original_gate_fixture_matches_its_structure() -> None:
    """The file the package was built against must keep validating."""
    # ARRANGE
    names, dtypes = tree_layout(GATE_HITS_FIXTURE, "Hits")

    # ACT
    detection = validate_hits_tree(names, dtypes)

    # ASSERT
    assert detection.variant is HitsTreeVariant.NO_SYSTEM


def test_missing_branch_is_refused() -> None:
    """A branch the structure describes but the tree lacks is an error."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.NO_SYSTEM)
    without_energy = [name for name in names if name != "edep"]

    # ACT
    with pytest.raises(HitsTreeValidationError) as raised:
        validate_hits_tree(without_energy, dtypes)

    # ASSERT
    assert "Branches missing from the tree: ['edep']" in str(raised.value)


def test_branch_of_another_type_is_refused() -> None:
    """A branch stored with another type means the data is not what it seems."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.NO_SYSTEM)
    dtypes["edep"] = "text"

    # ACT
    with pytest.raises(HitsTreeValidationError) as raised:
        validate_hits_tree(names, dtypes)

    # ASSERT
    assert "'edep' expected float32, stored as text" in str(raised.value)


def test_branch_of_an_unsupported_type_is_refused() -> None:
    """A type the package cannot represent is reported like any other."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.NO_SYSTEM)
    dtypes["volumeID"] = "unsupported"

    # ACT
    with pytest.raises(HitsTreeValidationError) as raised:
        validate_hits_tree(names, dtypes)

    # ASSERT
    assert "'volumeID' expected int32[10], stored as unsupported" in str(raised.value)


def test_branch_of_unknown_type_is_refused() -> None:
    """A branch nothing is known about cannot be taken as matching."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.NO_SYSTEM)
    del dtypes["eventID"]

    # ACT
    with pytest.raises(HitsTreeValidationError) as raised:
        validate_hits_tree(names, dtypes)

    # ASSERT
    assert "'eventID' expected int32, stored as None" in str(raised.value)


def test_failure_names_the_structure_it_was_checked_against() -> None:
    """The user has to see what the tree was taken for to judge the report."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.SYSTEM, GateSystemType.CYLINDRICAL_PET)
    without_gantry = [name for name in names if name != "gantryID"]
    detection = detect_hits_variant(names, "Hits")

    # ACT
    with pytest.raises(HitsTreeValidationError) as raised:
        validate_hits_tree(without_gantry, dtypes, detection)

    # ASSERT
    assert "Tree 'Hits' was recognised as the 'System' structure (A2)" in str(raised.value)


def test_extra_branch_is_reported_but_accepted(caplog: pytest.LogCaptureFixture) -> None:
    """GATE builds add branches, and that must not stop the package."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.NO_SYSTEM)
    names.append("multiPhotonFlag")
    dtypes["multiPhotonFlag"] = "int32"

    # ACT
    with caplog.at_level(logging.WARNING):
        detection = validate_hits_tree(names, dtypes)

    # ASSERT
    assert detection.variant is HitsTreeVariant.NO_SYSTEM
    assert "multiPhotonFlag" in caplog.text


def test_a_matching_tree_is_reported_without_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file that matches its structure should produce no noise."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.NO_SYSTEM)

    # ACT
    with caplog.at_level(logging.WARNING):
        validate_hits_tree(names, dtypes)

    # ASSERT
    assert caplog.text == ""


def test_shallow_identifier_block_matches_the_gate_to_tree_output() -> None:
    """The GateToTree output writes one identifier per level of the system."""
    # ARRANGE
    names, dtypes = schema_layout(
        HitsTreeVariant.TREE_COMMON, GateSystemType.CYLINDRICAL_PET, system_depth=4
    )

    # ACT
    detection = validate_hits_tree(names, dtypes)

    # ASSERT
    assert detection.variant is HitsTreeVariant.TREE_COMMON
    assert detection.system_depth == 4


def test_structure_is_recognised_when_it_is_not_given() -> None:
    """Validation can be asked about a tree nothing is known about yet."""
    # ARRANGE
    names, dtypes = schema_layout(HitsTreeVariant.SYSTEM_CC, GateSystemType.SPECT_HEAD)

    # ACT
    detection = validate_hits_tree(names, dtypes)

    # ASSERT
    assert detection.variant is HitsTreeVariant.SYSTEM_CC
    assert detection.system is GateSystemType.SPECT_HEAD


def test_unsupported_structure_is_refused_before_it_is_checked() -> None:
    """There is nothing to check a tree against when its structure is unknown."""
    # ARRANGE
    names: Sequence[str] = ["runID", "eventID", "sinogramTheta"]

    # ACT / ASSERT
    with pytest.raises(UnknownHitsVariantError):
        validate_hits_tree(names, dict.fromkeys(names, "int32"))
