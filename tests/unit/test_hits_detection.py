"""Unit tests for recognising the structure of a "Hits" tree.

Recognition decides what every later step works with, so the tests run it
against the branch lists of real simulation output, and state what happens for
the structures the package knows about but does not support.
"""

import pytest
import uproot
from conftest import GATE_HITS_FIXTURE, HITS_VARIANT_LAYOUTS, HitsVariantLayout

from opengate_gate_tree.errors import UnknownHitsVariantError
from opengate_gate_tree.tree.hits.detection import (
    describe_hits_tree,
    detect_hits_variant,
    find_hits_variant,
)
from opengate_gate_tree.tree.hits.schema import expected_branches
from opengate_gate_tree.tree.hits.variant import GateSystemType, HitsTreeVariant

# Structure and identifier naming scheme each fixture holds.
FIXTURE_STRUCTURES: dict[str, tuple[HitsTreeVariant, GateSystemType | None]] = {
    "a1": (HitsTreeVariant.NO_SYSTEM, None),
    "a2": (HitsTreeVariant.SYSTEM, GateSystemType.CYLINDRICAL_PET),
    "a3": (HitsTreeVariant.SYSTEM_SEPTAL, GateSystemType.SPECT_HEAD),
    "a4": (HitsTreeVariant.NO_SYSTEM_CC, None),
    "a5": (HitsTreeVariant.SYSTEM_CC, GateSystemType.CYLINDRICAL_PET),
    "b1": (HitsTreeVariant.TREE_COMMON, GateSystemType.CYLINDRICAL_PET),
    "multi-run": (HitsTreeVariant.NO_SYSTEM, None),
    "multi-sd": (HitsTreeVariant.NO_SYSTEM, None),
}


def branch_names_of(path: object, tree_name: str) -> list[str]:
    """Return the branch names of a tree stored in a file."""
    with uproot.open(path) as opened:
        return [str(name) for name in opened[tree_name].keys()]


def names_of(variant: HitsTreeVariant, system: GateSystemType | None) -> list[str]:
    """Return the branch names of a schema, as a tree would hold them."""
    return [spec.name for spec in expected_branches(variant, system)]


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_fixture_structure_is_recognised(layout: HitsVariantLayout) -> None:
    """Every reference file should be recognised as the structure it has."""
    # ARRANGE
    expected_variant, expected_system = FIXTURE_STRUCTURES[layout.key]
    tree_name = layout.tree_names[0]
    names = branch_names_of(layout.path, tree_name)

    # ACT
    detection = detect_hits_variant(names, tree_name)

    # ASSERT
    assert detection.variant is expected_variant
    assert detection.system is expected_system
    assert detection.tree_name == tree_name
    assert detection.branch_count == layout.branch_count


@pytest.mark.parametrize(
    "layout",
    [layout for layout in HITS_VARIANT_LAYOUTS if len(layout.tree_names) > 1],
    ids=lambda layout: layout.key,
)
def test_every_tree_of_a_split_file_is_recognised(layout: HitsVariantLayout) -> None:
    """Trees split per run or per detector each hold the whole structure."""
    # ARRANGE
    expected_variant, _ = FIXTURE_STRUCTURES[layout.key]

    # ACT
    variants = {
        tree_name: detect_hits_variant(branch_names_of(layout.path, tree_name)).variant
        for tree_name in layout.tree_names
    }

    # ASSERT
    assert variants == dict.fromkeys(layout.tree_names, expected_variant)


def test_the_original_gate_fixture_is_recognised() -> None:
    """The file the package was built against must keep working."""
    # ARRANGE
    names = branch_names_of(GATE_HITS_FIXTURE, "Hits")

    # ACT
    detection = detect_hits_variant(names, "Hits")

    # ASSERT
    assert detection.variant is HitsTreeVariant.NO_SYSTEM
    assert detection.system is None


def test_split_volume_id_marks_the_gate_to_tree_output() -> None:
    """The split volumeID is what tells the GateToTree output apart."""
    # ARRANGE
    names = names_of(HitsTreeVariant.TREE_COMMON, GateSystemType.CYLINDRICAL_PET)

    # ACT
    detection = detect_hits_variant(names)

    # ASSERT
    assert detection.variant is HitsTreeVariant.TREE_COMMON
    assert detection.system_depth == 6


def test_shallow_system_is_reported_with_its_depth() -> None:
    """The GateToTree output writes one identifier per level of the system."""
    # ARRANGE
    names = [
        spec.name
        for spec in expected_branches(
            HitsTreeVariant.TREE_COMMON, GateSystemType.CYLINDRICAL_PET, system_depth=3
        )
    ]

    # ACT
    detection = detect_hits_variant(names)

    # ASSERT
    assert detection.variant is HitsTreeVariant.TREE_COMMON
    assert detection.system_depth == 3


def test_compton_camera_layout_is_named_in_the_refusal() -> None:
    """A structure that is known but unsupported should be named as such."""
    # ARRANGE
    # The layer name is written by the Compton camera output and by nothing else.
    names = [*names_of(HitsTreeVariant.NO_SYSTEM_CC, None), "layerName"]

    # ACT / ASSERT
    with pytest.raises(UnknownHitsVariantError, match="Compton camera hits layout"):
        detect_hits_variant(names, "Hits")


def test_layer_name_outweighs_the_classic_volume_id() -> None:
    """The Compton camera output also holds volumeID, and must not pass as A4."""
    # ARRANGE
    names = [*names_of(HitsTreeVariant.NO_SYSTEM_CC, None), "layerName"]

    # ACT / ASSERT
    assert "volumeID" in names
    with pytest.raises(UnknownHitsVariantError, match="Compton camera"):
        detect_hits_variant(names)


def test_per_collection_output_with_camera_columns_is_named_in_the_refusal() -> None:
    """The per-collection GateToTree output has no reference file to confirm it."""
    # ARRANGE
    names = [
        *names_of(HitsTreeVariant.TREE_COMMON, GateSystemType.CYLINDRICAL_PET),
        "sourceEnergy",
        "sourcePDG",
        "nCrystalConv",
        "postStepProcess",
    ]

    # ACT / ASSERT
    with pytest.raises(UnknownHitsVariantError, match="per-collection GateToTree output"):
        detect_hits_variant(names, "Hits")


def test_refusal_lists_the_supported_structures() -> None:
    """Being told what is supported is the point of naming the structure."""
    # ARRANGE
    names = [*names_of(HitsTreeVariant.NO_SYSTEM_CC, None), "layerName"]

    # ACT
    with pytest.raises(UnknownHitsVariantError) as raised:
        detect_hits_variant(names)

    # ASSERT
    assert "A1 (No system)" in str(raised.value)
    assert "B1 (GateToTree common output)" in str(raised.value)


def test_tree_without_a_hierarchy_branch_is_refused() -> None:
    """Neither volumeID nor its split form means this is not a Hits tree."""
    # ARRANGE
    names = ["runID", "eventID", "time1", "energy1", "sinogramTheta"]

    # ACT
    with pytest.raises(UnknownHitsVariantError) as raised:
        detect_hits_variant(names, "Coincidences")

    # ASSERT
    assert "Coincidences" in str(raised.value)
    assert "volumeID" in str(raised.value)


def test_refusal_reports_the_branches_it_saw() -> None:
    """The message has to let the user see what the file actually holds."""
    # ARRANGE
    names = ["total_nb_primaries", "latest_event_ID", "start_time_sec", "stop_time_sec"]

    # ACT
    with pytest.raises(UnknownHitsVariantError) as raised:
        detect_hits_variant(names, "pet_data")

    # ASSERT
    assert "start_time_sec" in str(raised.value)


def test_long_branch_lists_are_summarised_in_the_refusal() -> None:
    """A message naming forty branches would hide what it is trying to say."""
    # ARRANGE
    names = [f"branch{index}" for index in range(40)]

    # ACT
    with pytest.raises(UnknownHitsVariantError) as raised:
        detect_hits_variant(names)

    # ASSERT
    assert "28 more" in str(raised.value)


def test_empty_tree_layout_is_refused() -> None:
    """A tree without branches cannot be any structure."""
    # ARRANGE
    names: list[str] = []

    # ACT
    with pytest.raises(UnknownHitsVariantError) as raised:
        detect_hits_variant(names)

    # ASSERT
    assert "no branches" in str(raised.value)


def test_undecided_naming_scheme_is_refused() -> None:
    """A structure using a system is only usable once its scheme is known."""
    # ARRANGE
    # gantryID starts four schemes, so on its own it decides none of them.
    names = [*names_of(HitsTreeVariant.NO_SYSTEM, None), "gantryID"]

    # ACT
    with pytest.raises(UnknownHitsVariantError) as raised:
        detect_hits_variant(names, "Hits")

    # ASSERT
    assert "naming scheme could not be told apart" in str(raised.value)


def test_partial_identifier_block_is_not_read_as_a_system_less_tree() -> None:
    """Identifier branches mean a system was used, whichever scheme they follow."""
    # ARRANGE
    names = [*names_of(HitsTreeVariant.NO_SYSTEM, None), "gantryID"]

    # ACT / ASSERT
    # Reading this as A1 would answer a question that stayed open.
    with pytest.raises(UnknownHitsVariantError, match="carries system identifier branches"):
        detect_hits_variant(names)


def test_septal_counter_without_a_system_is_refused() -> None:
    """Septal penetration counting only ever appears alongside a system."""
    # ARRANGE
    names = [*names_of(HitsTreeVariant.NO_SYSTEM, None), "septalNb"]

    # ACT / ASSERT
    with pytest.raises(UnknownHitsVariantError, match="no system identifier branches"):
        detect_hits_variant(names)


def test_finding_a_variant_answers_none_where_detection_refuses() -> None:
    """Looking for a hits tree must not raise on trees that are not one."""
    # ARRANGE
    refused = [
        [],
        ["runID", "eventID", "sinogramTheta"],
        [*names_of(HitsTreeVariant.NO_SYSTEM_CC, None), "layerName"],
    ]

    # ACT
    results = [find_hits_variant(names) for names in refused]

    # ASSERT
    assert results == [None, None, None]


def test_finding_a_variant_answers_for_a_supported_tree() -> None:
    """A supported structure is reported the same way detection reports it."""
    # ARRANGE
    names = names_of(HitsTreeVariant.SYSTEM, GateSystemType.SPECT_HEAD)

    # ACT
    detection = find_hits_variant(names)

    # ASSERT
    assert detection is not None
    assert detection.variant is HitsTreeVariant.SYSTEM
    assert detection.system is GateSystemType.SPECT_HEAD


def test_description_states_the_structure_and_the_branches() -> None:
    """The description is what the command line shows about a loaded tree."""
    # ARRANGE
    names = branch_names_of(HITS_VARIANT_LAYOUTS[1].path, "Hits")
    detection = detect_hits_variant(names, "Hits")

    # ACT
    description = describe_hits_tree(detection, entry_count=500)

    # ASSERT
    assert "Detected Hits tree variant: System (A2)" in description
    assert "Tree name in the file: Hits" in description
    assert "System identifier scheme: cylindricalPET / OPET" in description
    assert "Entries: 500" in description
    assert "Branches (46):" in description
    assert "  - volumeID / int32[10]" in description
    assert "  - processName / text" in description


def test_description_falls_back_to_the_schema_types() -> None:
    """A variant can be described without a file being open."""
    # ARRANGE
    names = names_of(HitsTreeVariant.NO_SYSTEM, None)
    detection = detect_hits_variant(names)

    # ACT
    description = describe_hits_tree(detection)

    # ASSERT
    assert "  - trackLocalTime / float64" in description
    assert "Tree name in the file:" not in description
    assert "Entries:" not in description


def test_description_reports_the_types_the_file_was_read_with() -> None:
    """Given the types of a file, the description states those, not the schema."""
    # ARRANGE
    names = names_of(HitsTreeVariant.NO_SYSTEM, None)
    detection = detect_hits_variant(names)
    dtypes = {"eventID": "int64", "edep": "float32"}

    # ACT
    description = describe_hits_tree(detection, dtypes=dtypes)

    # ASSERT
    assert "Branches (2):" in description
    assert "  - eventID / int64" in description


def test_description_names_the_system_of_the_gate_to_tree_output() -> None:
    """A shallow identifier block must not be described as the full scheme."""
    # ARRANGE
    names = [
        spec.name
        for spec in expected_branches(
            HitsTreeVariant.TREE_COMMON, GateSystemType.CYLINDRICAL_PET, system_depth=2
        )
    ]
    detection = detect_hits_variant(names, "tree")

    # ACT
    description = describe_hits_tree(detection)

    # ASSERT
    assert "Branches (50):" in description
    assert "  - rsectorID / int32" in description
    assert "moduleID" not in description
