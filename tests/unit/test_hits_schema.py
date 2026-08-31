"""Unit tests for the branch schemas of the "Hits" tree variants.

The schemas describe files a simulation produced, so most of the tests here
compare them against the variant fixtures rather than against a list retyped
from the same source the schemas were written from.
"""

import pytest
import uproot
from conftest import HITS_VARIANT_LAYOUTS, HitsVariantLayout, branch_type_name

from opengate_gate_tree.tree.hits.schema import (
    SYSTEM_ID_PLACEHOLDER,
    VARIANT_BRANCHES,
    BranchKind,
    branch_spec,
    expected_branches,
    supported_variants,
    uses_system,
    variant_reference,
)
from opengate_gate_tree.tree.hits.variant import (
    SYSTEM_ALIASES,
    SYSTEM_ID_BRANCHES,
    GateSystemType,
    HitsTreeVariant,
    find_system_type,
    system_id_depth,
)

# Structure each fixture holds, as read off the simulation that wrote it.
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

# Number of branches every variant holds, counted on the fixtures.
BRANCH_COUNTS: dict[HitsTreeVariant, int] = {
    HitsTreeVariant.NO_SYSTEM: 40,
    HitsTreeVariant.SYSTEM: 46,
    HitsTreeVariant.SYSTEM_SEPTAL: 47,
    HitsTreeVariant.NO_SYSTEM_CC: 30,
    HitsTreeVariant.SYSTEM_CC: 36,
    HitsTreeVariant.TREE_COMMON: 54,
}


def schema_of(layout: HitsVariantLayout) -> tuple[str, ...]:
    """Return the branch names the schema of a fixture's structure holds."""
    variant, system = FIXTURE_STRUCTURES[layout.key]
    return tuple(spec.name for spec in expected_branches(variant, system))


@pytest.mark.parametrize("variant", supported_variants(), ids=lambda variant: variant.name)
def test_schema_holds_the_measured_number_of_branches(variant: HitsTreeVariant) -> None:
    """Each schema should be as long as the tree it describes."""
    # ARRANGE
    system = GateSystemType.CYLINDRICAL_PET if uses_system(variant) else None

    # ACT
    branches = expected_branches(variant, system)

    # ASSERT
    assert len(branches) == BRANCH_COUNTS[variant]


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_schema_matches_the_fixture_branch_names(layout: HitsVariantLayout) -> None:
    """Schemas should name the branches of a real file, in the same order."""
    # ARRANGE
    expected = schema_of(layout)

    # ACT
    with uproot.open(layout.path) as fixture:
        names = {
            tree_name: tuple(str(name) for name in fixture[tree_name].keys())
            for tree_name in layout.tree_names
        }

    # ASSERT
    assert names == dict.fromkeys(layout.tree_names, expected)


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_schema_matches_the_fixture_branch_types(layout: HitsVariantLayout) -> None:
    """Schemas should give each branch the type the file stores it with."""
    # ARRANGE
    variant, system = FIXTURE_STRUCTURES[layout.key]
    expected = {spec.name: spec.dtype for spec in expected_branches(variant, system)}

    # ACT
    with uproot.open(layout.path) as fixture:
        tree = fixture[layout.tree_names[0]]
        types = {name: branch_type_name(branch.interpretation) for name, branch in tree.items()}

    # ASSERT
    assert types == expected


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_fixture_system_scheme_is_recognised(layout: HitsVariantLayout) -> None:
    """The naming scheme should be read off the branch names of a real file."""
    # ARRANGE
    _, expected = FIXTURE_STRUCTURES[layout.key]

    # ACT
    with uproot.open(layout.path) as fixture:
        names = [str(name) for name in fixture[layout.tree_names[0]].keys()]
    system = find_system_type(names)

    # ASSERT
    assert system == expected


def test_identifier_block_follows_the_process_name() -> None:
    """The classic output writes the identifiers right after processName."""
    # ARRANGE
    # GATE inserts the block into an otherwise unchanged branch list.

    # ACT
    names = [spec.name for spec in expected_branches(HitsTreeVariant.SYSTEM, GateSystemType.ECAT)]
    position = names.index("processName")

    # ASSERT
    assert tuple(names[position + 1 : position + 7]) == SYSTEM_ID_BRANCHES[GateSystemType.ECAT]


def test_septal_counter_precedes_the_decay_information() -> None:
    """Septal penetration counting inserts one branch before sourceType."""
    # ARRANGE
    # The A3 layout differs from A2 by this branch alone.

    # ACT
    names = [
        spec.name
        for spec in expected_branches(HitsTreeVariant.SYSTEM_SEPTAL, GateSystemType.SPECT_HEAD)
    ]

    # ASSERT
    assert names[names.index("septalNb") + 1] == "sourceType"


def test_tree_common_schema_follows_the_depth_of_the_system() -> None:
    """The GateToTree output writes as many identifiers as the system is deep."""
    # ARRANGE
    # A system of depth four leaves out the two deepest identifier branches.

    # ACT
    branches = expected_branches(
        HitsTreeVariant.TREE_COMMON,
        GateSystemType.CYLINDRICAL_PET,
        system_depth=4,
    )
    names = [spec.name for spec in branches]

    # ASSERT
    assert len(branches) == BRANCH_COUNTS[HitsTreeVariant.TREE_COMMON] - 2
    assert "crystalID" not in names
    assert names[names.index("nCrystalRayleigh") + 1] == "gantryID"


def test_schema_never_holds_the_placeholder() -> None:
    """The placeholder is an internal marker, not a branch of any tree."""
    # ARRANGE
    # Every variant is completed with a scheme where it needs one.

    # ACT
    names = {
        spec.name
        for variant in supported_variants()
        for spec in expected_branches(
            variant, GateSystemType.CPET if uses_system(variant) else None
        )
    }

    # ASSERT
    assert SYSTEM_ID_PLACEHOLDER not in names


@pytest.mark.parametrize("variant", supported_variants(), ids=lambda variant: variant.name)
def test_schema_names_no_branch_twice(variant: HitsTreeVariant) -> None:
    """A tree cannot hold two branches of the same name."""
    # ARRANGE
    system = GateSystemType.SCANNER if uses_system(variant) else None

    # ACT
    names = [spec.name for spec in expected_branches(variant, system)]

    # ASSERT
    assert len(names) == len(set(names))


def test_missing_naming_scheme_is_refused() -> None:
    """A variant that uses a system cannot be described without a scheme."""
    # ARRANGE
    # Identifier names depend on the system, so there is nothing to fall back on.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="naming scheme"):
        expected_branches(HitsTreeVariant.SYSTEM)


def test_naming_scheme_for_a_variant_without_a_system_is_refused() -> None:
    """A scheme given for a system-less variant points at a mistake."""
    # ARRANGE
    # Silently ignoring it would hide a wrongly detected variant.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="does not use a system"):
        expected_branches(HitsTreeVariant.NO_SYSTEM, GateSystemType.CYLINDRICAL_PET)


def test_depth_for_a_variant_without_a_system_is_refused() -> None:
    """A depth given for a system-less variant points at a mistake too."""
    # ARRANGE
    # No identifier block exists to be cut short.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="no depth applies"):
        expected_branches(HitsTreeVariant.NO_SYSTEM, system_depth=3)


def test_depth_for_the_classic_output_is_refused() -> None:
    """The classic output always writes six identifiers, whatever the system."""
    # ARRANGE
    # GateHitTree fills the unused levels instead of leaving them out.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="always holds 6"):
        expected_branches(HitsTreeVariant.SYSTEM, GateSystemType.CYLINDRICAL_PET, system_depth=4)


@pytest.mark.parametrize("depth", [0, 7, -1])
def test_depth_outside_the_system_is_refused(depth: int) -> None:
    """A system cannot be shallower than one level or deeper than the scheme."""
    # ARRANGE
    # The block is a prefix of the scheme, so its length is bounded by it.

    # ACT / ASSERT
    with pytest.raises(ValueError, match="System depth"):
        expected_branches(
            HitsTreeVariant.TREE_COMMON,
            GateSystemType.CYLINDRICAL_PET,
            system_depth=depth,
        )


def test_branch_spec_types_follow_the_gate_naming_rule() -> None:
    """Types are decided by the branch name, the same way GATE decides them."""
    # ARRANGE
    # volumeID is an array only in the classic layout, not once it is split.

    # ACT
    specs = {
        name: branch_spec(name)
        for name in ("eventID", "edep", "time", "processName", "volumeID", "volumeID[0]")
    }

    # ASSERT
    assert (specs["eventID"].kind, specs["eventID"].dtype) == (BranchKind.INTEGER, "int32")
    assert (specs["edep"].kind, specs["edep"].dtype) == (BranchKind.FLOAT, "float32")
    assert (specs["time"].kind, specs["time"].dtype) == (BranchKind.FLOAT, "float64")
    assert (specs["processName"].kind, specs["processName"].dtype) == (BranchKind.TEXT, "text")
    assert (specs["volumeID"].kind, specs["volumeID"].dtype) == (
        BranchKind.INTEGER_ARRAY,
        "int32[10]",
    )
    assert (specs["volumeID[0]"].kind, specs["volumeID[0]"].dtype) == (BranchKind.INTEGER, "int32")


def test_every_scheme_names_six_distinct_levels() -> None:
    """GATE fills six identifier levels, and a scheme must name each of them."""
    # ARRANGE
    # Unused levels carry placeholder names such as "unused4ID".

    # ACT
    sizes = {system: (len(names), len(set(names))) for system, names in SYSTEM_ID_BRANCHES.items()}

    # ASSERT
    assert sizes == dict.fromkeys(GateSystemType, (6, 6))


def test_schemes_can_be_told_apart() -> None:
    """Two schemes naming their levels alike could never be distinguished."""
    # ARRANGE
    # Detection relies on the names alone, with no other clue in the file.

    # ACT
    named = set(SYSTEM_ID_BRANCHES.values())

    # ASSERT
    assert len(named) == len(SYSTEM_ID_BRANCHES)


def test_shared_first_level_leaves_the_scheme_undecided() -> None:
    """Several schemes start with gantryID, so it identifies none of them."""
    # ARRANGE
    branches = ["eventID", "edep", "gantryID"]

    # ACT
    system = find_system_type(branches)

    # ASSERT
    assert system is None


def test_tree_without_identifiers_has_no_scheme() -> None:
    """A simulation without a system writes no identifier branch at all."""
    # ARRANGE
    branches = ["eventID", "edep", "processName"]

    # ACT
    system = find_system_type(branches)

    # ASSERT
    assert system is None


def test_partial_scheme_is_measured_by_its_depth() -> None:
    """The GateToTree output stops at the depth of the system."""
    # ARRANGE
    branches = ["gantryID", "rsectorID", "moduleID", "edep"]

    # ACT
    depth = system_id_depth(GateSystemType.CYLINDRICAL_PET, branches)
    system = find_system_type(branches)

    # ASSERT
    assert depth == 3
    assert system is GateSystemType.CYLINDRICAL_PET


def test_gap_in_the_scheme_stops_the_count() -> None:
    """Levels are written top down, so a gap means another scheme."""
    # ARRANGE
    branches = ["gantryID", "moduleID", "submoduleID"]

    # ACT
    depth = system_id_depth(GateSystemType.CYLINDRICAL_PET, branches)

    # ASSERT
    assert depth == 1


def test_every_variant_carries_a_reference_label() -> None:
    """The labels tie the schemas back to the material they were read from."""
    # ARRANGE
    # No additional setup required.

    # ACT
    labels = [variant_reference(variant) for variant in supported_variants()]

    # ASSERT
    assert labels == ["A1", "A2", "A3", "A4", "A5", "B1"]


def test_supported_variants_covers_the_whole_enum() -> None:
    """Every declared variant needs a schema, or detection could return one."""
    # ARRANGE
    # No additional setup required.

    # ACT
    covered = set(supported_variants())

    # ASSERT
    assert covered == set(HitsTreeVariant)
    assert covered == set(VARIANT_BRANCHES)


def test_system_aliases_cover_every_scheme() -> None:
    """Descriptions name the systems behind a scheme, so none may be missing."""
    # ARRANGE
    # cylindricalPET and OPET, for one, are indistinguishable by branch names.

    # ACT
    described = set(SYSTEM_ALIASES)

    # ASSERT
    assert described == set(GateSystemType)
    assert all(SYSTEM_ALIASES[system] for system in GateSystemType)
