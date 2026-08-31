"""Unit tests for the "Hits" tree variant fixtures.

The tests pin down what each fixture holds. A fixture regenerated with other
parameters, or cut from another simulation, changes the ground truth every
later test relies on, and these tests are what makes that visible.
"""

from itertools import combinations
from pathlib import Path

import pytest
import uproot
from conftest import (
    HITS_VARIANT_LAYOUTS,
    HitsVariantLayout,
    branch_type_name,
    expected_hits_branch_type,
)

# Variants stored under a single tree, used where one tree per file is needed.
SINGLE_TREE_LAYOUTS = tuple(
    layout for layout in HITS_VARIANT_LAYOUTS if len(layout.tree_names) == 1
)


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_fixture_holds_only_trees(layout: HitsVariantLayout) -> None:
    """Variant fixtures should carry the trees alone, without histograms."""
    # ARRANGE
    # GATE stores TH1D histograms next to its trees; they are not copied here.

    # ACT
    with uproot.open(layout.path) as fixture:
        class_names = set(fixture.classnames().values())

    # ASSERT
    assert class_names == {"TTree"}


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_fixture_matches_declared_trees(layout: HitsVariantLayout) -> None:
    """Tree names and their order should be the ones the layout declares."""
    # ARRANGE
    # The B1 fixture stores its tree under the name "tree", not "Hits".

    # ACT
    with uproot.open(layout.path) as fixture:
        names = tuple(str(key).split(";")[0] for key in fixture.classnames())

    # ASSERT
    assert names == layout.tree_names


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_fixture_matches_declared_size(layout: HitsVariantLayout) -> None:
    """Every tree should hold the declared number of entries and branches."""
    # ARRANGE
    # All files but B1 were cut to 500 entries.

    # ACT
    with uproot.open(layout.path) as fixture:
        sizes = {
            name: (fixture[name].num_entries, len(fixture[name].keys()))
            for name in layout.tree_names
        }

    # ASSERT
    assert sizes == dict.fromkeys(layout.tree_names, (layout.entries, layout.branch_count))


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_fixture_follows_gate_branch_types(layout: HitsVariantLayout) -> None:
    """Branch types should follow the rule measured on the GATE output."""
    # ARRANGE
    # In the GateToTree layout, volumeID is split into ten scalar branches.

    # ACT
    with uproot.open(layout.path) as fixture:
        types = {
            name: branch_type_name(branch.interpretation)
            for tree_name in layout.tree_names
            for name, branch in fixture[tree_name].items()
        }

    # ASSERT
    expected = {
        name: "int32" if name.startswith("volumeID[") else expected_hits_branch_type(name)
        for name in types
    }
    assert types == expected


@pytest.mark.parametrize(
    "layout",
    [layout for layout in HITS_VARIANT_LAYOUTS if len(layout.tree_names) > 1],
    ids=lambda layout: layout.key,
)
def test_multi_tree_fixture_repeats_one_branch_layout(layout: HitsVariantLayout) -> None:
    """Trees split per run or per detector should share one branch layout."""
    # ARRANGE
    # Merging these trees is only meaningful while their branches agree.

    # ACT
    with uproot.open(layout.path) as fixture:
        layouts = {
            tuple(str(name) for name in fixture[tree_name].keys())
            for tree_name in layout.tree_names
        }

    # ASSERT
    assert len(layouts) == 1


def test_variant_fixtures_differ_from_each_other() -> None:
    """Each variant fixture should describe a structure of its own."""
    # ARRANGE
    branch_sets: dict[str, frozenset[str]] = {}
    for layout in SINGLE_TREE_LAYOUTS:
        with uproot.open(layout.path) as fixture:
            branch_sets[layout.key] = frozenset(
                str(name) for name in fixture[layout.tree_names[0]].keys()
            )

    # ACT
    duplicates = [
        (first, second)
        for first, second in combinations(sorted(branch_sets), 2)
        if branch_sets[first] == branch_sets[second]
    ]

    # ASSERT
    assert duplicates == []


def test_variant_fixture_directory_holds_only_known_files() -> None:
    """The fixture directory should hold the declared files and the README."""
    # ARRANGE
    directory = HITS_VARIANT_LAYOUTS[0].path.parent
    expected = {layout.file_name for layout in HITS_VARIANT_LAYOUTS} | {"README.md"}

    # ACT
    present = {entry.name for entry in Path(directory).iterdir()}

    # ASSERT
    assert present == expected
