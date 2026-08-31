"""Unit tests for the PositroniumSource fixtures.

The tests pin down which values each file holds. Those values are the ground
truth the enum representations are tested against, so a file regenerated from
another simulation, or cut differently, has to be visible here rather than in
a puzzling failure somewhere else.
"""

from pathlib import Path

import pytest
import uproot
from conftest import POSITRONIUM_BRANCH_FIELDS, POSITRONIUM_LAYOUTS, PositroniumLayout

from opengate_gate_tree.io.rootfile import RootFile
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.variant import HitsTreeVariant

# File whose branch list every fixture is expected to repeat.
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
A1_FIXTURE = FIXTURES_DIR / "hits-variants" / "a1-no-system.root"

# Number of entries every file was cut to.
ENTRIES = 500


def distinct_values(path: Path, branch: str) -> tuple[int, ...]:
    """Return the distinct values a branch holds, sorted."""
    with uproot.open(path) as fixture:
        return tuple(sorted(set(fixture["Hits"][branch].array(library="np").tolist())))


@pytest.mark.parametrize("layout", POSITRONIUM_LAYOUTS, ids=lambda layout: layout.key)
def test_fixture_holds_the_hits_tree_alone(layout: PositroniumLayout) -> None:
    """The fixtures carry the hits, and the trees beside them are covered elsewhere."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with uproot.open(layout.path) as fixture:
        class_names = {str(key).split(";")[0]: value for key, value in fixture.classnames().items()}

    # ASSERT
    assert class_names == {"Hits": "TTree"}


@pytest.mark.parametrize("layout", POSITRONIUM_LAYOUTS, ids=lambda layout: layout.key)
def test_fixture_was_cut_to_the_declared_size(layout: PositroniumLayout) -> None:
    """Every file keeps the first 500 entries of its simulation."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with uproot.open(layout.path) as fixture:
        entries = fixture["Hits"].num_entries

    # ASSERT
    assert entries == ENTRIES


@pytest.mark.parametrize("layout", POSITRONIUM_LAYOUTS, ids=lambda layout: layout.key)
def test_fixture_holds_the_branches_of_the_a1_structure(layout: PositroniumLayout) -> None:
    """The scenes differ in their source, not in the shape of their output."""
    # ARRANGE
    with uproot.open(A1_FIXTURE) as reference:
        expected = [str(name) for name in reference["Hits"].keys()]

    # ACT
    with uproot.open(layout.path) as fixture:
        names = [str(name) for name in fixture["Hits"].keys()]

    # ASSERT
    assert names == expected


@pytest.mark.parametrize("layout", POSITRONIUM_LAYOUTS, ids=lambda layout: layout.key)
def test_fixture_holds_the_declared_values(layout: PositroniumLayout) -> None:
    """What each scene wrote into the four branches is the ground truth here."""
    # ARRANGE
    expected = {
        branch: getattr(layout, field) for branch, field in POSITRONIUM_BRANCH_FIELDS.items()
    }

    # ACT
    values = {branch: distinct_values(layout.path, branch) for branch in POSITRONIUM_BRANCH_FIELDS}

    # ASSERT
    assert values == expected


@pytest.mark.parametrize("layout", POSITRONIUM_LAYOUTS, ids=lambda layout: layout.key)
def test_fixture_is_read_as_the_structure_it_has(layout: PositroniumLayout) -> None:
    """The files are ordinary hits files, and the package reads them as such."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(layout.path) as root_file:
        detection = root_file.detect_hits_tree()
        data = root_file.read(GateTree.HITS, ["sourceType", "decayType", "gammaType", "decayIndex"])

    # ASSERT
    assert detection.variant is HitsTreeVariant.NO_SYSTEM
    assert data.entry_count == ENTRIES


def test_the_scenes_cover_every_value_the_package_will_have_to_read() -> None:
    """Together the files have to exercise the values the enums describe.

    Two values are missing on purpose and are named here, so that the gap is
    stated rather than discovered: a single gamma emitter and a single gamma
    come from the "sg" model, which none of these simulations uses.
    """
    # ARRANGE
    covered: dict[str, set[int]] = {branch: set() for branch in POSITRONIUM_BRANCH_FIELDS}

    # ACT
    for layout in POSITRONIUM_LAYOUTS:
        for branch, field in POSITRONIUM_BRANCH_FIELDS.items():
            covered[branch].update(getattr(layout, field))

    # ASSERT
    assert covered["sourceType"] == {0, 2, 3, 4}
    assert covered["decayType"] == {0, 1, 2}
    assert covered["gammaType"] == {0, 2, 3}
    assert covered["decayIndex"] == {-1, 0, 1}


def test_one_scene_holds_data_written_by_no_positronium_source() -> None:
    """Telling a PositroniumSource row from any other needs a file without one."""
    # ARRANGE
    layout = next(item for item in POSITRONIUM_LAYOUTS if item.key == "back-to-back")

    # ACT
    indices = distinct_values(layout.path, "decayIndex")

    # ASSERT
    assert indices == (-1,)


def test_the_fixture_directory_holds_only_known_files() -> None:
    """A file nothing declares is a file no test looks at."""
    # ARRANGE
    directory = POSITRONIUM_LAYOUTS[0].path.parent
    expected = {layout.file_name for layout in POSITRONIUM_LAYOUTS} | {"README.md"}

    # ACT
    present = {entry.name for entry in directory.iterdir()}

    # ASSERT
    assert present == expected
