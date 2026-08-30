"""Unit tests for GATE ROOT file consistency checks."""

from pathlib import Path
from typing import Any

import pytest
import uproot
from uproot.interpretation.jagged import AsJagged
from uproot.interpretation.numerical import AsDtype
from uproot.interpretation.strings import AsStrings

from opengate_gate_tree.errors import (
    BranchNotFoundError,
    RootFileError,
    TreeNotFoundError,
    UnsupportedBranchTypeError,
)
from opengate_gate_tree.io.validation import (
    find_tree_name,
    resolve_tree_name,
    validate_branch_interpretations,
    validate_branches_present,
    validate_root_file_path,
)
from opengate_gate_tree.tree.gatetree import GateTree


def test_validate_root_file_path_accepts_an_existing_root_file(tmp_path: Path) -> None:
    """A regular file with a .root extension should pass."""
    # ARRANGE
    path = tmp_path / "simulation.root"
    path.write_bytes(b"")

    # ACT & ASSERT
    validate_root_file_path(path)


def test_validate_root_file_path_rejects_a_wrong_extension(tmp_path: Path) -> None:
    """Files that are not named as ROOT files should be rejected."""
    # ARRANGE
    path = tmp_path / "simulation.txt"
    path.write_bytes(b"")

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="must have a '.root' extension"):
        validate_root_file_path(path)


def test_validate_root_file_path_rejects_a_missing_file(tmp_path: Path) -> None:
    """A path that does not exist should be reported."""
    # ARRANGE
    path = tmp_path / "missing.root"

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="does not exist"):
        validate_root_file_path(path)


def test_validate_root_file_path_rejects_a_directory(tmp_path: Path) -> None:
    """A directory named like a ROOT file should be reported."""
    # ARRANGE
    path = tmp_path / "simulation.root"
    path.mkdir()

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="not a regular file"):
        validate_root_file_path(path)


def test_resolve_tree_name_returns_an_exact_match() -> None:
    """A tree listed in the file should resolve to its own name."""
    # ARRANGE
    available = ["pet_data", "Hits", "OpticalData"]

    # ACT
    resolved = resolve_tree_name(available, GateTree.HITS)

    # ASSERT
    assert resolved == "Hits"


def test_resolve_tree_name_matches_regardless_of_case() -> None:
    """Tree names should be matched without regard to case."""
    # ARRANGE
    available = ["hits"]

    # ACT
    resolved = resolve_tree_name(available, GateTree.HITS)

    # ASSERT
    assert resolved == "hits"


def test_resolve_tree_name_lists_available_trees_when_missing() -> None:
    """A missing tree should point the caller at what the file holds."""
    # ARRANGE
    available = ["pet_data", "Hits", "OpticalData"]

    # ACT & ASSERT
    with pytest.raises(TreeNotFoundError) as error_info:
        resolve_tree_name(available, GateTree.SINGLES)

    message = str(error_info.value)
    assert "Singles" in message
    assert "pet_data" in message
    assert "OpticalData" in message


def test_resolve_tree_name_names_the_source_file_when_given(tmp_path: Path) -> None:
    """The error should say which file was inspected when the path is known."""
    # ARRANGE
    source = tmp_path / "simulation.root"

    # ACT & ASSERT
    with pytest.raises(TreeNotFoundError) as error_info:
        resolve_tree_name(["Hits"], GateTree.SINGLES, source)

    assert str(source) in str(error_info.value)


def test_find_tree_name_reports_absence_without_raising() -> None:
    """Looking a tree up should be possible without handling an exception."""
    # ARRANGE
    available = ["Hits"]

    # ACT
    found = find_tree_name(available, GateTree.SINGLES)

    # ASSERT
    assert found is None


def test_validate_branches_present_accepts_known_branches() -> None:
    """Requested branches that exist should pass."""
    # ARRANGE
    available = ["eventID", "edep", "posX"]

    # ACT & ASSERT
    validate_branches_present(available, ["edep", "eventID"])


def test_validate_branches_present_reports_every_missing_branch() -> None:
    """All missing branches should be reported at once."""
    # ARRANGE
    available = ["eventID", "edep"]

    # ACT & ASSERT
    with pytest.raises(BranchNotFoundError) as error_info:
        validate_branches_present(available, ["eventID", "missing", "absent"])

    message = str(error_info.value)
    assert "missing" in message
    assert "absent" in message
    assert "edep" in message


def test_validate_branch_interpretations_accepts_types_used_by_gate(
    gate_hits_file: Path,
) -> None:
    """Every branch type appearing in a real GATE file should pass."""
    # ARRANGE
    with uproot.open(gate_hits_file) as root_file:
        hits = root_file["Hits"]
        interpretations = {name: hits[name].interpretation for name in hits.keys()}

    # ACT & ASSERT
    validate_branch_interpretations(interpretations)


def test_validate_branch_interpretations_accepts_text_branches() -> None:
    """Text branches are supported even though they load as object arrays."""
    # ARRANGE
    interpretations: dict[str, Any] = {"processName": AsStrings()}

    # ACT & ASSERT
    validate_branch_interpretations(interpretations)


def test_validate_branch_interpretations_rejects_branches_of_varying_length() -> None:
    """Branches whose length varies per entry have no supported representation."""
    # ARRANGE
    interpretations: dict[str, Any] = {
        "eventID": AsDtype(">i4"),
        "hitTimes": AsJagged(AsDtype(">f8")),
    }

    # ACT & ASSERT
    with pytest.raises(UnsupportedBranchTypeError) as error_info:
        validate_branch_interpretations(interpretations)

    message = str(error_info.value)
    assert "hitTimes" in message
    assert "eventID" not in message
