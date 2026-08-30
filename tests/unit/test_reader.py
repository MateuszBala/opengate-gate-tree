"""Unit tests for tree extraction."""

from pathlib import Path

import pytest
from conftest import GateHitsLayout

from opengate_gate_tree.errors import (
    BranchNotFoundError,
    RootFileError,
    TreeNotFoundError,
)
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.io.rootfile import RootFile
from opengate_gate_tree.tree.gatetree import GateTree


def test_read_tree_loads_every_branch_by_default(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """Omitting the selection should read the whole tree."""
    # ARRANGE
    # No additional setup required.

    # ACT
    data = read_tree(gate_hits_file, GateTree.HITS)

    # ASSERT
    assert data.tree == GateTree.HITS
    assert data.entry_count == gate_hits_layout.entries
    assert len(data.branch_names) == gate_hits_layout.branch_count


def test_read_tree_reads_only_the_requested_branches(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """A selection should be honoured exactly, including its order."""
    # ARRANGE
    requested = ["eventID", "edep", "volumeID"]

    # ACT
    data = read_tree(gate_hits_file, GateTree.HITS, requested)

    # ASSERT
    assert data.branch_names == tuple(requested)
    assert data["volumeID"].shape == (
        gate_hits_layout.entries,
        gate_hits_layout.volume_id_width,
    )


def test_read_tree_collapses_repeated_branches(gate_hits_file: Path) -> None:
    """A repeated branch should be read once, at its first position."""
    # ARRANGE
    requested = ["eventID", "edep", "eventID"]

    # ACT
    data = read_tree(gate_hits_file, GateTree.HITS, requested)

    # ASSERT
    assert data.branch_names == ("eventID", "edep")


def test_read_tree_treats_an_empty_selection_as_every_branch(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """An empty list should behave like no selection at all."""
    # ARRANGE
    requested: list[str] = []

    # ACT
    data = read_tree(gate_hits_file, GateTree.HITS, requested)

    # ASSERT
    assert len(data.branch_names) == gate_hits_layout.branch_count


def test_read_tree_raises_for_an_unknown_branch(gate_hits_file: Path) -> None:
    """An unknown branch should be reported together with the alternatives."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with pytest.raises(BranchNotFoundError) as error_info:
        read_tree(gate_hits_file, GateTree.HITS, ["eventID", "notInTheTree"])

    message = str(error_info.value)
    assert "notInTheTree" in message
    assert "eventID" in message


def test_read_tree_raises_for_an_empty_branch_name(gate_hits_file: Path) -> None:
    """A branch name that carries no value should be refused."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with pytest.raises(ValueError, match="must not be empty"):
        read_tree(gate_hits_file, GateTree.HITS, ["eventID", ""])


def test_read_tree_raises_for_a_missing_tree(gate_hits_file: Path) -> None:
    """A tree absent from the file should be reported with what the file holds."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with pytest.raises(TreeNotFoundError) as error_info:
        read_tree(gate_hits_file, GateTree.SINGLES)

    message = str(error_info.value)
    assert "Singles" in message
    assert "Hits" in message


def test_read_tree_raises_for_a_missing_file(tmp_path: Path) -> None:
    """A missing input file should be reported as a package error."""
    # ARRANGE
    path = tmp_path / "missing.root"

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="does not exist"):
        read_tree(path, GateTree.HITS)


def test_read_tree_closes_the_file(
    gate_hits_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file should be closed once the tree has been read."""
    # ARRANGE
    closed: list[Path] = []
    original_close = RootFile.close

    def record_close(self: RootFile) -> None:
        closed.append(self.path)
        original_close(self)

    monkeypatch.setattr(RootFile, "close", record_close)

    # ACT
    read_tree(gate_hits_file, GateTree.HITS, ["eventID"])

    # ASSERT
    assert closed == [gate_hits_file]


def test_read_tree_closes_the_file_when_reading_fails(
    gate_hits_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed read should not leave the file open."""
    # ARRANGE
    closed: list[Path] = []
    original_close = RootFile.close

    def record_close(self: RootFile) -> None:
        closed.append(self.path)
        original_close(self)

    monkeypatch.setattr(RootFile, "close", record_close)

    # ACT
    with pytest.raises(TreeNotFoundError):
        read_tree(gate_hits_file, GateTree.SINGLES)

    # ASSERT
    assert closed == [gate_hits_file]
