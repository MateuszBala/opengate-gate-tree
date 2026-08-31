"""Unit tests for tree extraction."""

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest
from conftest import GateHitsLayout

from opengate_gate_tree.errors import (
    AmbiguousTreeError,
    BranchNotFoundError,
    RootFileError,
    TreeNotFoundError,
    UnknownHitsVariantError,
)
from opengate_gate_tree.io.reader import read_hits_trees, read_tree
from opengate_gate_tree.io.rootfile import RootFile
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.merge import SOURCE_TREE_BRANCH


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


def test_read_tree_checks_the_structure_by_default(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """The entry point used by the command line validates like the reader."""
    # ARRANGE
    path = make_hits_root_file({"Hits": ["eventID", "edep"]})

    # ACT / ASSERT
    with pytest.raises(UnknownHitsVariantError):
        read_tree(path, GateTree.HITS)


def test_read_tree_can_skip_the_structure_check(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """Extraction has to stay possible for a structure nothing describes."""
    # ARRANGE
    path = make_hits_root_file({"Hits": ["eventID", "edep"]})

    # ACT
    data = read_tree(path, GateTree.HITS, validate=False)

    # ASSERT
    assert data.branch_names == ("eventID", "edep")


def test_validation_leaves_the_data_untouched(gate_hits_file: Path) -> None:
    """Checking a tree must not change a single value it holds."""
    # ARRANGE
    branches = ["eventID", "runID", "edep", "volumeID", "processName"]

    # ACT
    checked = read_tree(gate_hits_file, GateTree.HITS, branches)
    unchecked = read_tree(gate_hits_file, GateTree.HITS, branches, validate=False)

    # ASSERT
    assert checked.branch_names == unchecked.branch_names
    assert np.array_equal(checked["eventID"], unchecked["eventID"])
    assert np.array_equal(checked["volumeID"], unchecked["volumeID"])
    assert list(checked["processName"]) == list(unchecked["processName"])


def test_read_tree_finds_hits_stored_under_another_name(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The entry point used by the command line resolves names like the reader."""
    # ARRANGE
    # The GateToTree output calls its tree "tree".

    # ACT
    data = read_tree(hits_variant_files["b1"], GateTree.HITS)

    # ASSERT
    assert len(data.branch_names) == 54


def test_read_tree_reads_the_named_tree(hits_variant_files: Mapping[str, Path]) -> None:
    """One run out of three is singled out by naming its tree."""
    # ARRANGE
    # No additional setup required.

    # ACT
    data = read_tree(hits_variant_files["multi-run"], GateTree.HITS, tree_name="Hits_run1")

    # ASSERT
    assert set(data["runID"].tolist()) == {1}


def test_read_tree_refuses_to_choose_between_trees_of_hits(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A file split per detector needs the caller to say which one to read."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with pytest.raises(AmbiguousTreeError):
        read_tree(hits_variant_files["multi-sd"], GateTree.HITS)


def test_read_hits_trees_joins_the_trees_of_a_file(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The entry point used by the command line merges like the reader."""
    # ARRANGE
    # No additional setup required.

    # ACT
    data = read_hits_trees(hits_variant_files["multi-run"])

    # ASSERT
    assert data.entry_count == 1500
    assert sorted(set(data["runID"].tolist())) == [0, 1, 2]


def test_read_hits_trees_reads_a_file_holding_one_tree(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Merging is not something the caller has to know the file needs."""
    # ARRANGE
    # No additional setup required.

    # ACT
    data = read_hits_trees(hits_variant_files["b1"])

    # ASSERT
    assert data.entry_count == 4441
    assert set(data[SOURCE_TREE_BRANCH]) == {"tree"}


def test_read_hits_trees_closes_the_file(
    hits_variant_files: Mapping[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading several trees should leave no handle behind either."""
    # ARRANGE
    closed: list[Path] = []
    original_close = RootFile.close

    def record_close(self: RootFile) -> None:
        closed.append(self.path)
        original_close(self)

    monkeypatch.setattr(RootFile, "close", record_close)

    # ACT
    read_hits_trees(hits_variant_files["a1"])

    # ASSERT
    assert closed == [hits_variant_files["a1"]]
