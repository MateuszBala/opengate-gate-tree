"""Unit tests for reading GATE ROOT files."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import GateHitsLayout

from opengate_gate_tree.errors import (
    BranchNotFoundError,
    RootFileError,
    TreeNotFoundError,
    UnsupportedBranchTypeError,
)
from opengate_gate_tree.io.rootfile import RootFile
from opengate_gate_tree.tree.gatetree import GateTree


def test_tree_names_reports_only_trees(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """Histograms stored by GATE should not be reported as trees."""
    # ARRANGE
    # The fixture holds two TH1D histograms next to three trees.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        names = root_file.tree_names

    # ASSERT
    assert names == gate_hits_layout.tree_names
    assert "latest_event_ID" not in names
    assert "total_nb_primaries" not in names


def test_tree_names_strips_cycle_numbers(gate_hits_file: Path) -> None:
    """ROOT cycle suffixes should not leak into the reported names."""
    # ARRANGE
    # Keys are stored as "Hits;1" in the file.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        names = root_file.tree_names

    # ASSERT
    assert all(";" not in name for name in names)


def test_has_tree_answers_for_present_and_absent_trees(gate_hits_file: Path) -> None:
    """Presence of a tree should be reported without raising."""
    # ARRANGE
    # The fixture holds a "Hits" tree but no "Singles" tree.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        has_hits = root_file.has_tree(GateTree.HITS)
        has_singles = root_file.has_tree(GateTree.SINGLES)

    # ASSERT
    assert has_hits is True
    assert has_singles is False


def test_resolve_tree_name_returns_the_file_key(gate_hits_file: Path) -> None:
    """An exactly matching name should resolve to itself."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        resolved = root_file.resolve_tree_name(GateTree.HITS)

    # ASSERT
    assert resolved == "Hits"


def test_resolve_tree_name_matches_regardless_of_case(
    make_gate_root_file: Callable[..., Path],
) -> None:
    """A differently cased tree name should still resolve."""
    # ARRANGE
    path = make_gate_root_file({"hits": {"eventID": np.arange(3, dtype=np.int32)}})

    # ACT
    with RootFile(path) as root_file:
        resolved = root_file.resolve_tree_name(GateTree.HITS)

    # ASSERT
    assert resolved == "hits"


def test_resolve_tree_name_lists_available_trees_when_missing(gate_hits_file: Path) -> None:
    """A missing tree should point the caller at what the file holds."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with RootFile(gate_hits_file) as root_file, pytest.raises(TreeNotFoundError) as error_info:
        root_file.resolve_tree_name(GateTree.SINGLES)

    message = str(error_info.value)
    assert "Singles" in message
    assert "pet_data" in message
    assert "OpticalData" in message


def test_branch_names_returns_the_gate_hits_branches(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """The reader should expose the branches GATE writes for hits."""
    # ARRANGE
    expected_branches = {"eventID", "edep", "posX", "volumeID", "processName"}

    # ACT
    with RootFile(gate_hits_file) as root_file:
        names = root_file.branch_names(GateTree.HITS)

    # ASSERT
    assert len(names) == gate_hits_layout.branch_count
    assert expected_branches.issubset(set(names))


def test_read_loads_every_branch_by_default(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """Reading without a selection should load the whole tree."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS)

    # ASSERT
    assert data.tree == GateTree.HITS
    assert data.entry_count == gate_hits_layout.entries
    assert len(data.branch_names) == gate_hits_layout.branch_count


def test_read_keeps_the_fixed_width_array_branch(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """The volumeID branch should stay two-dimensional."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, ["volumeID"])

    # ASSERT
    assert data["volumeID"].shape == (
        gate_hits_layout.entries,
        gate_hits_layout.volume_id_width,
    )
    assert data.array_branches == {"volumeID": gate_hits_layout.volume_id_width}


def test_read_loads_text_branches(gate_hits_file: Path) -> None:
    """Text branches should load as readable values."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, ["processName"])

    # ASSERT
    assert data.dtypes["processName"].kind == "O"
    assert isinstance(data["processName"][0], str)
    assert "Compton" in set(data["processName"])


def test_read_keeps_the_requested_branch_order(gate_hits_file: Path) -> None:
    """The selection order should be preserved, not the file order."""
    # ARRANGE
    requested = ["edep", "eventID", "posX"]

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, requested)

    # ASSERT
    assert data.branch_names == tuple(requested)


def test_read_keeps_repeated_branches_once(gate_hits_file: Path) -> None:
    """A repeated branch should be read once, at its first position."""
    # ARRANGE
    requested = ["edep", "eventID", "edep"]

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, requested)

    # ASSERT
    assert data.branch_names == ("edep", "eventID")


def test_read_raises_for_an_unknown_branch(gate_hits_file: Path) -> None:
    """An unknown branch should be reported before any data is loaded."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with RootFile(gate_hits_file) as root_file, pytest.raises(BranchNotFoundError) as error_info:
        root_file.read(GateTree.HITS, ["eventID", "missing"])

    assert "missing" in str(error_info.value)


def test_read_raises_for_a_branch_of_varying_length(
    make_jagged_root_file: Callable[..., Path],
) -> None:
    """Branches whose length varies per entry should be rejected."""
    # ARRANGE
    path = make_jagged_root_file()

    # ACT & ASSERT
    with RootFile(path) as root_file, pytest.raises(UnsupportedBranchTypeError) as error_info:
        root_file.read(GateTree.HITS)

    assert "hitTimes" in str(error_info.value)


def test_read_warns_for_an_empty_tree(
    make_gate_root_file: Callable[..., Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reading a tree without entries should leave a trace in the log."""
    # ARRANGE
    path = make_gate_root_file({"Hits": {"eventID": np.array([], dtype=np.int32)}})

    # ACT
    with caplog.at_level("WARNING"), RootFile(path) as root_file:
        data = root_file.read(GateTree.HITS)

    # ASSERT
    assert data.entry_count == 0
    assert "no entries" in caplog.text


def test_opening_a_missing_file_reports_a_package_error(tmp_path: Path) -> None:
    """A missing input file should raise a package error."""
    # ARRANGE
    path = tmp_path / "missing.root"

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="does not exist"):
        RootFile(path)


def test_opening_a_file_that_is_not_root_reports_a_package_error(tmp_path: Path) -> None:
    """A file that only looks like a ROOT file should raise a package error."""
    # ARRANGE
    path = tmp_path / "simulation.root"
    path.write_text("this is not a ROOT file\n", encoding="utf-8")

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="could not be read as a ROOT file"):
        RootFile(path)


def test_reader_exposes_the_file_path(gate_hits_file: Path) -> None:
    """The reader should report the file it was opened on."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        reported_path = root_file.path

    # ASSERT
    assert reported_path == gate_hits_file


def test_close_can_be_called_directly(gate_hits_file: Path) -> None:
    """The reader should be usable without the context manager."""
    # ARRANGE
    root_file = RootFile(gate_hits_file)

    # ACT
    names = root_file.tree_names
    root_file.close()

    # ASSERT
    assert "Hits" in names
