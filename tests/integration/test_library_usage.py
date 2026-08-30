"""End-to-end tests of the package used as a library.

The first test mirrors the example in the "Library Usage" section of the
README, so that the documented code cannot quietly stop working.
"""

from pathlib import Path

import h5py
import pandas as pd
import pytest
from conftest import GateHitsLayout

from opengate_gate_tree import (
    GateTree,
    GateTreeError,
    OutputFileFormat,
    RootFile,
    TreeNotFoundError,
    read_tree,
    write_tree,
)


def test_readme_example_runs(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
    tmp_path: Path,
) -> None:
    """The example documented in the README should work as written."""
    # ARRANGE
    output_file = tmp_path / "out" / "hits.hdf5"

    # ACT
    data = read_tree(
        gate_hits_file,
        GateTree.HITS,
        ["eventID", "edep", "posX", "posY", "posZ"],
    )
    energies = data["edep"]
    frame = data.to_dataframe()
    written = write_tree(data, output_file, OutputFileFormat.HDF5)

    # ASSERT
    assert data.entry_count == gate_hits_layout.entries
    assert data.branch_names == ("eventID", "edep", "posX", "posY", "posZ")
    assert len(energies) == gate_hits_layout.entries
    assert isinstance(frame, pd.DataFrame)
    assert written.is_file()
    with h5py.File(written) as stored:
        assert list(stored["Hits"]) == list(data.branch_names)


def test_reading_every_branch_without_a_selection(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """Omitting the branch list should read the whole tree."""
    # ARRANGE
    # No additional setup required.

    # ACT
    data = read_tree(gate_hits_file, GateTree.HITS)

    # ASSERT
    assert len(data.branch_names) == gate_hits_layout.branch_count


def test_reading_several_trees_from_one_open_file(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """RootFile should let a caller inspect and read without reopening."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        tree_names = root_file.tree_names
        branch_names = root_file.branch_names(GateTree.HITS)
        hits = root_file.read(GateTree.HITS, ["eventID", "edep"])

    # ASSERT
    assert tree_names == gate_hits_layout.tree_names
    assert len(branch_names) == gate_hits_layout.branch_count
    assert hits.branch_names == ("eventID", "edep")


def test_failures_are_catchable_through_the_base_error(gate_hits_file: Path) -> None:
    """One except clause should be enough to handle any package failure."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with pytest.raises(GateTreeError):
        read_tree(gate_hits_file, GateTree.SINGLES)

    with pytest.raises(TreeNotFoundError):
        read_tree(gate_hits_file, GateTree.SINGLES)


def test_library_use_writes_nothing_to_the_console(
    gate_hits_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A quiet application should stay quiet while the package works."""
    # ARRANGE
    output_file = tmp_path / "hits.csv"

    # ACT
    data = read_tree(gate_hits_file, GateTree.HITS, ["eventID"])
    write_tree(data, output_file, OutputFileFormat.CSV)
    captured = capsys.readouterr()

    # ASSERT
    assert captured.out == ""
    assert captured.err == ""


def test_a_selection_survives_the_whole_round_trip(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """Reading, selecting and writing should compose without surprises."""
    # ARRANGE
    output_file = tmp_path / "hits.root"

    # ACT
    data = read_tree(gate_hits_file, GateTree.HITS)
    reduced = data.select(["edep", "eventID"])
    write_tree(reduced, output_file, OutputFileFormat.ROOT)
    restored = read_tree(output_file, GateTree.HITS)

    # ASSERT
    assert restored.branch_names == ("edep", "eventID")
    assert restored.entry_count == data.entry_count
