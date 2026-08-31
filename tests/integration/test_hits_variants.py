"""End-to-end coverage of every supported "Hits" tree structure.

One simulation output per structure goes the whole way: the tree is found in
the file, its structure recognised and checked, the data read, and written out
again. What the unit tests state one piece at a time is stated here as one
path, on files a simulation produced.
"""

from collections.abc import Mapping
from pathlib import Path

import h5py
import pandas as pd
import pytest
from conftest import HITS_VARIANT_LAYOUTS, HitsVariantLayout

from opengate_gate_tree import (
    GateTree,
    OutputFileFormat,
    RootFile,
    describe_hits_tree,
    read_tree,
    write_tree,
)

# Tree to read for fixtures that hold hits under more than one name.
TREE_NAMES: dict[str, str | None] = {
    "multi-sd": "Hits_DET_INNER",
}

# Structures each fixture is expected to be recognised as.
REFERENCES: dict[str, str] = {
    "a1": "A1",
    "a2": "A2",
    "a3": "A3",
    "a4": "A4",
    "a5": "A5",
    "b1": "B1",
    "multi-run": "A1",
    "multi-sd": "A1",
}


def tree_name_of(layout: HitsVariantLayout) -> str | None:
    """Return the tree to read for a fixture, when it has to be named."""
    return TREE_NAMES.get(layout.key)


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_is_read_recognised_and_described(layout: HitsVariantLayout) -> None:
    """Every supported structure should read and describe itself."""
    # ARRANGE
    tree_name = tree_name_of(layout)

    # ACT
    with RootFile(layout.path) as root_file:
        detection = root_file.detect_hits_tree(tree_name)
        data = root_file.read(GateTree.HITS, tree_name=tree_name)
    description = describe_hits_tree(detection, entry_count=data.entry_count)

    # ASSERT
    assert data.entry_count == layout.entries
    assert len(data.branch_names) == layout.branch_count
    assert f"({REFERENCES[layout.key]})" in description
    assert f"Entries: {layout.entries}" in description


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_survives_a_write_to_csv(layout: HitsVariantLayout, tmp_path: Path) -> None:
    """Every structure should reach a CSV file with all of its entries."""
    # ARRANGE
    data = read_tree(layout.path, GateTree.HITS, tree_name=tree_name_of(layout))
    output_file = tmp_path / "hits.csv"

    # ACT
    write_tree(data, output_file, OutputFileFormat.CSV)

    # ASSERT
    written = pd.read_csv(output_file)
    assert len(written) == layout.entries


@pytest.mark.parametrize("layout", HITS_VARIANT_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_survives_a_write_to_hdf5(layout: HitsVariantLayout, tmp_path: Path) -> None:
    """Every structure should reach an HDF5 file with all of its branches."""
    # ARRANGE
    data = read_tree(layout.path, GateTree.HITS, tree_name=tree_name_of(layout))
    output_file = tmp_path / "hits.h5"

    # ACT
    write_tree(data, output_file, OutputFileFormat.HDF5)

    # ASSERT
    with h5py.File(output_file) as written:
        assert list(written["Hits"]) == list(data.branch_names)
        assert written["Hits"].attrs["entries"] == layout.entries


@pytest.mark.parametrize(
    "layout",
    # The GateToTree layout names branches "volumeID[0]" and up, which the
    # ROOT writer cannot store faithfully; that case is covered where the
    # writer refuses it.
    [layout for layout in HITS_VARIANT_LAYOUTS if layout.key != "b1"],
    ids=lambda layout: layout.key,
)
def test_variant_survives_a_write_to_root(layout: HitsVariantLayout, tmp_path: Path) -> None:
    """A structure written to ROOT should read back as the same structure."""
    # ARRANGE
    data = read_tree(layout.path, GateTree.HITS, tree_name=tree_name_of(layout))
    output_file = tmp_path / "hits.root"

    # ACT
    write_tree(data, output_file, OutputFileFormat.ROOT)
    restored = read_tree(output_file, GateTree.HITS)

    # ASSERT
    assert restored.branch_names == data.branch_names
    assert restored.entry_count == layout.entries


def test_a_file_split_per_run_reads_one_run_at_a_time(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Naming the tree is what tells the runs apart until they are merged."""
    # ARRANGE
    path = hits_variant_files["multi-run"]

    # ACT
    runs = {
        tree_name: read_tree(path, GateTree.HITS, tree_name=tree_name)
        for tree_name in ("Hits", "Hits_run1", "Hits_run2")
    }

    # ASSERT
    assert {name: set(data["runID"].tolist()) for name, data in runs.items()} == {
        "Hits": {0},
        "Hits_run1": {1},
        "Hits_run2": {2},
    }
