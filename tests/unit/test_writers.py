"""Unit tests for the output writers."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import uproot
from conftest import GateHitsLayout

from opengate_gate_tree import __version__
from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.io.writers import WRITERS, get_writer, write_tree
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.treedata import TreeData

# Width of the volumeID branch produced by GATE.
VOLUME_ID_WIDTH = 4

# File extension used for each output format.
EXTENSIONS = {
    OutputFileFormat.CSV: "csv",
    OutputFileFormat.HDF5: "h5",
    OutputFileFormat.ROOT: "root",
}


def make_tree_data(entries: int = 3) -> TreeData:
    """Build data holding a scalar, a text and a fixed-width array branch."""
    columns: dict[str, npt.NDArray[Any]] = {
        "eventID": np.arange(entries, dtype=np.int32),
        "edep": np.linspace(0.1, 1.0, entries).astype(np.float32),
        "processName": np.array(
            [["Compton", "PhotoElectric"][index % 2] for index in range(entries)],
            dtype=object,
        ),
        "volumeID": np.tile(np.arange(VOLUME_ID_WIDTH, dtype=np.int32), (entries, 1)),
    }
    return TreeData(GateTree.HITS, columns)


def make_empty_tree_data() -> TreeData:
    """Build data with branches but no entries, including a text branch."""
    return TreeData(
        GateTree.HITS,
        {
            "eventID": np.array([], dtype=np.int32),
            "processName": np.array([], dtype=object),
        },
    )


@pytest.mark.parametrize("file_format", list(OutputFileFormat))
def test_get_writer_returns_a_writer_for_every_format(file_format: OutputFileFormat) -> None:
    """Every format declared by the package should have a writer."""
    # ARRANGE
    # No additional setup required.

    # ACT
    writer = get_writer(file_format)

    # ASSERT
    assert writer.file_format == file_format


@pytest.mark.parametrize("file_format", list(OutputFileFormat))
def test_write_tree_creates_the_output_directory(
    file_format: OutputFileFormat,
    tmp_path: Path,
) -> None:
    """A missing output directory should be created rather than reported."""
    # ARRANGE
    path = tmp_path / "nested" / "output" / f"hits.{EXTENSIONS[file_format]}"

    # ACT
    written = write_tree(make_tree_data(), path, file_format)

    # ASSERT
    assert written == path
    assert path.is_file()


@pytest.mark.parametrize("file_format", list(OutputFileFormat))
def test_write_tree_overwrites_an_existing_file(
    file_format: OutputFileFormat,
    tmp_path: Path,
) -> None:
    """Writing twice should replace the previous file without complaint."""
    # ARRANGE
    path = tmp_path / f"hits.{EXTENSIONS[file_format]}"
    write_tree(make_tree_data(entries=10), path, file_format)

    # ACT
    write_tree(make_tree_data(entries=3), path, file_format)

    # ASSERT
    assert path.is_file()


@pytest.mark.parametrize("file_format", list(OutputFileFormat))
def test_write_tree_reports_an_unusable_output_directory(
    file_format: OutputFileFormat,
    tmp_path: Path,
) -> None:
    """A directory that cannot be created should be reported as an export error."""
    # ARRANGE
    blocking_file = tmp_path / "blocker"
    blocking_file.write_text("", encoding="utf-8")
    path = blocking_file / "output" / f"hits.{EXTENSIONS[file_format]}"

    # ACT & ASSERT
    with pytest.raises(ExportError, match="Output directory could not be created"):
        write_tree(make_tree_data(), path, file_format)


@pytest.mark.parametrize("file_format", list(OutputFileFormat))
def test_write_tree_reports_a_file_that_cannot_be_written(
    file_format: OutputFileFormat,
    tmp_path: Path,
) -> None:
    """A failing write should surface as an export error, not a raw OS error."""
    # ARRANGE
    path = tmp_path / f"hits.{EXTENSIONS[file_format]}"
    path.mkdir()

    # ACT & ASSERT
    with pytest.raises(ExportError, match="could not be written"):
        write_tree(make_tree_data(), path, file_format)


def test_get_writer_reports_a_format_without_a_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A format left without a writer should be reported, not raise a key error."""
    # ARRANGE
    monkeypatch.delitem(WRITERS, OutputFileFormat.CSV)

    # ACT & ASSERT
    with pytest.raises(ExportError, match="No writer is available"):
        get_writer(OutputFileFormat.CSV)


@pytest.mark.parametrize("file_format", [OutputFileFormat.CSV, OutputFileFormat.HDF5])
def test_write_tree_accepts_an_empty_tree_with_a_text_branch(
    file_format: OutputFileFormat,
    tmp_path: Path,
) -> None:
    """A tree without entries should still produce a file."""
    # ARRANGE
    path = tmp_path / f"hits.{EXTENSIONS[file_format]}"

    # ACT
    write_tree(make_empty_tree_data(), path, file_format)

    # ASSERT
    assert path.is_file()


def test_csv_writes_one_column_per_array_component(tmp_path: Path) -> None:
    """A fixed-width array branch has no scalar cell, so it is expanded."""
    # ARRANGE
    path = tmp_path / "hits.csv"

    # ACT
    write_tree(make_tree_data(entries=3), path, OutputFileFormat.CSV)

    # ASSERT
    frame = pd.read_csv(path)
    expected_columns = [
        "eventID",
        "edep",
        "processName",
        *[f"volumeID_{index}" for index in range(VOLUME_ID_WIDTH)],
    ]
    assert list(frame.columns) == expected_columns
    assert len(frame) == 3


def test_csv_round_trip_preserves_values(tmp_path: Path) -> None:
    """Values should survive a write and a read with pandas."""
    # ARRANGE
    data = make_tree_data(entries=5)
    path = tmp_path / "hits.csv"

    # ACT
    write_tree(data, path, OutputFileFormat.CSV)
    frame = pd.read_csv(path)

    # ASSERT
    assert np.array_equal(frame["eventID"].to_numpy(), data["eventID"])
    assert list(frame["processName"]) == list(data["processName"])


def test_csv_writes_only_a_header_for_an_empty_tree(tmp_path: Path) -> None:
    """A tree without entries should produce a header and nothing else."""
    # ARRANGE
    path = tmp_path / "hits.csv"

    # ACT
    write_tree(make_empty_tree_data(), path, OutputFileFormat.CSV)

    # ASSERT
    assert path.read_text(encoding="utf-8") == "eventID,processName\n"


def test_hdf5_writes_one_group_per_tree(tmp_path: Path) -> None:
    """The tree should appear as a named group holding one dataset per branch."""
    # ARRANGE
    data = make_tree_data(entries=3)
    path = tmp_path / "hits.h5"

    # ACT
    write_tree(data, path, OutputFileFormat.HDF5)

    # ASSERT
    with h5py.File(path) as output_file:
        assert list(output_file) == ["Hits"]
        assert list(output_file["Hits"]) == list(data.branch_names)


def test_hdf5_keeps_the_array_branch_two_dimensional(tmp_path: Path) -> None:
    """An array branch should stay one dataset, not be split into columns."""
    # ARRANGE
    data = make_tree_data(entries=6)
    path = tmp_path / "hits.h5"

    # ACT
    write_tree(data, path, OutputFileFormat.HDF5)

    # ASSERT
    with h5py.File(path) as output_file:
        stored = output_file["Hits"]["volumeID"]
        assert stored.shape == (6, VOLUME_ID_WIDTH)
        assert np.array_equal(stored[:], data["volumeID"])


def test_hdf5_round_trip_preserves_numbers_and_text(tmp_path: Path) -> None:
    """Numeric types and text should both survive the write."""
    # ARRANGE
    data = make_tree_data(entries=5)
    path = tmp_path / "hits.h5"

    # ACT
    write_tree(data, path, OutputFileFormat.HDF5)

    # ASSERT
    with h5py.File(path) as output_file:
        group = output_file["Hits"]
        assert group["eventID"].dtype == np.dtype(np.int32)
        assert np.array_equal(group["edep"][:], data["edep"])
        assert list(group["processName"].asstr()[:]) == list(data["processName"])


def test_hdf5_records_the_tree_metadata(tmp_path: Path) -> None:
    """The group should describe what it holds."""
    # ARRANGE
    data = make_tree_data(entries=7)
    path = tmp_path / "hits.h5"

    # ACT
    write_tree(data, path, OutputFileFormat.HDF5)

    # ASSERT
    with h5py.File(path) as output_file:
        attributes = output_file["Hits"].attrs
        assert attributes["gate_tree"] == "Hits"
        assert attributes["entries"] == 7
        assert list(attributes["branches"]) == list(data.branch_names)
        assert attributes["package_version"] == __version__


def test_root_writes_a_ttree_and_not_an_rntuple(tmp_path: Path) -> None:
    """The output must be a TTree; uproot writes an RNTuple by default."""
    # ARRANGE
    path = tmp_path / "hits.root"

    # ACT
    write_tree(make_tree_data(entries=3), path, OutputFileFormat.ROOT)

    # ASSERT
    with uproot.open(path) as output_file:
        assert list(output_file.classnames().values()) == ["TTree"]


def test_root_names_the_tree_after_the_gate_tree(tmp_path: Path) -> None:
    """The written tree should carry the name of the tree it came from."""
    # ARRANGE
    path = tmp_path / "hits.root"

    # ACT
    write_tree(make_tree_data(), path, OutputFileFormat.ROOT)

    # ASSERT
    with uproot.open(path) as output_file:
        assert list(output_file.keys(cycle=False)) == [GateTree.HITS.value]


def test_root_round_trip_preserves_every_branch_kind(tmp_path: Path) -> None:
    """Scalar, text and array branches should all come back unchanged."""
    # ARRANGE
    data = make_tree_data(entries=5)
    path = tmp_path / "hits.root"

    # ACT
    write_tree(data, path, OutputFileFormat.ROOT)
    restored = read_tree(path, GateTree.HITS)

    # ASSERT
    assert restored.branch_names == data.branch_names
    assert np.array_equal(restored["eventID"], data["eventID"])
    assert np.array_equal(restored["volumeID"], data["volumeID"])
    assert list(restored["processName"]) == list(data["processName"])
    assert restored.dtypes["eventID"] == data.dtypes["eventID"]


def test_root_accepts_an_empty_tree_with_a_text_branch(tmp_path: Path) -> None:
    """Declaring the branches is enough; appending an empty batch is not attempted."""
    # ARRANGE
    data = make_empty_tree_data()
    path = tmp_path / "hits.root"

    # ACT
    write_tree(data, path, OutputFileFormat.ROOT)

    # ASSERT
    with uproot.open(path) as output_file:
        written_tree = output_file["Hits"]
        assert written_tree.num_entries == 0
        assert list(written_tree.keys()) == list(data.branch_names)


def test_root_reports_data_without_branches(tmp_path: Path) -> None:
    """A tree with no branches cannot be expressed in ROOT."""
    # ARRANGE
    data = TreeData(GateTree.HITS, {})
    path = tmp_path / "hits.root"

    # ACT & ASSERT
    with pytest.raises(ExportError, match="without branches"):
        write_tree(data, path, OutputFileFormat.ROOT)


def test_root_output_holds_the_tree_only(tmp_path: Path) -> None:
    """Histograms from the input file must not appear in the output."""
    # ARRANGE
    path = tmp_path / "hits.root"

    # ACT
    write_tree(make_tree_data(), path, OutputFileFormat.ROOT)

    # ASSERT
    with uproot.open(path) as output_file:
        assert len(output_file.classnames()) == 1


@pytest.mark.parametrize("file_format", list(OutputFileFormat))
def test_gate_data_survives_a_write_for_every_format(
    file_format: OutputFileFormat,
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
    tmp_path: Path,
) -> None:
    """Every format should hold a real GATE tree without losing entries."""
    # ARRANGE
    data = read_tree(gate_hits_file, GateTree.HITS)
    path = tmp_path / f"hits.{EXTENSIONS[file_format]}"

    # ACT
    write_tree(data, path, file_format)

    # ASSERT
    assert path.is_file()
    if file_format is OutputFileFormat.ROOT:
        restored = read_tree(path, GateTree.HITS)
        assert restored.entry_count == gate_hits_layout.entries
        assert restored.branch_names == data.branch_names
        assert restored["volumeID"].shape == data["volumeID"].shape
    elif file_format is OutputFileFormat.HDF5:
        with h5py.File(path) as output_file:
            assert output_file["Hits"].attrs["entries"] == gate_hits_layout.entries
    else:
        assert len(pd.read_csv(path)) == gate_hits_layout.entries
