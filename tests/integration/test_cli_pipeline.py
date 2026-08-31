"""End-to-end tests running the command line against a real GATE file."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import uproot
from conftest import GateHitsLayout

from opengate_gate_tree import cli
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.gatetree import GateTree

# File name extension produced for each output format.
EXTENSIONS = {
    OutputFileFormat.CSV: "csv",
    OutputFileFormat.HDF5: "hdf5",
    OutputFileFormat.ROOT: "root",
}


def run_cli(
    input_file: Path,
    output_dir: Path,
    output_file_format: OutputFileFormat,
    *,
    title: str = "patient_01",
    branches: list[str] | None = None,
) -> int:
    """Run the command line once and return its exit code."""
    arguments = [
        "--input-gate-root-file",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--output-file-title",
        title,
        "--gate-tree",
        "Hits",
        "--output-file-format",
        output_file_format.value,
    ]
    if branches is not None:
        arguments += ["--branches-to-extract", *branches]
    return cli.main(arguments)


@pytest.mark.parametrize("output_file_format", list(OutputFileFormat))
def test_pipeline_writes_every_entry_for_each_format(
    output_file_format: OutputFileFormat,
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
    tmp_path: Path,
) -> None:
    """A full run should carry every entry of the tree into the output file."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    exit_code = run_cli(gate_hits_file, output_dir, output_file_format)

    # ASSERT
    output_file = output_dir / f"patient_01.hits.{EXTENSIONS[output_file_format]}"
    assert exit_code == 0
    assert output_file.is_file()

    if output_file_format is OutputFileFormat.ROOT:
        restored = read_tree(output_file, GateTree.HITS)
        assert restored.entry_count == gate_hits_layout.entries
        assert len(restored.branch_names) == gate_hits_layout.branch_count
    elif output_file_format is OutputFileFormat.HDF5:
        with h5py.File(output_file) as written:
            assert written["Hits"].attrs["entries"] == gate_hits_layout.entries
    else:
        assert len(pd.read_csv(output_file)) == gate_hits_layout.entries


@pytest.mark.parametrize("output_file_format", list(OutputFileFormat))
def test_pipeline_honours_a_branch_selection(
    output_file_format: OutputFileFormat,
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """Only the requested branches should reach the output file."""
    # ARRANGE
    output_dir = tmp_path / "output"
    requested = ["eventID", "edep", "posX", "posY", "posZ"]

    # ACT
    exit_code = run_cli(gate_hits_file, output_dir, output_file_format, branches=requested)

    # ASSERT
    output_file = output_dir / f"patient_01.hits.{EXTENSIONS[output_file_format]}"
    assert exit_code == 0

    if output_file_format is OutputFileFormat.ROOT:
        # A file holding a selection is no longer a whole hits structure, so
        # it is read back without the structure check.
        restored_names = read_tree(output_file, GateTree.HITS, validate=False).branch_names
        assert list(restored_names) == requested
    elif output_file_format is OutputFileFormat.HDF5:
        with h5py.File(output_file) as written:
            assert list(written["Hits"]) == requested
    else:
        assert list(pd.read_csv(output_file).columns) == requested


@pytest.mark.parametrize("output_file_format", list(OutputFileFormat))
def test_pipeline_handles_the_array_branch(
    output_file_format: OutputFileFormat,
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
    tmp_path: Path,
) -> None:
    """The volumeID branch should reach every format in its natural shape."""
    # ARRANGE
    output_dir = tmp_path / "output"
    width = gate_hits_layout.volume_id_width

    # ACT
    exit_code = run_cli(
        gate_hits_file,
        output_dir,
        output_file_format,
        branches=["eventID", "volumeID"],
    )

    # ASSERT
    output_file = output_dir / f"patient_01.hits.{EXTENSIONS[output_file_format]}"
    assert exit_code == 0

    if output_file_format is OutputFileFormat.ROOT:
        restored = read_tree(output_file, GateTree.HITS, validate=False)
        assert restored["volumeID"].shape == (gate_hits_layout.entries, width)
    elif output_file_format is OutputFileFormat.HDF5:
        with h5py.File(output_file) as written:
            assert written["Hits"]["volumeID"].shape == (gate_hits_layout.entries, width)
    else:
        columns = list(pd.read_csv(output_file).columns)
        assert columns == ["eventID", *[f"volumeID_{index}" for index in range(width)]]


def test_pipeline_preserves_values_through_root(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """A ROOT export should be readable again with the same values."""
    # ARRANGE
    output_dir = tmp_path / "output"
    source = read_tree(gate_hits_file, GateTree.HITS)

    # ACT
    exit_code = run_cli(gate_hits_file, output_dir, OutputFileFormat.ROOT)

    # ASSERT
    assert exit_code == 0
    restored = read_tree(output_dir / "patient_01.hits.root", GateTree.HITS)
    assert restored.branch_names == source.branch_names
    for name in source.branch_names:
        if source.dtypes[name].kind == "O":
            assert list(restored[name]) == list(source[name])
        else:
            assert np.array_equal(restored[name], source[name])


def test_pipeline_writes_a_ttree_not_an_rntuple(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """The ROOT output must be readable by analysis code expecting a TTree."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    run_cli(gate_hits_file, output_dir, OutputFileFormat.ROOT)

    # ASSERT
    with uproot.open(output_dir / "patient_01.hits.root") as written:
        assert list(written.classnames().values()) == ["TTree"]


def test_pipeline_does_not_copy_histograms(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """Histograms stored beside the tree should not reach the output file."""
    # ARRANGE
    output_dir = tmp_path / "output"
    with uproot.open(gate_hits_file) as source:
        assert "latest_event_ID;1" in source.classnames()

    # ACT
    run_cli(gate_hits_file, output_dir, OutputFileFormat.ROOT)

    # ASSERT
    with uproot.open(output_dir / "patient_01.hits.root") as written:
        assert list(written.keys(cycle=False)) == ["Hits"]


def test_pipeline_creates_a_missing_output_directory(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """A nested output directory should be created rather than reported."""
    # ARRANGE
    output_dir = tmp_path / "results" / "run" / "output"

    # ACT
    exit_code = run_cli(gate_hits_file, output_dir, OutputFileFormat.CSV)

    # ASSERT
    assert exit_code == 0
    assert (output_dir / "patient_01.hits.csv").is_file()


def test_pipeline_overwrites_an_existing_output_file(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """Running twice should replace the previous output without complaint."""
    # ARRANGE
    output_dir = tmp_path / "output"
    output_file = output_dir / "patient_01.hits.csv"
    run_cli(gate_hits_file, output_dir, OutputFileFormat.CSV, branches=["eventID"])
    first_size = output_file.stat().st_size

    # ACT
    exit_code = run_cli(gate_hits_file, output_dir, OutputFileFormat.CSV)

    # ASSERT
    assert exit_code == 0
    assert output_file.stat().st_size != first_size


def test_pipeline_reports_progress_on_the_console(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """A real run must tell the user what happened.

    The package attaches a NullHandler on import so library use stays silent.
    This test runs the console script in a separate process, because caplog
    captures records through the root logger and would pass even if the
    package logger had no output handler at all.
    """
    # ARRANGE
    output_dir = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "opengate_gate_tree",
        "--input-gate-root-file",
        str(gate_hits_file),
        "--output-dir",
        str(output_dir),
        "--output-file-title",
        "patient_01",
        "--gate-tree",
        "Hits",
        "--output-file-format",
        "csv",
    ]

    # ACT
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # ASSERT
    assert result.returncode == 0
    assert "Extracted 2000 entries" in result.stderr
    assert "Done. Output saved to file" in result.stderr


def test_pipeline_reports_failures_on_the_console(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """A failing run must say why, not just return a non-zero exit code."""
    # ARRANGE
    command = [
        sys.executable,
        "-m",
        "opengate_gate_tree",
        "--input-gate-root-file",
        str(gate_hits_file),
        "--output-dir",
        str(tmp_path / "output"),
        "--output-file-title",
        "patient_01",
        "--gate-tree",
        "Singles",
        "--output-file-format",
        "csv",
    ]

    # ACT
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # ASSERT
    assert result.returncode == 1
    assert "Singles" in result.stderr
    assert "OpticalData" in result.stderr


@pytest.mark.parametrize("title", ["../escaped", "/tmp/absolute", "nested/name"])
def test_pipeline_rejects_a_title_with_directory_components(
    title: str,
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """A title naming another directory would write outside the output directory."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    exit_code = run_cli(gate_hits_file, output_dir, OutputFileFormat.CSV, title=title)

    # ASSERT
    assert exit_code == 1
    assert not output_dir.exists()


def test_pipeline_reports_an_unknown_branch(
    gate_hits_file: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown branch should stop the run and name what is available."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    exit_code = run_cli(
        gate_hits_file,
        output_dir,
        OutputFileFormat.CSV,
        branches=["eventID", "notInTheTree"],
    )

    # ASSERT
    assert exit_code == 1
    assert "notInTheTree" in caplog.text
    assert not output_dir.exists()
