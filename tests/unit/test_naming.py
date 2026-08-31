"""Unit tests for the names of the files a run writes."""

from pathlib import Path

import pytest

from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.naming import (
    build_output_file_name,
    build_output_file_path,
    build_statistics_file_path,
)
from opengate_gate_tree.tree.gatetree import GateTree


@pytest.mark.parametrize(
    ("file_format", "expected"),
    [
        (OutputFileFormat.CSV, "patient_01.hits.csv"),
        (OutputFileFormat.HDF5, "patient_01.hits.hdf5"),
        (OutputFileFormat.ROOT, "patient_01.hits.root"),
    ],
)
def test_the_name_states_the_title_the_tree_and_the_format(
    file_format: OutputFileFormat,
    expected: str,
) -> None:
    """The name says what the file holds and how it is written."""
    # ARRANGE
    # No additional setup required.

    # ACT
    name = build_output_file_name("patient_01", GateTree.HITS, file_format)

    # ASSERT
    assert name == expected


def test_the_tree_keeps_the_names_of_two_runs_apart() -> None:
    """One input file holds several trees, and each extraction is its own file."""
    # ARRANGE
    # No additional setup required.

    # ACT
    hits = build_output_file_name("patient_01", GateTree.HITS, OutputFileFormat.CSV)
    singles = build_output_file_name("patient_01", GateTree.SINGLES, OutputFileFormat.CSV)

    # ASSERT
    assert hits != singles
    assert singles == "patient_01.singles.csv"


def test_a_title_holding_dots_is_left_alone() -> None:
    """A title is what the caller chose, not something to be parsed."""
    # ARRANGE
    # A date in the title is a common way to tell runs apart.

    # ACT
    name = build_output_file_name("run.2026-08-31", GateTree.HITS, OutputFileFormat.CSV)

    # ASSERT
    assert name == "run.2026-08-31.hits.csv"


def test_the_path_lands_in_the_requested_directory() -> None:
    """The directory comes from the caller, the name from the rule."""
    # ARRANGE
    directory = Path("out") / "analysis"

    # ACT
    path = build_output_file_path(directory, "patient_01", GateTree.HITS, OutputFileFormat.HDF5)

    # ASSERT
    assert path == directory / "patient_01.hits.hdf5"


def test_the_report_sits_next_to_the_data_under_the_same_title() -> None:
    """A report is found by the name of the run it describes."""
    # ARRANGE
    directory = Path("out")

    # ACT
    report = build_statistics_file_path(directory, "patient_01", GateTree.HITS)
    data = build_output_file_path(directory, "patient_01", GateTree.HITS, OutputFileFormat.CSV)

    # ASSERT
    assert report == directory / "patient_01.hits.stats.json"
    assert report.parent == data.parent
