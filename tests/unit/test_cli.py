"""Unit tests for the command-line interface."""

import logging
from pathlib import Path

import pytest

from opengate_gate_tree import cli
from opengate_gate_tree.config import RunConfig
from opengate_gate_tree.errors import RootFileError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.tree.gatetree import GateTree


def test_build_parser_parses_all_supported_options(tmp_path: Path) -> None:
    """Parser should correctly parse all supported CLI options."""
    input_file = tmp_path / "input.root"
    output_dir = tmp_path / "output"

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--input-gate-root-file",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--output-file-title",
            "run_01",
            "--gate-tree",
            "Hits",
            "--output-file-format",
            "csv",
            "--branches-to-extract",
            "eventID",
            "trackID",
        ]
    )

    assert args.input_gate_root_file == input_file
    assert args.output_dir == output_dir
    assert args.output_file_title == "run_01"
    assert args.gate_tree == GateTree.HITS
    assert args.output_file_format == OutputFileFormat.CSV
    assert args.branches_to_extract == ["eventID", "trackID"]


def test_main_returns_success_for_valid_required_arguments(tmp_path: Path) -> None:
    """CLI should return 0 for a valid set of required arguments."""
    input_file = tmp_path / "input.root"
    input_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "output"

    exit_code = cli.main(
        [
            "--input-gate-root-file",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--output-file-title",
            "run_01",
            "--gate-tree",
            "Singles",
            "--output-file-format",
            "hdf5",
        ]
    )

    assert exit_code == 0
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_main_returns_error_when_input_file_is_missing(tmp_path: Path) -> None:
    """CLI should return 1 when input ROOT file does not exist."""
    missing_input_file = tmp_path / "missing.root"
    output_dir = tmp_path / "output"

    exit_code = cli.main(
        [
            "--input-gate-root-file",
            str(missing_input_file),
            "--output-dir",
            str(output_dir),
            "--output-file-title",
            "run_01",
            "--gate-tree",
            "Coincidences",
            "--output-file-format",
            "root",
        ]
    )

    assert exit_code == 1


def test_main_returns_error_when_input_extension_is_not_root(tmp_path: Path) -> None:
    """CLI should return 1 when input file extension is not .root."""
    wrong_extension_file = tmp_path / "input.txt"
    wrong_extension_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "output"

    exit_code = cli.main(
        [
            "--input-gate-root-file",
            str(wrong_extension_file),
            "--output-dir",
            str(output_dir),
            "--output-file-title",
            "run_01",
            "--gate-tree",
            "Hits",
            "--output-file-format",
            "csv",
        ]
    )

    assert exit_code == 1


def test_main_accepts_a_branch_selection(tmp_path: Path) -> None:
    """Requested branches should be accepted; the file decides whether they exist."""
    # ARRANGE
    input_file = tmp_path / "input.root"
    input_file.write_text("", encoding="utf-8")

    # ACT
    exit_code = cli.main(
        [
            "--input-gate-root-file",
            str(input_file),
            "--output-dir",
            str(tmp_path / "output"),
            "--output-file-title",
            "run_01",
            "--gate-tree",
            "Hits",
            "--output-file-format",
            "csv",
            "--branches-to-extract",
            "eventID",
            "edep",
        ]
    )

    # ASSERT
    assert exit_code == 0


def test_main_returns_error_for_an_empty_branch_name(tmp_path: Path) -> None:
    """A branch name that carries no value should stop the run."""
    # ARRANGE
    input_file = tmp_path / "input.root"
    input_file.write_text("", encoding="utf-8")

    # ACT
    exit_code = cli.main(
        [
            "--input-gate-root-file",
            str(input_file),
            "--output-dir",
            str(tmp_path / "output"),
            "--output-file-title",
            "run_01",
            "--gate-tree",
            "Hits",
            "--output-file-format",
            "csv",
            "--branches-to-extract",
            "eventID",
            "   ",
        ]
    )

    # ASSERT
    assert exit_code == 1


def test_main_returns_error_when_run_raises_package_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI should report package errors as an error exit code instead of crashing."""
    # ARRANGE
    input_file = tmp_path / "input.root"
    input_file.write_text("", encoding="utf-8")
    expected_message = "input file is not a valid ROOT file"

    def raise_root_file_error(config: RunConfig) -> Path:
        raise RootFileError(expected_message)

    monkeypatch.setattr(cli, "_run", raise_root_file_error)

    # ACT
    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(
            [
                "--input-gate-root-file",
                str(input_file),
                "--output-dir",
                str(tmp_path / "output"),
                "--output-file-title",
                "run_01",
                "--gate-tree",
                "Hits",
                "--output-file-format",
                "csv",
            ]
        )

    # ASSERT
    assert exit_code == 1
    assert expected_message in caplog.text
