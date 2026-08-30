"""Unit tests for the command-line interface."""

import logging
from pathlib import Path

import pytest

from opengate_gate_tree import cli
from opengate_gate_tree.config import RunConfig
from opengate_gate_tree.errors import RootFileError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.tree.gatetree import GateTree


def build_arguments(
    input_file: Path,
    output_dir: Path,
    *,
    gate_tree: str = "Hits",
    output_file_format: str = "csv",
    title: str = "run_01",
    branches: list[str] | None = None,
) -> list[str]:
    """Build a command line for a single run."""
    arguments = [
        "--input-gate-root-file",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--output-file-title",
        title,
        "--gate-tree",
        gate_tree,
        "--output-file-format",
        output_file_format,
    ]
    if branches is not None:
        arguments += ["--branches-to-extract", *branches]
    return arguments


def test_build_parser_parses_all_supported_options(tmp_path: Path) -> None:
    """Parser should correctly parse all supported CLI options."""
    input_file = tmp_path / "input.root"
    output_dir = tmp_path / "output"

    parser = cli.build_parser()
    args = parser.parse_args(
        build_arguments(input_file, output_dir, branches=["eventID", "trackID"])
    )

    assert args.input_gate_root_file == input_file
    assert args.output_dir == output_dir
    assert args.output_file_title == "run_01"
    assert args.gate_tree == GateTree.HITS
    assert args.output_file_format == OutputFileFormat.CSV
    assert args.branches_to_extract == ["eventID", "trackID"]


def test_main_writes_the_output_file(gate_hits_file: Path, tmp_path: Path) -> None:
    """A valid run should report success and leave the output file behind."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    exit_code = cli.main(build_arguments(gate_hits_file, output_dir))

    # ASSERT
    assert exit_code == 0
    assert (output_dir / "run_01.csv").is_file()


def test_main_accepts_a_branch_selection(gate_hits_file: Path, tmp_path: Path) -> None:
    """Requested branches should be honoured rather than rejected."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    exit_code = cli.main(build_arguments(gate_hits_file, output_dir, branches=["eventID", "edep"]))

    # ASSERT
    assert exit_code == 0
    assert (output_dir / "run_01.csv").is_file()


def test_main_returns_error_when_input_file_is_missing(tmp_path: Path) -> None:
    """CLI should return 1 when the input ROOT file does not exist."""
    # ARRANGE
    missing_input_file = tmp_path / "missing.root"

    # ACT
    exit_code = cli.main(build_arguments(missing_input_file, tmp_path / "output"))

    # ASSERT
    assert exit_code == 1


def test_main_returns_error_when_input_extension_is_not_root(tmp_path: Path) -> None:
    """CLI should return 1 when the input file is not named as a ROOT file."""
    # ARRANGE
    wrong_extension_file = tmp_path / "input.txt"
    wrong_extension_file.write_text("", encoding="utf-8")

    # ACT
    exit_code = cli.main(build_arguments(wrong_extension_file, tmp_path / "output"))

    # ASSERT
    assert exit_code == 1


def test_main_returns_error_for_a_missing_tree(
    gate_hits_file: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tree absent from the input file should stop the run and say what is there."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(build_arguments(gate_hits_file, output_dir, gate_tree="Singles"))

    # ASSERT
    assert exit_code == 1
    assert not output_dir.exists()
    assert "Singles" in caplog.text
    assert "Hits" in caplog.text


def test_main_returns_error_for_an_empty_branch_name(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """A branch name that carries no value should stop the run."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    exit_code = cli.main(build_arguments(gate_hits_file, output_dir, branches=["eventID", "   "]))

    # ASSERT
    assert exit_code == 1


def test_main_returns_error_for_an_empty_output_title(
    gate_hits_file: Path,
    tmp_path: Path,
) -> None:
    """An output file title that carries no value should stop the run."""
    # ARRANGE
    output_dir = tmp_path / "output"

    # ACT
    exit_code = cli.main(build_arguments(gate_hits_file, output_dir, title="   "))

    # ASSERT
    assert exit_code == 1


@pytest.mark.parametrize(
    "omitted_option",
    [
        "--input-gate-root-file",
        "--output-dir",
        "--output-file-title",
        "--gate-tree",
        "--output-file-format",
    ],
)
def test_main_returns_error_when_a_required_option_is_omitted(
    omitted_option: str,
    gate_hits_file: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every required option should be reported by name when it is missing."""
    # ARRANGE
    arguments = build_arguments(gate_hits_file, tmp_path / "output")
    position = arguments.index(omitted_option)
    del arguments[position : position + 2]

    # ACT
    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(arguments)

    # ASSERT
    assert exit_code == 1
    assert "is required" in caplog.text


def test_main_rejects_an_unknown_gate_tree(gate_hits_file: Path, tmp_path: Path) -> None:
    """An unparsable option value should stop the parser, not the pipeline."""
    # ARRANGE
    arguments = build_arguments(gate_hits_file, tmp_path / "output", gate_tree="NotATree")

    # ACT & ASSERT
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)

    assert exit_info.value.code == 2


def test_main_returns_error_when_run_raises_package_error(
    gate_hits_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI should report package errors as an error exit code instead of crashing."""
    # ARRANGE
    expected_message = "input file is not a valid ROOT file"

    def raise_root_file_error(config: RunConfig) -> Path:
        raise RootFileError(expected_message)

    monkeypatch.setattr(cli, "_run", raise_root_file_error)

    # ACT
    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(build_arguments(gate_hits_file, tmp_path / "output"))

    # ASSERT
    assert exit_code == 1
    assert expected_message in caplog.text
