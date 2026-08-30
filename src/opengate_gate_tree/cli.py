"""Package command-line interface.

The module defines the entry point used by ``python -m
opengate_gate_tree`` and the ``opengate-gate-tree``
console script.

Public functions
----------------
build_parser() -> argparse.ArgumentParser
    Builds the command-line argument parser.
main(argv: list[str] | None = None) -> int
    Parses arguments, runs the processing pipeline, and returns an exit code.
"""

import argparse
import sys
from pathlib import Path
from typing import Final, NoReturn

from .config import RunConfig
from .io.fileformat import OutputFileFormat, parse_output_file_format
from .logger import log
from .logging_setup import configure_logging
from .tree.branch import validate_branch_name
from .tree.gatetree import GateTree, parse_gate_tree

# Program name displayed in help output.
PROG_NAME: Final[str] = "opengate-gate-tree"


class _ArgumentParser(argparse.ArgumentParser):
    """Argument parser that emits error messages."""

    def error(self, message: str) -> NoReturn:
        """Print usage and error message, then exit with status 2."""
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all tool flags.
    """
    parser = _ArgumentParser(
        prog=PROG_NAME,
        description=(""),
    )

    parser.add_argument(
        "--input-gate-root-file",
        type=Path,
        help="Path to the gate '*.root' file to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the directory where the output files will be saved",
    )
    parser.add_argument(
        "--output-file-title",
        type=str,
        help="Title of the output file (without extension)",
    )
    parser.add_argument(
        "--gate-tree",
        type=parse_gate_tree,
        choices=list(GateTree),
        help="Gate tree structure loaded from the input file",
    )
    parser.add_argument(
        "--output-file-format",
        type=parse_output_file_format,
        choices=list(OutputFileFormat),
        help="Format of the output file",
    )
    parser.add_argument(
        "--branches-to-extract",
        type=str,
        nargs="+",
        help="List of branch names to extract from the gate tree",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tool from the command line.

    Parameters
    ----------
    argv : list[str] | None
        List of arguments to parse. If ``None``, arguments from ``sys.argv``
        are used.

    Returns
    -------
    int
        Process exit code (0 means success, 1 means processing error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging()
    logger = log()

    try:
        run_config = RunConfig(
            input_gate_root_file=args.input_gate_root_file,
            output_dir=args.output_dir,
            output_file_title=args.output_file_title,
            gate_tree=args.gate_tree,
            output_file_format=args.output_file_format,
            branches_to_extract=args.branches_to_extract or [],
        )
        logger.info("Run configuration: %s", run_config)
        output_path = _run(run_config)
    except ValueError as e:
        logger.error("An error occurred: %s", e)
        return 1  # Error

    if output_path.is_file():
        logger.info("Done. Output saved to file: %s", output_path)
    else:
        logger.info("Done. Output saved to directory: %s", output_path)
    return 0


def _run(config: RunConfig) -> Path:
    """Run program according to ``config`` and return the output path.

    Raises
    ------
    ValueError
        If the configuration is invalid (wrong flag combination, missing
        required paths, or a macro file referenced by the input is
        missing).
    """
    _validate_config(config)

    assert config.output_dir is not None
    assert config.output_file_title is not None
    assert config.output_file_format is not None

    output_file_path = (
        config.output_dir / f"{config.output_file_title}.{config.output_file_format.value}"
    )
    # Below, program logic will be added to process the input file
    # and generate output based on the configuration.
    return output_file_path


def _validate_config(config: RunConfig) -> None:
    """Validate the run configuration.

    Raises
    ------
    ValueError
        If the configuration is invalid (wrong flag combination, missing
        required paths, or a macro file referenced by the input is
        missing).
    """
    logger = log()

    if config.input_gate_root_file is None:
        raise ValueError("Input gate root file path is required.")
    if config.input_gate_root_file.suffix != ".root":
        raise ValueError("Input gate root file must have a '.root' extension.")
    if not (config.input_gate_root_file.exists() and config.input_gate_root_file.is_file()):
        raise ValueError(f"Input gate root file does not exist: {config.input_gate_root_file}")
    if config.output_dir is None:
        raise ValueError("Output directory path is required.")
    if not config.output_dir.exists():
        config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", config.output_dir)
    if not config.output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {config.output_dir}")
    if config.output_file_title is None:
        raise ValueError("Output file title is required.")
    if config.output_file_title.strip() == "":
        raise ValueError("Output file title cannot be empty.")
    if config.gate_tree is None:
        raise ValueError("Gate tree structure is required.")
    if config.gate_tree not in GateTree:
        raise ValueError(f"Invalid gate tree structure: {config.gate_tree}")
    if config.output_file_format is None:
        raise ValueError("Output file format is required.")
    if config.output_file_format not in OutputFileFormat:
        raise ValueError(f"Invalid output file format: {config.output_file_format}")
    correct_branches, invalid_branches = validate_branch_name(
        config.branches_to_extract,
        config.gate_tree,
    )
    if not correct_branches:
        raise ValueError(
            f"Invalid branch names for gate tree {config.gate_tree.value}: {invalid_branches}"
        )
