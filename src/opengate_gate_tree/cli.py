"""Package command-line interface.

The module defines the entry point used by ``python -m
opengate_gate_tree`` and the ``opengate-gate-tree``
console script.

Public functions:

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
from .errors import GateTreeError
from .io.fileformat import OutputFileFormat, parse_output_file_format
from .io.reader import read_tree
from .io.writers import write_tree
from .logger import log
from .logging_setup import configure_logging
from .tree.branch import validate_branch_selection
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
    except (ValueError, GateTreeError) as e:
        logger.error("An error occurred: %s", e)
        return 1  # Error

    logger.info("Done. Output saved to file: %s", output_path)
    return 0


def _run(config: RunConfig) -> Path:
    """Extract the requested tree and export it, returning the output path.

    The function is a thin sequence of calls to the package API, so that
    everything the command line does is also reachable from code using the
    package as a library.

    Raises
    ------
    ValueError
        If the configuration is incomplete or a requested branch name is
        empty.
    RootFileError
        If the input file is not a readable ROOT file.
    TreeNotFoundError
        If the requested tree is not present in the input file.
    BranchNotFoundError
        If a requested branch is not present in the tree.
    UnsupportedBranchTypeError
        If a requested branch uses an unsupported type.
    ExportError
        If the output file cannot be written.
    """
    _validate_config(config)

    assert config.input_gate_root_file is not None
    assert config.output_dir is not None
    assert config.output_file_title is not None
    assert config.gate_tree is not None
    assert config.output_file_format is not None

    logger = log()

    data = read_tree(
        config.input_gate_root_file,
        config.gate_tree,
        config.branches_to_extract,
    )
    logger.info(
        "Extracted %d entries and %d branches from tree '%s'.",
        data.entry_count,
        len(data.branch_names),
        config.gate_tree.value,
    )

    output_file_path = (
        config.output_dir / f"{config.output_file_title}.{config.output_file_format.value}"
    )
    return write_tree(data, output_file_path, config.output_file_format)


def _validate_config(config: RunConfig) -> None:
    """Check that the run configuration is complete.

    Only what the command line itself owns is checked here: that every
    required option was given, and that the values which do not need the input
    file make sense. Whether the input file is readable, the tree exists, the
    requested branches exist, or the output directory can be created is
    decided by the package API and reported from there, so that the library
    and the command line behave the same way.

    Raises
    ------
    ValueError
        If a required option is missing, the output file title is empty, or a
        requested branch name is empty.
    """
    if config.input_gate_root_file is None:
        raise ValueError("Input gate root file path is required.")
    if config.output_dir is None:
        raise ValueError("Output directory path is required.")
    if config.output_file_title is None:
        raise ValueError("Output file title is required.")
    if config.output_file_title.strip() == "":
        raise ValueError("Output file title cannot be empty.")
    if config.gate_tree is None:
        raise ValueError("Gate tree structure is required.")
    if config.output_file_format is None:
        raise ValueError("Output file format is required.")

    validate_branch_selection(config.branches_to_extract)
