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
from pathlib import Path, PurePath
from typing import Final, NoReturn

from .config import RunConfig
from .errors import AmbiguousTreeError, GateTreeError
from .io.fileformat import OutputFileFormat, parse_output_file_format
from .io.naming import build_output_file_path, build_statistics_file_path
from .io.rootfile import RootFile
from .io.statistics import write_statistics
from .io.writers import write_tree
from .logger import log
from .logging_setup import configure_logging
from .tree.branch import validate_branch_selection
from .tree.gatetree import GateTree, parse_gate_tree
from .tree.hits.detection import HitsTreeDetection, summarise_hits_tree
from .tree.statistics import compute_statistics, format_statistics
from .tree.treedata import TreeData

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
    parser.add_argument(
        "--input-tree-name",
        type=str,
        help=(
            "Name of the tree in the input file, when it differs from the standard one "
            "or when the file holds several trees of hits"
        ),
    )
    parser.add_argument(
        "--merge-hits-trees",
        action="store_true",
        help="Read every tree of hits in the input file as a single dataset",
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Write a report describing the extracted data next to the output file",
    )
    parser.add_argument(
        "--skip-hits-validation",
        action="store_true",
        help=(
            "Extract the branches without recognising and checking the structure of the 'Hits' tree"
        ),
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
            input_tree_name=args.input_tree_name,
            merge_hits_trees=args.merge_hits_trees,
            write_statistics=args.statistics,
            skip_hits_validation=args.skip_hits_validation,
        )
        logger.info("Run configuration: %s", run_config)
        output_path = _run(run_config)
    except AmbiguousTreeError as e:
        logger.error("An error occurred: %s", e)
        logger.error(
            "Use --input-tree-name to read one of them, or --merge-hits-trees to read them all."
        )
        return 1  # Error
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
        If the configuration is incomplete, a requested branch name is empty,
        or two options contradict each other.
    RootFileError
        If the input file is not a readable ROOT file.
    TreeNotFoundError
        If the requested tree is not present in the input file.
    AmbiguousTreeError
        If several trees hold hits and none of them was named.
    UnknownHitsVariantError
        If the structure of the "Hits" tree is not a supported one.
    HitsTreeValidationError
        If the "Hits" tree does not match the structure it was recognised as.
    BranchNotFoundError
        If a requested branch is not present in the tree.
    UnsupportedBranchTypeError
        If a requested branch uses an unsupported type.
    TreeMergeError
        If the trees of hits cannot be read as one dataset.
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
    detection, data = _extract(config)

    if detection is not None:
        logger.info("%s", summarise_hits_tree(detection))
    logger.info(
        "Extracted %d entries and %d branches from tree '%s'.",
        data.entry_count,
        len(data.branch_names),
        config.gate_tree.value,
    )

    output_file_path = build_output_file_path(
        config.output_dir,
        config.output_file_title,
        config.gate_tree,
        config.output_file_format,
    )
    written = write_tree(data, output_file_path, config.output_file_format)

    if config.write_statistics:
        _report(config, data, detection)

    return written


def _extract(config: RunConfig) -> tuple[HitsTreeDetection | None, TreeData]:
    """Read the requested data, and the structure it was recognised as."""
    assert config.input_gate_root_file is not None
    assert config.gate_tree is not None

    validate = not config.skip_hits_validation
    with RootFile(config.input_gate_root_file) as root_file:
        detection = _recognise(root_file, config)
        if config.merge_hits_trees:
            data = root_file.read_hits(config.branches_to_extract, validate=validate)
        else:
            data = root_file.read(
                config.gate_tree,
                config.branches_to_extract,
                tree_name=config.input_tree_name,
                validate=validate,
            )
    return detection, data


def _recognise(root_file: RootFile, config: RunConfig) -> HitsTreeDetection | None:
    """Return the structure of the tree about to be read, when it is known.

    Nothing is recognised for a tree the package describes no structure for,
    or when the check was turned off. While the trees of a file are merged,
    the first of them stands for the structure they share.
    """
    if config.gate_tree is not GateTree.HITS or config.skip_hits_validation:
        return None

    tree_name = config.input_tree_name
    if config.merge_hits_trees and tree_name is None:
        names = root_file.hits_tree_names()
        tree_name = names[0] if names else None
    return root_file.detect_hits_tree(tree_name)


def _report(
    config: RunConfig,
    data: TreeData,
    detection: HitsTreeDetection | None,
) -> None:
    """Summarise the extracted data, in the log and in a file of its own."""
    assert config.output_dir is not None
    assert config.output_file_title is not None
    assert config.gate_tree is not None

    statistics = compute_statistics(data, detection)
    log().info("%s", format_statistics(statistics))
    report_path = build_statistics_file_path(
        config.output_dir,
        config.output_file_title,
        config.gate_tree,
    )
    write_statistics(statistics, report_path)
    log().info("Statistics saved to file: %s", report_path)


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
        If a required option is missing, the output file title is empty, a
        requested branch name is empty, or two options contradict each other.
    """
    if config.input_gate_root_file is None:
        raise ValueError("Input gate root file path is required.")
    if config.output_dir is None:
        raise ValueError("Output directory path is required.")
    if config.output_file_title is None:
        raise ValueError("Output file title is required.")
    if config.output_file_title.strip() == "":
        raise ValueError("Output file title cannot be empty.")
    # The title names a file inside the output directory. A separator or a
    # drive would make the path leave that directory, and since an existing
    # output file is overwritten without asking, it could replace a file the
    # caller never meant to touch.
    title_path = PurePath(config.output_file_title)
    if len(title_path.parts) != 1 or title_path.is_absolute():
        raise ValueError(
            f"Output file title must be a file name without directory components, "
            f"got: {config.output_file_title!r}"
        )
    if config.gate_tree is None:
        raise ValueError("Gate tree structure is required.")
    if config.output_file_format is None:
        raise ValueError("Output file format is required.")

    validate_branch_selection(config.branches_to_extract)

    if config.merge_hits_trees and config.input_tree_name is not None:
        raise ValueError(
            "Naming one tree and merging every tree of hits are two different runs; "
            "use either --input-tree-name or --merge-hits-trees."
        )
    if config.merge_hits_trees and config.gate_tree is not GateTree.HITS:
        raise ValueError(
            f"Merging is available for the '{GateTree.HITS.value}' tree only, "
            f"and '{config.gate_tree.value}' was requested."
        )
