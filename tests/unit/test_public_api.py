"""Unit tests for the public library API.

The names exported here are what code using the package as a library depends
on, so the tests state the contract explicitly: adding to it is a decision,
and removing from it is a breaking change.
"""

import importlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest

import opengate_gate_tree
from opengate_gate_tree.logging_setup import LOGGER_NAME

# Names the package promises to export.
EXPECTED_EXPORTS = {
    "ACCESSOR_NAME",
    "AmbiguousTreeError",
    "BranchKind",
    "BranchNotFoundError",
    "BranchSpec",
    "BranchStatistics",
    "DECAY_INDEX_BRANCH",
    "DecayType",
    "EVENT_COLUMN",
    "ExportError",
    "GammaType",
    "GateSystemType",
    "GateTree",
    "GateTreeError",
    "HitsSummary",
    "HitsTreeDetection",
    "HitsTreeValidationError",
    "HitsTreeVariant",
    "InclusiveSide",
    "NO_POSITRONIUM_METADATA",
    "OutputFileFormat",
    "POSITION_COLUMNS",
    "POSITRONIUM_BRANCHES",
    "RUN_COLUMN",
    "RootFile",
    "RootFileError",
    "SOURCE_TREE_BRANCH",
    "SourceType",
    "TreeData",
    "TreeMergeError",
    "TreeNotFoundError",
    "TreeStatistics",
    "UnknownHitsVariantError",
    "UnsupportedBranchTypeError",
    "__version__",
    "build_output_file_name",
    "build_output_file_path",
    "build_statistics_file_path",
    "by_event",
    "by_run",
    "compute_statistics",
    "decode_positronium_column",
    "decode_positronium_value",
    "describe_hits_tree",
    "detect_hits_variant",
    "expected_branches",
    "format_statistics",
    "has_decay_metadata",
    "has_positronium_metadata",
    "in_box",
    "in_cylinder",
    "in_range",
    "in_sphere",
    "is_decay_type",
    "is_from_event",
    "is_from_run",
    "is_gamma_type",
    "is_in_box",
    "is_in_cylinder",
    "is_in_range",
    "is_in_sphere",
    "is_process",
    "is_source_type",
    "merge_tree_data",
    "parse_gate_tree",
    "parse_output_file_format",
    "positronium_enum",
    "read_hits_trees",
    "read_tree",
    "select_by_decay_type",
    "select_by_gamma_type",
    "select_by_process",
    "select_by_source_type",
    "statistics_to_dict",
    "summarise_hits_tree",
    "supported_variants",
    "validate_hits_tree",
    "variant_reference",
    "with_decay_metadata",
    "write_statistics",
    "write_tree",
}

# Directory holding the package sources.
SOURCE_DIR = Path(opengate_gate_tree.__file__).parent


def test_public_api_matches_the_declared_contract() -> None:
    """The exported names should be exactly the ones promised."""
    # ARRANGE
    # No additional setup required.

    # ACT
    exported = set(opengate_gate_tree.__all__)

    # ASSERT
    assert exported == EXPECTED_EXPORTS


@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
def test_every_exported_name_is_importable(name: str) -> None:
    """Every promised name should resolve on the package itself."""
    # ARRANGE
    # No additional setup required.

    # ACT
    attribute = getattr(opengate_gate_tree, name, None)

    # ASSERT
    assert attribute is not None


def test_errors_are_reachable_from_the_package_root() -> None:
    """Library users need the exception hierarchy to catch failures."""
    # ARRANGE
    # No additional setup required.

    # ACT
    base_error = opengate_gate_tree.GateTreeError

    # ASSERT
    assert issubclass(opengate_gate_tree.TreeNotFoundError, base_error)
    assert issubclass(opengate_gate_tree.ExportError, base_error)


def test_package_logger_has_a_null_handler() -> None:
    """A library must not write to stderr when the application stays silent."""
    # ARRANGE
    package_logger = logging.getLogger(LOGGER_NAME)

    # ACT
    handler_types = [type(handler) for handler in package_logger.handlers]

    # ASSERT
    assert logging.NullHandler in handler_types


def test_importing_the_package_leaves_logging_alone() -> None:
    """Importing a library should not reconfigure the application's logging."""
    # ARRANGE
    program = (
        "import logging;"
        "before = (list(logging.root.handlers), logging.root.level);"
        "import opengate_gate_tree;"
        "after = (list(logging.root.handlers), logging.root.level);"
        "print(before == after)"
    )

    # ACT
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    # ASSERT
    assert result.stdout.strip() == "True"


def test_importing_the_package_writes_nothing() -> None:
    """Importing a library should produce no output of its own."""
    # ARRANGE
    program = "import opengate_gate_tree"

    # ACT
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    # ASSERT
    assert result.stdout == ""
    assert result.stderr == ""


def test_configuring_logging_after_the_import_still_produces_output() -> None:
    """The NullHandler must not stop an application from getting log output.

    configure_logging() only installs a handler when the logger has none, so
    the placeholder attached on import could leave the logger writing nowhere.
    """
    # ARRANGE
    program = (
        "import logging, opengate_gate_tree;"
        "from opengate_gate_tree.logging_setup import LOGGER_NAME, configure_logging;"
        "configure_logging();"
        "handlers = logging.getLogger(LOGGER_NAME).handlers;"
        "print(any(type(h) is logging.StreamHandler for h in handlers))"
    )

    # ACT
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    # ASSERT
    assert result.stdout.strip() == "True"


def test_only_the_command_line_module_exits_the_process() -> None:
    """Library code reports failures by raising, not by ending the process."""
    # ARRANGE
    offenders: list[str] = []

    # ACT
    for source_file in sorted(SOURCE_DIR.rglob("*.py")):
        if source_file.name in {"cli.py", "__main__.py"}:
            continue
        text = source_file.read_text(encoding="utf-8")
        if "sys.exit(" in text or "print(" in text:
            offenders.append(source_file.name)

    # ASSERT
    assert offenders == []


def test_the_package_can_be_imported_without_the_command_line() -> None:
    """Library use should not depend on the command line module being loaded."""
    # ARRANGE
    program = "import opengate_gate_tree, sys;print('opengate_gate_tree.cli' in sys.modules)"

    # ACT
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    # ASSERT
    assert result.stdout.strip() == "False"


def test_the_package_reports_its_version() -> None:
    """The version should stay available to code checking compatibility."""
    # ARRANGE
    module = importlib.import_module("opengate_gate_tree")

    # ACT
    version = module.__version__

    # ASSERT
    assert isinstance(version, str)
    assert version.count(".") == 2
