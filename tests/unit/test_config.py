"""Unit tests for runtime configuration dataclass."""

from dataclasses import FrozenInstanceError

import pytest

from opengate_gate_tree.config import RunConfig
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.tree.gatetree import GateTree


def test_run_config_defaults_to_empty_branches_list() -> None:
    """RunConfig should initialize branches_to_extract as an empty list."""
    config = RunConfig(
        input_gate_root_file=None,
        output_dir=None,
        output_file_title=None,
        gate_tree=None,
        output_file_format=None,
    )

    assert config.branches_to_extract == []


def test_run_config_is_frozen() -> None:
    """RunConfig should be immutable because it is a frozen dataclass."""
    config = RunConfig(
        input_gate_root_file=None,
        output_dir=None,
        output_file_title="run_01",
        gate_tree=GateTree.HITS,
        output_file_format=OutputFileFormat.CSV,
    )

    with pytest.raises(FrozenInstanceError):
        config.output_file_title = "new_title"
