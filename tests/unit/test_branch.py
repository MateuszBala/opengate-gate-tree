"""Unit tests for branch name validation."""

import pytest

from opengate_gate_tree.tree.branch import validate_branch_name
from opengate_gate_tree.tree.gatetree import GateTree


@pytest.mark.parametrize("gate_tree", [GateTree.HITS, GateTree.SINGLES, GateTree.COINCIDENCES])
def test_validate_branch_name_accepts_empty_branch_list(gate_tree: GateTree) -> None:
    """Empty branch lists should always be valid."""
    is_valid, invalid = validate_branch_name([], gate_tree)
    assert is_valid is True
    assert invalid == []


@pytest.mark.parametrize("gate_tree", [GateTree.HITS, GateTree.SINGLES, GateTree.COINCIDENCES])
def test_validate_branch_name_rejects_unknown_branch(gate_tree: GateTree) -> None:
    """Unknown branches should be reported as invalid."""
    is_valid, invalid = validate_branch_name(["unknown_branch"], gate_tree)
    assert is_valid is False
    assert invalid == ["unknown_branch"]
