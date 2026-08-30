"""Unit tests for gate tree parsing."""

import pytest

from opengate_gate_tree.tree.gatetree import GateTree, parse_gate_tree


def test_parse_gate_tree_is_case_insensitive() -> None:
    """Known tree names should parse regardless of case."""
    assert parse_gate_tree("hits") == GateTree.HITS
    assert parse_gate_tree("SINGLES") == GateTree.SINGLES
    assert parse_gate_tree("Coincidences") == GateTree.COINCIDENCES


def test_parse_gate_tree_raises_for_unknown_value() -> None:
    """Unknown tree names should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown GateTree member"):
        parse_gate_tree("unknown")
