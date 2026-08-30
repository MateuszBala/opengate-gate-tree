"""Unit tests for branch selection."""

import pytest

from opengate_gate_tree.tree.branch import (
    normalize_branch_selection,
    validate_branch_selection,
)

# Branch names as they appear in a GATE "Hits" tree.
AVAILABLE_BRANCHES = ["eventID", "edep", "posX", "volumeID"]


def test_validate_branch_selection_accepts_an_empty_selection() -> None:
    """Asking for no branch in particular is a valid request."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    validate_branch_selection([])


def test_validate_branch_selection_accepts_repeated_names() -> None:
    """Repeating a branch name is not an error."""
    # ARRANGE
    requested = ["edep", "eventID", "edep"]

    # ACT & ASSERT
    validate_branch_selection(requested)


@pytest.mark.parametrize("branch_name", ["", "   "])
def test_validate_branch_selection_rejects_empty_names(branch_name: str) -> None:
    """A branch name that carries no value cannot be resolved."""
    # ARRANGE
    requested = ["eventID", branch_name]

    # ACT & ASSERT
    with pytest.raises(ValueError, match="must not be empty"):
        validate_branch_selection(requested)


def test_normalize_branch_selection_expands_an_empty_selection() -> None:
    """An empty selection should mean every branch of the tree."""
    # ARRANGE
    requested: list[str] = []

    # ACT
    selected = normalize_branch_selection(requested, AVAILABLE_BRANCHES)

    # ASSERT
    assert selected == AVAILABLE_BRANCHES


def test_normalize_branch_selection_keeps_the_requested_order() -> None:
    """The caller decides the order of the selected branches."""
    # ARRANGE
    requested = ["posX", "eventID"]

    # ACT
    selected = normalize_branch_selection(requested, AVAILABLE_BRANCHES)

    # ASSERT
    assert selected == ["posX", "eventID"]


def test_normalize_branch_selection_collapses_repeated_names() -> None:
    """A repeated name should be kept once, at its first position."""
    # ARRANGE
    requested = ["edep", "eventID", "edep"]

    # ACT
    selected = normalize_branch_selection(requested, AVAILABLE_BRANCHES)

    # ASSERT
    assert selected == ["edep", "eventID"]


def test_normalize_branch_selection_does_not_check_existence() -> None:
    """Whether a branch exists is decided by the file, not by this function."""
    # ARRANGE
    requested = ["notInTheTree"]

    # ACT
    selected = normalize_branch_selection(requested, AVAILABLE_BRANCHES)

    # ASSERT
    assert selected == ["notInTheTree"]


def test_normalize_branch_selection_rejects_empty_names() -> None:
    """Normalization should refuse a selection it cannot resolve."""
    # ARRANGE
    requested = ["eventID", ""]

    # ACT & ASSERT
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_branch_selection(requested, AVAILABLE_BRANCHES)
