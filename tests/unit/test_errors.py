"""Unit tests for the package exception hierarchy."""

import pytest

from opengate_gate_tree.errors import (
    AmbiguousTreeError,
    BranchNotFoundError,
    ExportError,
    GateTreeError,
    HitsTreeValidationError,
    RootFileError,
    TreeMergeError,
    TreeNotFoundError,
    UnknownHitsVariantError,
    UnsupportedBranchTypeError,
)

PACKAGE_ERRORS: list[type[GateTreeError]] = [
    RootFileError,
    TreeNotFoundError,
    AmbiguousTreeError,
    BranchNotFoundError,
    UnsupportedBranchTypeError,
    UnknownHitsVariantError,
    HitsTreeValidationError,
    TreeMergeError,
    ExportError,
]


@pytest.mark.parametrize("error_type", PACKAGE_ERRORS)
def test_package_errors_derive_from_base_error(error_type: type[GateTreeError]) -> None:
    """Every package error should be catchable as GateTreeError."""
    # ARRANGE
    # No additional setup required.

    # ACT
    raised = error_type("failure details")

    # ASSERT
    assert isinstance(raised, GateTreeError)


@pytest.mark.parametrize("error_type", [GateTreeError, *PACKAGE_ERRORS])
def test_package_errors_are_not_value_errors(error_type: type[GateTreeError]) -> None:
    """Package errors should stay distinguishable from argument validation errors."""
    # ARRANGE
    # No additional setup required.

    # ACT
    raised = error_type("failure details")

    # ASSERT
    assert not isinstance(raised, ValueError)


@pytest.mark.parametrize("error_type", PACKAGE_ERRORS)
def test_package_errors_keep_their_message(error_type: type[GateTreeError]) -> None:
    """Error messages should be preserved so callers can report them."""
    # ARRANGE
    expected_message = "failure details"

    # ACT
    raised = error_type(expected_message)

    # ASSERT
    assert str(raised) == expected_message


def test_base_error_catches_every_package_error() -> None:
    """A single except clause should be enough to catch any package error."""
    # ARRANGE
    expected_message = "tree is missing"

    # ACT
    with pytest.raises(GateTreeError) as error_info:
        raise TreeNotFoundError(expected_message)

    # ASSERT
    assert str(error_info.value) == expected_message
