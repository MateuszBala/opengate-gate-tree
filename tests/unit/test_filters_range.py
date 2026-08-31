"""Unit tests for the range filters."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.filters import InclusiveSide, in_range, is_in_range
from opengate_gate_tree.tree.gatetree import GateTree

# Values spanning both ends of the range used in most of the tests.
VALUES = pd.Series([0.0, 1.0, 1.5, 2.0, 3.0])


@pytest.mark.parametrize(
    ("inclusive", "expected"),
    [
        ("both", [1.0, 1.5, 2.0]),
        ("neither", [1.5]),
        ("left", [1.0, 1.5]),
        ("right", [1.5, 2.0]),
    ],
)
def test_the_ends_belong_to_the_range_as_asked(
    inclusive: InclusiveSide,
    expected: list[float],
) -> None:
    """Which ends count is the caller's choice, in the vocabulary of pandas."""
    # ARRANGE
    # The values hold both ends of the range and one value between them.

    # ACT
    selected = in_range(VALUES, 1.0, 2.0, inclusive)

    # ASSERT
    assert list(selected) == expected


def test_the_mask_covers_every_row() -> None:
    """A mask is what combines with other conditions, so it keeps its shape."""
    # ARRANGE
    # No additional setup required.

    # ACT
    mask = is_in_range(VALUES, 1.0, 2.0)

    # ASSERT
    assert len(mask) == len(VALUES)
    assert list(mask.index) == list(VALUES.index)
    assert mask.dtype == bool


def test_the_selection_keeps_the_index_it_had() -> None:
    """The index is what ties a selection back to the rest of the data."""
    # ARRANGE
    values = pd.Series([5.0, 1.5, 7.0], index=[10, 11, 12])

    # ACT
    selected = in_range(values, 1.0, 2.0)

    # ASSERT
    assert list(selected.index) == [11]


def test_a_value_that_is_not_a_number_falls_outside() -> None:
    """A missing value is not in any range, as it is not in any comparison."""
    # ARRANGE
    values = pd.Series([1.5, np.nan])

    # ACT
    mask = is_in_range(values, 1.0, 2.0)

    # ASSERT
    assert list(mask) == [True, False]


def test_a_range_that_ends_before_it_starts_holds_nothing() -> None:
    """An empty range is a legitimate question with an empty answer."""
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = in_range(VALUES, 2.0, 1.0)

    # ASSERT
    assert list(selected) == []


def test_a_range_of_one_point_holds_that_point() -> None:
    """Both ends closed on the same value is how a single value is asked for."""
    # ARRANGE
    # No additional setup required.

    # ACT
    selected = in_range(VALUES, 1.5, 1.5)

    # ASSERT
    assert list(selected) == [1.5]


def test_an_empty_column_answers_empty() -> None:
    """An extraction can end up with no rows, and filtering them is not an error."""
    # ARRANGE
    values = pd.Series([], dtype=float)

    # ACT
    mask = is_in_range(values, 1.0, 2.0)
    selected = in_range(values, 1.0, 2.0)

    # ASSERT
    assert len(mask) == 0
    assert len(selected) == 0


def test_selections_chain(positronium_files: Mapping[str, Path]) -> None:
    """Chaining is the shape the API promises: a column in, a column out."""
    # ARRANGE
    frame = read_tree(positronium_files["all-variants"], GateTree.HITS, ["edep"]).to_dataframe()

    # ACT
    narrowed = in_range(in_range(frame["edep"], 0.0, 0.4), 0.2, 0.4)

    # ASSERT
    assert isinstance(narrowed, pd.Series)
    assert list(narrowed) == list(in_range(frame["edep"], 0.2, 0.4))


def test_a_real_column_is_filtered_the_way_a_comparison_would(
    positronium_files: Mapping[str, Path],
) -> None:
    """The filter has to agree with the comparison it saves the reader from writing."""
    # ARRANGE
    frame = read_tree(positronium_files["ops"], GateTree.HITS, ["edep"]).to_dataframe()
    energies = frame["edep"]

    # ACT
    selected = in_range(energies, 0.2, 0.4)

    # ASSERT
    expected = energies[(energies >= 0.2) & (energies <= 0.4)]
    assert list(selected) == list(expected)
    assert 0 < len(selected) < len(energies)
