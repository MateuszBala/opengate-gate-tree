"""Unit tests for the lazy chain every analysis is built on.

A small analysis of sets of numbers stands in for a real one: each operation
keeps the numbers a condition holds for, so the order of operations, the
combinations and the laziness can be seen without any data of GATE.
"""

import pickle
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from opengate_gate_tree.analysis.base import Analysis, Combination


@dataclass(frozen=True)
class Keep:
    """Keep the numbers a named condition holds for."""

    condition: str

    def __repr__(self) -> str:
        return f"keep({self.condition!r})"


CONDITIONS = {
    "even": lambda number: number % 2 == 0,
    "small": lambda number: number < 5,
    "positive": lambda number: number > 0,
}


class Numbers:
    """Numbers that count how often they were read."""

    def __init__(self, values: set[int]) -> None:
        self._values = values
        self.reads = 0

    @property
    def values(self) -> set[int]:
        self.reads += 1
        return self._values


class NumberAnalysis(Analysis[frozenset[int], frozenset[int]]):
    """Analysis of a set of numbers."""

    def keep(self, condition: str) -> "NumberAnalysis":
        return self._appended(Keep(condition))

    def _target(self, source: Any) -> Numbers:
        return source if isinstance(source, Numbers) else Numbers(set(source))

    def _initial(self, target: Numbers) -> frozenset[int]:
        return frozenset(target.values)

    def _operate(self, target: Numbers, state: frozenset[int], operation: object) -> frozenset[int]:
        assert isinstance(operation, Keep)
        return frozenset(number for number in state if CONDITIONS[operation.condition](number))

    def _combine(
        self,
        target: Numbers,
        kind: Literal["and", "or"],
        left: frozenset[int],
        right: frozenset[int],
    ) -> frozenset[int]:
        return left & right if kind == "and" else left | right

    def _finish(self, target: Numbers, state: frozenset[int]) -> frozenset[int]:
        return state


class OtherAnalysis(NumberAnalysis):
    """Another kind of analysis, which does not combine with the first."""


NUMBERS = range(-3, 10)


def test_a_call_returns_a_new_analysis_and_leaves_the_old_one() -> None:
    """A part shared by two analyses stays the same in both."""
    # ARRANGE
    shared = NumberAnalysis().keep("positive")

    # ACT
    even = shared.keep("even")
    small = shared.keep("small")

    # ASSERT
    assert shared.steps == (Keep("positive"),)
    assert even.apply(NUMBERS) == {2, 4, 6, 8}
    assert small.apply(NUMBERS) == {1, 2, 3, 4}


def test_nothing_is_read_until_the_analysis_is_given_data() -> None:
    """Building a chain is describing it."""
    # ARRANGE
    numbers = Numbers({1, 2, 3})

    # ACT
    analysis = NumberAnalysis().keep("even").keep("small")
    reads_before = numbers.reads
    analysis.apply(numbers)

    # ASSERT
    assert reads_before == 0
    assert numbers.reads == 1


def test_and_keeps_what_both_keep_and_or_what_either_keeps() -> None:
    """Each side is performed on its own, from the same numbers."""
    # ARRANGE
    even = NumberAnalysis().keep("even")
    small = NumberAnalysis().keep("small")

    # ACT
    both = (even & small).apply(NUMBERS)
    either = (even | small).apply(NUMBERS)

    # ASSERT
    assert both == {-2, 0, 2, 4}
    assert either == set(range(-3, 5)) | {6, 8}


def test_the_operators_are_associative_and_commutative() -> None:
    """The way three analyses are grouped does not change what they keep."""
    # ARRANGE
    even, small, positive = (NumberAnalysis().keep(name) for name in CONDITIONS)

    # ACT
    left = ((even & small) & positive).apply(NUMBERS)
    right = (even & (small & positive)).apply(NUMBERS)
    swapped = (positive & even & small).apply(NUMBERS)
    mixed = ((even & small) | positive).apply(NUMBERS)

    # ASSERT
    assert left == right == swapped == {2, 4}
    assert mixed == {-2, 0} | set(range(1, 10))


def test_then_performs_the_second_analysis_on_what_the_first_kept() -> None:
    """Its operations follow those of the first, in order."""
    # ARRANGE
    even = NumberAnalysis().keep("even")
    small_positive = NumberAnalysis().keep("small").keep("positive")

    # ACT
    chained = even.then(small_positive)

    # ASSERT
    assert chained.steps == (Keep("even"), Keep("small"), Keep("positive"))
    assert chained.apply(NUMBERS) == {2, 4}


def test_a_combination_inside_a_chain_starts_from_where_the_chain_is() -> None:
    """``positive`` first, then either half of the combination on what it kept."""
    # ARRANGE
    combined = NumberAnalysis().keep("even") | NumberAnalysis().keep("small")

    # ACT
    chained = NumberAnalysis().keep("positive").then(combined)

    # ASSERT
    assert chained.apply(NUMBERS) == {1, 2, 3, 4, 6, 8}


def test_analyses_of_different_kinds_do_not_combine() -> None:
    """They describe different things."""
    # ARRANGE
    numbers = NumberAnalysis().keep("even")
    other = OtherAnalysis().keep("even")

    # ACT
    with pytest.raises(TypeError) as raised:
        numbers & other

    # ASSERT
    assert "NumberAnalysis combines only with another NumberAnalysis" in str(raised.value)
    with pytest.raises(TypeError):
        numbers | other
    with pytest.raises(TypeError):
        numbers.then(other)


def test_an_analysis_is_shown_as_it_is_written() -> None:
    """The chain, and the brackets of a combination."""
    # ARRANGE
    even = NumberAnalysis().keep("even")
    small = NumberAnalysis().keep("small")

    # ACT
    chained = repr(even.keep("small"))
    combined = repr(even | small)
    nested = repr(NumberAnalysis().keep("positive").then(even & small))

    # ASSERT
    assert chained == "NumberAnalysis().keep('even').keep('small')"
    assert combined == "(NumberAnalysis().keep('even') | NumberAnalysis().keep('small'))"
    assert nested == (
        "NumberAnalysis().keep('positive')"
        ".then(NumberAnalysis().keep('even') & NumberAnalysis().keep('small'))"
    )
    assert repr(NumberAnalysis()) == "NumberAnalysis()"


def test_analyses_with_the_same_steps_are_equal() -> None:
    """And hash alike, so they can key a dictionary of results."""
    # ARRANGE
    first = NumberAnalysis().keep("even").keep("small")
    second = NumberAnalysis().keep("even").keep("small")

    # ACT
    results = {first: "kept"}

    # ASSERT
    assert first == second
    assert results[second] == "kept"
    assert first != NumberAnalysis().keep("small").keep("even")
    assert first != OtherAnalysis().keep("even").keep("small")
    assert first != "NumberAnalysis().keep('even').keep('small')"


def test_an_analysis_survives_pickling() -> None:
    """A description can be saved and performed later, elsewhere."""
    # ARRANGE
    analysis = (
        NumberAnalysis()
        .keep("positive")
        .then(NumberAnalysis().keep("even") | NumberAnalysis().keep("small"))
    )

    # ACT
    restored = pickle.loads(pickle.dumps(analysis))

    # ASSERT
    assert restored == analysis
    assert restored.apply(NUMBERS) == analysis.apply(NUMBERS)


def test_a_combination_names_its_kind_and_its_sides() -> None:
    """What the operators build."""
    # ARRANGE
    even = NumberAnalysis().keep("even")
    small = NumberAnalysis().keep("small")

    # ACT
    (combination,) = (even & small).steps

    # ASSERT
    assert combination == Combination("and", even, small)
