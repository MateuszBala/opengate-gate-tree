"""What every analysis has in common: a lazy chain of operations.

An analysis is a description, not a computation. It is built by calling one
operation after another - ``analysis.op_a(...).op_b(...).op_c(...)`` - and
keeps them in the order they were called, because that order is part of what
the analysis asks. Nothing is read until the analysis is given data with
:meth:`Analysis.apply`.

An analysis never changes. Every call returns a new analysis, so a part shared
by two analyses stays the same in both::

    pairs = EventTrackHitAnalysis().tracks(2)
    scattered = pairs.hits(ge=2)          # pairs is unchanged
    absorbed = pairs.process(Process.PHOTOELECTRIC)

Analyses combine in three ways:

- ``a & b`` keeps what both keep, and ``a | b`` what either keeps. Each of them
  is performed on its own, from the same starting point, and their results are
  combined, so both operators are associative and commutative;
- ``a.then(b)`` performs ``b`` on what ``a`` kept: the operations of ``b`` are
  appended to those of ``a``, in order, and start from where ``b`` starts on
  its own, whatever ``a`` ended with.

Each side of ``a & b`` and ``a | b`` likewise starts from where an analysis
starts, also inside a chain, so that an analysis means the same wherever it
is used.

Only analyses of the same kind combine: an analysis of events and one of
coincidences describe different things.

The same skeleton serves every kind of analysis. A subclass says what it is
performed on and what each of its operations does; how an analysis is built,
combined, compared, shown and performed is said here.

Public objects:

Analysis
    Base class of every lazy analysis.
Combination
    Two analyses performed on their own and combined.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self


@dataclass(frozen=True)
class Combination:
    """Two analyses performed on their own and combined.

    Attributes
    ----------
    kind : {"and", "or"}
        Whether what both keep is kept, or what either keeps.
    left, right : Analysis
        The analyses combined.
    """

    kind: Literal["and", "or"]
    left: "Analysis[Any, Any]"
    right: "Analysis[Any, Any]"

    def __repr__(self) -> str:
        """Return the combination as it is written."""
        operator = "&" if self.kind == "and" else "|"
        return f"({self.left!r} {operator} {self.right!r})"


class Analysis[Result, State](ABC):
    """A lazy chain of operations, performed when it is given data.

    Subclasses add the methods that append their operations, and say how an
    operation, a combination and the end of the chain act on the state the
    analysis is performed in.

    Parameters
    ----------
    steps : Sequence[object]
        Operations, and combinations of analyses, in the order they are
        performed. A new analysis starts with none; the methods of a subclass
        append to it, and check what they append.

    Raises
    ------
    TypeError
        If a step is neither a combination nor an operation of the subclass.
    """

    def __init__(self, steps: Sequence[object] = ()) -> None:
        self._steps: tuple[object, ...] = tuple(steps)
        for step in self._steps:
            if not isinstance(step, Combination) and not self._accepts(step):
                raise TypeError(
                    f"{type(self).__name__} cannot perform {step!r}: a step is a combination "
                    "or one of its own operations."
                )

    @property
    def steps(self) -> tuple[object, ...]:
        """Operations and combinations, in the order they are performed."""
        return self._steps

    def then(self, other: Self) -> Self:
        """Return an analysis performing ``other`` on what this one keeps.

        Raises
        ------
        TypeError
            If the two analyses are not of the same kind.
        """
        self._require_same_kind(other, "then")
        return self._appended(*self._restart_steps(), *other.steps)

    def __and__(self, other: Self) -> Self:
        """Return an analysis keeping what both this one and ``other`` keep.

        Raises
        ------
        TypeError
            If the two analyses are not of the same kind.
        """
        self._require_same_kind(other, "&")
        return type(self)((Combination("and", self, other),))

    def __or__(self, other: Self) -> Self:
        """Return an analysis keeping what this one or ``other`` keeps.

        Raises
        ------
        TypeError
            If the two analyses are not of the same kind.
        """
        self._require_same_kind(other, "|")
        return type(self)((Combination("or", self, other),))

    def apply(self, source: Any) -> Result:
        """Perform the analysis on data.

        Parameters
        ----------
        source : Any
            What the analysis is performed on; a subclass says what it takes.

        Returns
        -------
        Result
            What the analysis kept.
        """
        target = self._target(source)
        state = self._perform(self._steps, target, self._initial(target))
        return self._finish(target, state)

    def __eq__(self, other: object) -> bool:
        """Return whether two analyses are of one kind and hold the same steps."""
        return type(other) is type(self) and other.steps == self._steps  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        """Return a hash of the kind and the steps."""
        return hash((type(self), self._steps))

    def __repr__(self) -> str:
        """Return the chain as it would be written."""
        if len(self._steps) == 1 and isinstance(self._steps[0], Combination):
            return repr(self._steps[0])
        text = f"{type(self).__name__}()"
        for step in self._steps:
            text = f"{text}.then{step!r}" if isinstance(step, Combination) else f"{text}.{step!r}"
        return text

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle an analysis as its kind and its steps."""
        return (type(self), (self._steps,))

    def _appended(self, *steps: object) -> Self:
        """Return a new analysis of the same kind with steps added at the end."""
        return type(self)(self._steps + steps)

    def _perform(self, steps: Sequence[object], target: Any, state: State) -> State:
        """Perform steps one after another, starting from a state."""
        for step in steps:
            if isinstance(step, Combination):
                start = self._restart(state)
                left = self._perform(step.left.steps, target, start)
                right = self._perform(step.right.steps, target, start)
                state = self._combine(target, step.kind, left, right)
            else:
                state = self._operate(target, state, step)
        return state

    def _require_same_kind(self, other: object, operator: str) -> None:
        """Refuse to combine analyses of different kinds.

        Raises
        ------
        TypeError
            If ``other`` is not an analysis of the same kind.
        """
        if type(other) is not type(self):
            raise TypeError(
                f"{type(self).__name__} combines only with another {type(self).__name__} "
                f"through {operator!r}, not with {type(other).__name__}."
            )

    def _accepts(self, step: object) -> bool:
        """Return whether a step is an operation this kind of analysis performs."""
        return False

    def _restart_steps(self) -> tuple[object, ...]:
        """Return the steps that bring this analysis back to where an analysis starts.

        ``then`` puts them between two analyses, so that the second one starts
        where it would on its own. None are needed when the analysis is there.
        """
        return ()

    def _restart(self, state: State) -> State:
        """Return a state brought back to where an analysis starts, keeping what it kept."""
        return state

    @abstractmethod
    def _target(self, source: Any) -> Any:
        """Return what the analysis is performed on, built from the source."""

    @abstractmethod
    def _initial(self, target: Any) -> State:
        """Return the state an analysis starts from."""

    @abstractmethod
    def _operate(self, target: Any, state: State, operation: object) -> State:
        """Return the state after one operation."""

    @abstractmethod
    def _combine(self, target: Any, kind: Literal["and", "or"], left: State, right: State) -> State:
        """Return the state two analyses performed on their own combine into."""

    @abstractmethod
    def _finish(self, target: Any, state: State) -> Result:
        """Return the result of the state the analysis ended in."""
