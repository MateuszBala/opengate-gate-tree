"""The operations an analysis of events, tracks and hits is made of.

Every operation is a frozen record of what was asked, and shows itself as the
call that asked it, so an analysis prints as the chain it was written as. What
an operation does to the data is said by the analysis that performs it; here
it only is.

Public objects:

Navigate
    Move to the events, the tracks or the hits, or to one position of them.
ValueTest
    Comparisons a value has to pass.
FieldCondition
    Keep the elements whose field passes a value test.
COMPARISONS
    Comparisons a value or a count can be asked to pass.
"""

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from opengate_gate_tree.representation.fields import ElementLevel, EventTrackHitField

FieldLike = EventTrackHitField | str

# Comparisons of one value against one bound.
_BOUND_COMPARISONS: Final[Mapping[str, Callable[[Any, Any], Any]]] = MappingProxyType(
    {
        "eq": operator.eq,
        "ne": operator.ne,
        "ge": operator.ge,
        "le": operator.le,
        "gt": operator.gt,
        "lt": operator.lt,
    }
)

# Comparisons a value or a count can be asked to pass, in the order they are shown.
COMPARISONS: Final[tuple[str, ...]] = (*_BOUND_COMPARISONS, "isin", "between")


@dataclass(frozen=True)
class Navigate:
    """Move to the events, the tracks or the hits, or to one position of them.

    Attributes
    ----------
    level : ElementLevel
        Level the analysis moves to.
    position : int | None
        Position of the element within its parent, from 1 and from -1
        backwards; ``None`` stands at every element of the level.
    call : str
        The call that asked for it.
    """

    level: ElementLevel
    position: int | None
    call: str

    def __repr__(self) -> str:
        """Return the call that asked for the move."""
        return self.call


@dataclass(frozen=True)
class ValueTest:
    """Comparisons a value has to pass.

    Attributes
    ----------
    comparisons : tuple[tuple[str, Any], ...]
        Pairs of a comparison from :data:`COMPARISONS` and what it compares
        against, all of which a value has to pass. ``isin`` takes a tuple of
        values, ``between`` a pair of ends that belong to the range.
    """

    comparisons: tuple[tuple[str, Any], ...]

    def passes(self, values: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        """Return which values pass every comparison.

        A value that is not a number passes none of them, except ``ne``.

        Raises
        ------
        ValueError
            If the values hold several components per element.
        """
        if values.ndim != 1:
            raise ValueError(
                f"A value test needs one value per element, got {values.shape[1]} components "
                "per element; test one of them instead."
            )
        passed = np.ones(len(values), dtype=bool)
        for name, bound in self.comparisons:
            if name == "isin":
                passed &= np.isin(values, list(bound))
            elif name == "between":
                passed &= pd.Series(values).between(*bound).to_numpy()
            else:
                passed &= np.asarray(_BOUND_COMPARISONS[name](values, bound), dtype=bool)
        return passed

    def arguments(self) -> str:
        """Return the comparisons as the keyword arguments that asked for them."""
        return ", ".join(f"{name}={bound!r}" for name, bound in self.comparisons)


@dataclass(frozen=True)
class FieldCondition:
    """Keep the elements whose field passes a value test.

    Attributes
    ----------
    field : EventTrackHitField | str
        Field tested; a name that is not the value of a field is a branch
        describing hits.
    test : ValueTest
        What the field has to pass.
    call : str
        The call that asked for it.
    """

    field: FieldLike
    test: ValueTest
    call: str

    def __repr__(self) -> str:
        """Return the call that asked for the condition."""
        return self.call
