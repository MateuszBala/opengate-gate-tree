"""The ``gate`` namespace on a pandas column and on a pandas frame.

Importing the package registers two accessors, so that a filter reads as
something the data does rather than as something done to it::

    frame["edep"].gate.in_range(0.2, 0.4)
    frame.gate.in_cylinder(centre=(0, 0), radius=500.0, inner_radius=409.0)

The split follows what a filter needs to know. A range is a question about one
column, so it lives on the column; a shape is a question about three of them
at once, and the identity of an event about two, so those live on the frame.

Every method calls the function of the same name in
:mod:`opengate_gate_tree.tree.filters` and adds nothing. The accessors are a
way of writing, and nothing is reachable only through them.

Registering a name in pandas is a change to something the whole process
shares, and it happens when this package is imported. Should ``gate`` already
be taken, pandas says so with a warning of its own, which is not silenced
here: it reports a real collision in somebody's code.

Public objects:

ACCESSOR_NAME
    The name both accessors are registered under.
GateSeriesAccessor
    What ``Series.gate`` gives.
GateFrameAccessor
    What ``DataFrame.gate`` gives.
"""

from collections.abc import Sequence
from typing import Final

import pandas as pd

from opengate_gate_tree.tree.filters import (
    POSITION_COLUMNS,
    InclusiveSide,
    by_event,
    by_run,
    has_decay_metadata,
    in_box,
    in_cylinder,
    in_range,
    in_sphere,
    is_decay_type,
    is_from_event,
    is_from_run,
    is_gamma_type,
    is_in_box,
    is_in_cylinder,
    is_in_range,
    is_in_sphere,
    is_process,
    is_source_type,
    select_by_decay_type,
    select_by_gamma_type,
    select_by_process,
    select_by_source_type,
    with_decay_metadata,
)
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType

# Name both accessors answer to.
ACCESSOR_NAME: Final[str] = "gate"


@pd.api.extensions.register_series_accessor(ACCESSOR_NAME)
class GateSeriesAccessor:
    """Filters of one column of a GATE tree."""

    def __init__(self, values: pd.Series) -> None:
        """Keep the column the filters will be asked about."""
        self._values = values

    def is_in_range(
        self,
        low: float,
        high: float,
        inclusive: InclusiveSide = "both",
    ) -> pd.Series:
        """Return which values fall in a range."""
        return is_in_range(self._values, low, high, inclusive)

    def in_range(
        self,
        low: float,
        high: float,
        inclusive: InclusiveSide = "both",
    ) -> pd.Series:
        """Return the values that fall in a range."""
        return in_range(self._values, low, high, inclusive)

    def is_source_type(self, *types: SourceType) -> pd.Series:
        """Return which values name one of the given source types."""
        return is_source_type(self._values, *types)

    def select_by_source_type(self, *types: SourceType) -> pd.Series:
        """Return the values that name one of the given source types."""
        return select_by_source_type(self._values, *types)

    def is_decay_type(self, *types: DecayType) -> pd.Series:
        """Return which values name one of the given decay channels."""
        return is_decay_type(self._values, *types)

    def select_by_decay_type(self, *types: DecayType) -> pd.Series:
        """Return the values that name one of the given decay channels."""
        return select_by_decay_type(self._values, *types)

    def is_gamma_type(self, *types: GammaType) -> pd.Series:
        """Return which values name one of the given kinds of gamma."""
        return is_gamma_type(self._values, *types)

    def select_by_gamma_type(self, *types: GammaType) -> pd.Series:
        """Return the values that name one of the given kinds of gamma."""
        return select_by_gamma_type(self._values, *types)

    def is_process(self, *names: str) -> pd.Series:
        """Return which values name one of the given processes."""
        return is_process(self._values, *names)

    def select_by_process(self, *names: str) -> pd.Series:
        """Return the values that name one of the given processes."""
        return select_by_process(self._values, *names)


@pd.api.extensions.register_dataframe_accessor(ACCESSOR_NAME)
class GateFrameAccessor:
    """Filters reading several columns of a GATE tree at once."""

    def __init__(self, frame: pd.DataFrame) -> None:
        """Keep the frame the filters will be asked about."""
        self._frame = frame

    def is_in_box(
        self,
        centre: Sequence[float],
        sides: Sequence[float] | float,
        columns: Sequence[str] = POSITION_COLUMNS,
    ) -> pd.Series:
        """Return which rows lie in a box."""
        return is_in_box(self._frame, centre, sides, columns)

    def in_box(
        self,
        centre: Sequence[float],
        sides: Sequence[float] | float,
        columns: Sequence[str] = POSITION_COLUMNS,
    ) -> pd.DataFrame:
        """Return the rows that lie in a box."""
        return in_box(self._frame, centre, sides, columns)

    def is_in_sphere(
        self,
        centre: Sequence[float],
        radius: float,
        columns: Sequence[str] = POSITION_COLUMNS,
    ) -> pd.Series:
        """Return which rows lie in a sphere."""
        return is_in_sphere(self._frame, centre, radius, columns)

    def in_sphere(
        self,
        centre: Sequence[float],
        radius: float,
        columns: Sequence[str] = POSITION_COLUMNS,
    ) -> pd.DataFrame:
        """Return the rows that lie in a sphere."""
        return in_sphere(self._frame, centre, radius, columns)

    def is_in_cylinder(
        self,
        centre: Sequence[float],
        radius: float,
        z_range: tuple[float, float] | None = None,
        inner_radius: float = 0.0,
        columns: Sequence[str] = POSITION_COLUMNS,
    ) -> pd.Series:
        """Return which rows lie in a cylinder, or in a ring."""
        return is_in_cylinder(self._frame, centre, radius, z_range, inner_radius, columns)

    def in_cylinder(
        self,
        centre: Sequence[float],
        radius: float,
        z_range: tuple[float, float] | None = None,
        inner_radius: float = 0.0,
        columns: Sequence[str] = POSITION_COLUMNS,
    ) -> pd.DataFrame:
        """Return the rows that lie in a cylinder, or in a ring."""
        return in_cylinder(self._frame, centre, radius, z_range, inner_radius, columns)

    def is_from_run(self, run_id: int) -> pd.Series:
        """Return which rows come from a run."""
        return is_from_run(self._frame, run_id)

    def by_run(self, run_id: int) -> pd.DataFrame:
        """Return the rows of a run."""
        return by_run(self._frame, run_id)

    def is_from_event(self, run_id: int, event_id: int) -> pd.Series:
        """Return which rows come from an event."""
        return is_from_event(self._frame, run_id, event_id)

    def by_event(self, run_id: int, event_id: int) -> pd.DataFrame:
        """Return the rows of an event."""
        return by_event(self._frame, run_id, event_id)

    def has_decay_metadata(self) -> pd.Series:
        """Return which rows carry the decay metadata of a PositroniumSource."""
        return has_decay_metadata(self._frame)

    def with_decay_metadata(self) -> pd.DataFrame:
        """Return the rows that carry the decay metadata of a PositroniumSource."""
        return with_decay_metadata(self._frame)
