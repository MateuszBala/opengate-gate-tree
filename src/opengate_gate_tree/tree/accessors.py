"""The ``gate`` namespace on a pandas column and on a pandas frame.

Importing the package registers two accessors, so that the work reads as
something the data does rather than as something done to it::

    frame["edep"].gate.in_range(0.2, 0.4).gate.MeV_to_keV()
    frame.gate.in_cylinder(centre=(0, 0), radius=500.0, inner_radius=409.0)
    frame.gate.position().angle_to(frame.gate.momentum_direction())

The split follows what the work needs to know. A range and a conversion are
questions about one column, so they live on the column; a shape is a question
about three of them at once, the identity of an event about two, and a vector
is three columns read as one thing, so those live on the frame.

Every method calls the function of the same name in
:mod:`opengate_gate_tree.tree.filters` or in :mod:`opengate_gate_tree.units`
and adds nothing. The accessors are a way of writing, and nothing is reachable
only through them.

The five that read vectors are the exception, and only in their names: they
call :meth:`~opengate_gate_tree.geometry.vectorview.VectorView.from_frame` with
the triple of columns their name stands for, which is the one thing an accessor
knows that a function taking a frame would have to be told.

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

from opengate_gate_tree.geometry.vectorview import VectorView
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
from opengate_gate_tree.units import (
    MeV_to_keV,
    cm_to_m,
    cm_to_mm,
    deg_to_rad,
    keV_to_MeV,
    m_to_cm,
    m_to_mm,
    mm_to_cm,
    mm_to_m,
    ms_to_ns,
    ms_to_s,
    ns_to_ms,
    ns_to_s,
    rad_to_deg,
    s_to_ms,
    s_to_ns,
)

# Name both accessors answer to.
ACCESSOR_NAME: Final[str] = "gate"

# The triples of columns a "Hits" tree carries, each read as vectors by the
# method named after it.
LOCAL_POSITION_COLUMNS: Final[tuple[str, str, str]] = ("localPosX", "localPosY", "localPosZ")
SOURCE_POSITION_COLUMNS: Final[tuple[str, str, str]] = ("sourcePosX", "sourcePosY", "sourcePosZ")
MOMENTUM_DIRECTION_COLUMNS: Final[tuple[str, str, str]] = ("momDirX", "momDirY", "momDirZ")


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

    def MeV_to_keV(self) -> pd.Series:
        """Convert energy from megaelectronvolts to kiloelectronvolts."""
        return MeV_to_keV(self._values)

    def keV_to_MeV(self) -> pd.Series:
        """Convert energy from kiloelectronvolts to megaelectronvolts."""
        return keV_to_MeV(self._values)

    def mm_to_cm(self) -> pd.Series:
        """Convert length from millimetres to centimetres."""
        return mm_to_cm(self._values)

    def cm_to_mm(self) -> pd.Series:
        """Convert length from centimetres to millimetres."""
        return cm_to_mm(self._values)

    def mm_to_m(self) -> pd.Series:
        """Convert length from millimetres to metres."""
        return mm_to_m(self._values)

    def m_to_mm(self) -> pd.Series:
        """Convert length from metres to millimetres."""
        return m_to_mm(self._values)

    def cm_to_m(self) -> pd.Series:
        """Convert length from centimetres to metres."""
        return cm_to_m(self._values)

    def m_to_cm(self) -> pd.Series:
        """Convert length from metres to centimetres."""
        return m_to_cm(self._values)

    def s_to_ms(self) -> pd.Series:
        """Convert time from seconds to milliseconds."""
        return s_to_ms(self._values)

    def ms_to_s(self) -> pd.Series:
        """Convert time from milliseconds to seconds."""
        return ms_to_s(self._values)

    def s_to_ns(self) -> pd.Series:
        """Convert time from seconds to nanoseconds."""
        return s_to_ns(self._values)

    def ns_to_s(self) -> pd.Series:
        """Convert time from nanoseconds to seconds."""
        return ns_to_s(self._values)

    def ms_to_ns(self) -> pd.Series:
        """Convert time from milliseconds to nanoseconds."""
        return ms_to_ns(self._values)

    def ns_to_ms(self) -> pd.Series:
        """Convert time from nanoseconds to milliseconds."""
        return ns_to_ms(self._values)

    def rad_to_deg(self) -> pd.Series:
        """Convert angle from radians to degrees."""
        return rad_to_deg(self._values)

    def deg_to_rad(self) -> pd.Series:
        """Convert angle from degrees to radians."""
        return deg_to_rad(self._values)


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

    def vector(self, x: str, y: str, z: str) -> VectorView:
        """Read three columns as vectors.

        Parameters
        ----------
        x, y, z : str
            The columns holding the components, in that order.

        Returns
        -------
        VectorView
            The vectors, with the index of the frame.
        """
        return VectorView.from_frame(self._frame, (x, y, z))

    def position(self) -> VectorView:
        """Read where the hits happened, as vectors."""
        return VectorView.from_frame(self._frame, POSITION_COLUMNS)

    def local_position(self) -> VectorView:
        """Read where the hits happened inside their volume, as vectors."""
        return VectorView.from_frame(self._frame, LOCAL_POSITION_COLUMNS)

    def source_position(self) -> VectorView:
        """Read where the gammas were born, as vectors."""
        return VectorView.from_frame(self._frame, SOURCE_POSITION_COLUMNS)

    def momentum_direction(self) -> VectorView:
        """Read where the particles went after the hits, as vectors.

        GATE writes this direction as it is after the interaction at the hit,
        so the one a particle arrived with is rebuilt from positions by
        :func:`~opengate_gate_tree.geometry.momentum.momentum_direction_from_positions`.
        """
        return VectorView.from_frame(self._frame, MOMENTUM_DIRECTION_COLUMNS)
