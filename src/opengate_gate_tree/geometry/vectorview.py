"""Three columns of a frame, read and worked with as vectors.

A GATE tree carries vectors as three columns side by side: ``posX, posY,
posZ`` for where a hit happened, ``momDirX, momDirY, momDirZ`` for where the
particle was going. A view reads such a triple as what it is, so that the
arithmetic reads like the physics::

    direction = frame.gate.position() - frame.gate.source_position()
    along = direction.unit().dot(frame.gate.momentum_direction())

A view owns nothing. It holds the values as an ``(N, 3)`` array of ``float64``
and the index of the rows they came from - the array it hands back is
read-only, and every answer carries that index
back: a length or a scalar product comes back as a column, a vector as another
view, three components as a frame. That is what lets a selection, a
computation and another selection stand in one chain.

Two views can be combined only when they describe the same rows. Vectors of
different rows would be zipped by position, which is the mistake pandas
alignment exists to prevent, so it is refused instead.

Public objects:

VectorView
    Vectors read from three columns, with the index of their rows.
"""

from typing import Final, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from opengate_gate_tree.geometry.angles import angle_between
from opengate_gate_tree.geometry.components import (
    parallel_component,
    perpendicular_component,
    spherical_components,
)
from opengate_gate_tree.geometry.vectors import (
    VECTOR_DIMENSION,
    as_vectors,
    cross,
    dot,
    norm,
    normalize,
)
from opengate_gate_tree.tree.filters import POSITION_COLUMNS

# What the components are called when a view no longer describes the columns it
# was read from, as after adding two of them together.
GENERIC_NAMES: Final[tuple[str, str, str]] = ("x", "y", "z")

# The suffixes a triple of columns carries in a GATE tree, and the ones
# ``to_frame`` writes back.
COMPONENT_SUFFIXES: Final[tuple[str, str, str]] = ("X", "Y", "Z")

# What the spherical components are called. The names say which angle is
# which, where "theta" and "phi" would leave it to the reader to guess.
SPHERICAL_COLUMNS: Final[tuple[str, str, str]] = ("radius", "polar", "azimuth")


class VectorView:
    """Vectors read from three columns, carrying the index of their rows.

    Parameters
    ----------
    values : array_like
        Vectors of shape ``(N, 3)``.
    index : pandas.Index
        The rows the vectors describe, one per vector.
    names : tuple[str, str, str]
        What the three components are called, used when the vectors are
        written back into a frame.

    Raises
    ------
    ValueError
        If the values do not hold vectors, or hold a different number of them
        than the index holds rows.
    """

    def __init__(
        self,
        values: ArrayLike,
        index: pd.Index,
        names: tuple[str, str, str] = GENERIC_NAMES,
    ) -> None:
        """Keep the vectors, the rows they came from and what to call them."""
        array = as_vectors(values)
        if len(array) != len(index):
            raise ValueError(
                f"A view holds one vector per row: got {len(array)} vector(s) "
                f"for {len(index)} row(s)."
            )
        self._array = array
        self._index = index
        self._names = names

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        columns: tuple[str, str, str] = POSITION_COLUMNS,
    ) -> Self:
        """Read three columns of a frame as vectors.

        Parameters
        ----------
        frame : pandas.DataFrame
            Frame holding the columns.
        columns : tuple[str, str, str]
            The three columns, in the order ``(x, y, z)``. A "Hits" tree
            carries three such triples: where the hit happened, where it
            happened inside its volume, and where the gamma was born.

        Returns
        -------
        VectorView
            The vectors, with the index of the frame.

        Raises
        ------
        ValueError
            If the number of columns is not three.
        KeyError
            If the frame holds no column of one of those names.
        """
        if len(columns) != VECTOR_DIMENSION:
            raise ValueError(
                f"A vector is given by {VECTOR_DIMENSION} columns, got {len(columns)}: "
                f"{list(columns)}."
            )
        values = frame.loc[:, list(columns)]
        return cls(values, frame.index, tuple(columns))  # type: ignore[arg-type]

    @property
    def array(self) -> np.ndarray:
        """Return the vectors as an ``(N, 3)`` array of ``float64``.

        The array is read-only, and is the one the view holds rather than a
        copy of it: a view owns nothing, and handing out something writable
        would let one caller change what another one is looking at. Values
        given to the constructor are referenced too, as
        :class:`~opengate_gate_tree.tree.treedata.TreeData` references the
        arrays it is built from - write to them and the view sees it.
        """
        readable = self._array.view()
        readable.flags.writeable = False
        return readable

    @property
    def index(self) -> pd.Index:
        """Return the rows the vectors describe."""
        return self._index

    @property
    def names(self) -> tuple[str, str, str]:
        """Return what the three components are called."""
        return self._names

    @property
    def x(self) -> pd.Series:
        """Return the first component, as a column."""
        return self._component(0)

    @property
    def y(self) -> pd.Series:
        """Return the second component, as a column."""
        return self._component(1)

    @property
    def z(self) -> pd.Series:
        """Return the third component, as a column."""
        return self._component(2)

    def __len__(self) -> int:
        """Return how many vectors the view holds."""
        return len(self._array)

    def __repr__(self) -> str:
        """Return what the view holds and where it came from."""
        return f"VectorView({len(self)} vectors, names={self._names})"

    def __add__(self, other: "VectorView") -> "VectorView":
        """Return the sum of two views, row by row."""
        return self._combined(self._array + self._matching(other), GENERIC_NAMES)

    def __sub__(self, other: "VectorView") -> "VectorView":
        """Return the difference of two views, row by row.

        This is how a direction is built out of two positions: where a gamma
        went is where it arrived minus where it started.
        """
        return self._combined(self._array - self._matching(other), GENERIC_NAMES)

    def __mul__(self, factor: float | pd.Series) -> "VectorView":
        """Return the vectors scaled, by one number or by one per row."""
        return self._combined(self._array * self._factor(factor), self._names)

    def __rmul__(self, factor: float | pd.Series) -> "VectorView":
        """Return the vectors scaled, with the factor written first."""
        return self.__mul__(factor)

    def __neg__(self) -> "VectorView":
        """Return the vectors pointing the other way."""
        return self._combined(-self._array, self._names)

    def norm(self) -> pd.Series:
        """Return the length of every vector.

        Returns
        -------
        pandas.Series
            Lengths, with the index of the rows.
        """
        return pd.Series(norm(self._array), index=self._index, name="norm")

    def unit(self) -> "VectorView":
        """Return the same vectors, of unit length.

        A vector of no length has no direction, so such a row comes back as
        ``nan`` and is reported. See
        :func:`~opengate_gate_tree.geometry.vectors.normalize`.

        Returns
        -------
        VectorView
            Vectors of unit length, describing the same rows.
        """
        return self._combined(normalize(self._array), self._names)

    def dot(self, other: "VectorView") -> pd.Series:
        """Return the scalar product with another view, row by row.

        Parameters
        ----------
        other : VectorView
            Vectors describing the same rows.

        Returns
        -------
        pandas.Series
            Scalar products, with the index of the rows.

        Raises
        ------
        ValueError
            If the two views do not describe the same rows.
        """
        return pd.Series(dot(self._array, self._matching(other)), index=self._index, name="dot")

    def cross(self, other: "VectorView") -> "VectorView":
        """Return the vector product with another view, row by row.

        Parameters
        ----------
        other : VectorView
            Vectors describing the same rows.

        Returns
        -------
        VectorView
            Vector products, describing the same rows.

        Raises
        ------
        ValueError
            If the two views do not describe the same rows.
        """
        return self._combined(cross(self._array, self._matching(other)), GENERIC_NAMES)

    def angle_to(self, other: "VectorView") -> pd.Series:
        """Return the angle to another view, row by row.

        Parameters
        ----------
        other : VectorView
            Vectors describing the same rows.

        Returns
        -------
        pandas.Series
            Angles in radians, in ``[0, pi]``, with the index of the rows.
            :func:`~opengate_gate_tree.units.rad_to_deg` reads them in degrees.

        Raises
        ------
        ValueError
            If the two views do not describe the same rows.
        """
        return pd.Series(
            angle_between(self._array, self._matching(other)),
            index=self._index,
            name="angle",
        )

    def parallel_to(self, axis: "VectorView | ArrayLike") -> "VectorView":
        """Return the part of every vector that lies along an axis.

        Parameters
        ----------
        axis : VectorView | array_like
            The direction to read along: another view of the same rows, one
            vector for the whole column, or one per row.

        Returns
        -------
        VectorView
            The part along the axis, describing the same rows.

        Raises
        ------
        ValueError
            If a view of other rows, or an axis of another length, is given.
        """
        return self._combined(parallel_component(self._array, self._axis(axis)), self._names)

    def perpendicular_to(self, axis: "VectorView | ArrayLike") -> "VectorView":
        """Return the part of every vector that is left across an axis.

        Together with :meth:`parallel_to` this adds back up to the vectors it
        was taken from.

        Parameters
        ----------
        axis : VectorView | array_like
            The direction to read across: another view of the same rows, one
            vector for the whole column, or one per row.

        Returns
        -------
        VectorView
            The part across the axis, describing the same rows.

        Raises
        ------
        ValueError
            If a view of other rows, or an axis of another length, is given.
        """
        return self._combined(perpendicular_component(self._array, self._axis(axis)), self._names)

    def spherical(self) -> pd.DataFrame:
        """Return the radius, polar angle and azimuth of every vector.

        The polar angle is measured from the ``z`` axis and the azimuth runs
        in ``[0, 2*pi)``; see
        :func:`~opengate_gate_tree.geometry.components.spherical_components`.

        Returns
        -------
        pandas.DataFrame
            Columns ``radius``, ``polar`` and ``azimuth``, with the index of
            the rows. Angles are in radians, as everywhere in this package.
        """
        radius, polar, azimuth = spherical_components(self._array)
        return pd.DataFrame(
            dict(zip(SPHERICAL_COLUMNS, (radius, polar, azimuth), strict=True)),
            index=self._index,
        )

    def to_frame(self, prefix: str | None = None) -> pd.DataFrame:
        """Return the vectors as three columns of a frame.

        Parameters
        ----------
        prefix : str | None
            Start of the column names, completed with ``X``, ``Y`` and ``Z``.
            When omitted, the names the view carries are used.

        Returns
        -------
        pandas.DataFrame
            Three columns, with the index of the rows. The frame holds values
            of its own: writing to it does not reach the view it came from.
        """
        names = self._names if prefix is None else _prefixed(prefix)
        # A copy: the frame is the caller's to write to, and pandas before 3.0
        # would otherwise hand it the storage this view is holding.
        return pd.DataFrame(self._array.copy(), index=self._index, columns=list(names))

    def _component(self, position: int) -> pd.Series:
        """Return one component of every vector, as a column."""
        return pd.Series(
            self._array[:, position].copy(), index=self._index, name=self._names[position]
        )

    def _combined(self, values: np.ndarray, names: tuple[str, str, str]) -> "VectorView":
        """Return another view of the same rows."""
        return VectorView(values, self._index, names)

    def _axis(self, axis: "VectorView | ArrayLike") -> ArrayLike:
        """Return an axis, holding a view of other rows to the usual rule."""
        if isinstance(axis, VectorView):
            return self._matching(axis)
        return axis

    def _matching(self, other: "VectorView") -> np.ndarray:
        """Return the vectors of another view, refusing one about other rows."""
        if not self._index.equals(other.index):
            raise ValueError(
                "Two views are combined row by row, so they describe the same rows: "
                f"got {len(self)} row(s) against {len(other)}, with different index values. "
                "Select the rows first, then read the vectors."
            )
        return other.array

    def _factor(self, factor: float | pd.Series) -> np.ndarray | float:
        """Return a factor as one number or as one per row, refusing a mismatch."""
        if isinstance(factor, pd.Series):
            if not self._index.equals(factor.index):
                raise ValueError(
                    "A factor given per row describes the same rows as the vectors it "
                    f"scales: got {len(factor)} value(s) against {len(self)} row(s)."
                )
            return np.asarray(factor, dtype=np.float64)[:, np.newaxis]
        return float(factor)


def _prefixed(prefix: str) -> tuple[str, str, str]:
    """Return the three column names a prefix stands for."""
    first, second, third = (f"{prefix}{suffix}" for suffix in COMPONENT_SUFFIXES)
    return first, second, third
