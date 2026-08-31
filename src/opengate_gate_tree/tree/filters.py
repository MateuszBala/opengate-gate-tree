"""Filters and selectors for the data of a "Hits" tree.

The functions here work on the pandas view of extracted data. They come in
pairs, and the pair is the same everywhere:

- ``is_<something>`` answers with a boolean column of the same length and the
  same index as its input, which is what combines with other conditions
  (``&``, ``|``) and what indexes other columns;
- the other name of the pair answers with the rows themselves, which is what
  chains.

Only what pandas has no name for is added. Comparing, ``isin`` and combining
masks already work on a column read from a GATE file, so the package does not
restate them; a filter earns its place by naming something the data means -
a closed range, a shape in space, the identity of an event, the meaning of a
code.

Public functions:

is_in_range(values, low, high, inclusive) -> pandas.Series
    Which values fall in a range.
in_range(values, low, high, inclusive) -> pandas.Series
    The values that fall in a range.
is_in_box(frame, centre, sides, columns) -> pandas.Series
    Which rows lie in a box.
in_box(frame, centre, sides, columns) -> pandas.DataFrame
    The rows that lie in a box.
is_in_sphere(frame, centre, radius, columns) -> pandas.Series
    Which rows lie in a sphere.
in_sphere(frame, centre, radius, columns) -> pandas.DataFrame
    The rows that lie in a sphere.
is_in_cylinder(frame, centre, radius, z_range, inner_radius, columns) -> pandas.Series
    Which rows lie in a cylinder or a ring.
in_cylinder(frame, centre, radius, z_range, inner_radius, columns) -> pandas.DataFrame
    The rows that lie in a cylinder or a ring.
"""

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd

# Which ends of a range belong to it, in the vocabulary of ``pandas.Series.between``.
InclusiveSide = Literal["both", "neither", "left", "right"]

# Columns holding the position of a hit, used by the shape filters unless the
# caller names others. A "Hits" tree carries three such triples: where the hit
# happened, where it happened inside its volume, and where the gamma was born.
POSITION_COLUMNS: Final[tuple[str, str, str]] = ("posX", "posY", "posZ")


def is_in_range(
    values: pd.Series,
    low: float,
    high: float,
    inclusive: InclusiveSide = "both",
) -> pd.Series:
    """Return which values fall in a range.

    Parameters
    ----------
    values : pandas.Series
        Column to test.
    low, high : float
        Ends of the range.
    inclusive : {"both", "neither", "left", "right"}
        Which ends belong to the range. The vocabulary is the one of
        :meth:`pandas.Series.between`, so a reader of pandas needs no second
        convention, and a value that is not a number falls outside the range
        the same way it does there.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``values``.
    """
    return values.between(low, high, inclusive=inclusive)


def in_range(
    values: pd.Series,
    low: float,
    high: float,
    inclusive: InclusiveSide = "both",
) -> pd.Series:
    """Return the values that fall in a range.

    Parameters
    ----------
    values : pandas.Series
        Column to select from.
    low, high : float
        Ends of the range.
    inclusive : {"both", "neither", "left", "right"}
        Which ends belong to the range.

    Returns
    -------
    pandas.Series
        The values in the range, with the index they had.
    """
    return values[is_in_range(values, low, high, inclusive)]


def is_in_box(
    frame: pd.DataFrame,
    centre: Sequence[float],
    sides: Sequence[float] | float,
    columns: Sequence[str] = POSITION_COLUMNS,
) -> pd.Series:
    """Return which rows lie in a box.

    The box is described the way the other shapes are: by where it sits and
    how big it is. Each side reaches half its length either way from the
    centre, so a box of sides 100 centred on the origin runs from -50 to 50.

    The faces belong to the box: a hit sitting exactly on one is inside. The
    other convention would drop hits on a boundary, and a simulation puts them
    there.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to test.
    centre : Sequence[float]
        Centre of the box, one value per column.
    sides : Sequence[float] | float
        Length of each side, one per column, or a single length for a cube.
    columns : Sequence[str]
        Columns holding the coordinates, in the order the centre gives them.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``frame``.

    Raises
    ------
    ValueError
        If the centre does not give one value per column, the sides do not
        give one length per column, or a side is negative.
    KeyError
        If the frame holds no column of one of those names.
    """
    coordinates = _coordinates(frame, columns)
    middle = _matching(centre, columns, "centre")
    lengths = _side_lengths(sides, columns)

    inside = pd.Series(True, index=frame.index)
    for values, position, length in zip(coordinates, middle, lengths, strict=True):
        half = length / 2
        inside &= is_in_range(values, position - half, position + half)
    return inside


def in_box(
    frame: pd.DataFrame,
    centre: Sequence[float],
    sides: Sequence[float] | float,
    columns: Sequence[str] = POSITION_COLUMNS,
) -> pd.DataFrame:
    """Return the rows that lie in a box.

    See :func:`is_in_box` for the parameters and for which points count as
    inside.
    """
    return frame[is_in_box(frame, centre, sides, columns)]


def is_in_sphere(
    frame: pd.DataFrame,
    centre: Sequence[float],
    radius: float,
    columns: Sequence[str] = POSITION_COLUMNS,
) -> pd.Series:
    """Return which rows lie in a sphere.

    The surface belongs to the sphere, for the reason the faces belong to a
    box.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to test.
    centre : Sequence[float]
        Centre of the sphere, one value per column.
    radius : float
        Radius of the sphere.
    columns : Sequence[str]
        Columns holding the coordinates.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``frame``.

    Raises
    ------
    ValueError
        If the centre does not give one value per column, or the radius is
        negative.
    KeyError
        If the frame holds no column of one of those names.
    """
    _positive_radius(radius, "radius")
    coordinates = _coordinates(frame, columns)
    middle = _matching(centre, columns, "centre")
    squared = sum(
        (values - position) ** 2 for values, position in zip(coordinates, middle, strict=True)
    )
    return pd.Series(squared <= radius**2, index=frame.index)


def in_sphere(
    frame: pd.DataFrame,
    centre: Sequence[float],
    radius: float,
    columns: Sequence[str] = POSITION_COLUMNS,
) -> pd.DataFrame:
    """Return the rows that lie in a sphere.

    See :func:`is_in_sphere` for the parameters.
    """
    return frame[is_in_sphere(frame, centre, radius, columns)]


def is_in_cylinder(
    frame: pd.DataFrame,
    centre: Sequence[float],
    radius: float,
    z_range: tuple[float, float] | None = None,
    inner_radius: float = 0.0,
    columns: Sequence[str] = POSITION_COLUMNS,
) -> pd.Series:
    """Return which rows lie in a cylinder, or in a ring.

    The cylinder runs along the **third** of the columns, so another axis is a
    matter of naming the columns in another order rather than of another
    parameter: ``columns=("posX", "posZ", "posY")`` stands it along y.

    An inner radius turns the cylinder into a ring, which is how a layer of a
    detector is usually asked for. The surfaces belong to the shape.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to test.
    centre : Sequence[float]
        Where the axis crosses the plane of the first two columns.
    radius : float
        Outer radius.
    z_range : tuple[float, float] | None
        Ends of the cylinder along its axis. Unbounded when omitted.
    inner_radius : float
        Inner radius, which makes the shape a ring.
    columns : Sequence[str]
        Columns holding the coordinates, the axis last.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``frame``.

    Raises
    ------
    ValueError
        If the centre does not give one value per plane column, a radius is
        negative, or the inner radius is larger than the outer one - which is
        a ring that could hold nothing, and reads more like two arguments
        swapped than like a question.
    KeyError
        If the frame holds no column of one of those names.
    """
    _positive_radius(radius, "radius")
    _positive_radius(inner_radius, "inner_radius")
    if inner_radius > radius:
        raise ValueError(
            f"The inner radius of a ring cannot exceed its outer radius, "
            f"got {inner_radius} and {radius}."
        )

    *across, along = _coordinates(frame, columns)
    middle = _matching(centre, columns[:2], "centre")
    distance = np.sqrt(
        sum((values - position) ** 2 for values, position in zip(across, middle, strict=True))
    )

    inside = pd.Series((distance >= inner_radius) & (distance <= radius), index=frame.index)
    if z_range is not None:
        inside &= is_in_range(along, z_range[0], z_range[1])
    return inside


def in_cylinder(
    frame: pd.DataFrame,
    centre: Sequence[float],
    radius: float,
    z_range: tuple[float, float] | None = None,
    inner_radius: float = 0.0,
    columns: Sequence[str] = POSITION_COLUMNS,
) -> pd.DataFrame:
    """Return the rows that lie in a cylinder, or in a ring.

    See :func:`is_in_cylinder` for the parameters.
    """
    return frame[is_in_cylinder(frame, centre, radius, z_range, inner_radius, columns)]


def _coordinates(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.Series, ...]:
    """Return the coordinate columns of a frame, refusing a wrong count."""
    if len(columns) != len(POSITION_COLUMNS):
        raise ValueError(
            f"A position is given by {len(POSITION_COLUMNS)} columns, got {len(columns)}: "
            f"{list(columns)}."
        )
    return tuple(frame[column] for column in columns)


def _matching(values: Sequence[float], columns: Sequence[str], name: str) -> tuple[float, ...]:
    """Return values given per column, refusing a count that does not fit them."""
    if len(values) != len(columns):
        raise ValueError(
            f"'{name}' needs one value per column: {len(columns)} expected, {len(values)} given."
        )
    return tuple(float(value) for value in values)


def _side_lengths(sides: Sequence[float] | float, columns: Sequence[str]) -> tuple[float, ...]:
    """Return the length of every side, refusing lengths that describe no box."""
    if isinstance(sides, int | float):
        lengths = (float(sides),) * len(columns)
    else:
        lengths = _matching(sides, columns, "sides")
    for length in lengths:
        if length < 0:
            raise ValueError(f"The side of a box cannot be negative, got {length}.")
    return lengths


def _positive_radius(radius: float, name: str) -> None:
    """Refuse a radius that is not a distance."""
    if radius < 0:
        raise ValueError(f"The '{name}' of a shape cannot be negative, got {radius}.")
