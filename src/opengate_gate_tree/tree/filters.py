r"""Filters and selectors for the data of a "Hits" tree.

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
is_from_run(frame, run_id) -> pandas.Series
    Which rows come from a run.
by_run(frame, run_id) -> pandas.DataFrame
    The rows of a run.
is_from_event(frame, run_id, event_id) -> pandas.Series
    Which rows come from an event.
by_event(frame, run_id, event_id) -> pandas.DataFrame
    The rows of an event.
has_decay_metadata(frame) -> pandas.Series
    Which rows carry the decay metadata of a PositroniumSource.
with_decay_metadata(frame) -> pandas.DataFrame
    The rows that carry it.
is_source_type(values, \*types) -> pandas.Series
    Which values name one of the given source types.
select_by_source_type(values, \*types) -> pandas.Series
    The values that name one of them.
is_decay_type(values, \*types), select_by_decay_type(values, \*types)
    The same for the decay channel.
is_gamma_type(values, \*types), select_by_gamma_type(values, \*types)
    The same for the kind of gamma.
is_process(values, \*names) -> pandas.Series
    Which values name one of the given processes.
select_by_process(values, \*names) -> pandas.Series
    The values that name one of them.
"""

from collections.abc import Sequence
from enum import IntEnum
from typing import Final, Literal

import numpy as np
import pandas as pd

from opengate_gate_tree.tree.hits.positronium import (
    DECAY_INDEX_BRANCH,
    DecayType,
    GammaType,
    SourceType,
    has_positronium_metadata,
)

# Which ends of a range belong to it, in the vocabulary of ``pandas.Series.between``.
InclusiveSide = Literal["both", "neither", "left", "right"]

# Columns naming the run and the event a row belongs to.
RUN_COLUMN: Final[str] = "runID"
EVENT_COLUMN: Final[str] = "eventID"

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


def is_from_run(frame: pd.DataFrame, run_id: int) -> pd.Series:
    """Return which rows come from a run.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to test.
    run_id : int
        Run to look for, as GATE numbered it.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``frame``.

    Raises
    ------
    KeyError
        If the frame holds no ``runID`` column.
    """
    return frame[RUN_COLUMN] == run_id


def by_run(frame: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Return the rows of a run.

    See :func:`is_from_run` for the parameters.
    """
    return frame[is_from_run(frame, run_id)]


def is_from_event(frame: pd.DataFrame, run_id: int, event_id: int) -> pd.Series:
    """Return which rows come from an event.

    An event is named by **both** identifiers. GATE numbers events within a
    run, so a file holding more than one run holds an event 5 in each of them,
    and they are different decays. There is deliberately no filter taking the
    event identifier alone: writing ``frame["eventID"] == 5`` is one
    comparison, and it should look like the guess it is.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to test.
    run_id, event_id : int
        Run and event to look for, as GATE numbered them.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``frame``.

    Raises
    ------
    KeyError
        If the frame holds no ``runID`` or no ``eventID`` column.
    """
    return is_from_run(frame, run_id) & (frame[EVENT_COLUMN] == event_id)


def by_event(frame: pd.DataFrame, run_id: int, event_id: int) -> pd.DataFrame:
    """Return the rows of an event.

    See :func:`is_from_event` for the parameters and for why an event needs
    both identifiers.
    """
    return frame[is_from_event(frame, run_id, event_id)]


def has_decay_metadata(frame: pd.DataFrame) -> pd.Series:
    """Return which rows carry the decay metadata of a PositroniumSource.

    The frame-wide counterpart of
    :func:`~opengate_gate_tree.tree.hits.positronium.has_positronium_metadata`,
    which answers about a column. Both say the same thing, and it is narrower
    than "written by a PositroniumSource": such a source writes the metadata
    for every gamma it emits, and GATE leaves it out for a gamma of another
    source **or** for a particle it never reached.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to test.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``frame``.

    Raises
    ------
    KeyError
        If the frame holds no ``decayIndex`` column.
    """
    return pd.Series(
        has_positronium_metadata(frame[DECAY_INDEX_BRANCH]),
        index=frame.index,
    )


def with_decay_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the rows that carry the decay metadata of a PositroniumSource.

    See :func:`has_decay_metadata` for what the answer covers.
    """
    return frame[has_decay_metadata(frame)]


def is_source_type(values: pd.Series, *types: SourceType) -> pd.Series:
    """Return which values name one of the given source types.

    Parameters
    ----------
    values : pandas.Series
        Column of the ``sourceType`` branch.
    *types : SourceType
        One or more members to look for.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``values``.

    Raises
    ------
    ValueError
        If no member was given, or one of them belongs to another class. The
        classes share their numbers - a source type of 2 is a positronium and
        a gamma type of 2 is an annihilation gamma - so a member of the wrong
        class would select the right rows for the wrong reason, or the wrong
        rows outright.
    """
    return _is_of_enum(values, SourceType, types)


def select_by_source_type(values: pd.Series, *types: SourceType) -> pd.Series:
    """Return the values that name one of the given source types.

    See :func:`is_source_type` for the parameters.
    """
    return values[is_source_type(values, *types)]


def is_decay_type(values: pd.Series, *types: DecayType) -> pd.Series:
    """Return which values name one of the given decay channels.

    See :func:`is_source_type`; this one reads the ``decayType`` branch.
    """
    return _is_of_enum(values, DecayType, types)


def select_by_decay_type(values: pd.Series, *types: DecayType) -> pd.Series:
    """Return the values that name one of the given decay channels."""
    return values[is_decay_type(values, *types)]


def is_gamma_type(values: pd.Series, *types: GammaType) -> pd.Series:
    """Return which values name one of the given kinds of gamma.

    See :func:`is_source_type`; this one reads the ``gammaType`` branch.
    """
    return _is_of_enum(values, GammaType, types)


def select_by_gamma_type(values: pd.Series, *types: GammaType) -> pd.Series:
    """Return the values that name one of the given kinds of gamma."""
    return values[is_gamma_type(values, *types)]


def is_process(values: pd.Series, *names: str) -> pd.Series:
    """Return which values name one of the given processes.

    Parameters
    ----------
    values : pandas.Series
        Column of the ``processName`` branch.
    *names : str
        One or more process names, as GATE writes them.

    Returns
    -------
    pandas.Series
        Boolean column of the same length and index as ``values``.

    Raises
    ------
    ValueError
        If no name was given.
    """
    if not names:
        raise ValueError("Selecting by process needs at least one name to select.")
    return values.isin(names)


def select_by_process(values: pd.Series, *names: str) -> pd.Series:
    """Return the values that name one of the given processes.

    See :func:`is_process` for the parameters.
    """
    return values[is_process(values, *names)]


def _is_of_enum(
    values: pd.Series,
    enum_class: type[IntEnum],
    members: Sequence[IntEnum],
) -> pd.Series:
    """Return which values are one of the given members of a class."""
    if not members:
        raise ValueError(f"Selecting by {enum_class.__name__} needs at least one member to select.")
    wrong = [member for member in members if not isinstance(member, enum_class)]
    if wrong:
        named = ", ".join(_describe(member) for member in wrong)
        raise ValueError(
            f"Selecting by {enum_class.__name__} takes its own members, got {named}. "
            f"The classes share their numbers, so another class would select by a value that "
            f"means something else."
        )
    return values.isin([int(member) for member in members])


def _describe(member: object) -> str:
    """Return how a value is named in a message about the wrong class."""
    if isinstance(member, IntEnum):
        return f"{type(member).__name__}.{member.name}"
    return repr(member)
