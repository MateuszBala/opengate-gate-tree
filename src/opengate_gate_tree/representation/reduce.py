"""Questions asked of every track or every event at once.

The index of a "Hits" tree keeps the hits of a track, and the tracks of an
event, as neighbours. A value known for every hit is therefore summarised for
every track by reducing consecutive segments, and a value known for every
track is spread over its hits by repeating it. Both are single vectorised
operations, whatever the number of tracks: nothing loops over events.

The reductions take the name of what they answer:

- ``"any"`` / ``"all"`` - whether some, or every, element of a segment is
  ``True``; a segment with no elements answers ``False`` and ``True``;
- ``"count"`` - how many elements of a segment are ``True``;
- ``"sum"`` - the sum of a segment, in ``float64`` for floating point values
  and ``int64`` for whole numbers, so that summing many ``float32`` deposits
  loses nothing; an empty segment sums to zero;
- ``"first"`` / ``"last"`` - the first or last element of a segment, which an
  empty segment does not have.

A position counts elements a mask keeps, from 1, the way "the first hit" is
said: elements the mask leaves out have position 0.

Public objects:

Reduction
    Name of a reduction.
REDUCTIONS
    Every reduction, in the order the module lists them.
reduce_hits_to_tracks(index, values, how) -> numpy.ndarray
    Summarise a value of every hit for every track.
reduce_tracks_to_events(index, values, how) -> numpy.ndarray
    Summarise a value of every track for every event.
broadcast_tracks_to_hits(index, values) -> numpy.ndarray
    Give every hit the value of its track.
broadcast_events_to_tracks(index, values) -> numpy.ndarray
    Give every track the value of its event.
position_in_track(index, hit_mask, from_end) -> numpy.ndarray
    Position of every kept hit among the kept hits of its track.
position_in_event(index, track_mask, from_end) -> numpy.ndarray
    Position of every kept track among the kept tracks of its event.
"""

from typing import Any, Final, Literal, get_args

import numpy as np
import numpy.typing as npt

from opengate_gate_tree.representation.index import EventTrackHitIndex, IndexArray

Reduction = Literal["any", "all", "count", "sum", "first", "last"]

# Every reduction, in the order the module lists them.
REDUCTIONS: Final[tuple[Reduction, ...]] = get_args(Reduction)

# Reductions that read their values as a mask.
_MASK_REDUCTIONS: Final[frozenset[str]] = frozenset({"any", "all", "count"})


def reduce_hits_to_tracks(
    index: EventTrackHitIndex, values: npt.ArrayLike, how: Reduction
) -> npt.NDArray[Any]:
    """Summarise a value of every hit for every track.

    Parameters
    ----------
    index : EventTrackHitIndex
        Hits and tracks the values belong to.
    values : numpy.typing.ArrayLike
        One value per hit, hits in the order of the index. ``"any"``,
        ``"all"`` and ``"count"`` take a mask.
    how : Reduction
        What to answer for every track.

    Returns
    -------
    numpy.ndarray
        One answer per track.

    Raises
    ------
    ValueError
        If there is not one value per hit, the reduction is unknown, a mask
        reduction is given values that are not a mask, or ``"first"`` or
        ``"last"`` meets a track without hits.

    Examples
    --------
    >>> import numpy as np
    >>> index = EventTrackHitIndex(
    ...     hit_rows=np.arange(3), track_offsets=np.array([0, 2, 3]), event_offsets=np.array([0, 2])
    ... )
    >>> reduce_hits_to_tracks(index, np.array([True, False, True]), "count").tolist()
    [1, 1]
    """
    return _reduce_segments("hit", values, index.track_offsets, how)


def reduce_tracks_to_events(
    index: EventTrackHitIndex, values: npt.ArrayLike, how: Reduction
) -> npt.NDArray[Any]:
    """Summarise a value of every track for every event.

    Parameters
    ----------
    index : EventTrackHitIndex
        Tracks and events the values belong to.
    values : numpy.typing.ArrayLike
        One value per track, tracks in the order of the index. ``"any"``,
        ``"all"`` and ``"count"`` take a mask.
    how : Reduction
        What to answer for every event.

    Returns
    -------
    numpy.ndarray
        One answer per event.

    Raises
    ------
    ValueError
        If there is not one value per track, the reduction is unknown, a mask
        reduction is given values that are not a mask, or ``"first"`` or
        ``"last"`` meets an event without tracks.
    """
    return _reduce_segments("track", values, index.event_offsets, how)


def broadcast_tracks_to_hits(index: EventTrackHitIndex, values: npt.ArrayLike) -> npt.NDArray[Any]:
    """Give every hit the value of its track.

    Parameters
    ----------
    index : EventTrackHitIndex
        Hits and tracks the values belong to.
    values : numpy.typing.ArrayLike
        One value per track.

    Returns
    -------
    numpy.ndarray
        One value per hit, hits in the order of the index.

    Raises
    ------
    ValueError
        If there is not one value per track.
    """
    return np.repeat(_level_values("track", values, index.track_count), index.hits_per_track())


def broadcast_events_to_tracks(
    index: EventTrackHitIndex, values: npt.ArrayLike
) -> npt.NDArray[Any]:
    """Give every track the value of its event.

    Parameters
    ----------
    index : EventTrackHitIndex
        Tracks and events the values belong to.
    values : numpy.typing.ArrayLike
        One value per event.

    Returns
    -------
    numpy.ndarray
        One value per track, tracks in the order of the index.

    Raises
    ------
    ValueError
        If there is not one value per event.
    """
    return np.repeat(_level_values("event", values, index.event_count), index.tracks_per_event())


def position_in_track(
    index: EventTrackHitIndex, hit_mask: npt.ArrayLike, from_end: bool = False
) -> IndexArray:
    """Position of every kept hit among the kept hits of its track.

    Parameters
    ----------
    index : EventTrackHitIndex
        Hits and tracks the mask belongs to.
    hit_mask : numpy.typing.ArrayLike
        Which hits are kept, one value per hit.
    from_end : bool
        Count from the last kept hit instead of the first.

    Returns
    -------
    numpy.ndarray
        Position of every kept hit, from 1; 0 for a hit the mask leaves out.

    Raises
    ------
    ValueError
        If there is not one value per hit, or the values are not a mask.

    Examples
    --------
    >>> import numpy as np
    >>> index = EventTrackHitIndex(
    ...     hit_rows=np.arange(4), track_offsets=np.array([0, 3, 4]), event_offsets=np.array([0, 2])
    ... )
    >>> position_in_track(index, np.array([True, False, True, True])).tolist()
    [1, 0, 2, 1]
    """
    return _position_in_segments("hit", hit_mask, index.track_offsets, from_end)


def position_in_event(
    index: EventTrackHitIndex, track_mask: npt.ArrayLike, from_end: bool = False
) -> IndexArray:
    """Position of every kept track among the kept tracks of its event.

    Parameters
    ----------
    index : EventTrackHitIndex
        Tracks and events the mask belongs to.
    track_mask : numpy.typing.ArrayLike
        Which tracks are kept, one value per track.
    from_end : bool
        Count from the last kept track instead of the first.

    Returns
    -------
    numpy.ndarray
        Position of every kept track, from 1; 0 for a track the mask leaves
        out.

    Raises
    ------
    ValueError
        If there is not one value per track, or the values are not a mask.
    """
    return _position_in_segments("track", track_mask, index.event_offsets, from_end)


def _reduce_segments(
    level: str, values: npt.ArrayLike, offsets: IndexArray, how: Reduction
) -> npt.NDArray[Any]:
    """Reduce every segment the offsets mark out.

    Raises
    ------
    ValueError
        If the values do not match the segments, the reduction is unknown, a
        mask reduction is given another kind of values, or ``"first"`` or
        ``"last"`` meets an empty segment.
    """
    if how not in REDUCTIONS:
        raise ValueError(f"Unknown reduction {how!r}; expected one of {list(REDUCTIONS)}.")
    column = _level_values(level, values, int(offsets[-1]))
    if how in _MASK_REDUCTIONS:
        _require_mask(level, column)

    counts = np.diff(offsets)
    filled = counts > 0
    starts = offsets[:-1]

    if how in ("first", "last"):
        if not np.all(filled):
            raise ValueError(
                f"A segment without a {level} has no {how} {level}; "
                f"{int(np.count_nonzero(~filled))} segments are empty."
            )
        return column[starts] if how == "first" else column[offsets[1:] - 1]

    operation: np.ufunc
    kind: type[np.generic]
    if how == "any":
        operation, kind, empty = np.logical_or, np.bool_, False
    elif how == "all":
        operation, kind, empty = np.logical_and, np.bool_, True
    else:
        operation, empty = np.add, False
        kind = np.float64 if np.issubdtype(column.dtype, np.floating) else np.int64
    answer = np.full(len(counts), empty, dtype=kind)
    # reduceat answers an empty segment with the element it starts at, so
    # only the segments holding something are reduced.
    if np.any(filled):
        answer[filled] = operation.reduceat(column, starts[filled], dtype=kind)
    return answer


def _position_in_segments(
    level: str, mask: npt.ArrayLike, offsets: IndexArray, from_end: bool
) -> IndexArray:
    """Number the kept elements of every segment, from 1.

    Raises
    ------
    ValueError
        If the mask does not match the segments, or is not a mask.
    """
    kept = _level_values(level, mask, int(offsets[-1]))
    _require_mask(level, kept)

    running = np.cumsum(kept, dtype=np.int64)
    # Kept elements before every segment, to count each segment from zero.
    before = np.concatenate(([0], running))[offsets[:-1]]
    counts = np.diff(offsets)
    positions = running - np.repeat(before, counts)
    if from_end:
        in_segment = np.concatenate(([0], running))[offsets[1:]] - before
        positions = np.repeat(in_segment, counts) - positions + 1
    return np.where(kept, positions, 0)


def _level_values(level: str, values: npt.ArrayLike, expected: int) -> npt.NDArray[Any]:
    """Return values as an array holding one value per element of a level.

    Raises
    ------
    ValueError
        If the values are not one-dimensional or their number is not the one
        expected.
    """
    column = np.asarray(values)
    if column.ndim != 1 or len(column) != expected:
        raise ValueError(
            f"Expected one value per {level}, {expected} in all, "
            f"but got an array of shape {column.shape}."
        )
    return column


def _require_mask(level: str, column: npt.NDArray[Any]) -> None:
    """Check that values are a mask.

    Raises
    ------
    ValueError
        If the values are not booleans.
    """
    if column.dtype != np.bool_:
        raise ValueError(f"Expected a mask of the {level}s, got values of type {column.dtype}.")
