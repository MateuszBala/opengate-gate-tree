"""Which rows of a "Hits" tree make up which hit, track and event.

A row of the "Hits" tree is one hit. The hits of one track share its run,
event and track identifiers, and the tracks of one event share the run and
event ones. The index records that grouping in three arrays, so that nothing
has to be grouped again later:

- ``hit_rows`` - the row of the data every hit was read from, hits in order;
- ``track_offsets`` - where the hits of every track begin and end in
  ``hit_rows``, as ``track_offsets[k]:track_offsets[k + 1]``;
- ``event_offsets`` - the same for the tracks of every event.

This is the layout sparse matrices use. The hits of a track, and the tracks of
an event, are neighbours, so a question asked of every track at once - how
many hits it has, whether all of them are Compton scatterings - is one
vectorised reduction over consecutive segments rather than a grouping.

The order is fixed, and it is the one an analysis counts in:

- events by their run, then their event identifier;
- tracks within an event by their track identifier, so track 1 is the first
  primary particle the source emitted;
- hits within a track by their time, and rows written at the same time in the
  order of the data.

GATE writes the hits of a track one after another, in the order they happened,
but the tracks of an event from the last one to the first. The index is
therefore built in one of two ways. When every track is a single block of rows
whose time does not decrease, only the blocks are sorted, which is as many
items as there are tracks. Otherwise, as in hits merged from one tree per
sensitive detector, where a track has rows in both, every row is sorted. Both
ways give the same index for the same data.

Public objects:

EventTrackHitIndex
    Hits, tracks and events of a "Hits" tree, as positions in its rows.
build_event_track_hit_index(run_ids, event_ids, track_ids, times, force_sort) -> EventTrackHitIndex
    Group the rows of a "Hits" tree into hits, tracks and events.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

IndexArray = npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class EventTrackHitIndex:
    """Hits, tracks and events of a "Hits" tree, as positions in its rows.

    Equality is not defined, for the reason :class:`TreeData` gives: arrays
    compared with ``==`` answer with arrays.

    Attributes
    ----------
    hit_rows : numpy.ndarray
        Row of the data each hit was read from, hits in order.
    track_offsets : numpy.ndarray
        Boundaries of the hits of every track in ``hit_rows``; one more entry
        than there are tracks.
    event_offsets : numpy.ndarray
        Boundaries of the tracks of every event in the track arrays; one more
        entry than there are events.

    Raises
    ------
    ValueError
        If the offsets do not start at zero, decrease, or do not end at the
        number of elements they divide.
    """

    hit_rows: IndexArray
    track_offsets: IndexArray
    event_offsets: IndexArray

    def __post_init__(self) -> None:
        """Check that the offsets divide the levels below them."""
        _validate_offsets("track", self.track_offsets, len(self.hit_rows))
        _validate_offsets("event", self.event_offsets, len(self.track_offsets) - 1)

    @property
    def event_count(self) -> int:
        """Number of events."""
        return len(self.event_offsets) - 1

    @property
    def track_count(self) -> int:
        """Number of tracks."""
        return len(self.track_offsets) - 1

    @property
    def hit_count(self) -> int:
        """Number of hits."""
        return len(self.hit_rows)

    def tracks_per_event(self) -> IndexArray:
        """Return the number of tracks of every event."""
        return np.diff(self.event_offsets)

    def hits_per_track(self) -> IndexArray:
        """Return the number of hits of every track."""
        return np.diff(self.track_offsets)

    def event_of_track(self) -> IndexArray:
        """Return the event every track belongs to."""
        return np.repeat(np.arange(self.event_count, dtype=np.int64), self.tracks_per_event())

    def track_of_hit(self) -> IndexArray:
        """Return the track every hit belongs to."""
        return np.repeat(np.arange(self.track_count, dtype=np.int64), self.hits_per_track())

    def first_hit_rows(self) -> IndexArray:
        """Return the row of the first hit of every track.

        A track without hits, which only a selection keeping empty tracks
        leaves, has no first hit.

        Raises
        ------
        ValueError
            If a track has no hits.
        """
        if np.any(self.hits_per_track() == 0):
            raise ValueError("A track without hits has no first hit to read its values from.")
        return self.hit_rows[self.track_offsets[:-1]]

    def first_track_rows(self) -> IndexArray:
        """Return the row of the first hit of the first track of every event.

        Raises
        ------
        ValueError
            If an event has no tracks, or its first track has no hits.
        """
        if np.any(self.tracks_per_event() == 0):
            raise ValueError("An event without tracks has no first track to read its values from.")
        return self.first_hit_rows()[self.event_offsets[:-1]]

    def __repr__(self) -> str:
        """Return a compact representation that does not render the arrays."""
        return (
            f"{type(self).__name__}(events={self.event_count}, "
            f"tracks={self.track_count}, hits={self.hit_count})"
        )


def build_event_track_hit_index(
    run_ids: npt.ArrayLike,
    event_ids: npt.ArrayLike,
    track_ids: npt.ArrayLike,
    times: npt.ArrayLike,
    force_sort: bool = False,
) -> EventTrackHitIndex:
    """Group the rows of a "Hits" tree into hits, tracks and events.

    Parameters
    ----------
    run_ids, event_ids, track_ids : numpy.typing.ArrayLike
        Identifier columns of the rows, as GATE wrote them.
    times : numpy.typing.ArrayLike
        Time of every row, which orders the hits of a track.
    force_sort : bool
        Sort every row even when only the blocks of rows a track occupies
        would need sorting. The result is the same either way; this is there
        to compare the two, and to measure them.

    Returns
    -------
    EventTrackHitIndex
        Hits, tracks and events, in the order the module describes.

    Raises
    ------
    ValueError
        If the columns are not one-dimensional or differ in length.
    """
    runs, events, tracks, hit_times = (
        _as_column(name, values)
        for name, values in (
            ("run_ids", run_ids),
            ("event_ids", event_ids),
            ("track_ids", track_ids),
            ("times", times),
        )
    )
    lengths = {len(runs), len(events), len(tracks), len(hit_times)}
    if len(lengths) != 1:
        raise ValueError(
            "The identifier and time columns must hold one value per row, "
            f"but their lengths are {sorted(lengths)}."
        )

    rows = None if force_sort else _rows_by_sorting_blocks(runs, events, tracks, hit_times)
    if rows is None:
        # lexsort is stable, so rows written at the same time keep their order.
        rows = np.lexsort((hit_times, tracks, events, runs)).astype(np.int64)
    return _index_from_rows(rows, runs, events, tracks)


def _as_column(name: str, values: npt.ArrayLike) -> npt.NDArray[Any]:
    """Return values as a one-dimensional array.

    Raises
    ------
    ValueError
        If the values are not one-dimensional.
    """
    column = np.asarray(values)
    if column.ndim != 1:
        raise ValueError(f"'{name}' must be one-dimensional, got shape {column.shape}.")
    return column


def _rows_by_sorting_blocks(
    runs: npt.NDArray[Any],
    events: npt.NDArray[Any],
    tracks: npt.NDArray[Any],
    times: npt.NDArray[Any],
) -> IndexArray | None:
    """Return the rows in order when every track is one block in time order.

    Returns ``None`` when a track occupies several blocks, or its time
    decreases within one: then only sorting every row orders the hits.
    """
    count = len(runs)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    changes = (runs[1:] != runs[:-1]) | (events[1:] != events[:-1]) | (tracks[1:] != tracks[:-1])
    if np.any((times[1:] < times[:-1]) & ~changes):
        return None

    starts = np.concatenate(([0], np.flatnonzero(changes) + 1)).astype(np.int64)
    lengths = np.diff(np.append(starts, count))
    order = np.lexsort((tracks[starts], events[starts], runs[starts]))

    ordered = starts[order]
    repeated = (
        (runs[ordered][1:] == runs[ordered][:-1])
        & (events[ordered][1:] == events[ordered][:-1])
        & (tracks[ordered][1:] == tracks[ordered][:-1])
    )
    if np.any(repeated):
        return None

    # Lay the blocks one after another in their new order: every row moves by
    # the distance between where its block starts and where it is placed.
    ordered_lengths = lengths[order]
    placed_at = np.cumsum(ordered_lengths) - ordered_lengths
    shift = np.repeat(ordered - placed_at, ordered_lengths)
    rows: IndexArray = np.arange(count, dtype=np.int64) + shift
    return rows


def _index_from_rows(
    rows: IndexArray,
    runs: npt.NDArray[Any],
    events: npt.NDArray[Any],
    tracks: npt.NDArray[Any],
) -> EventTrackHitIndex:
    """Find where tracks and events begin among rows already in order."""
    count = len(rows)
    ordered_runs, ordered_events, ordered_tracks = runs[rows], events[rows], tracks[rows]

    new_event = np.ones(count, dtype=bool)
    new_event[1:] = (ordered_runs[1:] != ordered_runs[:-1]) | (
        ordered_events[1:] != ordered_events[:-1]
    )
    new_track = new_event.copy()
    new_track[1:] |= ordered_tracks[1:] != ordered_tracks[:-1]

    track_starts = np.flatnonzero(new_track)
    event_starts = np.flatnonzero(new_event[track_starts])
    return EventTrackHitIndex(
        hit_rows=rows,
        track_offsets=np.append(track_starts, count).astype(np.int64),
        event_offsets=np.append(event_starts, len(track_starts)).astype(np.int64),
    )


def _validate_offsets(level: str, offsets: IndexArray, divided: int) -> None:
    """Check that offsets divide a level of the given size into segments.

    Raises
    ------
    ValueError
        If the offsets are empty, do not start at zero, decrease, or do not
        end at the size of the level.
    """
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError(f"The {level} offsets must be a non-empty one-dimensional array.")
    if offsets[0] != 0 or offsets[-1] != divided:
        raise ValueError(
            f"The {level} offsets must run from 0 to {divided}, "
            f"but run from {offsets[0]} to {offsets[-1]}."
        )
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"The {level} offsets must not decrease.")
