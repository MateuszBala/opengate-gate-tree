"""One row per event, with the variables asked for of its tracks and hits.

The result of an analysis is a table in which every row is an event that
passed it. A variable describing events is one column; one describing tracks
is one column per track, ``t1_<name>``, ``t2_<name>``…; one describing hits is
one column per hit of every track, ``t1h1_<name>``, ``t1h2_<name>``…. The name
is the branch the variable is read from, so the columns of one hit form the
triples the ``gate`` namespace reads as vectors::

    table.gate.vector("t1h1_posX", "t1h1_posY", "t1h1_posZ")

Tracks and hits are numbered from 1 among the ones the analysis kept, in the
order of the representation: tracks by their identifier, hits by their time.

Events seldom keep the same number of tracks, nor tracks of hits, and a table
has one width. It is the smallest number of tracks an event of the result
kept, and the smallest number of hits a track entering the table kept, so the
table has no empty cell and every column keeps the type of its branch. What
does not fit is left out, and the log says how much. An analysis that ends
with fixed counts, such as ``tracks(2)`` and ``hits(2)``, loses nothing.

Public objects:

extract_table(representation, variables, max_tracks, max_hits) -> pandas.DataFrame
    One row per event of a representation, with the variables asked for.
EVENT_KEY
    Columns every table starts with.
"""

from collections.abc import Sequence, Set
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from opengate_gate_tree.logger import log
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import ElementLevel, EventTrackHitField

FieldLike = EventTrackHitField | str

# Columns every table starts with: an event is the pair of both.
EVENT_KEY: Final[tuple[EventTrackHitField, ...]] = (
    EventTrackHitField.RUN_ID,
    EventTrackHitField.EVENT_ID,
)


def extract_table(
    representation: EventTrackHit,
    variables: Sequence[FieldLike],
    max_tracks: int | None = None,
    max_hits: int | None = None,
) -> pd.DataFrame:
    """Return one row per event of a representation, with the variables asked for.

    Parameters
    ----------
    representation : EventTrackHit
        Events, tracks and hits to tabulate, usually what an analysis kept.
    variables : Sequence[EventTrackHitField | str]
        Fields to put in the table, in order; a name that is not the value of
        a field is a branch describing hits. ``runID`` and ``eventID`` always
        come first, whether asked for or not.
    max_tracks, max_hits : int | None
        Narrower widths than the ones the data allows, to keep fewer tracks
        per event or hits per track.

    Returns
    -------
    pandas.DataFrame
        One row per event; columns in the order variable, track, hit.

    Raises
    ------
    ValueError
        If the variables are not an ordered sequence, or a width asked for is
        negative or wider than the data allows, which would leave cells empty.
    BranchNotFoundError
        If the data does not hold the branch of a variable.
    """
    if isinstance(variables, Set | str) or not isinstance(variables, Sequence):
        raise ValueError(
            "The variables of a table are a list or a tuple: their order is the order of the "
            f"columns, and a {type(variables).__name__} has none."
        )
    index = representation.index
    track_numbers = representation.values(EventTrackHitField.TRACK_NUMBER)
    tracks_wide = _width(
        "tracks", "event", representation.track_counts(), max_tracks, index.event_count
    )
    entering = track_numbers <= tracks_wide
    hits_wide = _width(
        "hits", "track", representation.hit_counts()[entering], max_hits, int(entering.sum())
    )
    _report_left_out(representation, tracks_wide, hits_wide, entering)

    columns: dict[str, npt.NDArray[Any]] = {}
    chosen = [*EVENT_KEY, *(field for field in variables if _resolve(field) not in EVENT_KEY)]
    for field in dict.fromkeys(_resolve(field) for field in chosen):
        _add(columns, representation, field, tracks_wide, hits_wide)
    return pd.DataFrame(columns, index=pd.RangeIndex(index.event_count))


def _width(
    elements: str, parent: str, counts: npt.NDArray[np.int64], asked: int | None, parents: int
) -> int:
    """Return how many elements of every parent the table has room for.

    Raises
    ------
    ValueError
        If the width asked for is negative, or wider than the data allows.
    """
    allowed = int(counts.min()) if parents else 0
    if asked is None:
        return allowed
    if asked < 0:
        raise ValueError(f"A table cannot hold {asked} {elements} per {parent}.")
    if asked > allowed:
        raise ValueError(
            f"A table of {asked} {elements} per {parent} would leave cells empty: at least one "
            f"{parent} of the result kept only {allowed}. Ask for {allowed} or fewer."
        )
    return asked


def _report_left_out(
    representation: EventTrackHit,
    tracks_wide: int,
    hits_wide: int,
    entering: npt.NDArray[np.bool_],
) -> None:
    """Log how many tracks and hits did not fit in the table, when any did not."""
    track_counts = representation.track_counts()
    hit_counts = representation.hit_counts()[entering]
    crowded_events = int(np.count_nonzero(track_counts > tracks_wide))
    left_tracks = int(np.clip(track_counts - tracks_wide, 0, None).sum())
    crowded_tracks = int(np.count_nonzero(hit_counts > hits_wide))
    left_hits = int(np.clip(hit_counts - hits_wide, 0, None).sum())
    if left_tracks or left_hits:
        log().warning(
            "A table of %d track(s) per event and %d hit(s) per track left out %d track(s) of "
            "%d event(s) and %d hit(s) of %d track(s); end the analysis with tracks(n) and "
            "hits(n) to keep a fixed number of each.",
            tracks_wide,
            hits_wide,
            left_tracks,
            crowded_events,
            left_hits,
            crowded_tracks,
        )


def _add(
    columns: dict[str, npt.NDArray[Any]],
    representation: EventTrackHit,
    field: FieldLike,
    tracks_wide: int,
    hits_wide: int,
) -> None:
    """Add the columns of one variable, track after track, hit after hit."""
    name = field.value if isinstance(field, EventTrackHitField) else field
    level = field.level if isinstance(field, EventTrackHitField) else ElementLevel.HIT
    values = representation.values(field)
    index = representation.index
    if level is ElementLevel.EVENT:
        _put(columns, name, values)
        return

    track_numbers = representation.values(EventTrackHitField.TRACK_NUMBER)
    if level is ElementLevel.TRACK:
        kept = track_numbers <= tracks_wide
        table = np.empty((index.event_count, tracks_wide, *values.shape[1:]), dtype=values.dtype)
        table[index.event_of_track()[kept], track_numbers[kept] - 1] = values[kept]
        for track in range(tracks_wide):
            _put(columns, f"t{track + 1}_{name}", table[:, track])
        return

    hit_tracks = representation.values(EventTrackHitField.TRACK_NUMBER, ElementLevel.HIT)
    hit_numbers = representation.values(EventTrackHitField.HIT_NUMBER)
    event_of_hit = np.repeat(index.event_of_track(), index.hits_per_track())
    kept = (hit_tracks <= tracks_wide) & (hit_numbers <= hits_wide)
    table = np.empty(
        (index.event_count, tracks_wide, hits_wide, *values.shape[1:]), dtype=values.dtype
    )
    table[event_of_hit[kept], hit_tracks[kept] - 1, hit_numbers[kept] - 1] = values[kept]
    for track in range(tracks_wide):
        for hit in range(hits_wide):
            _put(columns, f"t{track + 1}h{hit + 1}_{name}", table[:, track, hit])


def _put(columns: dict[str, npt.NDArray[Any]], name: str, values: npt.NDArray[Any]) -> None:
    """Put one column, or one per component of a variable with several."""
    if values.ndim == 1:
        columns[name] = values
        return
    for component in range(values.shape[1]):
        columns[f"{name}_{component}"] = values[:, component]


def _resolve(field: FieldLike) -> FieldLike:
    """Return the field a name stands for, or the name of a branch of hits."""
    if isinstance(field, EventTrackHitField):
        return field
    try:
        return EventTrackHitField(field)
    except ValueError:
        return field
