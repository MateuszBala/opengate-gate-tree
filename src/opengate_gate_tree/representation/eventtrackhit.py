"""The "Hits" tree read as events, the tracks in them, and the hits they left.

:class:`EventTrackHit` is built from the columns of a "Hits" tree, or from a
frame of them, and is light: it keeps a reference to the columns and an index
of which rows make up which hit, track and event. Building it reads four
columns - the run, event and track identifiers and the time - and nothing
else is read until it is asked for. What is known from the start is what the
grouping itself says: how many tracks every event has, and how many hits
every track has.

Everything else is read on request, in one of three shapes:

- :meth:`EventTrackHit.values` - one field for every element of its level, as
  an array in the order of the index;
- :meth:`EventTrackHit.events_frame`, :meth:`EventTrackHit.tracks_frame` and
  :meth:`EventTrackHit.hits_frame` - the fields asked for, as a table with a
  row per element; the hits table is labelled with the rows of the data, so a
  hit leads back to the row it was read from;
- :attr:`EventTrackHit.events` - the events as objects, made one at a time,
  for looking at the data the way it is thought of.

Public objects:

EventTrackHit
    The "Hits" tree read as events, tracks and hits.
ConsistencyReport
    Fields whose value differs within a track or an event that should share it.
INDEX_BRANCHES
    Branches the representation is built from.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from opengate_gate_tree.errors import RepresentationError
from opengate_gate_tree.representation.base import DataRepresentation
from opengate_gate_tree.representation.elements import ElementSequence, Event
from opengate_gate_tree.representation.fields import ElementLevel, EventTrackHitField
from opengate_gate_tree.representation.index import (
    EventTrackHitIndex,
    build_event_track_hit_index,
)
from opengate_gate_tree.representation.reduce import (
    reduce_hits_to_tracks,
    reduce_tracks_to_events,
)
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.treedata import TreeData

FieldLike = EventTrackHitField | str

# Branches the representation is built from.
INDEX_BRANCHES: Final[tuple[str, ...]] = ("runID", "eventID", "trackID", "time")

# Columns the tables add to say where an element stands.
EVENT_INDEX_COLUMN: Final[str] = "eventIndex"
TRACK_INDEX_COLUMN: Final[str] = "trackIndex"

# Depth of every level, to tell whether a value can be spread over another.
_DEPTH: Final[Mapping[ElementLevel, int]] = {
    ElementLevel.EVENT: 0,
    ElementLevel.TRACK: 1,
    ElementLevel.HIT: 2,
}

DEFAULT_EVENT_FIELDS: Final[tuple[EventTrackHitField, ...]] = (
    EventTrackHitField.RUN_ID,
    EventTrackHitField.EVENT_ID,
    EventTrackHitField.TRACK_COUNT,
)

DEFAULT_TRACK_FIELDS: Final[tuple[EventTrackHitField, ...]] = (
    EventTrackHitField.TRACK_NUMBER,
    EventTrackHitField.RUN_ID,
    EventTrackHitField.EVENT_ID,
    EventTrackHitField.TRACK_ID,
    EventTrackHitField.HIT_COUNT,
)


@dataclass(frozen=True)
class ConsistencyReport:
    """Fields whose value differs within a track or an event that should share it.

    Attributes
    ----------
    checked : tuple[EventTrackHitField, ...]
        Fields that were checked.
    inconsistent : Mapping[EventTrackHitField, int]
        Number of tracks or events in which each field takes more than one
        value; a field that agrees everywhere is not listed.
    examples : Mapping[EventTrackHitField, int]
        Position of the first such track or event, for every listed field.
    """

    checked: tuple[EventTrackHitField, ...]
    inconsistent: Mapping[EventTrackHitField, int]
    examples: Mapping[EventTrackHitField, int]

    @property
    def is_consistent(self) -> bool:
        """Whether every checked field agrees within every track and event."""
        return not self.inconsistent


class EventTrackHit(DataRepresentation):
    """The "Hits" tree read as events, the tracks in them, and the hits they left.

    Parameters
    ----------
    source : TreeData | pandas.DataFrame
        Columns of a "Hits" tree, or a frame of them, possibly left by a
        selection. They need the branches in :data:`INDEX_BRANCHES`.

    Raises
    ------
    RepresentationError
        If the data was read from another tree, or misses a branch the
        grouping needs.

    Examples
    --------
    >>> import numpy as np
    >>> from opengate_gate_tree import GateTree, TreeData
    >>> data = TreeData(
    ...     GateTree.HITS,
    ...     {
    ...         "runID": np.array([0, 0, 0]),
    ...         "eventID": np.array([0, 0, 0]),
    ...         "trackID": np.array([2, 1, 1]),
    ...         "time": np.array([1.0, 1.0, 2.0]),
    ...     },
    ... )
    >>> representation = EventTrackHit(data)
    >>> representation.track_counts().tolist(), representation.hit_counts().tolist()
    ([2], [2, 1])
    """

    TREE: ClassVar[GateTree] = GateTree.HITS

    def __init__(self, source: TreeData | pd.DataFrame) -> None:
        super().__init__(source)
        missing = [name for name in INDEX_BRANCHES if name not in self.data.columns]
        if missing:
            raise RepresentationError(
                f"Grouping hits into tracks and events needs the branches {missing}, "
                f"which the data does not hold. Available branches: {list(self.data.columns)}."
            )
        self._index = build_event_track_hit_index(
            self.data["runID"], self.data["eventID"], self.data["trackID"], self.data["time"]
        )

    @property
    def index(self) -> EventTrackHitIndex:
        """Which rows make up which hit, track and event."""
        return self._index

    @property
    def event_count(self) -> int:
        """Number of events."""
        return self._index.event_count

    @property
    def track_count(self) -> int:
        """Number of tracks."""
        return self._index.track_count

    @property
    def hit_count(self) -> int:
        """Number of hits."""
        return self._index.hit_count

    def __len__(self) -> int:
        """Return the number of events."""
        return self.event_count

    def track_counts(self) -> npt.NDArray[np.int64]:
        """Return the number of tracks of every event."""
        return self._index.tracks_per_event()

    def hit_counts(self) -> npt.NDArray[np.int64]:
        """Return the number of hits of every track."""
        return self._index.hits_per_track()

    @property
    def events(self) -> ElementSequence[Event]:
        """Events as objects, made when they are asked for."""
        return ElementSequence(lambda position: Event(self, position), 0, self.event_count)

    def values(self, field: FieldLike, level: ElementLevel | None = None) -> npt.NDArray[Any]:
        """Return a field for every element of a level, in the order of the index.

        Parameters
        ----------
        field : EventTrackHitField | str
            Field to read. A name that is not the value of a field is read as
            a branch describing hits.
        level : ElementLevel | None
            Level to read the field at; by default, the level it describes. A
            field is spread over the levels below its own: an event field read
            at the hit level gives every hit the value of its event.

        Returns
        -------
        numpy.ndarray
            One value per element of the level; two-dimensional for an array
            branch such as ``volumeID``.

        Raises
        ------
        ValueError
            If the level is above the one the field describes.
        BranchNotFoundError
            If the data does not hold the branch of the field.
        """
        resolved, own_level = _resolve(field)
        target = own_level if level is None else level
        if _DEPTH[target] < _DEPTH[own_level]:
            raise ValueError(
                f"'{_name(resolved)}' describes every {own_level.value}, so it has no single "
                f"value per {target.value}; reduce it to one first."
            )

        answer = self._own_level_values(resolved, own_level)
        if own_level is ElementLevel.EVENT and _DEPTH[target] >= _DEPTH[ElementLevel.TRACK]:
            answer = np.repeat(answer, self._index.tracks_per_event(), axis=0)
        if own_level is not ElementLevel.HIT and target is ElementLevel.HIT:
            answer = np.repeat(answer, self._index.hits_per_track(), axis=0)
        return answer

    def element_value(self, field: FieldLike, level: ElementLevel, position: int) -> Any:
        """Return the value of a field for one element.

        Parameters
        ----------
        field : EventTrackHitField | str
            Field to read.
        level : ElementLevel
            Level of the element.
        position : int
            Position of the element among every element of its level.

        Returns
        -------
        Any
            Value of the field; a row of values for an array branch.

        Raises
        ------
        ValueError
            If the field describes a level below the one of the element.
        BranchNotFoundError
            If the data does not hold the branch of the field.
        """
        resolved, own_level = _resolve(field)
        if _DEPTH[level] < _DEPTH[own_level]:
            raise ValueError(
                f"'{_name(resolved)}' describes every {own_level.value}, "
                f"so a {level.value} has no single value of it."
            )
        index = self._index
        track = position
        if level is ElementLevel.HIT:
            if resolved is EventTrackHitField.HIT_NUMBER:
                return (
                    position - int(index.track_offsets[_parent(index.track_offsets, position)]) + 1
                )
            if own_level is ElementLevel.HIT:
                return self.data[_name(resolved)][index.hit_rows[position]]
            track = _parent(index.track_offsets, position)
        event = track if level is ElementLevel.EVENT else _parent(index.event_offsets, track)

        if resolved is EventTrackHitField.TRACK_COUNT:
            return int(index.event_offsets[event + 1] - index.event_offsets[event])
        if resolved is EventTrackHitField.HIT_COUNT:
            return int(index.track_offsets[track + 1] - index.track_offsets[track])
        if resolved is EventTrackHitField.TRACK_NUMBER:
            return track - int(index.event_offsets[event]) + 1
        if own_level is ElementLevel.EVENT:
            track = int(index.event_offsets[event])
        return self.data[_name(resolved)][index.hit_rows[index.track_offsets[track]]]

    def events_frame(self, fields: Sequence[FieldLike] = DEFAULT_EVENT_FIELDS) -> pd.DataFrame:
        """Return a table with a row per event.

        Parameters
        ----------
        fields : Sequence[EventTrackHitField | str]
            Event fields to put in the table, in order.

        Returns
        -------
        pandas.DataFrame
            One column per field, named after its branch; the index is the
            position of every event, named ``eventIndex``.

        Raises
        ------
        ValueError
            If a field describes tracks or hits.
        """
        columns: dict[str, npt.NDArray[Any]] = {}
        self._add_fields(columns, fields, ElementLevel.EVENT)
        return pd.DataFrame(columns, index=pd.RangeIndex(self.event_count, name=EVENT_INDEX_COLUMN))

    def tracks_frame(self, fields: Sequence[FieldLike] = DEFAULT_TRACK_FIELDS) -> pd.DataFrame:
        """Return a table with a row per track.

        Parameters
        ----------
        fields : Sequence[EventTrackHitField | str]
            Track and event fields to put in the table, in order. An event
            field gives every track the value of its event.

        Returns
        -------
        pandas.DataFrame
            A ``eventIndex`` column naming the event of every track, then one
            column per field; the index is the position of every track, named
            ``trackIndex``.

        Raises
        ------
        ValueError
            If a field describes hits.
        """
        columns: dict[str, npt.NDArray[Any]] = {EVENT_INDEX_COLUMN: self._index.event_of_track()}
        self._add_fields(columns, fields, ElementLevel.TRACK)
        return pd.DataFrame(columns, index=pd.RangeIndex(self.track_count, name=TRACK_INDEX_COLUMN))

    def hits_frame(self, fields: Sequence[FieldLike] | None = None) -> pd.DataFrame:
        """Return a table with a row per hit.

        Parameters
        ----------
        fields : Sequence[EventTrackHitField | str] | None
            Fields of any level to put in the table, in order. By default,
            every branch of the data, in its order.

        Returns
        -------
        pandas.DataFrame
            ``eventIndex``, ``trackIndex``, ``trackNumber`` and ``hitNumber``
            columns saying where every hit stands, then one column per field,
            an array branch expanded as :meth:`TreeData.to_dataframe` does. The
            index is the label of the row of the data each hit was read from.
        """
        index = self._index
        columns: dict[str, npt.NDArray[Any]] = {
            EVENT_INDEX_COLUMN: np.repeat(index.event_of_track(), index.hits_per_track()),
            TRACK_INDEX_COLUMN: index.track_of_hit(),
        }
        structure = (EventTrackHitField.TRACK_NUMBER, EventTrackHitField.HIT_NUMBER)
        self._add_fields(columns, structure, ElementLevel.HIT)
        chosen = list(self.data.branch_names) if fields is None else fields
        self._add_fields(columns, chosen, ElementLevel.HIT)
        return pd.DataFrame(columns, index=self.row_labels[index.hit_rows])

    def check_consistency(self, fields: Sequence[FieldLike] | None = None) -> ConsistencyReport:
        """Check that track and event fields agree within their track and event.

        A track or event field is read from the first row of its track or
        event, which only means something when every row of it holds the same
        value. GATE writes them that way; data assembled by other means may
        not.

        Parameters
        ----------
        fields : Sequence[EventTrackHitField | str] | None
            Fields to check. By default, every track and event field whose
            branch the data holds.

        Returns
        -------
        ConsistencyReport
            Fields that disagree somewhere, with how often and where first.

        Raises
        ------
        ValueError
            If a field describes hits or is derived from the grouping.
        BranchNotFoundError
            If the data does not hold the branch of a field asked for.
        """
        if fields is None:
            chosen = [
                field
                for field in EventTrackHitField
                if field.level is not ElementLevel.HIT
                and not field.is_derived
                and field.value in self.data.columns
            ]
        else:
            chosen = [_checkable(field) for field in fields]

        inconsistent: dict[EventTrackHitField, int] = {}
        examples: dict[EventTrackHitField, int] = {}
        for field in chosen:
            differs = self._differs_within(field)
            if differs.any():
                inconsistent[field] = int(np.count_nonzero(differs))
                examples[field] = int(np.flatnonzero(differs)[0])
        return ConsistencyReport(tuple(chosen), inconsistent, examples)

    def __repr__(self) -> str:
        """Return the number of events, tracks and hits."""
        return (
            f"{type(self).__name__}(events={self.event_count}, "
            f"tracks={self.track_count}, hits={self.hit_count})"
        )

    def _own_level_values(self, field: FieldLike, level: ElementLevel) -> npt.NDArray[Any]:
        """Return a field at the level it describes."""
        index = self._index
        if field is EventTrackHitField.TRACK_COUNT:
            return index.tracks_per_event()
        if field is EventTrackHitField.HIT_COUNT:
            return index.hits_per_track()
        if field is EventTrackHitField.TRACK_NUMBER:
            return _numbers(index.event_offsets)
        if field is EventTrackHitField.HIT_NUMBER:
            return _numbers(index.track_offsets)
        column = self.data[_name(field)]
        if level is ElementLevel.HIT:
            return column[index.hit_rows]
        if level is ElementLevel.TRACK:
            return column[index.first_hit_rows()]
        return column[index.first_track_rows()]

    def _add_fields(
        self, columns: dict[str, npt.NDArray[Any]], fields: Sequence[FieldLike], level: ElementLevel
    ) -> None:
        """Add fields read at a level to the columns of a table.

        A field whose column is already there is not added twice. An array
        branch is expanded into one column per component.

        Raises
        ------
        ValueError
            If a field describes a level below the one of the table.
        """
        for field in fields:
            name = _name(_resolve(field)[0])
            if name in columns:
                continue
            values = self.values(field, level)
            if values.ndim == 1:
                columns[name] = values
            else:
                for component in range(values.shape[1]):
                    columns[f"{name}_{component}"] = values[:, component]

    def _differs_within(self, field: EventTrackHitField) -> npt.NDArray[np.bool_]:
        """Return which tracks or events hold more than one value of a field."""
        index = self._index
        column = self.data[field.value][index.hit_rows]
        shared = self.values(field, ElementLevel.HIT)
        differs = column != shared
        if np.issubdtype(column.dtype, np.floating):
            differs &= ~(np.isnan(column) & np.isnan(shared))
        per_track = reduce_hits_to_tracks(index, differs, "any")
        if field.level is ElementLevel.TRACK:
            return per_track
        return reduce_tracks_to_events(index, per_track, "any")


def _checkable(field: FieldLike) -> EventTrackHitField:
    """Return a field whose value can disagree within a track or an event.

    Raises
    ------
    ValueError
        If the field describes hits or is derived from the grouping.
    """
    resolved, level = _resolve(field)
    if not isinstance(resolved, EventTrackHitField) or level is ElementLevel.HIT:
        raise ValueError(
            f"'{_name(resolved)}' describes hits; only track and event fields "
            "can disagree within a track or an event."
        )
    if resolved.is_derived:
        raise ValueError(f"'{resolved.value}' is given by the grouping and cannot disagree.")
    return resolved


def _resolve(field: FieldLike) -> tuple[FieldLike, ElementLevel]:
    """Return a field and the level it describes.

    A name that is the value of a field is that field; any other name is a
    branch describing hits.
    """
    if isinstance(field, EventTrackHitField):
        return field, field.level
    try:
        member = EventTrackHitField(field)
    except ValueError:
        return field, ElementLevel.HIT
    return member, member.level


def _name(field: FieldLike) -> str:
    """Return the branch or column name of a field."""
    return field.value if isinstance(field, EventTrackHitField) else field


def _numbers(offsets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the position of every element within its segment, from 1."""
    counts = np.diff(offsets)
    return np.arange(int(offsets[-1]), dtype=np.int64) - np.repeat(offsets[:-1], counts) + 1


def _parent(offsets: npt.NDArray[np.int64], position: int) -> int:
    """Return the segment of the offsets a position falls in."""
    return int(np.searchsorted(offsets, position, side="right")) - 1
