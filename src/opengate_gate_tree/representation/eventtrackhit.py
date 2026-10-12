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

Selecting follows the convention of the filters of the pandas view: a method
named ``is_*`` answers with a mask of the level it asks about, and one named
``select_*`` or ``with_*`` answers with a new representation of what was kept.
A selection never changes the representation it was made on, and shares its
columns. Leaving out every hit of a track leaves out the track, and leaving
out every track of an event leaves out the event, unless ``keep_empty`` says
otherwise. A count asked for after a selection counts what the selection kept,
so the order of the calls is part of what they ask.

Public objects:

EventTrackHit
    The "Hits" tree read as events, tracks and hits.
ConsistencyReport
    Fields whose value differs within a track or an event that should share it.
INDEX_BRANCHES
    Branches the representation is built from.
STRUCTURE_COLUMNS
    Columns a written representation adds to say where every hit stands.
"""

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
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
    position_in_event,
    position_in_track,
    reduce_hits_to_tracks,
    reduce_tracks_to_events,
)
from opengate_gate_tree.tree.filters import (
    InclusiveSide,
    is_decay_type,
    is_gamma_type,
    is_in_range,
    is_process,
    is_source_type,
)
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process
from opengate_gate_tree.tree.treedata import TreeData

FieldLike = EventTrackHitField | str
MaskLike = npt.NDArray[np.bool_] | pd.Series

# Branches the representation is built from.
INDEX_BRANCHES: Final[tuple[str, ...]] = ("runID", "eventID", "trackID", "time")

# Columns the tables add to say where an element stands.
EVENT_INDEX_COLUMN: Final[str] = "eventIndex"
TRACK_INDEX_COLUMN: Final[str] = "trackIndex"

# Columns a written representation adds to say where every hit stands, with
# the type they are written in.
STRUCTURE_COLUMNS: Final[Mapping[str, type[np.integer[Any]]]] = MappingProxyType(
    {
        EVENT_INDEX_COLUMN: np.int64,
        EventTrackHitField.TRACK_NUMBER.value: np.int32,
        EventTrackHitField.HIT_NUMBER.value: np.int32,
        EventTrackHitField.TRACK_COUNT.value: np.int32,
        EventTrackHitField.HIT_COUNT.value: np.int32,
    }
)

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

    @classmethod
    def _from_index(
        cls, data: TreeData, row_labels: pd.Index, index: EventTrackHitIndex
    ) -> "EventTrackHit":
        """Return a representation of the same data under another index."""
        representation = cls.__new__(cls)
        representation._data = data
        representation._row_labels = row_labels
        representation._index = index
        return representation

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
            return self.data[_name(resolved)][index.event_rows[event]]
        return self.data[_name(resolved)][index.track_rows[track]]

    def select_events(self, mask: MaskLike, keep_empty: bool = False) -> "EventTrackHit":
        """Return the events a mask keeps, with their tracks and hits.

        Parameters
        ----------
        mask : numpy.ndarray | pandas.Series
            One boolean per event, in the order of the events; a column of the
            events table works as well.
        keep_empty : bool
            Accepted for every selection alike; leaving out events leaves no
            track or event empty.

        Returns
        -------
        EventTrackHit
            Representation of the same data, reduced to the events kept.

        Raises
        ------
        ValueError
            If the mask is not one boolean per event.
        """
        return self._selected(event_mask=_as_mask(mask), keep_empty=keep_empty)

    def select_tracks(self, mask: MaskLike, keep_empty: bool = False) -> "EventTrackHit":
        """Return the tracks a mask keeps, with their hits.

        Parameters
        ----------
        mask : numpy.ndarray | pandas.Series
            One boolean per track, in the order of the tracks.
        keep_empty : bool
            Keep an event whose tracks were all left out.

        Returns
        -------
        EventTrackHit
            Representation of the same data, reduced to the tracks kept.

        Raises
        ------
        ValueError
            If the mask is not one boolean per track.
        """
        return self._selected(track_mask=_as_mask(mask), keep_empty=keep_empty)

    def select_hits(self, mask: MaskLike, keep_empty: bool = False) -> "EventTrackHit":
        """Return the hits a mask keeps.

        Parameters
        ----------
        mask : numpy.ndarray | pandas.Series
            One boolean per hit, in the order of the hits.
        keep_empty : bool
            Keep a track whose hits were all left out, and an event whose
            tracks were.

        Returns
        -------
        EventTrackHit
            Representation of the same data, reduced to the hits kept.

        Raises
        ------
        ValueError
            If the mask is not one boolean per hit.
        """
        return self._selected(hit_mask=_as_mask(mask), keep_empty=keep_empty)

    def is_source_type(self, *types: SourceType) -> npt.NDArray[np.bool_]:
        """Return which events a PositroniumSource emitted with one of the given models.

        Raises
        ------
        ValueError
            If no model is given, or a value is not a :class:`SourceType`.
        """
        return _mask_of(is_source_type, self.values(EventTrackHitField.SOURCE_TYPE), types)

    def is_gamma_type(self, *types: GammaType) -> npt.NDArray[np.bool_]:
        """Return which tracks are one of the given kinds of gamma.

        Raises
        ------
        ValueError
            If no kind is given, or a value is not a :class:`GammaType`.
        """
        return _mask_of(is_gamma_type, self.values(EventTrackHitField.GAMMA_TYPE), types)

    def is_decay_type(self, *types: DecayType) -> npt.NDArray[np.bool_]:
        """Return which tracks came through one of the given decay channels.

        Raises
        ------
        ValueError
            If no channel is given, or a value is not a :class:`DecayType`.
        """
        return _mask_of(is_decay_type, self.values(EventTrackHitField.DECAY_TYPE), types)

    def is_particle(self, *pdg_codes: int) -> npt.NDArray[np.bool_]:
        """Return which tracks are one of the given particles, by PDG code.

        Raises
        ------
        ValueError
            If no code is given.
        """
        if not pdg_codes:
            raise ValueError("Selecting by particle needs at least one PDG code to select.")
        return np.isin(self.values(EventTrackHitField.PDG_ENCODING), pdg_codes)

    def is_process(self, *processes: Process | str) -> npt.NDArray[np.bool_]:
        """Return which hits were made by one of the given processes.

        Raises
        ------
        ValueError
            If no process is given.
        """
        return _mask_of(is_process, self.values(EventTrackHitField.PROCESS_NAME), processes)

    def is_value(
        self,
        field: FieldLike,
        *,
        eq: Any = None,
        isin: Sequence[Any] | None = None,
        between: tuple[float, float] | None = None,
        inclusive: InclusiveSide = "both",
    ) -> npt.NDArray[np.bool_]:
        """Return which elements of the level of a field hold a value asked for.

        Exactly one of ``eq``, ``isin`` and ``between`` says what is asked for.

        Parameters
        ----------
        field : EventTrackHitField | str
            Field to test, at the level it describes.
        eq : Any
            Value the field has to equal.
        isin : Sequence[Any] | None
            Values one of which the field has to equal.
        between : tuple[float, float] | None
            Range the field has to fall in; a missing value falls outside.
        inclusive : {"both", "neither", "left", "right"}
            Which ends of ``between`` belong to the range.

        Returns
        -------
        numpy.ndarray
            One boolean per element of the level of the field.

        Raises
        ------
        ValueError
            If not exactly one of ``eq``, ``isin`` and ``between`` is given.
        """
        asked = [
            name
            for name, given in (("eq", eq), ("isin", isin), ("between", between))
            if given is not None
        ]
        if len(asked) != 1:
            raise ValueError(
                "Say what a field is tested for with exactly one of eq, isin and between; "
                f"got {asked or 'none'}."
            )
        values = self.values(field)
        if eq is not None:
            return np.asarray(values == eq, dtype=bool)
        if isin is not None:
            return np.isin(values, list(isin))
        low, high = between  # type: ignore[misc]
        return is_in_range(pd.Series(values), low, high, inclusive).to_numpy()

    def with_source_type(self, *types: SourceType, keep_empty: bool = False) -> "EventTrackHit":
        """Return the events emitted with one of the given source models."""
        return self.select_events(self.is_source_type(*types), keep_empty)

    def with_gamma_type(self, *types: GammaType, keep_empty: bool = False) -> "EventTrackHit":
        """Return the tracks that are one of the given kinds of gamma."""
        return self.select_tracks(self.is_gamma_type(*types), keep_empty)

    def with_decay_type(self, *types: DecayType, keep_empty: bool = False) -> "EventTrackHit":
        """Return the tracks that came through one of the given decay channels."""
        return self.select_tracks(self.is_decay_type(*types), keep_empty)

    def with_particle(self, *pdg_codes: int, keep_empty: bool = False) -> "EventTrackHit":
        """Return the tracks of the given particles, by PDG code."""
        return self.select_tracks(self.is_particle(*pdg_codes), keep_empty)

    def with_process(self, *processes: Process | str, keep_empty: bool = False) -> "EventTrackHit":
        """Return the hits made by one of the given processes."""
        return self.select_hits(self.is_process(*processes), keep_empty)

    def with_value(
        self,
        field: FieldLike,
        *,
        eq: Any = None,
        isin: Sequence[Any] | None = None,
        between: tuple[float, float] | None = None,
        inclusive: InclusiveSide = "both",
        keep_empty: bool = False,
    ) -> "EventTrackHit":
        """Return the elements of the level of a field that hold a value asked for.

        See :meth:`is_value` for the parameters.
        """
        mask = self.is_value(field, eq=eq, isin=isin, between=between, inclusive=inclusive)
        level = _resolve(field)[1]
        if level is ElementLevel.EVENT:
            return self.select_events(mask, keep_empty)
        if level is ElementLevel.TRACK:
            return self.select_tracks(mask, keep_empty)
        return self.select_hits(mask, keep_empty)

    def is_track_count(
        self,
        eq: int | None = None,
        *,
        ne: int | None = None,
        ge: int | None = None,
        le: int | None = None,
        gt: int | None = None,
        lt: int | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return which events have a number of tracks asked for.

        The tracks counted are the ones the representation holds, so a count
        asked for after a selection counts what the selection kept.

        Parameters
        ----------
        eq : int | None
            Number of tracks an event has to have; it cannot be combined with
            the other comparisons.
        ne, ge, le, gt, lt : int | None
            Comparisons the number has to pass, all of them at once.

        Returns
        -------
        numpy.ndarray
            One boolean per event.

        Raises
        ------
        ValueError
            If no comparison is given, or ``eq`` is given with another one.
        """
        return _count_mask(self.track_counts(), eq=eq, ne=ne, ge=ge, le=le, gt=gt, lt=lt)

    def is_hit_count(
        self,
        eq: int | None = None,
        *,
        ne: int | None = None,
        ge: int | None = None,
        le: int | None = None,
        gt: int | None = None,
        lt: int | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return which tracks have a number of hits asked for.

        See :meth:`is_track_count` for the parameters; this one counts the
        hits of every track.
        """
        return _count_mask(self.hit_counts(), eq=eq, ne=ne, ge=ge, le=le, gt=gt, lt=lt)

    def is_nth_track(self, position: int) -> npt.NDArray[np.bool_]:
        """Return which tracks stand at a position in their event.

        Parameters
        ----------
        position : int
            Position from 1, in the order of the track identifiers; a negative
            one counts from the last track, -1 being the last.

        Returns
        -------
        numpy.ndarray
            One boolean per track; an event with fewer tracks has none.

        Raises
        ------
        ValueError
            If the position is 0.
        """
        every_track = np.ones(self.track_count, dtype=bool)
        return _at_position(position_in_event, self._index, every_track, position)

    def is_nth_hit(self, position: int) -> npt.NDArray[np.bool_]:
        """Return which hits stand at a position in their track.

        See :meth:`is_nth_track` for the parameter; hits are in the order of
        their time.
        """
        every_hit = np.ones(self.hit_count, dtype=bool)
        return _at_position(position_in_track, self._index, every_hit, position)

    def with_track_count(
        self,
        eq: int | None = None,
        *,
        ne: int | None = None,
        ge: int | None = None,
        le: int | None = None,
        gt: int | None = None,
        lt: int | None = None,
    ) -> "EventTrackHit":
        """Return the events that have a number of tracks asked for.

        See :meth:`is_track_count` for the parameters.
        """
        return self.select_events(self.is_track_count(eq, ne=ne, ge=ge, le=le, gt=gt, lt=lt))

    def with_hit_count(
        self,
        eq: int | None = None,
        *,
        ne: int | None = None,
        ge: int | None = None,
        le: int | None = None,
        gt: int | None = None,
        lt: int | None = None,
        keep_empty: bool = False,
    ) -> "EventTrackHit":
        """Return the tracks that have a number of hits asked for.

        See :meth:`is_track_count` for the parameters; ``keep_empty`` keeps
        an event whose tracks were all left out.
        """
        mask = self.is_hit_count(eq, ne=ne, ge=ge, le=le, gt=gt, lt=lt)
        return self.select_tracks(mask, keep_empty)

    def nth_track(self, position: int, keep_empty: bool = False) -> "EventTrackHit":
        """Return the track at a position of every event.

        See :meth:`is_nth_track` for the parameter.
        """
        return self.select_tracks(self.is_nth_track(position), keep_empty)

    def nth_hit(self, position: int, keep_empty: bool = False) -> "EventTrackHit":
        """Return the hit at a position of every track.

        See :meth:`is_nth_track` for the parameter.
        """
        return self.select_hits(self.is_nth_hit(position), keep_empty)

    def first_hits(self) -> "EventTrackHit":
        """Return the first hit of every track."""
        return self.nth_hit(1)

    def last_hits(self) -> "EventTrackHit":
        """Return the last hit of every track."""
        return self.nth_hit(-1)

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

    def to_tree_data(self) -> TreeData:
        """Return the hits as a flat tree, the way the representation is written.

        Every branch of the data is kept, its rows in the order of the
        representation, and the columns in :data:`STRUCTURE_COLUMNS` are added:
        the position of the event of every hit from 0, the position of its
        track in the event and of the hit in the track from 1, and the number
        of tracks of its event and of hits of its track. Identifiers written by
        GATE are kept as they are. A column of the data already named like one
        of them is replaced, since it described another grouping.

        Returns
        -------
        TreeData
            One entry per hit.

        Raises
        ------
        RepresentationError
            If a track holds no hits or an event no tracks: a flat tree has no
            row to write them in.
        """
        index = self._index
        if np.any(index.hits_per_track() == 0) or np.any(index.tracks_per_event() == 0):
            raise RepresentationError(
                "A track without hits, or an event without tracks, has no row to be written "
                "in; select without keep_empty before writing."
            )
        columns = {
            name: column[index.hit_rows]
            for name, column in self.data.columns.items()
            if name not in STRUCTURE_COLUMNS
        }
        derived = {
            EVENT_INDEX_COLUMN: np.repeat(index.event_of_track(), index.hits_per_track()),
            EventTrackHitField.TRACK_NUMBER.value: self.values(
                EventTrackHitField.TRACK_NUMBER, ElementLevel.HIT
            ),
            EventTrackHitField.HIT_NUMBER.value: self.values(EventTrackHitField.HIT_NUMBER),
            EventTrackHitField.TRACK_COUNT.value: self.values(
                EventTrackHitField.TRACK_COUNT, ElementLevel.HIT
            ),
            EventTrackHitField.HIT_COUNT.value: self.values(
                EventTrackHitField.HIT_COUNT, ElementLevel.HIT
            ),
        }
        for name, values in derived.items():
            columns[name] = values.astype(STRUCTURE_COLUMNS[name])
        return TreeData(self.TREE, columns)

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

    def _selected(
        self,
        event_mask: npt.NDArray[Any] | None = None,
        track_mask: npt.NDArray[Any] | None = None,
        hit_mask: npt.NDArray[Any] | None = None,
        keep_empty: bool = False,
    ) -> "EventTrackHit":
        """Return the representation of what the masks keep."""
        index = self._index.select(event_mask, track_mask, hit_mask, keep_empty)
        return type(self)._from_index(self.data, self.row_labels, index)

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
            return column[index.track_rows]
        return column[index.event_rows]

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


def _count_mask(
    counts: npt.NDArray[np.int64],
    eq: int | None,
    ne: int | None,
    ge: int | None,
    le: int | None,
    gt: int | None,
    lt: int | None,
) -> npt.NDArray[np.bool_]:
    """Return which counts pass every comparison given.

    Raises
    ------
    ValueError
        If no comparison is given, or ``eq`` is given with another one.
    """
    comparisons = {"ne": ne, "ge": ge, "le": le, "gt": gt, "lt": lt}
    given = {name: value for name, value in comparisons.items() if value is not None}
    if eq is not None:
        if given:
            raise ValueError(
                f"A count equal to {eq} cannot also be compared with {sorted(given)}; "
                "give either eq or the other comparisons."
            )
        return np.asarray(counts == eq, dtype=bool)
    if not given:
        raise ValueError("Selecting by a count needs eq, ne, ge, le, gt or lt.")
    passed = np.ones(len(counts), dtype=bool)
    for name, value in given.items():
        passed &= getattr(operator, name)(counts, value)
    return passed


def _at_position(
    number: Callable[..., npt.NDArray[np.int64]],
    index: EventTrackHitIndex,
    kept: npt.NDArray[np.bool_],
    position: int,
) -> npt.NDArray[np.bool_]:
    """Return which elements stand at a position from 1, or from the end when negative.

    Raises
    ------
    ValueError
        If the position is 0.
    """
    if position == 0:
        raise ValueError("Positions count from 1, and from -1 backwards; 0 is neither.")
    positions = number(index, kept, from_end=position < 0)
    return np.asarray(positions == abs(position), dtype=bool)


def _as_mask(mask: MaskLike) -> npt.NDArray[Any]:
    """Return a mask as an array; a column is read by position, not by label."""
    return mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)


def _mask_of(
    test: Callable[..., pd.Series], values: npt.NDArray[Any], wanted: Sequence[Any]
) -> npt.NDArray[np.bool_]:
    """Return what a filter of the pandas view answers about an array of values."""
    return np.asarray(test(pd.Series(values), *wanted).to_numpy(), dtype=bool)


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
