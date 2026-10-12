"""An event, a track and a hit, as objects to look at one at a time.

The elements are views: each holds the representation it belongs to and its
own position in it, and reads a value from the columns when it is asked for
one. Nothing is built ahead - a representation of ten million hits creates no
objects until one is asked for, and an object asked for costs two fields.

They are there to look at the data the way it is thought of - this event,
its second track, the first hit of it. Questions asked of every event at once
are answered by the selectors and tables of the representation instead, which
do not go through objects at all.

Public objects:

Event
    One event: a run and an event identifier, and the tracks recorded in it.
Track
    One track of an event, and the hits it left.
Hit
    One hit of a track: one row of the "Hits" tree.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np
import numpy.typing as npt

from opengate_gate_tree.representation.fields import ElementLevel, EventTrackHitField
from opengate_gate_tree.tree.hits.positronium import decode_positronium_value

if TYPE_CHECKING:
    from opengate_gate_tree.representation.eventtrackhit import EventTrackHit

Element = TypeVar("Element")


class ElementSequence(Sequence[Element]):
    """Elements made on request, one position at a time.

    Parameters
    ----------
    make : Callable[[int], Element]
        Makes the element at a position of the representation.
    start, stop : int
        Positions the sequence covers.
    """

    def __init__(self, make: Callable[[int], Element], start: int, stop: int) -> None:
        self._make = make
        self._start = start
        self._stop = stop

    def __len__(self) -> int:
        """Return the number of elements."""
        return self._stop - self._start

    @overload
    def __getitem__(self, position: int) -> Element: ...

    @overload
    def __getitem__(self, position: slice) -> list[Element]: ...

    def __getitem__(self, position: int | slice) -> Element | list[Element]:
        """Return the element at a position, or a list of them for a slice.

        Raises
        ------
        IndexError
            If the position is outside the sequence.
        """
        if isinstance(position, slice):
            return [self[found] for found in range(*position.indices(len(self)))]
        if not -len(self) <= position < len(self):
            raise IndexError(f"Position {position} is outside {len(self)} element(s).")
        return self._make(self._start + position % len(self))

    def __iter__(self) -> Iterator[Element]:
        """Make the elements one after another."""
        for position in range(self._start, self._stop):
            yield self._make(position)

    def __repr__(self) -> str:
        """Return a representation that makes no element."""
        return f"{type(self).__name__}(elements={len(self)})"


@dataclass(frozen=True, slots=True)
class Event:
    """One event: a run and an event identifier, and the tracks recorded in it.

    Attributes
    ----------
    representation : EventTrackHit
        Representation the event belongs to.
    index : int
        Position of the event in it, from 0.
    """

    representation: "EventTrackHit"
    index: int

    @property
    def run_id(self) -> int:
        """Run the event belongs to."""
        return int(self.value(EventTrackHitField.RUN_ID))

    @property
    def event_id(self) -> int:
        """Identifier of the event within its run."""
        return int(self.value(EventTrackHitField.EVENT_ID))

    @property
    def source_type(self) -> IntEnum:
        """Model of the PositroniumSource that emitted the event."""
        return _decoded(self.value(EventTrackHitField.SOURCE_TYPE), EventTrackHitField.SOURCE_TYPE)

    @property
    def track_count(self) -> int:
        """Number of tracks of the event."""
        return int(self.value(EventTrackHitField.TRACK_COUNT))

    @property
    def tracks(self) -> ElementSequence["Track"]:
        """Tracks of the event, in the order of their identifiers."""
        offsets = self.representation.index.event_offsets
        return ElementSequence(
            lambda position: Track(self.representation, position),
            int(offsets[self.index]),
            int(offsets[self.index + 1]),
        )

    def value(self, field: EventTrackHitField | str) -> Any:
        """Return the value of an event field.

        Raises
        ------
        ValueError
            If the field does not describe an event.
        """
        return self.representation.element_value(field, ElementLevel.EVENT, self.index)

    def __repr__(self) -> str:
        """Return the identifiers of the event and its number of tracks."""
        return f"Event(runID={self.run_id}, eventID={self.event_id}, tracks={self.track_count})"


@dataclass(frozen=True, slots=True)
class Track:
    """One track of an event, and the hits it left.

    Attributes
    ----------
    representation : EventTrackHit
        Representation the track belongs to.
    index : int
        Position of the track among every track of the representation, from 0.
    """

    representation: "EventTrackHit"
    index: int

    @property
    def event(self) -> Event:
        """Event the track belongs to."""
        offsets = self.representation.index.event_offsets
        return Event(self.representation, _parent(offsets, self.index))

    @property
    def number(self) -> int:
        """Position of the track in its event, from 1."""
        return int(self.value(EventTrackHitField.TRACK_NUMBER))

    @property
    def track_id(self) -> int:
        """Identifier of the track within its event."""
        return int(self.value(EventTrackHitField.TRACK_ID))

    @property
    def parent_id(self) -> int:
        """Identifier of the track that produced this one; 0 for a primary."""
        return int(self.value(EventTrackHitField.PARENT_ID))

    @property
    def pdg_encoding(self) -> int:
        """PDG code of the particle."""
        return int(self.value(EventTrackHitField.PDG_ENCODING))

    @property
    def gamma_type(self) -> IntEnum:
        """Kind of gamma, as a PositroniumSource names it."""
        return _decoded(self.value(EventTrackHitField.GAMMA_TYPE), EventTrackHitField.GAMMA_TYPE)

    @property
    def decay_type(self) -> IntEnum:
        """Decay channel the gamma came through."""
        return _decoded(self.value(EventTrackHitField.DECAY_TYPE), EventTrackHitField.DECAY_TYPE)

    @property
    def decay_index(self) -> int:
        """Component of the sampled decay; -1 when the track carries none."""
        return int(self.value(EventTrackHitField.DECAY_INDEX))

    @property
    def source_position(self) -> npt.NDArray[np.float64]:
        """Where the particle was born, in mm."""
        return _vector(
            self,
            EventTrackHitField.SOURCE_POS_X,
            EventTrackHitField.SOURCE_POS_Y,
            EventTrackHitField.SOURCE_POS_Z,
        )

    @property
    def hit_count(self) -> int:
        """Number of hits of the track."""
        return int(self.value(EventTrackHitField.HIT_COUNT))

    @property
    def hits(self) -> ElementSequence["Hit"]:
        """Hits of the track, in the order of their time."""
        offsets = self.representation.index.track_offsets
        return ElementSequence(
            lambda position: Hit(self.representation, position),
            int(offsets[self.index]),
            int(offsets[self.index + 1]),
        )

    def value(self, field: EventTrackHitField | str) -> Any:
        """Return the value of a track field, or of an event field of its event.

        Raises
        ------
        ValueError
            If the field describes a hit.
        """
        return self.representation.element_value(field, ElementLevel.TRACK, self.index)

    def __repr__(self) -> str:
        """Return the identifier of the track and its number of hits."""
        return f"Track(trackID={self.track_id}, hits={self.hit_count})"


@dataclass(frozen=True, slots=True)
class Hit:
    """One hit of a track: one row of the "Hits" tree.

    Attributes
    ----------
    representation : EventTrackHit
        Representation the hit belongs to.
    index : int
        Position of the hit among every hit of the representation, from 0.
    """

    representation: "EventTrackHit"
    index: int

    @property
    def track(self) -> Track:
        """Track the hit belongs to."""
        offsets = self.representation.index.track_offsets
        return Track(self.representation, _parent(offsets, self.index))

    @property
    def number(self) -> int:
        """Position of the hit in its track, from 1."""
        return int(self.value(EventTrackHitField.HIT_NUMBER))

    @property
    def row(self) -> Any:
        """Label of the row of the data the hit was read from."""
        rows = self.representation.index.hit_rows
        return self.representation.row_labels[rows[self.index]]

    @property
    def edep(self) -> float:
        """Energy deposited, in MeV."""
        return float(self.value(EventTrackHitField.EDEP))

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """Where the hit happened, in mm."""
        return _vector(
            self, EventTrackHitField.POS_X, EventTrackHitField.POS_Y, EventTrackHitField.POS_Z
        )

    @property
    def momentum_direction(self) -> npt.NDArray[np.float64]:
        """Direction of the particle after the interaction, as a unit vector."""
        return _vector(
            self,
            EventTrackHitField.MOM_DIR_X,
            EventTrackHitField.MOM_DIR_Y,
            EventTrackHitField.MOM_DIR_Z,
        )

    @property
    def time(self) -> float:
        """Time of the hit since the start of the run, in s."""
        return float(self.value(EventTrackHitField.TIME))

    @property
    def local_time(self) -> float:
        """Time of the hit since the track began, in s."""
        return float(self.value(EventTrackHitField.TRACK_LOCAL_TIME))

    @property
    def process_name(self) -> str:
        """Process of the interaction, as Geant4 names it."""
        return str(self.value(EventTrackHitField.PROCESS_NAME))

    def value(self, field: EventTrackHitField | str) -> Any:
        """Return the value of any field, read for this hit."""
        return self.representation.element_value(field, ElementLevel.HIT, self.index)

    def __repr__(self) -> str:
        """Return the row the hit was read from and its position in the track."""
        return f"Hit(row={self.row!r}, number={self.number})"


def _parent(offsets: npt.NDArray[np.int64], position: int) -> int:
    """Return the segment of the offsets a position falls in."""
    return int(np.searchsorted(offsets, position, side="right")) - 1


def _decoded(value: Any, field: EventTrackHitField) -> IntEnum:
    """Return what a value of a PositroniumSource branch means."""
    return decode_positronium_value(field.value, int(value))


def _vector(element: Track | Hit, *fields: EventTrackHitField) -> npt.NDArray[np.float64]:
    """Return three fields of an element as one vector."""
    return np.array([element.value(field) for field in fields], dtype=np.float64)
