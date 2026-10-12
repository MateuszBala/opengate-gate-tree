"""What an event, a track and a hit carry, and where each of it is read from.

A row of the "Hits" tree holds everything about one hit, and the same values
about its track and its event repeated on every row of them. A field names
one of those values together with the level it describes:

- an **event** field is the same on every row of the event, such as its run
  and event identifiers, or the model of the source that emitted it;
- a **track** field is the same on every row of the track, such as its
  identifier, the kind of gamma it is, or where it was born;
- a **hit** field is what the row itself says: the energy deposited, where,
  when, by which process.

The value of a field is the name of the branch it is read from, so a field
names a column of every table built from it, and that column is named as GATE
named it. A track or event field is read from the first row of its track or
event: those rows agree, which a representation checks on request rather than
on every reading.

Four fields are not read from a branch, they are what the grouping says:
how many tracks an event has and hits a track has, and the position of a
track in its event and of a hit in its track, counted from 1.

Public objects:

ElementLevel
    Element of the representation a field describes.
EventTrackHitField
    Value carried by an event, a track or a hit.
FIELD_LEVELS
    Level every field describes.
DERIVED_FIELDS
    Fields the grouping gives rather than a branch.
"""

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final


class ElementLevel(Enum):
    """Element of the representation a field describes."""

    EVENT = "event"
    TRACK = "track"
    HIT = "hit"


class EventTrackHitField(Enum):
    """Value carried by an event, a track or a hit.

    The value of a member is the branch it is read from, or the column name
    of a derived field.
    """

    # Event fields.
    RUN_ID = "runID"
    EVENT_ID = "eventID"
    SOURCE_TYPE = "sourceType"
    TRACK_COUNT = "trackCount"

    # Track fields.
    TRACK_ID = "trackID"
    PARENT_ID = "parentID"
    PDG_ENCODING = "PDGEncoding"
    GAMMA_TYPE = "gammaType"
    DECAY_TYPE = "decayType"
    DECAY_INDEX = "decayIndex"
    SOURCE_POS_X = "sourcePosX"
    SOURCE_POS_Y = "sourcePosY"
    SOURCE_POS_Z = "sourcePosZ"
    SOURCE_ENERGY = "sourceEnergy"
    HIT_COUNT = "hitCount"
    TRACK_NUMBER = "trackNumber"

    # Hit fields.
    EDEP = "edep"
    POS_X = "posX"
    POS_Y = "posY"
    POS_Z = "posZ"
    LOCAL_POS_X = "localPosX"
    LOCAL_POS_Y = "localPosY"
    LOCAL_POS_Z = "localPosZ"
    MOM_DIR_X = "momDirX"
    MOM_DIR_Y = "momDirY"
    MOM_DIR_Z = "momDirZ"
    TIME = "time"
    TRACK_LOCAL_TIME = "trackLocalTime"
    STEP_LENGTH = "stepLength"
    TRACK_LENGTH = "trackLength"
    PROCESS_NAME = "processName"
    N_PHANTOM_COMPTON = "nPhantomCompton"
    N_CRYSTAL_COMPTON = "nCrystalCompton"
    N_PHANTOM_RAYLEIGH = "nPhantomRayleigh"
    N_CRYSTAL_RAYLEIGH = "nCrystalRayleigh"
    N_INTERACTIONS = "nInteractions"
    VOLUME_ID = "volumeID"
    HIT_NUMBER = "hitNumber"

    @property
    def level(self) -> ElementLevel:
        """Level the field describes."""
        return FIELD_LEVELS[self]

    @property
    def is_derived(self) -> bool:
        """Whether the grouping gives the field rather than a branch."""
        return self in DERIVED_FIELDS


_EVENT_FIELDS: Final[tuple[EventTrackHitField, ...]] = (
    EventTrackHitField.RUN_ID,
    EventTrackHitField.EVENT_ID,
    EventTrackHitField.SOURCE_TYPE,
    EventTrackHitField.TRACK_COUNT,
)

_TRACK_FIELDS: Final[tuple[EventTrackHitField, ...]] = (
    EventTrackHitField.TRACK_ID,
    EventTrackHitField.PARENT_ID,
    EventTrackHitField.PDG_ENCODING,
    EventTrackHitField.GAMMA_TYPE,
    EventTrackHitField.DECAY_TYPE,
    EventTrackHitField.DECAY_INDEX,
    EventTrackHitField.SOURCE_POS_X,
    EventTrackHitField.SOURCE_POS_Y,
    EventTrackHitField.SOURCE_POS_Z,
    EventTrackHitField.SOURCE_ENERGY,
    EventTrackHitField.HIT_COUNT,
    EventTrackHitField.TRACK_NUMBER,
)

# Level every field describes.
FIELD_LEVELS: Final[Mapping[EventTrackHitField, ElementLevel]] = MappingProxyType(
    {
        field: (
            ElementLevel.EVENT
            if field in _EVENT_FIELDS
            else ElementLevel.TRACK
            if field in _TRACK_FIELDS
            else ElementLevel.HIT
        )
        for field in EventTrackHitField
    }
)

# Fields the grouping gives rather than a branch.
DERIVED_FIELDS: Final[frozenset[EventTrackHitField]] = frozenset(
    {
        EventTrackHitField.TRACK_COUNT,
        EventTrackHitField.HIT_COUNT,
        EventTrackHitField.TRACK_NUMBER,
        EventTrackHitField.HIT_NUMBER,
    }
)
