"""A lazy analysis of a "Hits" tree read as events, tracks and hits.

An analysis stands at one level at a time - the events, the tracks or the
hits - and every condition is asked of the elements of that level. It starts
at the events. ``tracks()`` and ``hits()`` move it down; ``events()`` moves it
back. Given a count - ``tracks(2)``, ``hits(ge=2)`` - they first keep the
events, or the tracks, with that number of tracks, or hits, kept so far.

A condition keeps the elements that pass it and leaves out the rest: it is a
filter. A track whose hits were all left out is left out with them, and an
event whose tracks were all left out is left out too. A condition on a field
of a level above the one the analysis stands at is asked of that level - the
source model of an event can be asked while standing at its tracks - but a
field of a level below has to be moved to first: a track has no single energy
deposit.

``track(i)`` and ``hit(j)`` stand at one position of every parent, counted
from 1 among the elements kept so far, and from -1 backwards. A condition
asked there decides about the parent: a track whose first hit is not a
Compton scattering is left out with all its hits, and so is a track that has
no first hit left.

Every operation acts on what the operations before it kept, so their order is
part of the question. Nothing is read until the analysis is given data:
:meth:`EventTrackHitAnalysis.apply` returns the events, tracks and hits it
kept, and :meth:`EventTrackHitAnalysis.extract` a table with one row per
event and the variables asked for of its tracks and hits.

Public objects:

EventTrackHitAnalysis
    A lazy analysis of events, tracks and hits.
Selection
    What an analysis has kept so far, and where it stands.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final, Literal, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from opengate_gate_tree.analysis.base import Analysis, Combination
from opengate_gate_tree.analysis.extract import extract_table
from opengate_gate_tree.analysis.operations import (
    COMPARISONS,
    CountCondition,
    FieldCondition,
    FieldLike,
    Navigate,
    ValueTest,
)
from opengate_gate_tree.representation.eventtrackhit import EventTrackHit
from opengate_gate_tree.representation.fields import ElementLevel, EventTrackHitField
from opengate_gate_tree.representation.index import EventTrackHitIndex
from opengate_gate_tree.representation.reduce import (
    position_in_event,
    position_in_track,
    reduce_hits_to_tracks,
    reduce_tracks_to_events,
)
from opengate_gate_tree.tree.hits.positronium import DecayType, GammaType, SourceType
from opengate_gate_tree.tree.hits.processes import Process
from opengate_gate_tree.tree.treedata import TreeData

Source = EventTrackHit | TreeData | pd.DataFrame

# Depth of every level, to tell which one is above another.
_DEPTH: Final[Mapping[ElementLevel, int]] = {
    ElementLevel.EVENT: 0,
    ElementLevel.TRACK: 1,
    ElementLevel.HIT: 2,
}

# How a level is called in a message, in the plural.
_PLURAL: Final[Mapping[ElementLevel, str]] = {
    ElementLevel.EVENT: "events",
    ElementLevel.TRACK: "tracks",
    ElementLevel.HIT: "hits",
}

# The method that moves an analysis to a level.
_MOVE: Final[Mapping[ElementLevel, str]] = {
    ElementLevel.EVENT: "events()",
    ElementLevel.TRACK: "tracks()",
    ElementLevel.HIT: "hits()",
}


@dataclass(frozen=True, eq=False)
class Selection:
    """What an analysis has kept so far, and where it stands.

    Attributes
    ----------
    events, tracks, hits : numpy.ndarray
        Which events, tracks and hits of the representation are kept.
    level : ElementLevel
        Level the analysis stands at.
    position : int | None
        Position the analysis stands at within every parent, or ``None``.
    """

    events: npt.NDArray[np.bool_]
    tracks: npt.NDArray[np.bool_]
    hits: npt.NDArray[np.bool_]
    level: ElementLevel
    position: int | None


class _Run:
    """One performance of an analysis: the representation, and the fields read for it."""

    def __init__(self, representation: EventTrackHit) -> None:
        self.representation = representation
        self._read: dict[FieldLike, npt.NDArray[Any]] = {}

    def values(self, field: FieldLike) -> npt.NDArray[Any]:
        """Return a field at its own level, reading it once per performance."""
        if field not in self._read:
            self._read[field] = self.representation.values(field)
        return self._read[field]


class EventTrackHitAnalysis(Analysis[EventTrackHit, Selection]):
    """A lazy analysis of events, tracks and hits.

    Examples
    --------
    >>> from opengate_gate_tree import GammaType, Process
    >>> analysis = (
    ...     EventTrackHitAnalysis().tracks().gamma_type(GammaType.ANNIHILATION)
    ...     .hits().process(Process.COMPTON)
    ... )
    >>> analysis
    EventTrackHitAnalysis().tracks().gamma_type(GammaType.ANNIHILATION).hits().process(Process.COMPTON)
    """

    def events(self) -> Self:
        """Move to the events."""
        return self._appended(Navigate(ElementLevel.EVENT, None, "events()"))

    def tracks(
        self,
        count: int | None = None,
        *,
        eq: int | None = None,
        ne: int | None = None,
        ge: int | None = None,
        le: int | None = None,
        gt: int | None = None,
        lt: int | None = None,
    ) -> Self:
        """Move to the tracks of every event, keeping the events with a number of them.

        Without a count, only moves. With one, keeps the events whose number
        of tracks kept so far passes it, then moves to their tracks.

        Parameters
        ----------
        count : int | None
            Number of tracks an event has to have; the same as ``eq``.
        eq : int | None
            Number of tracks an event has to have; it cannot be combined with
            the other comparisons.
        ne, ge, le, gt, lt : int | None
            Comparisons the number has to pass, all of them at once.

        Raises
        ------
        ValueError
            If ``eq`` (or ``count``) is given with another comparison.
        """
        comparisons = _count_comparisons(count, eq, ne, ge, le, gt, lt)
        if not comparisons:
            return self._appended(Navigate(ElementLevel.TRACK, None, "tracks()"))
        call = _count_call("tracks", count, comparisons)
        return self._appended(CountCondition(ElementLevel.TRACK, comparisons, call))

    def hits(
        self,
        count: int | None = None,
        *,
        eq: int | None = None,
        ne: int | None = None,
        ge: int | None = None,
        le: int | None = None,
        gt: int | None = None,
        lt: int | None = None,
    ) -> Self:
        """Move to the hits of every track, keeping the tracks with a number of them.

        Without a count, only moves. With one, keeps the tracks whose number
        of hits kept so far passes it, then moves to their hits. See
        :meth:`tracks` for the parameters.

        Raises
        ------
        ValueError
            If ``eq`` (or ``count``) is given with another comparison.
        """
        comparisons = _count_comparisons(count, eq, ne, ge, le, gt, lt)
        if not comparisons:
            return self._appended(Navigate(ElementLevel.HIT, None, "hits()"))
        call = _count_call("hits", count, comparisons)
        return self._appended(CountCondition(ElementLevel.HIT, comparisons, call))

    def track(self, position: int) -> Self:
        """Move to the track at a position of every event.

        A condition asked there decides about the event: an event whose
        track at that position does not pass it, or that has no track there,
        is left out with all its tracks.

        Parameters
        ----------
        position : int
            Position among the tracks kept so far, from 1 in the order of
            their identifiers; a negative one counts from the last, -1 being
            the last.

        Raises
        ------
        ValueError
            If the position is 0 or not a whole number.
        """
        return self._appended(
            Navigate(ElementLevel.TRACK, _position(position), f"track({position})")
        )

    def hit(self, position: int) -> Self:
        """Move to the hit at a position of every track.

        A condition asked there decides about the track: a track whose hit at
        that position does not pass it, or that has no hit there, is left out
        with all its hits. See :meth:`track` for the parameter; hits are in the
        order of their time.

        Raises
        ------
        ValueError
            If the position is 0 or not a whole number.
        """
        return self._appended(Navigate(ElementLevel.HIT, _position(position), f"hit({position})"))

    def source_type(self, *types: SourceType) -> Self:
        """Keep the events emitted with one of the given source models.

        Raises
        ------
        ValueError
            If no model is given, or a value is not a :class:`SourceType`.
        """
        return self._enum_condition(
            EventTrackHitField.SOURCE_TYPE, SourceType, types, "source_type"
        )

    def gamma_type(self, *types: GammaType) -> Self:
        """Keep the tracks that are one of the given kinds of gamma.

        Raises
        ------
        ValueError
            If no kind is given, or a value is not a :class:`GammaType`.
        """
        return self._enum_condition(EventTrackHitField.GAMMA_TYPE, GammaType, types, "gamma_type")

    def decay_type(self, *types: DecayType) -> Self:
        """Keep the tracks that came through one of the given decay channels.

        Raises
        ------
        ValueError
            If no channel is given, or a value is not a :class:`DecayType`.
        """
        return self._enum_condition(EventTrackHitField.DECAY_TYPE, DecayType, types, "decay_type")

    def particle(self, *pdg_codes: int) -> Self:
        """Keep the tracks of the given particles, by PDG code.

        Raises
        ------
        ValueError
            If no code is given.
        """
        if not pdg_codes:
            raise ValueError("Selecting by particle needs at least one PDG code to select.")
        call = f"particle({', '.join(repr(code) for code in pdg_codes)})"
        test = ValueTest((("isin", tuple(pdg_codes)),))
        return self._field_condition(EventTrackHitField.PDG_ENCODING, test, call)

    def process(self, *processes: Process | str) -> Self:
        """Keep the hits made by one of the given processes.

        Raises
        ------
        ValueError
            If no process is given.
        """
        if not processes:
            raise ValueError("Selecting by process needs at least one process to select.")
        shown = (
            f"Process.{process.name}" if isinstance(process, Process) else repr(process)
            for process in processes
        )
        test = ValueTest((("isin", tuple(str(process) for process in processes)),))
        return self._field_condition(
            EventTrackHitField.PROCESS_NAME, test, f"process({', '.join(shown)})"
        )

    def edep(self, **comparisons: Any) -> Self:
        """Keep the hits whose deposited energy, in MeV, passes the comparisons.

        See :meth:`where` for the comparisons.
        """
        test = _value_test(comparisons)
        return self._field_condition(EventTrackHitField.EDEP, test, f"edep({test.arguments()})")

    def time(self, **comparisons: Any) -> Self:
        """Keep the hits whose time, in s, passes the comparisons.

        See :meth:`where` for the comparisons.
        """
        test = _value_test(comparisons)
        return self._field_condition(EventTrackHitField.TIME, test, f"time({test.arguments()})")

    def where(self, field: FieldLike, **comparisons: Any) -> Self:
        """Keep the elements whose field passes the comparisons.

        Parameters
        ----------
        field : EventTrackHitField | str
            Field to test, at the level it describes; a name that is not the
            value of a field is a branch describing hits.
        **comparisons
            Any of ``eq``, ``ne``, ``ge``, ``le``, ``gt``, ``lt`` against one
            value, ``isin`` against several, and ``between`` against a pair of
            ends that belong to the range. A value has to pass all of them.

        Raises
        ------
        ValueError
            If no comparison is given, a comparison is unknown, or the field
            describes a level below the one the analysis stands at.
        """
        test = _value_test(comparisons)
        shown = (
            f"EventTrackHitField.{field.name}"
            if isinstance(field, EventTrackHitField)
            else repr(field)
        )
        return self._field_condition(field, test, f"where({shown}, {test.arguments()})")

    def extract(
        self,
        source: Source,
        variables: Sequence[FieldLike],
        *,
        max_tracks: int | None = None,
        max_hits: int | None = None,
    ) -> pd.DataFrame:
        """Perform the analysis and return one row per event that passed it.

        Parameters
        ----------
        source : EventTrackHit | TreeData | pandas.DataFrame
            Events, tracks and hits, or the data to build them from.
        variables : Sequence[EventTrackHitField | str]
            Fields to put in the table, in order. An event field is one column;
            a track field one per track, ``t1_<name>``…; a hit field one per
            hit of every track, ``t1h1_<name>``…. ``runID`` and ``eventID``
            always come first.
        max_tracks, max_hits : int | None
            Narrower widths than the ones the result allows.

        Returns
        -------
        pandas.DataFrame
            One row per event, columns in the order variable, track, hit. The
            table is as wide as the event with the fewest tracks and the track
            with the fewest hits allow; what does not fit is left out and
            reported in the log.

        Raises
        ------
        ValueError
            If the variables are not an ordered sequence, or a width asked
            for is wider than the result allows.
        BranchNotFoundError
            If the data does not hold the branch of a field the analysis or
            the table needs.
        """
        return extract_table(self.apply(source), variables, max_tracks, max_hits)

    def describe(self) -> str:
        """Return the analysis as numbered steps, one per line."""
        if not self.steps:
            return "Keep every event, track and hit."
        return "\n".join(
            f"{number}. {_describe(step)}" for number, step in enumerate(self.steps, start=1)
        )

    def _enum_condition(
        self,
        field: EventTrackHitField,
        enum_class: type[IntEnum],
        members: Sequence[Any],
        name: str,
    ) -> Self:
        """Return the analysis with a condition on the members of an enum appended.

        Raises
        ------
        ValueError
            If no member is given, or a value is not a member of the class.
        """
        if not members:
            raise ValueError(
                f"Selecting by {enum_class.__name__} needs at least one member to select."
            )
        wrong = [member for member in members if not isinstance(member, enum_class)]
        if wrong:
            raise ValueError(
                f"Selecting by {enum_class.__name__} takes its own members, got {wrong!r}. "
                "The classes share their numbers, so another class would select by a value "
                "that means something else."
            )
        shown = ", ".join(f"{enum_class.__name__}.{member.name}" for member in members)
        test = ValueTest((("isin", tuple(int(member) for member in members)),))
        return self._field_condition(field, test, f"{name}({shown})")

    def _field_condition(self, field: FieldLike, test: ValueTest, call: str) -> Self:
        """Return the analysis with a condition on a field appended.

        Raises
        ------
        ValueError
            If the field describes a level below the one the analysis stands at.
        """
        level = _level_of(field)
        standing, _ = _standing_after(self.steps)
        if _DEPTH[level] > _DEPTH[standing]:
            raise ValueError(
                f"'{_name(field)}' describes {_PLURAL[level]}, but the analysis stands at the "
                f"{_PLURAL[standing]}; move to them first with {_MOVE[level]}."
            )
        return self._appended(FieldCondition(field, test, call))

    def _target(self, source: Source) -> _Run:
        """Return one performance of the analysis on a representation."""
        representation = source if isinstance(source, EventTrackHit) else EventTrackHit(source)
        return _Run(representation)

    def _initial(self, target: _Run) -> Selection:
        """Return a selection of everything, standing at the events."""
        index = target.representation.index
        return Selection(
            np.ones(index.event_count, dtype=bool),
            np.ones(index.track_count, dtype=bool),
            np.ones(index.hit_count, dtype=bool),
            ElementLevel.EVENT,
            None,
        )

    def _operate(self, target: _Run, state: Selection, operation: object) -> Selection:
        """Return the selection after one operation."""
        if isinstance(operation, Navigate):
            return Selection(
                state.events, state.tracks, state.hits, operation.level, operation.position
            )
        if isinstance(operation, CountCondition):
            index = target.representation.index
            if operation.counted is ElementLevel.TRACK:
                counts = reduce_tracks_to_events(index, state.tracks, "count")
                parent = ElementLevel.EVENT
            else:
                counts = reduce_hits_to_tracks(index, state.hits, "count")
                parent = ElementLevel.TRACK
            passed = operation.passes(counts)
            if parent is state.level and state.position is not None:
                kept = self._kept_at_position(target, state, passed)
            else:
                kept = self._kept(target, state, parent, passed)
            return Selection(kept.events, kept.tracks, kept.hits, operation.counted, None)
        assert isinstance(operation, FieldCondition)
        passed = operation.test.passes(target.values(operation.field))
        level = _level_of(operation.field)
        if level is state.level and state.position is not None:
            return self._kept_at_position(target, state, passed)
        return self._kept(target, state, level, passed)

    def _combine(
        self, target: _Run, kind: Literal["and", "or"], left: Selection, right: Selection
    ) -> Selection:
        """Return what two selections keep together, standing at the events."""
        join = np.logical_and if kind == "and" else np.logical_or
        return _propagated(
            target.representation.index,
            Selection(
                join(left.events, right.events),
                join(left.tracks, right.tracks),
                join(left.hits, right.hits),
                ElementLevel.EVENT,
                None,
            ),
        )

    def _finish(self, target: _Run, state: Selection) -> EventTrackHit:
        """Return the representation of what the selection kept."""
        return target.representation.select(state.events, state.tracks, state.hits, keep_empty=True)

    def _kept_at_position(
        self, target: _Run, state: Selection, passed: npt.NDArray[np.bool_]
    ) -> Selection:
        """Return the selection keeping the parents whose element at the position passed."""
        index = target.representation.index
        assert state.position is not None
        from_end = state.position < 0
        if state.level is ElementLevel.TRACK:
            numbers = position_in_event(index, state.tracks, from_end=from_end)
            at = numbers == abs(state.position)
            parents = reduce_tracks_to_events(index, at & passed, "any")
            return self._kept(target, state, ElementLevel.EVENT, parents)
        numbers = position_in_track(index, state.hits, from_end=from_end)
        at = numbers == abs(state.position)
        parents = reduce_hits_to_tracks(index, at & passed, "any")
        return self._kept(target, state, ElementLevel.TRACK, parents)

    def _kept(
        self, target: _Run, state: Selection, level: ElementLevel, passed: npt.NDArray[np.bool_]
    ) -> Selection:
        """Return the selection keeping, at a level, only the elements that passed."""
        masks = {
            ElementLevel.EVENT: state.events,
            ElementLevel.TRACK: state.tracks,
            ElementLevel.HIT: state.hits,
        }
        masks[level] = masks[level] & passed
        return _propagated(
            target.representation.index,
            Selection(
                masks[ElementLevel.EVENT],
                masks[ElementLevel.TRACK],
                masks[ElementLevel.HIT],
                state.level,
                state.position,
            ),
        )


def _propagated(index: EventTrackHitIndex, state: Selection) -> Selection:
    """Return a selection whose levels agree with one another.

    A track of a left-out event, and a hit of a left-out track, are left out.
    A track whose hits were all left out is left out, and an event whose
    tracks were; one that held nothing in the representation stays.
    """
    tracks = state.tracks & np.repeat(state.events, index.tracks_per_event())
    hits = state.hits & np.repeat(tracks, index.hits_per_track())
    tracks &= reduce_hits_to_tracks(index, hits, "any") | (index.hits_per_track() == 0)
    events = state.events & (
        reduce_tracks_to_events(index, tracks, "any") | (index.tracks_per_event() == 0)
    )
    return Selection(events, tracks, hits, state.level, state.position)


def _standing_after(steps: Sequence[object]) -> tuple[ElementLevel, int | None]:
    """Return the level and position an analysis stands at after its steps."""
    level, position = ElementLevel.EVENT, None
    for step in steps:
        if isinstance(step, Navigate):
            level, position = step.level, step.position
        elif isinstance(step, CountCondition):
            level, position = step.counted, None
        elif isinstance(step, Combination):
            level, position = ElementLevel.EVENT, None
    return level, position


def _position(position: int) -> int:
    """Return a position from 1, or from -1 backwards.

    Raises
    ------
    ValueError
        If the position is 0 or not a whole number.
    """
    if isinstance(position, bool) or not isinstance(position, int | np.integer):
        raise ValueError(f"A position is a whole number, got {position!r}.")
    if position == 0:
        raise ValueError("Positions count from 1, and from -1 backwards; 0 is neither.")
    return int(position)


def _count_comparisons(
    count: int | None,
    eq: int | None,
    ne: int | None,
    ge: int | None,
    le: int | None,
    gt: int | None,
    lt: int | None,
) -> tuple[tuple[str, int], ...]:
    """Return the comparisons a count is asked to pass.

    Raises
    ------
    ValueError
        If ``eq`` or ``count`` is given with another comparison.
    """
    if count is not None and eq is not None:
        raise ValueError(f"A count of {count} and eq={eq} say the same thing twice; give one.")
    equal = count if count is not None else eq
    others = {"ne": ne, "ge": ge, "le": le, "gt": gt, "lt": lt}
    given = tuple((name, bound) for name, bound in others.items() if bound is not None)
    if equal is not None:
        if given:
            raise ValueError(
                f"A count equal to {equal} cannot also be compared with "
                f"{[name for name, _ in given]}; give either eq or the other comparisons."
            )
        return (("eq", equal),)
    return given


def _count_call(name: str, count: int | None, comparisons: tuple[tuple[str, int], ...]) -> str:
    """Return the call that asked for a count, as it was written."""
    if count is not None:
        return f"{name}({count})"
    return f"{name}({', '.join(f'{key}={bound}' for key, bound in comparisons)})"


def _value_test(comparisons: Mapping[str, Any]) -> ValueTest:
    """Return a value test from the keyword arguments that ask for it.

    Raises
    ------
    ValueError
        If no comparison is given, or one is unknown.
    """
    unknown = sorted(set(comparisons) - set(COMPARISONS))
    if unknown:
        raise ValueError(f"Unknown comparison(s) {unknown}; expected some of {list(COMPARISONS)}.")
    if not comparisons:
        raise ValueError(f"A value test needs at least one of {list(COMPARISONS)}.")
    pairs = []
    for name in COMPARISONS:
        if name in comparisons:
            bound = comparisons[name]
            if name in ("isin", "between"):
                bound = tuple(bound)
            if name == "between" and len(bound) != 2:
                raise ValueError(f"'between' takes the two ends of a range, got {bound!r}.")
            pairs.append((name, bound))
    return ValueTest(tuple(pairs))


def _level_of(field: FieldLike) -> ElementLevel:
    """Return the level a field describes; any other name is a branch of hits."""
    if isinstance(field, EventTrackHitField):
        return field.level
    try:
        return EventTrackHitField(field).level
    except ValueError:
        return ElementLevel.HIT


def _name(field: FieldLike) -> str:
    """Return the branch or column name of a field."""
    return field.value if isinstance(field, EventTrackHitField) else field


def _describe(step: object) -> str:
    """Return one step of an analysis in words."""
    if isinstance(step, Navigate):
        if step.position is None:
            return f"Move to the {_PLURAL[step.level]}."
        holder = "event" if step.level is ElementLevel.TRACK else "track"
        element = "track" if step.level is ElementLevel.TRACK else "hit"
        place = (
            f"{element} {step.position}"
            if step.position > 0
            else f"{element} {abs(step.position)} from the end"
        )
        return f"Move to {place} of every {holder}; a condition there decides about the {holder}."
    if isinstance(step, CountCondition):
        parent = ElementLevel.EVENT if step.counted is ElementLevel.TRACK else ElementLevel.TRACK
        counted = ", ".join(f"{name}={bound}" for name, bound in step.comparisons)
        return (
            f"Keep the {_PLURAL[parent]} whose number of {_PLURAL[step.counted]} passes "
            f"{counted}, and move to their {_PLURAL[step.counted]}."
        )
    if isinstance(step, FieldCondition):
        level = _level_of(step.field)
        return (
            f"Keep the {_PLURAL[level]} whose {_name(step.field)} passes {step.test.arguments()}."
        )
    assert isinstance(step, Combination)
    joined = "and" if step.kind == "and" else "or"
    return f"Keep what {step.left!r} {joined} {step.right!r} keep."
