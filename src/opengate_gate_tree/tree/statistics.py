"""Summaries of the data extracted from a tree.

Statistics answer what a file holds before anything is done with it: how many
entries, how many events behind them, what range each branch covers, which
process names appear and how often.

The per-branch part is not specific to hits and stays that way: the other
trees of a GATE file are summarised the same way once they are supported. The
part that reads the numbers as physics is separate and is filled in only for
hits, and only from the branches that were actually extracted.

Events and tracks are counted by the identifiers that name them together. GATE
numbers events within a run and tracks within an event, so an identifier on its
own counts far too few of them: every event has a track 1. The package leaves
those identifiers as they are, so a summary has to do the composing.

Public objects:

BranchStatistics
    Summary of a single branch.
HitsSummary
    Summary of what the hits describe, beyond their columns.
TreeStatistics
    Summary of an extracted tree.
compute_statistics(data, detection) -> TreeStatistics
    Summarise extracted data.
format_statistics(statistics) -> str
    Render a summary for reading.
statistics_to_dict(statistics) -> dict
    Render a summary for a file.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.detection import HitsTreeDetection, summarise_hits_tree
from opengate_gate_tree.tree.hits.positronium import POSITRONIUM_BRANCHES
from opengate_gate_tree.tree.hits.schema import BranchKind, variant_reference
from opengate_gate_tree.tree.hits.variant import SYSTEM_ALIASES
from opengate_gate_tree.tree.merge import SOURCE_TREE_BRANCH
from opengate_gate_tree.tree.treedata import TreeData

# Branch naming the run an entry belongs to.
RUN_BRANCH: Final[str] = "runID"

# Branch naming the event an entry belongs to.
EVENT_BRANCH: Final[str] = "eventID"

# Branch naming the track an entry belongs to.
TRACK_BRANCH: Final[str] = "trackID"

# Branch holding the energy deposited by an entry.
ENERGY_BRANCH: Final[str] = "edep"

# Branch holding the time of an entry.
TIME_BRANCH: Final[str] = "time"

# Number of values a branch of unbounded content reports as its most frequent
# ones. Branches whose values stand for something report all of them.
TOP_VALUE_COUNT: Final[int] = 5

# NumPy data type kinds holding text.
TEXT_DTYPE_KINDS: Final[frozenset[str]] = frozenset({"O", "U", "S"})

# NumPy data type kinds holding whole numbers. Booleans are whole numbers as
# far as a summary is concerned: they have a range and no missing value.
INTEGER_DTYPE_KINDS: Final[frozenset[str]] = frozenset({"i", "u", "b"})


@dataclass(frozen=True)
class BranchStatistics:
    """Summary of a single branch.

    Attributes
    ----------
    name : str
        Branch name.
    dtype : str
        Type the branch is held with.
    kind : BranchKind
        Kind of value the branch holds.
    entries : int
        Number of entries of the branch.
    non_finite_count : int | None
        Number of values that are not a finite number, for a floating point
        branch. Both "not a number" and the infinities are counted: neither
        can be written to a report, and an infinity would carry into the mean
        and turn the spread into "not a number".
    minimum, maximum, mean, std : float | None
        Range and spread of a numeric branch, ignoring values that are not
        finite. ``None`` for a text branch, and for a numeric one holding no
        usable value.
    unique_count : int | None
        Number of distinct values, for a text or whole number branch.
    top_values : tuple[tuple[str, int], ...]
        Most frequent values, with their counts, for a branch whose values
        can be named: a text branch, and a branch of the PositroniumSource
        whose numbers stand for something. A value the package cannot name is
        reported as the number it is.
    """

    name: str
    dtype: str
    kind: BranchKind
    entries: int
    non_finite_count: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    unique_count: int | None = None
    top_values: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class HitsSummary:
    """Summary of what the hits describe, beyond their columns.

    Attributes
    ----------
    event_key : tuple[str, ...]
        Branches the events were counted by. Empty when they could not be
        counted at all.
    event_count : int | None
        Number of distinct events.
    run_count : int | None
        Number of distinct runs.
    track_key : tuple[str, ...]
        Branches the tracks were counted by. Empty when they could not be
        counted at all.
    track_count : int | None
        Number of distinct tracks.
    total_edep : float | None
        Sum of the deposited energy, over the values that are finite.
    time_min, time_max : float | None
        Range of the times.
    source_trees : tuple[str, ...]
        Trees the entries came from, for a merged dataset.
    """

    event_key: tuple[str, ...] = ()
    event_count: int | None = None
    run_count: int | None = None
    track_key: tuple[str, ...] = ()
    track_count: int | None = None
    total_edep: float | None = None
    time_min: float | None = None
    time_max: float | None = None
    source_trees: tuple[str, ...] = ()


@dataclass(frozen=True)
class TreeStatistics:
    """Summary of an extracted tree.

    Attributes
    ----------
    tree : GateTree
        Tree the data was extracted from.
    entry_count : int
        Number of entries.
    branches : tuple[BranchStatistics, ...]
        Summary of every branch, in the order of the data.
    detection : HitsTreeDetection | None
        Structure the tree was recognised as, when it is known.
    hits_summary : HitsSummary | None
        Summary of the hits, for a "Hits" tree.
    """

    tree: GateTree
    entry_count: int
    branches: tuple[BranchStatistics, ...]
    detection: HitsTreeDetection | None = None
    hits_summary: HitsSummary | None = None


def compute_statistics(
    data: TreeData,
    detection: HitsTreeDetection | None = None,
) -> TreeStatistics:
    """Summarise extracted data.

    Parameters
    ----------
    data : TreeData
        Data to summarise.
    detection : HitsTreeDetection | None
        Structure the tree was recognised as, reported along with the numbers.

    Returns
    -------
    TreeStatistics
        Summary of the data.
    """
    branches = tuple(_branch_statistics(name, column) for name, column in data.columns.items())
    summary = _hits_summary(data) if data.tree is GateTree.HITS else None
    return TreeStatistics(
        tree=data.tree,
        entry_count=data.entry_count,
        branches=branches,
        detection=detection,
        hits_summary=summary,
    )


def statistics_to_dict(statistics: TreeStatistics) -> dict[str, Any]:
    """Render a summary as plain values, ready to be written to a file.

    Parameters
    ----------
    statistics : TreeStatistics
        Summary to render.

    Returns
    -------
    dict
        Summary as dictionaries, lists, strings and numbers. Values that are
        not a number are reported as ``null``, so that the result is valid
        JSON.
    """
    report: dict[str, Any] = {
        "tree": statistics.tree.value,
        "entries": statistics.entry_count,
        "branches": [_branch_to_dict(branch) for branch in statistics.branches],
    }
    if statistics.detection is not None:
        report["structure"] = _detection_to_dict(statistics.detection)
    if statistics.hits_summary is not None:
        report["hits"] = _summary_to_dict(statistics.hits_summary)
    return report


def format_statistics(statistics: TreeStatistics) -> str:
    """Render a summary for reading.

    Parameters
    ----------
    statistics : TreeStatistics
        Summary to render.

    Returns
    -------
    str
        Description spanning several lines.
    """
    lines = [f"Tree: {statistics.tree.value}", f"Entries: {statistics.entry_count}"]
    if statistics.detection is not None:
        lines.insert(1, summarise_hits_tree(statistics.detection))
    if statistics.hits_summary is not None:
        lines.extend(_summary_lines(statistics.hits_summary))

    lines.append(f"Branches ({len(statistics.branches)}):")
    lines.extend(f"  - {_branch_line(branch)}" for branch in statistics.branches)
    return "\n".join(lines)


def _branch_statistics(name: str, column: npt.NDArray[Any]) -> BranchStatistics:
    """Summarise a single branch."""
    entries = int(column.shape[0])
    if column.dtype.kind in TEXT_DTYPE_KINDS:
        counts = Counter(str(value) for value in column)
        return BranchStatistics(
            name=name,
            dtype="text",
            kind=BranchKind.TEXT,
            entries=entries,
            unique_count=len(counts),
            top_values=_most_common(counts),
        )

    is_integer = column.dtype.kind in INTEGER_DTYPE_KINDS
    is_array = column.ndim > 1
    kind = _numeric_kind(is_integer, is_array)
    dtype = str(column.dtype.name)
    if is_array:
        dtype = f"{dtype}{''.join(f'[{width}]' for width in column.shape[1:])}"

    flattened = column.reshape(-1)
    finite = np.isfinite(flattened) if not is_integer else None
    non_finite_count = None if finite is None else int(np.count_nonzero(~finite))
    usable = flattened if finite is None else flattened[finite]
    unique_count = len(np.unique(flattened)) if is_integer and not is_array else None
    top_values = _named_values(name, flattened) if is_integer and not is_array else ()

    if usable.size == 0:
        return BranchStatistics(
            name=name,
            dtype=dtype,
            kind=kind,
            entries=entries,
            non_finite_count=non_finite_count,
            unique_count=unique_count,
            top_values=top_values,
        )

    return BranchStatistics(
        name=name,
        dtype=dtype,
        kind=kind,
        entries=entries,
        non_finite_count=non_finite_count,
        top_values=top_values,
        minimum=float(np.min(usable)),
        maximum=float(np.max(usable)),
        mean=float(np.mean(usable)),
        std=float(np.std(usable)),
        unique_count=unique_count,
    )


def _numeric_kind(is_integer: bool, is_array: bool) -> BranchKind:
    """Return the kind of a numeric branch."""
    if is_integer:
        return BranchKind.INTEGER_ARRAY if is_array else BranchKind.INTEGER
    return BranchKind.FLOAT_ARRAY if is_array else BranchKind.FLOAT


def _named_values(name: str, column: npt.NDArray[Any]) -> tuple[tuple[str, int], ...]:
    """Return the most frequent values of a branch whose numbers have names.

    The three branches a PositroniumSource fills carry numbers that stand for
    something, and a report saying ``2`` where the package knows it means
    ``ANNIHILATION`` would keep that to itself. A value with no name is
    reported as the number it is: the report says what the file holds, and
    naming it something else would not make it true.

    Counting runs on the whole column, so it is done the way NumPy does it
    rather than a value at a time: at ten million rows the difference is
    seconds against a fraction of one.
    """
    enum_class = POSITRONIUM_BRANCHES.get(name)
    if enum_class is None:
        return ()
    known = {int(member): member.name for member in enum_class}
    values, counts = np.unique(column, return_counts=True)
    named = [
        (known.get(int(value), str(int(value))), int(count))
        for value, count in zip(values, counts, strict=True)
    ]
    # Every value is reported, not the most frequent five: the branch has a
    # handful of them by construction, and a value the package cannot name is
    # exactly the one worth seeing, however rare it is.
    return tuple(sorted(named, key=lambda item: (-item[1], item[0])))


def _most_common(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Return the most frequent values, ties resolved by value."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ordered[:TOP_VALUE_COUNT])


def _hits_summary(data: TreeData) -> HitsSummary:
    """Summarise what the hits describe, from the branches that are present."""
    columns = data.columns
    event_key, event_count = _count_by_key(columns, (RUN_BRANCH, EVENT_BRANCH), EVENT_BRANCH)
    track_key, track_count = _count_by_key(
        columns, (RUN_BRANCH, EVENT_BRANCH, TRACK_BRANCH), TRACK_BRANCH
    )

    time_min, time_max = _range(columns, TIME_BRANCH)
    return HitsSummary(
        event_key=event_key,
        event_count=event_count,
        run_count=_distinct_count(columns, RUN_BRANCH),
        track_key=track_key,
        track_count=track_count,
        total_edep=_total(columns, ENERGY_BRANCH),
        time_min=time_min,
        time_max=time_max,
        source_trees=_source_trees(columns),
    )


def _count_by_key(
    columns: Mapping[str, npt.NDArray[Any]],
    key: Sequence[str],
    required: str,
) -> tuple[tuple[str, ...], int | None]:
    """Count what the identifiers of a key name, and report the key used.

    The identifiers of GATE are numbered within the thing that holds them, so
    what names one event or one track is the whole key, not its last branch.
    A key is built from the branches that were extracted; when the branch that
    carries the count itself is missing, there is nothing to count.
    """
    if required not in columns:
        return (), None
    present = tuple(name for name in key if name in columns)
    stacked = np.stack([columns[name] for name in present], axis=1)
    return present, int(len(np.unique(stacked, axis=0)))


def _distinct_count(columns: Mapping[str, npt.NDArray[Any]], name: str) -> int | None:
    """Return how many distinct values a branch holds, when it is present."""
    if name not in columns:
        return None
    return int(len(np.unique(columns[name])))


def _total(columns: Mapping[str, npt.NDArray[Any]], name: str) -> float | None:
    """Return the sum of a branch, when it is present.

    Values that are not finite are left out, the same way they are left out of
    a range: a single infinity would otherwise carry into the total and make
    the report unwritable.
    """
    if name not in columns:
        return None
    values = columns[name]
    usable = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if usable.size == 0:
        return None
    return float(np.sum(usable))


def _range(
    columns: Mapping[str, npt.NDArray[Any]],
    name: str,
) -> tuple[float | None, float | None]:
    """Return both ends of the range of a branch, when it holds usable values."""
    if name not in columns:
        return None, None
    values = columns[name]
    usable = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if usable.size == 0:
        return None, None
    return float(np.min(usable)), float(np.max(usable))


def _source_trees(columns: Mapping[str, npt.NDArray[Any]]) -> tuple[str, ...]:
    """Return the trees a merged dataset came from, in the order they appear."""
    if SOURCE_TREE_BRANCH not in columns:
        return ()
    return tuple(dict.fromkeys(str(value) for value in columns[SOURCE_TREE_BRANCH]))


def _branch_to_dict(branch: BranchStatistics) -> dict[str, Any]:
    """Render the summary of a branch as plain values."""
    report: dict[str, Any] = {
        "name": branch.name,
        "type": branch.dtype,
        "kind": branch.kind.value,
        "entries": branch.entries,
        "minimum": branch.minimum,
        "maximum": branch.maximum,
        "mean": branch.mean,
        "std": branch.std,
    }
    if branch.non_finite_count is not None:
        report["not_finite"] = branch.non_finite_count
    if branch.unique_count is not None:
        report["distinct_values"] = branch.unique_count
    if branch.top_values:
        report["most_frequent"] = [
            {"value": value, "count": count} for value, count in branch.top_values
        ]
    return report


def _detection_to_dict(detection: HitsTreeDetection) -> dict[str, Any]:
    """Render a recognised structure as plain values."""
    report: dict[str, Any] = {
        "variant": detection.variant.value,
        "reference": variant_reference(detection.variant),
        "branches": detection.branch_count,
    }
    if detection.tree_name is not None:
        report["tree_name"] = detection.tree_name
    if detection.system is not None:
        report["system_scheme"] = list(SYSTEM_ALIASES[detection.system])
        report["system_depth"] = detection.system_depth
    return report


def _summary_to_dict(summary: HitsSummary) -> dict[str, Any]:
    """Render the summary of the hits as plain values."""
    report: dict[str, Any] = {
        "event_key": list(summary.event_key),
        "events": summary.event_count,
        "runs": summary.run_count,
        "track_key": list(summary.track_key),
        "tracks": summary.track_count,
        "total_edep": summary.total_edep,
        "time_min": summary.time_min,
        "time_max": summary.time_max,
    }
    if summary.source_trees:
        report["source_trees"] = list(summary.source_trees)
    return report


def _summary_lines(summary: HitsSummary) -> list[str]:
    """Return the lines describing what the hits add up to."""
    lines: list[str] = []
    if summary.event_count is not None:
        counted_by = ", ".join(summary.event_key)
        lines.append(f"Events: {summary.event_count} (counted by {counted_by})")
    if summary.run_count is not None:
        lines.append(f"Runs: {summary.run_count}")
    if summary.track_count is not None:
        lines.append(f"Tracks: {summary.track_count} (counted by {', '.join(summary.track_key)})")
    if summary.total_edep is not None:
        lines.append(f"Deposited energy: {summary.total_edep:.6g}")
    if summary.time_min is not None and summary.time_max is not None:
        lines.append(f"Time range: {summary.time_min:.6g} to {summary.time_max:.6g}")
    if summary.source_trees:
        lines.append(f"Source trees: {', '.join(summary.source_trees)}")
    return lines


def _branch_line(branch: BranchStatistics) -> str:
    """Return the line describing a single branch."""
    head = f"{branch.name} / {branch.dtype}"
    if branch.kind is BranchKind.TEXT:
        frequent = ", ".join(f"{value} ({count})" for value, count in branch.top_values)
        return f"{head}: {branch.unique_count} distinct, most frequent {frequent}"
    if branch.minimum is None:
        return f"{head}: no usable value"
    body = (
        f"min {branch.minimum:.6g}, max {branch.maximum:.6g}, "
        f"mean {branch.mean:.6g}, std {branch.std:.6g}"
    )
    if branch.non_finite_count:
        body += f", not finite {branch.non_finite_count}"
    if branch.unique_count is not None:
        body += f", {branch.unique_count} distinct"
    if branch.top_values:
        frequent = ", ".join(f"{value} ({count})" for value, count in branch.top_values)
        body += f", most frequent {frequent}"
    return f"{head}: {body}"
