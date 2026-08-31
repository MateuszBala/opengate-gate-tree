"""Unit tests for the filters that select rows by where they come from."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from opengate_gate_tree.io.reader import read_hits_trees, read_tree
from opengate_gate_tree.tree.filters import (
    by_event,
    by_run,
    has_decay_metadata,
    is_from_event,
    is_from_run,
    with_decay_metadata,
)
from opengate_gate_tree.tree.gatetree import GateTree

# The scene behind the multi-run fixture, from its application.mac: three runs
# of one time slice each, so run N covers the seconds [0.02 N, 0.02 (N + 1)].
RUN_LENGTH = 0.02  # setTimeSlice 0.02 s
RUN_COUNT = 3  # setTimeStop 0.06 s


def merged(files: Mapping[str, Path], key: str) -> pd.DataFrame:
    """Return every tree of hits of a file, as one frame."""
    return read_hits_trees(files[key]).to_dataframe()


def test_a_run_is_the_time_slice_its_macro_gives_it(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A run in GATE is a slice of time, and selecting one has to land in it."""
    # ARRANGE
    frame = merged(hits_variant_files, "multi-run")

    # ACT
    second_run = by_run(frame, 1)

    # ASSERT
    assert set(second_run["runID"]) == {1}
    assert second_run["time"].between(RUN_LENGTH, 2 * RUN_LENGTH).all()


def test_the_runs_of_a_file_add_up_to_it(hits_variant_files: Mapping[str, Path]) -> None:
    """Every row belongs to exactly one run, so the runs partition the file."""
    # ARRANGE
    frame = merged(hits_variant_files, "multi-run")

    # ACT
    counts = [len(by_run(frame, run_id)) for run_id in range(RUN_COUNT)]

    # ASSERT
    assert all(count > 0 for count in counts)
    assert sum(counts) == len(frame)


def test_a_run_that_never_ran_holds_nothing(hits_variant_files: Mapping[str, Path]) -> None:
    """Asking about a run the simulation did not have is a question, not a mistake."""
    # ARRANGE
    frame = merged(hits_variant_files, "multi-run")

    # ACT
    selected = by_run(frame, RUN_COUNT)

    # ASSERT
    assert len(selected) == 0


def test_an_event_needs_both_identifiers_to_be_itself(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """GATE numbers events within a run, so the identifier alone names several.

    This is the whole reason the filter takes a pair: on this file, event 5 of
    run 1 is a handful of rows, while "event 5" is every event 5 of every run.
    """
    # ARRANGE
    frame = merged(hits_variant_files, "multi-run")

    # ACT
    one_event = by_event(frame, 1, 5)
    every_fifth = frame[frame["eventID"] == 5]

    # ASSERT
    assert set(one_event["runID"]) == {1}
    assert 0 < len(one_event) < len(every_fifth)
    assert set(every_fifth["runID"]) == set(range(RUN_COUNT))


def test_the_rows_of_an_event_belong_to_one_decay(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Every row of one event carries the same pair of identifiers."""
    # ARRANGE
    frame = merged(hits_variant_files, "multi-run")

    # ACT
    one_event = by_event(frame, 2, 11)

    # ASSERT
    assert set(one_event["runID"]) == {2}
    assert set(one_event["eventID"]) == {11}


def test_the_masks_cover_every_row(hits_variant_files: Mapping[str, Path]) -> None:
    """A mask is what combines with other conditions, so it keeps its shape."""
    # ARRANGE
    frame = merged(hits_variant_files, "multi-run")

    # ACT
    of_run = is_from_run(frame, 1)
    of_event = is_from_event(frame, 1, 5)

    # ASSERT
    assert len(of_run) == len(frame)
    assert list(of_event.index) == list(frame.index)
    assert (of_event <= of_run).all()


def test_rows_of_a_positronium_source_are_kept(
    positronium_files: Mapping[str, Path],
) -> None:
    """A file can mix sources, and only some rows carry decay metadata."""
    # ARRANGE
    frame = read_tree(positronium_files["all-variants"], GateTree.HITS).to_dataframe()

    # ACT
    selected = with_decay_metadata(frame)

    # ASSERT
    assert list(selected.index) == list(frame[frame["decayIndex"] >= 0].index)
    assert 0 < len(selected) < len(frame)


def test_a_file_without_a_positronium_source_keeps_no_rows(
    positronium_files: Mapping[str, Path],
) -> None:
    """A back-to-back source writes no decay metadata at all."""
    # ARRANGE
    frame = read_tree(positronium_files["back-to-back"], GateTree.HITS).to_dataframe()

    # ACT
    selected = with_decay_metadata(frame)

    # ASSERT
    assert len(selected) == 0
    assert not has_decay_metadata(frame).any()


def test_a_file_of_one_positronium_source_keeps_every_row(
    positronium_files: Mapping[str, Path],
) -> None:
    """Every gamma of such a source carries the metadata, including its own."""
    # ARRANGE
    frame = read_tree(positronium_files["ops-prompt"], GateTree.HITS).to_dataframe()

    # ACT
    selected = with_decay_metadata(frame)

    # ASSERT
    assert len(selected) == len(frame)


@pytest.mark.parametrize(
    ("missing", "call"),
    [
        ("runID", lambda frame: is_from_run(frame, 0)),
        ("eventID", lambda frame: is_from_event(frame, 0, 0)),
        ("decayIndex", has_decay_metadata),
    ],
    ids=["run", "event", "decay-metadata"],
)
def test_a_missing_column_is_reported_by_its_name(
    missing: str,
    call: object,
    positronium_files: Mapping[str, Path],
) -> None:
    """A frame without the column cannot answer a question about it."""
    # ARRANGE
    frame = read_tree(positronium_files["pps"], GateTree.HITS).to_dataframe()
    without_it = frame.drop(columns=[missing])

    # ACT / ASSERT
    with pytest.raises(KeyError, match=missing):
        call(without_it)  # type: ignore[operator]


def test_an_empty_frame_answers_empty(positronium_files: Mapping[str, Path]) -> None:
    """A selection that removed every row is still a frame to ask questions of.

    The frame is taken from a file and emptied rather than built here: an
    empty frame built by hand holds columns of floating point numbers, and the
    identifiers of GATE are whole numbers, which the decay metadata filter
    checks for.
    """
    # ARRANGE
    frame = read_tree(positronium_files["pps"], GateTree.HITS).to_dataframe().iloc[:0]

    # ACT
    of_event = by_event(frame, 0, 0)
    with_metadata = with_decay_metadata(frame)

    # ASSERT
    assert len(of_event) == 0
    assert len(with_metadata) == 0
