"""Unit tests for writing hits as events, tracks and hits, and reading them back."""

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from opengate_gate_tree.errors import RepresentationError
from opengate_gate_tree.io.eth import read_event_track_hit, write_event_track_hit
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.naming import build_event_track_hit_file_path
from opengate_gate_tree.io.reader import read_hits_trees, read_tree
from opengate_gate_tree.io.writers import write_tree
from opengate_gate_tree.representation.eventtrackhit import STRUCTURE_COLUMNS, EventTrackHit
from opengate_gate_tree.representation.fields import EventTrackHitField
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.positronium import GammaType

Field = EventTrackHitField


@pytest.fixture(scope="module")
def ops_prompt(positronium_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the oPs scene with a prompt gamma as events, tracks and hits."""
    return EventTrackHit(read_tree(positronium_files["ops-prompt"], GateTree.HITS))


@pytest.fixture(scope="module")
def multi_sd(hits_variant_files: Mapping[str, Path]) -> EventTrackHit:
    """Return the hits of two detectors, merged, as events, tracks and hits."""
    return EventTrackHit(read_hits_trees(hits_variant_files["multi-sd"]))


def test_the_file_is_named_after_the_representation() -> None:
    """``eth`` stands where the tree stands in the name of a plain tree."""
    # ARRANGE
    directory = Path("out")

    # ACT
    path = build_event_track_hit_file_path(directory, "patient_01", OutputFileFormat.HDF5)

    # ASSERT
    assert path == Path("out/patient_01.eth.hdf5")


def test_the_flat_tree_carries_where_every_hit_stands(ops_prompt: EventTrackHit) -> None:
    """Five columns, after every branch of the data."""
    # ARRANGE
    branches = list(ops_prompt.data.branch_names)

    # ACT
    data = ops_prompt.to_tree_data()

    # ASSERT
    assert list(data.branch_names) == [*branches, *STRUCTURE_COLUMNS]
    assert data["eventIndex"].dtype == np.int64
    assert data["hitNumber"].dtype == np.int32
    assert data["eventIndex"][[0, -1]].tolist() == [0, 79]
    assert np.all(data["hitNumber"] <= data["hitCount"])
    assert np.all(data["trackNumber"] <= data["trackCount"])


def test_identifiers_are_written_as_gate_wrote_them(ops_prompt: EventTrackHit) -> None:
    """Only the order changes; the values of every row stay its own."""
    # ARRANGE
    rows = ops_prompt.index.hit_rows

    # ACT
    data = ops_prompt.to_tree_data()

    # ASSERT
    for name in ("runID", "eventID", "trackID", "edep", "processName"):
        assert np.array_equal(data[name], ops_prompt.data[name][rows])
        assert data[name].dtype == ops_prompt.data[name].dtype


@pytest.mark.parametrize("file_format", list(OutputFileFormat))
def test_a_written_file_is_read_back_as_it_was(
    multi_sd: EventTrackHit, file_format: OutputFileFormat, tmp_path: Path
) -> None:
    """The grouping and every value come back, in every format."""
    # ARRANGE
    path = build_event_track_hit_file_path(tmp_path, "scene", file_format)

    # ACT
    written = write_event_track_hit(multi_sd, path, file_format)
    read = read_event_track_hit(written)

    # ASSERT
    assert written == path
    assert np.array_equal(read.index.track_offsets, multi_sd.index.track_offsets)
    assert np.array_equal(read.index.event_offsets, multi_sd.index.event_offsets)
    assert not set(STRUCTURE_COLUMNS) & set(read.data.branch_names)
    original = multi_sd.hits_frame([Field.EDEP, Field.TIME, Field.PROCESS_NAME, "sourceTreeName"])
    again = read.hits_frame([Field.EDEP, Field.TIME, Field.PROCESS_NAME, "sourceTreeName"])
    # CSV keeps no types: single precision comes back as double, to the digits written.
    pd.testing.assert_frame_equal(
        again.reset_index(drop=True), original.reset_index(drop=True), check_dtype=False, rtol=1e-6
    )


def test_a_written_file_is_read_in_order_without_sorting(
    multi_sd: EventTrackHit, tmp_path: Path
) -> None:
    """The rows of the file are already in the order of the representation."""
    # ARRANGE
    path = write_event_track_hit(multi_sd, tmp_path / "scene.eth.root", OutputFileFormat.ROOT)

    # ACT
    read = read_event_track_hit(path)

    # ASSERT
    assert read.index.hit_rows.tolist() == list(range(read.hit_count))


def test_a_selection_is_written_with_its_own_positions(
    ops_prompt: EventTrackHit, tmp_path: Path
) -> None:
    """The prompt gammas alone: one track per event, events counted from 0 again."""
    # ARRANGE
    prompt = ops_prompt.with_gamma_type(GammaType.PROMPT)
    path = tmp_path / "prompt.eth.csv"

    # ACT
    write_event_track_hit(prompt, path, OutputFileFormat.CSV)
    frame = pd.read_csv(path)

    # ASSERT
    assert len(frame) == prompt.hit_count
    assert frame["eventIndex"].max() == prompt.event_count - 1 == 78
    assert set(frame["trackCount"]) == {1}
    assert set(frame["gammaType"]) == {int(GammaType.PROMPT)}


def test_tracks_left_empty_cannot_be_written(ops_prompt: EventTrackHit, tmp_path: Path) -> None:
    """A flat tree of hits has no row for a track without hits."""
    # ARRANGE
    empty = ops_prompt.nth_hit(5, keep_empty=True)

    # ACT
    with pytest.raises(RepresentationError) as raised:
        write_event_track_hit(empty, tmp_path / "empty.eth.root", OutputFileFormat.ROOT)

    # ASSERT
    assert "keep_empty" in str(raised.value)


def test_a_file_edited_after_it_was_written_is_refused(
    ops_prompt: EventTrackHit, tmp_path: Path
) -> None:
    """A hit number that disagrees with the rows says the grouping is not the one written."""
    # ARRANGE
    path = write_event_track_hit(ops_prompt, tmp_path / "scene.eth.csv", OutputFileFormat.CSV)
    frame = pd.read_csv(path)
    frame.loc[3, "hitNumber"] = 7
    frame.to_csv(path, index=False)

    # ACT
    with pytest.raises(RepresentationError) as raised:
        read_event_track_hit(path)

    # ASSERT
    assert "'hitNumber' column" in str(raised.value)


def test_a_plain_tree_of_hits_is_not_read_as_events(
    ops_prompt: EventTrackHit, tmp_path: Path
) -> None:
    """Without the columns, nothing says the rows were written grouped."""
    # ARRANGE
    path = write_tree(ops_prompt.data, tmp_path / "plain.hits.hdf5", OutputFileFormat.HDF5)

    # ACT
    with pytest.raises(RepresentationError) as raised:
        read_event_track_hit(path)

    # ASSERT
    assert "'eventIndex'" in str(raised.value)


def test_an_extension_naming_no_format_is_refused(tmp_path: Path) -> None:
    """The extension is what tells the format."""
    # ARRANGE
    path = tmp_path / "scene.eth.parquet"

    # ACT
    with pytest.raises(RepresentationError) as raised:
        read_event_track_hit(path)

    # ASSERT
    assert "names no format" in str(raised.value)


def test_a_file_that_cannot_be_read_is_reported(tmp_path: Path) -> None:
    """An HDF5 file without a group of hits is not one the package wrote."""
    # ARRANGE
    path = tmp_path / "other.eth.hdf5"
    with h5py.File(path, "w") as output_file:
        output_file.create_group("Singles")

    # ACT
    with pytest.raises(RepresentationError) as raised:
        read_event_track_hit(path)

    # ASSERT
    assert "could not be read" in str(raised.value)
