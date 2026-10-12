"""Hits written as events, tracks and hits, and read back.

A representation of events, tracks and hits is written as a flat tree of hits
in its own order - events by run and event, tracks by identifier, hits by
time - with five columns saying where every hit stands: ``eventIndex``,
``trackNumber``, ``hitNumber``, ``trackCount`` and ``hitCount``. Any of the
three output formats takes it, and a reader that knows nothing of this
package still sees which rows make one track and one event.

Reading a file back groups its rows again and compares the grouping with the
order of the rows and the columns the file carries. A file whose rows are out
of order, or whose columns disagree with them - an identifier, a time or a
position changed by hand, a file cut in the middle of a track, or one written
by something else - is refused rather than read as a grouping it does not
hold. Nothing checks the other values: an energy changed by hand is read as
it stands.

CSV keeps no types and no arrays. Reading one back gives every branch the
type the "Hits" tree schema gives it, folds the ``volumeID_0`` to
``volumeID_9`` columns back into ``volumeID``, and reads any other column in
the type pandas infers.

Public objects:

write_event_track_hit(representation, path, file_format) -> Path
    Write hits as events, tracks and hits.
read_event_track_hit(path) -> EventTrackHit
    Read back hits written as events, tracks and hits.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from opengate_gate_tree.errors import RepresentationError
from opengate_gate_tree.io.fileformat import OutputFileFormat
from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.io.writers import write_tree
from opengate_gate_tree.representation.eventtrackhit import STRUCTURE_COLUMNS, EventTrackHit
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.schema import (
    ARRAY_BRANCH_WIDTHS,
    VARIANT_BRANCHES,
    BranchKind,
    branch_spec,
)
from opengate_gate_tree.tree.treedata import TreeData


def write_event_track_hit(
    representation: EventTrackHit, path: Path, file_format: OutputFileFormat
) -> Path:
    """Write hits as events, tracks and hits.

    Parameters
    ----------
    representation : EventTrackHit
        Hits to write, possibly after a selection.
    path : Path
        Path of the output file; :func:`build_event_track_hit_file_path` names
        it. An existing file is overwritten.
    file_format : OutputFileFormat
        Format of the output file.

    Returns
    -------
    Path
        Path that was written to.

    Raises
    ------
    RepresentationError
        If a track holds no hits or an event no tracks.
    ExportError
        If the file cannot be written.
    """
    return write_tree(representation.to_tree_data(), path, file_format)


def read_event_track_hit(path: Path) -> EventTrackHit:
    """Read back hits written as events, tracks and hits.

    The format is told by the extension of the file: ``.root``, ``.hdf5`` or
    ``.csv``.

    Parameters
    ----------
    path : Path
        Path of a file :func:`write_event_track_hit` wrote.

    Returns
    -------
    EventTrackHit
        The hits of the file, grouped as the file says; the columns saying
        where every hit stands are not among its branches, since the grouping
        says the same.

    Raises
    ------
    RepresentationError
        If the extension names no format, the file cannot be read, a column
        saying where a hit stands is missing, or the columns disagree with the
        grouping of the rows.
    """
    data = _read_columns(path)
    missing = [name for name in STRUCTURE_COLUMNS if name not in data.columns]
    if missing:
        raise RepresentationError(
            f"{path} holds no {missing} column(s), so it was not written as events, tracks "
            f"and hits. Columns of the file: {list(data.columns)}."
        )

    branches = [name for name in data.branch_names if name not in STRUCTURE_COLUMNS]
    representation = EventTrackHit(data.select(branches))
    # A written file stands in the order of its representation. Rows that would
    # be put in another order were changed after writing - a time, a track or
    # an event identifier - whatever the five columns say.
    if not np.array_equal(representation.index.hit_rows, np.arange(len(data))):
        raise RepresentationError(
            f"The rows of {path} are not in the order of their events, tracks and times; "
            "the file was changed after it was written, or not written by this package."
        )
    for name, values in representation.structure_columns().items():
        if not np.array_equal(data[name], values):
            raise RepresentationError(
                f"The '{name}' column of {path} disagrees with the grouping of its rows; "
                "the file was changed after it was written, or not written by this package."
            )
    return representation


def _read_columns(path: Path) -> TreeData:
    """Return every column of a written file, in the format its extension names.

    Raises
    ------
    RepresentationError
        If the extension names no format, or the file cannot be read.
    """
    extension = path.suffix.lstrip(".").lower()
    try:
        file_format = OutputFileFormat(extension)
    except ValueError as err:
        formats = [member.value for member in OutputFileFormat]
        raise RepresentationError(
            f"The extension of {path} names no format the package writes: {formats}."
        ) from err

    if file_format is OutputFileFormat.ROOT:
        return read_tree(path, GateTree.HITS, validate=False)
    try:
        if file_format is OutputFileFormat.HDF5:
            return _read_hdf5(path)
        return _typed(pd.read_csv(path))
    except (OSError, KeyError, ValueError) as err:
        raise RepresentationError(f"{path} could not be read: {err}") from err


def _typed(frame: pd.DataFrame) -> TreeData:
    """Return the columns of a CSV file in the types the package writes them in."""
    gate_branches = {name for branches in VARIANT_BRANCHES.values() for name in branches}
    columns: dict[str, npt.NDArray[Any]] = {}
    folded: set[str] = set()
    for name, width in ARRAY_BRANCH_WIDTHS.items():
        parts = [f"{name}_{component}" for component in range(width)]
        if name not in frame.columns and all(part in frame.columns for part in parts):
            columns[name] = frame[parts].to_numpy().astype(np.int32)
            folded.update(parts)
    for name in frame.columns:
        if name in folded:
            continue
        column = frame[name].to_numpy()
        if name in STRUCTURE_COLUMNS:
            column = column.astype(STRUCTURE_COLUMNS[name])
        elif name in gate_branches and branch_spec(name).kind is not BranchKind.TEXT:
            column = column.astype(branch_spec(name).dtype)
        columns[str(name)] = column
    # A folded array goes where its first component stood.
    order: list[str] = []
    for name in frame.columns:
        if name not in folded:
            order.append(str(name))
        elif str(name).endswith("_0"):
            order.append(str(name)[: -len("_0")])
    return TreeData(GateTree.HITS, {name: columns[name] for name in order})


def _read_hdf5(path: Path) -> TreeData:
    """Return the columns of the group of hits of an HDF5 file the package wrote."""
    columns: dict[str, npt.NDArray[Any]] = {}
    with h5py.File(path, "r") as input_file:
        group = input_file[GateTree.HITS.value]
        for name, dataset in group.items():
            if h5py.check_string_dtype(dataset.dtype) is not None:
                columns[name] = np.asarray(dataset.asstr()[()], dtype=object)
            else:
                columns[name] = dataset[()]
    return TreeData(GateTree.HITS, columns)
