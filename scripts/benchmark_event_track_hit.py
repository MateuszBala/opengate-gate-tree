"""Measure how fast the event, track and hit representation is.

The script builds synthetic "Hits" trees of growing size, in the proportions
of the B1 reference output - one or two tracks per event, one to six hits per
track - laid out the way GATE writes them: the hits of a track one after
another in time, the tracks of an event from the last to the first. Each size
is measured twice: once in that order, which the index groups by sorting only
the blocks of rows a track occupies, and once with every row shuffled, which
it groups by sorting every row, as it does for hits merged from one tree per
sensitive detector.

For every size and order it reports, as a Markdown table:

- build: grouping the rows into events, tracks and hits;
- chain: a typical selection, ``with_track_count(2).with_hit_count(2).first_hits()``;
- table: the hits table of four fields;
- write: writing the representation as ``.eth.hdf5``;
- index: the memory the grouping keeps, per hit;
- peak: the most memory the grouping holds at once while it is built, per hit.

A "Hits" tree read from a GATE file takes about 200 bytes per hit, against
which the two memory columns are to be read. Every written file is removed as
soon as it is measured.

The script runs outside CI and is not part of ``make check``. Run it from the
root of the repository::

    uv run python scripts/benchmark_event_track_hit.py --sizes 100000 1000000 10000000
"""

import argparse
import platform
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from opengate_gate_tree import (
    EventTrackHit,
    EventTrackHitField,
    GateTree,
    OutputFileFormat,
    TreeData,
    write_event_track_hit,
)

# Share of events with one and with two tracks, and of tracks with one to six
# hits, rounded from the B1 reference output.
TRACKS_PER_EVENT = (np.array([1, 2]), np.array([0.30, 0.70]))
HITS_PER_TRACK = (
    np.array([1, 2, 3, 4, 5, 6]),
    np.array([0.427, 0.388, 0.139, 0.036, 0.009, 0.001]),
)

PROCESSES = np.array(["Compton", "PhotoElectric"], dtype=object)


def synthetic_hits(entries: int, seed: int = 2026) -> TreeData:
    """Return about the given number of hits, in the order GATE writes them."""
    rng = np.random.default_rng(seed)
    mean_hits = float(np.dot(*HITS_PER_TRACK)) * float(np.dot(*TRACKS_PER_EVENT))
    events = max(1, int(entries / mean_hits))

    track_choices, track_shares = TRACKS_PER_EVENT
    hit_choices, hit_shares = HITS_PER_TRACK
    tracks_per_event: npt.NDArray[np.int64] = rng.choice(track_choices, events, p=track_shares)
    tracks = int(tracks_per_event.sum())
    hits_per_track: npt.NDArray[np.int64] = rng.choice(hit_choices, tracks, p=hit_shares)
    hits = int(hits_per_track.sum())

    event_of_track = np.repeat(np.arange(events), tracks_per_event)
    first_track = np.repeat(np.cumsum(tracks_per_event) - tracks_per_event, tracks_per_event)
    position_in_event = np.arange(tracks) - first_track
    # GATE writes the tracks of an event from the last one to the first.
    track_ids = np.repeat(tracks_per_event, tracks_per_event) - position_in_event
    first_hit = np.repeat(np.cumsum(hits_per_track) - hits_per_track, hits_per_track)
    position_in_track = np.arange(hits) - first_hit

    event_of_hit = np.repeat(event_of_track, hits_per_track)
    times = event_of_hit * 1e-6 + position_in_track * 1e-10
    columns: dict[str, npt.NDArray[Any]] = {
        "runID": np.zeros(hits, dtype=np.int32),
        "eventID": event_of_hit.astype(np.int32),
        "trackID": np.repeat(track_ids, hits_per_track).astype(np.int32),
        "time": times,
        "edep": rng.uniform(0.0, 0.511, hits).astype(np.float32),
        "posX": rng.normal(0.0, 300.0, hits).astype(np.float32),
        "posY": rng.normal(0.0, 300.0, hits).astype(np.float32),
        "posZ": rng.normal(0.0, 100.0, hits).astype(np.float32),
        "processName": PROCESSES[(position_in_track == 0).astype(int) ^ 1],
    }
    return TreeData(GateTree.HITS, columns)


def shuffled(data: TreeData, seed: int = 7) -> TreeData:
    """Return the same hits with every row in another place."""
    order = np.random.default_rng(seed).permutation(len(data))
    return TreeData(data.tree, {name: column[order] for name, column in data.columns.items()})


def best_time(action: Callable[[], object], repeat: int) -> float:
    """Return the shortest of several runs of an action, in seconds."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        action()
        times.append(time.perf_counter() - start)
    return min(times)


def peak_memory(action: Callable[[], object]) -> int:
    """Return the peak memory an action allocates, in bytes."""
    tracemalloc.start()
    try:
        action()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def index_bytes(representation: EventTrackHit) -> int:
    """Return the memory the index of a representation keeps, in bytes."""
    index = representation.index
    arrays = (
        index.hit_rows,
        index.track_offsets,
        index.event_offsets,
        index.track_rows,
        index.event_rows,
    )
    return sum(array.nbytes for array in arrays)


def measure(data: TreeData, order: str, repeat: int, directory: Path) -> list[str]:
    """Return one row of the table for the given data."""
    representation = EventTrackHit(data)
    fields = [
        EventTrackHitField.EDEP,
        EventTrackHitField.POS_X,
        EventTrackHitField.POS_Y,
        EventTrackHitField.POS_Z,
    ]
    output = directory / f"benchmark-{len(data)}-{order}.eth.hdf5"

    build = best_time(lambda: EventTrackHit(data), repeat)
    chain = best_time(
        lambda: representation.with_track_count(2).with_hit_count(2).first_hits(), repeat
    )
    table = best_time(lambda: representation.hits_frame(fields), repeat)
    write = best_time(
        lambda: write_event_track_hit(representation, output, OutputFileFormat.HDF5), repeat
    )
    output.unlink()
    peak = peak_memory(lambda: EventTrackHit(data))
    return [
        f"{len(data):,}",
        order,
        f"{representation.event_count:,}",
        f"{build:.3f}",
        f"{chain:.3f}",
        f"{table:.3f}",
        f"{write:.3f}",
        f"{index_bytes(representation) / len(data):.1f} B",
        f"{peak / len(data):.0f} B",
    ]


def main(arguments: Sequence[str] | None = None) -> None:
    """Measure the representation at every size asked for and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory the measured files are written to; a temporary one by default",
    )
    options = parser.parse_args(arguments)

    header = [
        "hits",
        "order",
        "events",
        "build [s]",
        "chain [s]",
        "table [s]",
        "write [s]",
        "index per hit",
        "peak per hit",
    ]
    print(f"Python {platform.python_version()}, numpy {np.__version__}, {platform.machine()}")
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(" --- " for _ in header) + "|")
    with tempfile.TemporaryDirectory(dir=options.output_dir) as directory:
        for size in options.sizes:
            data = synthetic_hits(size)
            for order, ordered in (("as GATE writes", data), ("shuffled", shuffled(data))):
                row = measure(ordered, order, options.repeat, Path(directory))
                print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
