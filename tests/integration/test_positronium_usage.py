"""End-to-end use of the PositroniumSource representations.

The point of the enums is what a user does with them: read a file, keep the
rows describing one kind of gamma, and take that further. The tests here walk
that path on simulation output.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from opengate_gate_tree import (
    DECAY_INDEX_BRANCH,
    GammaType,
    GateTree,
    OutputFileFormat,
    SourceType,
    decode_positronium_column,
    is_positronium_source,
    read_tree,
    write_tree,
)


def test_prompt_gammas_are_selected_by_what_they_are(
    positronium_files: Mapping[str, Path],
) -> None:
    """Keeping the prompt gammas of a run is one comparison, on the read column."""
    # ARRANGE
    data = read_tree(
        positronium_files["ops-prompt"],
        GateTree.HITS,
        ["gammaType", "edep", "eventID"],
    )

    # ACT
    prompt = data["gammaType"] == GammaType.PROMPT
    energies = data["edep"][prompt]

    # ASSERT
    assert prompt.sum() == np.count_nonzero(data["gammaType"] == int(GammaType.PROMPT))
    assert energies.size == prompt.sum()
    assert 0 < energies.size < data.entry_count


def test_a_mixed_file_is_split_by_the_source_of_each_gamma(
    positronium_files: Mapping[str, Path],
) -> None:
    """A file mixing channels is separated by what emitted each gamma."""
    # ARRANGE
    data = read_tree(positronium_files["pps-direct"], GateTree.HITS, ["sourceType"])
    column = data["sourceType"]

    # ACT
    counts = {
        source: int(np.count_nonzero(column == source))
        for source in (SourceType.PARA_POSITRONIUM, SourceType.DIRECT_ANNIHILATION)
    }

    # ASSERT
    assert all(count > 0 for count in counts.values())
    assert sum(counts.values()) == data.entry_count


def test_rows_of_another_source_are_dropped_before_an_analysis(
    positronium_files: Mapping[str, Path],
) -> None:
    """A file can hold gammas written by another source, and they are told apart.

    The mask says which source wrote a row, not whether positronium was
    formed: a PositroniumSource configured with a direct annihilation channel
    writes those gammas too, and they carry a channel number like any other.
    """
    # ARRANGE
    data = read_tree(
        positronium_files["all-variants"],
        GateTree.HITS,
        [DECAY_INDEX_BRANCH, "sourceType"],
    )

    # ACT
    from_positronium = is_positronium_source(data[DECAY_INDEX_BRANCH])
    sources = set(data["sourceType"][from_positronium].tolist())
    others = set(data["sourceType"][~from_positronium].tolist())

    # ASSERT
    assert sources == {
        SourceType.PARA_POSITRONIUM,
        SourceType.ORTHO_POSITRONIUM,
        SourceType.DIRECT_ANNIHILATION,
    }
    assert others == {SourceType.NOT_DEFINED}


def test_a_data_frame_carries_the_meaning_alongside_the_values(
    positronium_files: Mapping[str, Path],
) -> None:
    """Reading the values as names is what a table meant for people needs."""
    # ARRANGE
    data = read_tree(positronium_files["all-variants"], GateTree.HITS, ["gammaType", "edep"])
    frame = data.to_dataframe()

    # ACT
    frame["gammaTypeName"] = [
        meaning.name for meaning in decode_positronium_column("gammaType", data["gammaType"])
    ]
    by_kind = frame.groupby("gammaTypeName")["edep"].sum()

    # ASSERT
    assert set(by_kind.index) == {"UNKNOWN", "ANNIHILATION", "PROMPT"}
    assert by_kind.sum() == frame["edep"].sum()


def test_written_files_hold_the_values_gate_wrote(
    positronium_files: Mapping[str, Path],
    tmp_path: Path,
) -> None:
    """The names are a reading of the data, and the data goes out as it came in."""
    # ARRANGE
    data = read_tree(positronium_files["ops-prompt"], GateTree.HITS, ["gammaType", "edep"])
    output_file = tmp_path / "hits.csv"

    # ACT
    write_tree(data, output_file, OutputFileFormat.CSV)

    # ASSERT
    written = pd.read_csv(output_file)
    assert set(written["gammaType"]) == {int(GammaType.ANNIHILATION), int(GammaType.PROMPT)}
    assert written["gammaType"].dtype.kind in "iu"
