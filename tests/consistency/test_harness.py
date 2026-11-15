"""Tests of the harness itself, on a reference counted with pandas.

The self-check below is not an analysis consistency test: its reference is
computed here, by pandas, not by an analysis made outside the package. It
checks that the comparison agrees when the tables do, and says so when they
do not - a missing event, an extra one, a value out of tolerance, a missing
column - so that a passing case means what it claims.
"""

import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest
from harness import compare_tables, discover_cases, load_case

from opengate_gate_tree import (
    DecayType,
    EventTrackHitAnalysis,
    EventTrackHitField,
    GammaType,
    GateTree,
    Process,
    SourceType,
    read_tree,
)

TRACK_KEY = ["runID", "eventID", "trackID"]

ANALYSIS = (
    EventTrackHitAnalysis()
    .tracks(2)
    .gamma_type(GammaType.ANNIHILATION)
    .decay_type(DecayType.STANDARD)
    .source_type(SourceType.PARA_POSITRONIUM)
    .hits(eq=2)
    .hit(1)
    .process(Process.COMPTON)
    .hit(2)
    .process(Process.PHOTOELECTRIC)
    .tracks(2)
)


def pandas_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the table of the first example of the concept, made with pandas alone."""
    hits = frame.sort_values([*TRACK_KEY, "time"], kind="stable").copy()
    hits["hit"] = hits.groupby(TRACK_KEY).cumcount() + 1
    tracks = hits.groupby(TRACK_KEY).agg(
        size=("hit", "size"),
        first=("processName", "first"),
        last=("processName", "last"),
        gamma=("gammaType", "first"),
        decay=("decayType", "first"),
        source=("sourceType", "first"),
    )
    passing = tracks[
        (tracks["size"] == 2)
        & (tracks["first"] == "Compton")
        & (tracks["last"] == "PhotoElectric")
        & (tracks["gamma"] == GammaType.ANNIHILATION)
        & (tracks["decay"] == DecayType.STANDARD)
        & (tracks["source"] == SourceType.PARA_POSITRONIUM)
    ].reset_index()
    all_tracks = tracks.reset_index().groupby(["runID", "eventID"]).size()
    kept = passing.groupby(["runID", "eventID"]).size()
    events = kept[(kept == 2) & (all_tracks.reindex(kept.index) == 2)].index
    passing = passing.set_index(["runID", "eventID"]).loc[events].reset_index()
    passing["track"] = passing.groupby(["runID", "eventID"]).cumcount() + 1
    chosen = hits.merge(passing[[*TRACK_KEY, "track"]], on=TRACK_KEY)
    chosen["column"] = "t" + chosen["track"].astype(str) + "h" + chosen["hit"].astype(str) + "_edep"
    table = chosen.pivot(index=["runID", "eventID"], columns="column", values="edep")
    return table.reset_index()


@pytest.fixture(scope="module")
def tables(positronium_files: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the table of the package and the one pandas made, for the pPs scene."""
    frame = read_tree(positronium_files["pps"], GateTree.HITS).to_dataframe()
    return ANALYSIS.extract(frame, [EventTrackHitField.EDEP]), pandas_reference(frame)


def test_the_harness_agrees_when_the_tables_do(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The self-check: the package and pandas find the same 18 events and deposits."""
    # ARRANGE
    table, reference = tables

    # ACT
    differences = compare_tables(table, reference, rtol=1e-6)

    # ASSERT
    assert differences == []
    assert len(reference) == 18


def test_a_missing_and_an_extra_event_are_named(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Both sides of a disagreement about which events pass."""
    # ARRANGE
    table, reference = tables
    first = tuple(int(value) for value in reference.loc[0, ["runID", "eventID"]])
    shifted = reference.copy()
    shifted.loc[1, "eventID"] = 99999

    # ACT
    differences = compare_tables(table, shifted.iloc[1:], rtol=1e-6)

    # ASSERT
    assert any(
        "2 event(s) the package kept that the reference does not" in line for line in differences
    )
    assert any(
        "1 event(s) of the reference the package did not keep" in line for line in differences
    )
    assert any(str(first) in line for line in differences)


def test_a_value_beyond_the_tolerance_is_named_and_one_within_it_is_not(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Single precision against double differs in the seventh digit, not the fifth."""
    # ARRANGE
    table, reference = tables
    within = reference.copy()
    within["t1h1_edep"] = within["t1h1_edep"] * (1 + 1e-7)
    beyond = reference.copy()
    beyond.loc[3, "t2h2_edep"] *= 1.001

    # ACT
    close = compare_tables(table, within, rtol=1e-6)
    far = compare_tables(table, beyond, rtol=1e-6)

    # ASSERT
    assert close == []
    changed = tuple(int(value) for value in reference.loc[3, ["runID", "eventID"]])
    assert far == [f"Column t2h2_edep differs in 1 event(s), first {[changed]}."]


def test_columns_one_side_lacks_are_named(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A column nobody compares would pass without being checked."""
    # ARRANGE
    table, reference = tables

    # ACT
    differences = compare_tables(table.drop(columns="t1h1_edep"), reference, rtol=1e-6)
    extra = compare_tables(table, reference.drop(columns="t2h1_edep"), rtol=1e-6)

    # ASSERT
    assert differences == ["Columns the package does not give: ['t1h1_edep']."]
    assert extra == ["Columns the reference does not hold: ['t2h1_edep']."]


def test_whole_numbers_and_text_are_compared_exactly(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A track identifier is right or wrong."""
    # ARRANGE
    table, reference = tables
    mine = table.assign(label="pair")
    theirs = reference.assign(label="pair")
    theirs.loc[0, "label"] = "single"

    # ACT
    differences = compare_tables(mine, theirs, rtol=1e-6)

    # ASSERT
    assert len(differences) == 1
    assert differences[0].startswith("Column label differs in 1 event(s)")


def test_an_event_named_twice_or_a_missing_key_is_refused(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Without one row per event, rows cannot be paired."""
    # ARRANGE
    table, reference = tables
    doubled = pd.concat([reference, reference.iloc[:1]])

    # ACT
    twice = compare_tables(table, doubled, rtol=1e-6)
    keyless = compare_tables(table, reference.drop(columns="runID"), rtol=1e-6)

    # ASSERT
    assert twice == ["The reference names an event more than once."]
    assert keyless == ["The reference lacks the event key column(s) ['runID']."]


def write_case(directory: Path, input_file: Path, analysis: str) -> Path:
    """Write a case directory and return it."""
    directory.mkdir()
    shutil.copy(input_file, directory / "input.root")
    (directory / "reference.csv").write_text("runID,eventID\n")
    (directory / "analysis.py").write_text(analysis)
    (directory / "README.md").write_text("# A case\n")
    return directory


def test_a_case_is_loaded_from_its_directory(
    tmp_path: Path, positronium_files: Mapping[str, Path]
) -> None:
    """Its analysis, its variables and its tolerance come from analysis.py."""
    # ARRANGE
    write_case(
        tmp_path / "pairs",
        positronium_files["pps"],
        "from opengate_gate_tree import EventTrackHitAnalysis\n"
        "ANALYSIS = EventTrackHitAnalysis().tracks(2)\n"
        "VARIABLES = ['edep']\n"
        "RTOL = 1e-5\n"
        "ATOL = 1e-9\n",
    )
    (tmp_path / ".hidden").mkdir()

    # ACT
    (case,) = discover_cases(tmp_path)

    # ASSERT
    assert case.name == "pairs"
    assert case.analysis == EventTrackHitAnalysis().tracks(2)
    assert case.variables == ["edep"]
    assert case.rtol == 1e-5
    assert case.atol == 1e-9
    assert list(case.reference().columns) == ["runID", "eventID"]
    assert case.input_path.name == "input.root"


def test_a_case_without_its_files_or_its_analysis_is_refused(
    tmp_path: Path, positronium_files: Mapping[str, Path]
) -> None:
    """A case that does not say what it compares compares nothing."""
    # ARRANGE
    empty = tmp_path / "empty"
    empty.mkdir()
    wrong = write_case(tmp_path / "wrong", positronium_files["pps"], "ANALYSIS = 1\n")

    # ACT
    with pytest.raises(ValueError) as lacking:
        load_case(empty)
    with pytest.raises(ValueError) as malformed:
        load_case(wrong)

    # ASSERT
    assert "lacks ['input.root', 'reference.csv', 'analysis.py', 'README.md']" in str(lacking.value)
    assert "has to define ANALYSIS" in str(malformed.value)


def test_a_value_missing_on_both_sides_agrees(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Nothing against nothing is not a difference."""
    # ARRANGE
    table, reference = tables
    mine = table.copy()
    theirs = reference.copy()
    mine.loc[0, "t1h1_edep"] = float("nan")
    theirs.loc[0, "t1h1_edep"] = float("nan")

    # ACT
    differences = compare_tables(mine, theirs, rtol=1e-6)

    # ASSERT
    assert differences == []


def test_text_against_an_empty_column_is_a_difference_not_a_crash(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A reference column CSV read as nothing at all is reported as differing."""
    # ARRANGE
    table, reference = tables
    mine = table.assign(process="Compton")
    theirs = reference.assign(process=float("nan"))

    # ACT
    differences = compare_tables(mine, theirs, rtol=1e-6)

    # ASSERT
    assert len(differences) == 1
    assert differences[0].startswith(f"Column process differs in {len(table)} event(s)")


def test_whole_numbers_are_compared_exactly_even_when_written_as_floats(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A track identifier of 1 against 1.0000001 is a different track."""
    # ARRANGE
    table, reference = tables
    mine = table.assign(t1_trackID=1)
    exact = reference.assign(t1_trackID=1.0)
    off = reference.assign(t1_trackID=1.0000001)

    # ACT
    agreeing = compare_tables(mine, exact, rtol=1e-6)
    differing = compare_tables(mine, off, rtol=1e-6)

    # ASSERT
    assert agreeing == []
    assert differing[0].startswith("Column t1_trackID differs in 18 event(s)")


def test_columns_in_another_order_are_named(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The order of the columns is part of what the table promises."""
    # ARRANGE
    table, reference = tables
    swapped = reference[["runID", "eventID", "t1h2_edep", "t1h1_edep", "t2h1_edep", "t2h2_edep"]]

    # ACT
    differences = compare_tables(table, swapped, rtol=1e-6)

    # ASSERT
    assert differences == [
        "The columns are in another order: column 2 is 't1h1_edep' in the table and "
        "'t1h2_edep' in the reference."
    ]


def test_an_absolute_tolerance_covers_values_near_zero(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """1e-12 against an exact zero differs relatively, and agrees absolutely."""
    # ARRANGE
    table, reference = tables
    mine = table.assign(offset=0.0)
    theirs = reference.assign(offset=1e-12)

    # ACT
    relative = compare_tables(mine, theirs, rtol=1e-6)
    absolute = compare_tables(mine, theirs, rtol=1e-6, atol=1e-9)

    # ASSERT
    assert relative[0].startswith("Column offset differs")
    assert absolute == []
