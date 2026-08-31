"""Unit tests for summarising extracted data."""

import json
from pathlib import Path

import numpy as np
import pytest

from opengate_gate_tree.errors import ExportError
from opengate_gate_tree.io.reader import read_hits_trees, read_tree
from opengate_gate_tree.io.statistics import write_statistics
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.detection import detect_hits_variant
from opengate_gate_tree.tree.hits.schema import BranchKind, expected_branches
from opengate_gate_tree.tree.hits.variant import GateSystemType, HitsTreeVariant
from opengate_gate_tree.tree.statistics import (
    compute_statistics,
    format_statistics,
    statistics_to_dict,
)
from opengate_gate_tree.tree.treedata import TreeData


def summarise(columns: dict[str, np.ndarray], tree: GateTree = GateTree.HITS) -> dict[str, object]:
    """Return the per-branch summaries of the given columns, keyed by name."""
    statistics = compute_statistics(TreeData(tree, columns))
    return {branch.name: branch for branch in statistics.branches}


def test_a_whole_number_branch_is_summarised_by_its_range() -> None:
    """Numbers are described by where they start, end and sit on average."""
    # ARRANGE
    columns = {"eventID": np.array([3, 1, 1, 7], dtype=np.int32)}

    # ACT
    branch = summarise(columns)["eventID"]

    # ASSERT
    assert (branch.minimum, branch.maximum) == (1.0, 7.0)
    assert branch.mean == 3.0
    assert branch.unique_count == 3
    assert branch.kind is BranchKind.INTEGER
    assert branch.nan_count is None


def test_values_that_are_not_a_number_are_counted_and_left_out() -> None:
    """A missing value must not turn the whole summary into nothing."""
    # ARRANGE
    columns = {"edep": np.array([1.0, np.nan, 3.0], dtype=np.float32)}

    # ACT
    branch = summarise(columns)["edep"]

    # ASSERT
    assert branch.nan_count == 1
    assert branch.mean == 2.0
    assert (branch.minimum, branch.maximum) == (1.0, 3.0)


def test_a_branch_of_nothing_but_missing_values_has_no_range() -> None:
    """JSON has no way to write "not a number", so none is reported."""
    # ARRANGE
    columns = {"edep": np.array([np.nan, np.nan], dtype=np.float32)}

    # ACT
    branch = summarise(columns)["edep"]

    # ASSERT
    assert branch.nan_count == 2
    assert (branch.minimum, branch.maximum, branch.mean, branch.std) == (None, None, None, None)


def test_a_text_branch_is_summarised_by_its_values() -> None:
    """Which processes occurred, and how often, is what the column says."""
    # ARRANGE
    values = ["Compton"] * 3 + ["Rayleigh"] * 2 + ["photoElectric"]
    columns = {"processName": np.array(values, dtype=object)}

    # ACT
    branch = summarise(columns)["processName"]

    # ASSERT
    assert branch.kind is BranchKind.TEXT
    assert branch.unique_count == 3
    assert branch.top_values[0] == ("Compton", 3)
    assert branch.minimum is None


def test_equally_frequent_values_are_ordered_by_name() -> None:
    """A report that changes between runs of the same data is no report."""
    # ARRANGE
    columns = {"processName": np.array(["Rayleigh", "Compton"], dtype=object)}

    # ACT
    branch = summarise(columns)["processName"]

    # ASSERT
    assert branch.top_values == (("Compton", 1), ("Rayleigh", 1))


def test_an_array_branch_is_summarised_as_a_whole() -> None:
    """The detector hierarchy is one branch, not ten of them."""
    # ARRANGE
    columns = {"volumeID": np.arange(20, dtype=np.int32).reshape(2, 10)}

    # ACT
    branch = summarise(columns)["volumeID"]

    # ASSERT
    assert branch.kind is BranchKind.INTEGER_ARRAY
    assert branch.dtype == "int32[10]"
    assert (branch.minimum, branch.maximum) == (0.0, 19.0)
    assert branch.entries == 2


def test_hits_are_summarised_by_what_they_describe(gate_hits_file: Path) -> None:
    """The numbers a physicist asks for first come from the columns."""
    # ARRANGE
    data = read_tree(gate_hits_file, GateTree.HITS)

    # ACT
    summary = compute_statistics(data).hits_summary

    # ASSERT
    assert summary is not None
    assert summary.run_count == len(np.unique(data["runID"]))
    assert summary.track_count == len(np.unique(data["trackID"]))
    assert summary.total_edep == pytest.approx(float(np.sum(data["edep"])))
    assert summary.time_min == pytest.approx(float(np.min(data["time"])))


def test_events_are_counted_by_run_and_event_together(
    hits_variant_files: dict[str, Path],
) -> None:
    """Event numbers restart with every run, so the pair is what counts."""
    # ARRANGE
    data = read_hits_trees(hits_variant_files["multi-run"])

    # ACT
    summary = compute_statistics(data).hits_summary

    # ASSERT
    assert summary is not None
    assert summary.event_key == ("runID", "eventID")
    assert summary.event_count > len(np.unique(data["eventID"]))


def test_one_event_in_two_detectors_is_counted_once(
    hits_variant_files: dict[str, Path],
) -> None:
    """Merging detectors must not double the events of a simulation."""
    # ARRANGE
    data = read_hits_trees(hits_variant_files["multi-sd"])

    # ACT
    summary = compute_statistics(data).hits_summary

    # ASSERT
    assert summary is not None
    assert summary.event_count == len(np.unique(data["eventID"]))
    assert summary.source_trees == ("Hits_DET_INNER", "Hits_DET_OUTER")


def test_events_are_counted_without_runs_when_runs_were_not_read() -> None:
    """A selection leaving out the run says so instead of guessing."""
    # ARRANGE
    columns = {"eventID": np.array([1, 1, 2], dtype=np.int32)}

    # ACT
    summary = compute_statistics(TreeData(GateTree.HITS, columns)).hits_summary

    # ASSERT
    assert summary is not None
    assert summary.event_key == ("eventID",)
    assert summary.event_count == 2
    assert summary.run_count is None


def test_a_selection_without_the_usual_branches_is_summarised_anyway() -> None:
    """Statistics work with what was extracted, whatever that is."""
    # ARRANGE
    columns = {"posX": np.array([1.0, 2.0], dtype=np.float32)}

    # ACT
    summary = compute_statistics(TreeData(GateTree.HITS, columns)).hits_summary

    # ASSERT
    assert summary is not None
    assert summary.event_key == ()
    assert (summary.event_count, summary.total_edep, summary.time_min) == (None, None, None)


def test_a_tree_of_another_kind_is_summarised_without_the_physics() -> None:
    """Only hits are read as hits."""
    # ARRANGE
    columns = {"energy1": np.array([1.0, 2.0], dtype=np.float32)}

    # ACT
    statistics = compute_statistics(TreeData(GateTree.COINCIDENCES, columns))

    # ASSERT
    assert statistics.hits_summary is None
    assert len(statistics.branches) == 1


def test_a_tree_without_entries_is_summarised_without_numbers() -> None:
    """An empty tree is a fact to report, not a failure."""
    # ARRANGE
    columns = {"eventID": np.array([], dtype=np.int32)}

    # ACT
    statistics = compute_statistics(TreeData(GateTree.HITS, columns))

    # ASSERT
    assert statistics.entry_count == 0
    assert statistics.branches[0].minimum is None


def test_the_recognised_structure_is_reported_with_the_numbers(
    hits_variant_files: dict[str, Path],
) -> None:
    """A summary is read together with what the tree was taken for."""
    # ARRANGE
    names = [spec.name for spec in expected_branches(HitsTreeVariant.NO_SYSTEM)]
    detection = detect_hits_variant(names, "Hits")
    data = read_tree(hits_variant_files["a1"], GateTree.HITS)

    # ACT
    statistics = compute_statistics(data, detection)
    rendered = format_statistics(statistics)

    # ASSERT
    assert statistics.detection is detection
    assert "Hits tree variant: No system (A1)" in rendered
    assert "Entries: 500" in rendered


def test_the_rendering_states_what_the_branches_hold(
    hits_variant_files: dict[str, Path],
) -> None:
    """The rendering is what a command line run shows about the data."""
    # ARRANGE
    data = read_tree(hits_variant_files["a1"], GateTree.HITS)

    # ACT
    rendered = format_statistics(compute_statistics(data))

    # ASSERT
    assert "Events: " in rendered
    assert "  - edep / float32: min " in rendered
    assert "  - processName / text: " in rendered
    assert "distinct" in rendered


def test_the_rendering_reports_a_branch_without_usable_values() -> None:
    """A column of missing values still deserves a line of its own."""
    # ARRANGE
    columns = {"edep": np.array([np.nan], dtype=np.float32)}

    # ACT
    rendered = format_statistics(compute_statistics(TreeData(GateTree.HITS, columns)))

    # ASSERT
    assert "no usable value" in rendered


def test_the_report_holds_plain_values_only(hits_variant_files: dict[str, Path]) -> None:
    """A report that cannot be written is not a report."""
    # ARRANGE
    data = read_tree(hits_variant_files["a1"], GateTree.HITS)
    statistics = compute_statistics(data)

    # ACT
    report = statistics_to_dict(statistics)
    written = json.dumps(report, allow_nan=False)

    # ASSERT
    assert json.loads(written)["entries"] == 500
    assert isinstance(report["branches"][0]["minimum"], float)


def test_the_report_survives_values_that_are_not_a_number() -> None:
    """JSON refuses NaN, and the report must not depend on it accepting one."""
    # ARRANGE
    columns = {"edep": np.array([np.nan, 1.0], dtype=np.float32)}
    statistics = compute_statistics(TreeData(GateTree.HITS, columns))

    # ACT
    written = json.dumps(statistics_to_dict(statistics), allow_nan=False)

    # ASSERT
    assert json.loads(written)["branches"][0]["not_a_number"] == 1


def test_the_report_names_the_structure_and_the_system(
    hits_variant_files: dict[str, Path],
) -> None:
    """A report read months later has to say what the file was."""
    # ARRANGE
    data = read_tree(hits_variant_files["a2"], GateTree.HITS)
    detection = detect_hits_variant(list(data.branch_names), "Hits")

    # ACT
    report = statistics_to_dict(compute_statistics(data, detection))

    # ASSERT
    assert report["structure"]["reference"] == "A2"
    assert report["structure"]["system_scheme"] == ["cylindricalPET", "OPET"]
    assert report["structure"]["system_depth"] == 6


def test_writing_a_report_creates_the_directory(
    hits_variant_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A report goes next to the data, wherever the data was asked to go."""
    # ARRANGE
    data = read_tree(hits_variant_files["a1"], GateTree.HITS)
    output_file = tmp_path / "reports" / "hits.stats.json"

    # ACT
    written = write_statistics(compute_statistics(data), output_file)

    # ASSERT
    assert written == output_file
    assert json.loads(output_file.read_text(encoding="utf-8"))["tree"] == "Hits"


def test_writing_a_report_replaces_an_earlier_one(
    hits_variant_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Running the same extraction twice leaves one report, not two."""
    # ARRANGE
    output_file = tmp_path / "hits.stats.json"
    output_file.write_text("stale", encoding="utf-8")
    data = read_tree(hits_variant_files["a1"], GateTree.HITS)

    # ACT
    write_statistics(compute_statistics(data), output_file)

    # ASSERT
    assert json.loads(output_file.read_text(encoding="utf-8"))["entries"] == 500


def test_a_report_that_cannot_be_written_is_reported(tmp_path: Path) -> None:
    """Failing to save a report is a package error like any other."""
    # ARRANGE
    columns = {"eventID": np.array([1], dtype=np.int32)}
    statistics = compute_statistics(TreeData(GateTree.HITS, columns))
    output_file = tmp_path / "hits.stats.json"
    output_file.mkdir()

    # ACT / ASSERT
    with pytest.raises(ExportError, match="Statistics file could not be written"):
        write_statistics(statistics, output_file)


def test_a_time_branch_of_missing_values_gives_no_range() -> None:
    """A time column of nothing but missing values has no first or last hit."""
    # ARRANGE
    columns = {"time": np.array([np.nan, np.nan], dtype=np.float64)}

    # ACT
    summary = compute_statistics(TreeData(GateTree.HITS, columns)).hits_summary

    # ASSERT
    assert summary is not None
    assert (summary.time_min, summary.time_max) == (None, None)


def test_the_report_names_the_trees_a_merged_dataset_came_from(
    hits_variant_files: dict[str, Path],
) -> None:
    """A saved report of merged data has to say what went into it."""
    # ARRANGE
    data = read_hits_trees(hits_variant_files["multi-run"])

    # ACT
    report = statistics_to_dict(compute_statistics(data))

    # ASSERT
    assert report["hits"]["source_trees"] == ["Hits", "Hits_run1", "Hits_run2"]


def test_the_rendering_names_the_system_and_the_source_trees(
    hits_variant_files: dict[str, Path],
) -> None:
    """What was read, and from where, belongs at the top of the rendering."""
    # ARRANGE
    data = read_hits_trees(hits_variant_files["multi-run"])
    detection = detect_hits_variant(
        [spec.name for spec in expected_branches(HitsTreeVariant.SYSTEM, GateSystemType.ECAT)],
        "Hits",
    )

    # ACT
    rendered = format_statistics(compute_statistics(data, detection))

    # ASSERT
    assert "system ecat / ecatAccel" in rendered
    assert "Source trees: Hits, Hits_run1, Hits_run2" in rendered


def test_the_rendering_reports_missing_values_of_a_branch() -> None:
    """A column that is partly missing should say so where it is described."""
    # ARRANGE
    columns = {"edep": np.array([1.0, np.nan], dtype=np.float32)}

    # ACT
    rendered = format_statistics(compute_statistics(TreeData(GateTree.HITS, columns)))

    # ASSERT
    assert "not a number 1" in rendered
