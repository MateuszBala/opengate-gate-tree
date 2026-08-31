"""Unit tests for reading GATE ROOT files."""

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import HITS_VARIANT_LAYOUTS, GateHitsLayout, HitsVariantLayout

from opengate_gate_tree.errors import (
    AmbiguousTreeError,
    BranchNotFoundError,
    HitsTreeValidationError,
    RootFileError,
    TreeNotFoundError,
    UnknownHitsVariantError,
    UnsupportedBranchTypeError,
)
from opengate_gate_tree.io.rootfile import RootFile
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.schema import expected_branches
from opengate_gate_tree.tree.hits.variant import HitsTreeVariant
from opengate_gate_tree.tree.merge import SOURCE_TREE_BRANCH

# Variant fixtures storing their tree under the standard name.
STANDARD_NAME_LAYOUTS = [
    layout for layout in HITS_VARIANT_LAYOUTS if layout.tree_names[0] == "Hits"
]

# Branch names of the structure generated trees are built from.
NO_SYSTEM_BRANCHES: Sequence[str] = [
    spec.name for spec in expected_branches(HitsTreeVariant.NO_SYSTEM)
]


def hits_file(
    factory: Callable[..., Path],
    branches: Sequence[str] = NO_SYSTEM_BRANCHES,
    tree_name: str = "Hits",
    **extra: Mapping[str, Sequence[str]],
) -> Path:
    """Write a file holding one tree of the given branches."""
    return factory({tree_name: list(branches)}, **extra)


def test_tree_names_reports_only_trees(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """Histograms stored by GATE should not be reported as trees."""
    # ARRANGE
    # The fixture holds two TH1D histograms next to three trees.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        names = root_file.tree_names

    # ASSERT
    assert names == gate_hits_layout.tree_names
    assert "latest_event_ID" not in names
    assert "total_nb_primaries" not in names


def test_tree_names_strips_cycle_numbers(gate_hits_file: Path) -> None:
    """ROOT cycle suffixes should not leak into the reported names."""
    # ARRANGE
    # Keys are stored as "Hits;1" in the file.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        names = root_file.tree_names

    # ASSERT
    assert all(";" not in name for name in names)


def test_has_tree_answers_for_present_and_absent_trees(gate_hits_file: Path) -> None:
    """Presence of a tree should be reported without raising."""
    # ARRANGE
    # The fixture holds a "Hits" tree but no "Singles" tree.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        has_hits = root_file.has_tree(GateTree.HITS)
        has_singles = root_file.has_tree(GateTree.SINGLES)

    # ASSERT
    assert has_hits is True
    assert has_singles is False


def test_resolve_tree_name_returns_the_file_key(gate_hits_file: Path) -> None:
    """An exactly matching name should resolve to itself."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        resolved = root_file.resolve_tree_name(GateTree.HITS)

    # ASSERT
    assert resolved == "Hits"


def test_resolve_tree_name_matches_regardless_of_case(
    make_gate_root_file: Callable[..., Path],
) -> None:
    """A differently cased tree name should still resolve."""
    # ARRANGE
    path = make_gate_root_file({"hits": {"eventID": np.arange(3, dtype=np.int32)}})

    # ACT
    with RootFile(path) as root_file:
        resolved = root_file.resolve_tree_name(GateTree.HITS)

    # ASSERT
    assert resolved == "hits"


def test_resolve_tree_name_lists_available_trees_when_missing(gate_hits_file: Path) -> None:
    """A missing tree should point the caller at what the file holds."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with RootFile(gate_hits_file) as root_file, pytest.raises(TreeNotFoundError) as error_info:
        root_file.resolve_tree_name(GateTree.SINGLES)

    message = str(error_info.value)
    assert "Singles" in message
    assert "pet_data" in message
    assert "OpticalData" in message


def test_branch_names_returns_the_gate_hits_branches(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """The reader should expose the branches GATE writes for hits."""
    # ARRANGE
    expected_branches = {"eventID", "edep", "posX", "volumeID", "processName"}

    # ACT
    with RootFile(gate_hits_file) as root_file:
        names = root_file.branch_names(GateTree.HITS)

    # ASSERT
    assert len(names) == gate_hits_layout.branch_count
    assert expected_branches.issubset(set(names))


def test_read_loads_every_branch_by_default(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """Reading without a selection should load the whole tree."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS)

    # ASSERT
    assert data.tree == GateTree.HITS
    assert data.entry_count == gate_hits_layout.entries
    assert len(data.branch_names) == gate_hits_layout.branch_count


def test_read_keeps_the_fixed_width_array_branch(
    gate_hits_file: Path,
    gate_hits_layout: GateHitsLayout,
) -> None:
    """The volumeID branch should stay two-dimensional."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, ["volumeID"])

    # ASSERT
    assert data["volumeID"].shape == (
        gate_hits_layout.entries,
        gate_hits_layout.volume_id_width,
    )
    assert data.array_branches == {"volumeID": gate_hits_layout.volume_id_width}


def test_read_loads_text_branches(gate_hits_file: Path) -> None:
    """Text branches should load as readable values."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, ["processName"])

    # ASSERT
    assert data.dtypes["processName"].kind == "O"
    assert isinstance(data["processName"][0], str)
    assert "Compton" in set(data["processName"])


def test_read_keeps_the_requested_branch_order(gate_hits_file: Path) -> None:
    """The selection order should be preserved, not the file order."""
    # ARRANGE
    requested = ["edep", "eventID", "posX"]

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, requested)

    # ASSERT
    assert data.branch_names == tuple(requested)


def test_read_keeps_repeated_branches_once(gate_hits_file: Path) -> None:
    """A repeated branch should be read once, at its first position."""
    # ARRANGE
    requested = ["edep", "eventID", "edep"]

    # ACT
    with RootFile(gate_hits_file) as root_file:
        data = root_file.read(GateTree.HITS, requested)

    # ASSERT
    assert data.branch_names == ("edep", "eventID")


def test_read_raises_for_an_unknown_branch(gate_hits_file: Path) -> None:
    """An unknown branch should be reported before any data is loaded."""
    # ARRANGE
    # No additional setup required.

    # ACT & ASSERT
    with RootFile(gate_hits_file) as root_file, pytest.raises(BranchNotFoundError) as error_info:
        root_file.read(GateTree.HITS, ["eventID", "missing"])

    assert "missing" in str(error_info.value)


def test_read_raises_for_a_branch_of_varying_length(
    make_jagged_root_file: Callable[..., Path],
) -> None:
    """Branches whose length varies per entry should be rejected."""
    # ARRANGE
    # The generated tree holds two branches, so it is no hits structure and
    # is read with the structure check turned off.
    path = make_jagged_root_file()

    # ACT & ASSERT
    with RootFile(path) as root_file, pytest.raises(UnsupportedBranchTypeError) as error_info:
        root_file.read(GateTree.HITS, validate=False)

    assert "hitTimes" in str(error_info.value)


def test_read_warns_for_an_empty_tree(
    make_gate_root_file: Callable[..., Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reading a tree without entries should leave a trace in the log."""
    # ARRANGE
    path = make_gate_root_file({"Hits": {"eventID": np.array([], dtype=np.int32)}})

    # ACT
    with caplog.at_level("WARNING"), RootFile(path) as root_file:
        data = root_file.read(GateTree.HITS, validate=False)

    # ASSERT
    assert data.entry_count == 0
    assert "no entries" in caplog.text


def test_opening_a_missing_file_reports_a_package_error(tmp_path: Path) -> None:
    """A missing input file should raise a package error."""
    # ARRANGE
    path = tmp_path / "missing.root"

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="does not exist"):
        RootFile(path)


def test_opening_a_file_that_is_not_root_reports_a_package_error(tmp_path: Path) -> None:
    """A file that only looks like a ROOT file should raise a package error."""
    # ARRANGE
    path = tmp_path / "simulation.root"
    path.write_text("this is not a ROOT file\n", encoding="utf-8")

    # ACT & ASSERT
    with pytest.raises(RootFileError, match="could not be read as a ROOT file"):
        RootFile(path)


def test_reader_exposes_the_file_path(gate_hits_file: Path) -> None:
    """The reader should report the file it was opened on."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        reported_path = root_file.path

    # ASSERT
    assert reported_path == gate_hits_file


def test_close_can_be_called_directly(gate_hits_file: Path) -> None:
    """The reader should be usable without the context manager."""
    # ARRANGE
    root_file = RootFile(gate_hits_file)

    # ACT
    names = root_file.tree_names
    root_file.close()

    # ASSERT
    assert "Hits" in names


@pytest.mark.parametrize("layout", STANDARD_NAME_LAYOUTS, ids=lambda layout: layout.key)
def test_variant_files_are_read_with_validation_on(layout: HitsVariantLayout) -> None:
    """Every structure the package supports should read without complaint."""
    # ARRANGE
    # Validation is on by default and covers the whole tree.

    # ACT
    with RootFile(layout.path) as root_file:
        data = root_file.read(GateTree.HITS)

    # ASSERT
    assert data.entry_count == layout.entries
    assert len(data.branch_names) == layout.branch_count


def test_reading_recognises_the_structure_of_the_hits_tree(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The structure is reported for a file, not only for a branch list."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["a3"]) as root_file:
        detection = root_file.detect_hits_tree()

    # ASSERT
    assert detection.variant is HitsTreeVariant.SYSTEM_SEPTAL
    assert detection.tree_name == "Hits"


def test_recognising_a_missing_hits_tree_reports_it(
    make_gate_root_file: Callable[..., Path],
) -> None:
    """A file without hits cannot have their structure recognised."""
    # ARRANGE
    path = make_gate_root_file({"Singles": {"eventID": np.arange(3, dtype=np.int32)}})

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(TreeNotFoundError):
        root_file.detect_hits_tree()


def test_reading_an_unknown_structure_is_refused(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """A tree that is not a hits tree should be reported, not half read."""
    # ARRANGE
    path = hits_file(make_hits_root_file, ["eventID", "edep", "sinogramTheta"])

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(UnknownHitsVariantError):
        root_file.read(GateTree.HITS)


def test_reading_an_unknown_structure_is_possible_without_validation(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """A file from a build the package does not know still has to be usable."""
    # ARRANGE
    path = hits_file(make_hits_root_file, ["eventID", "edep", "sinogramTheta"])

    # ACT
    with RootFile(path) as root_file:
        data = root_file.read(GateTree.HITS, validate=False)

    # ASSERT
    assert data.branch_names == ("eventID", "edep", "sinogramTheta")


def test_reading_a_tree_missing_a_branch_is_refused(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """A tree that almost matches its structure is reported as such."""
    # ARRANGE
    incomplete = [name for name in NO_SYSTEM_BRANCHES if name != "edep"]
    path = hits_file(make_hits_root_file, incomplete)

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(HitsTreeValidationError, match="edep"):
        root_file.read(GateTree.HITS)


def test_reading_a_tree_with_an_extra_branch_warns_and_continues(
    make_hits_root_file: Callable[..., Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GATE build adding a branch must not stop the extraction."""
    # ARRANGE
    path = hits_file(make_hits_root_file, [*NO_SYSTEM_BRANCHES, "multiPhotonFlag"])

    # ACT
    with caplog.at_level(logging.WARNING), RootFile(path) as root_file:
        data = root_file.read(GateTree.HITS)

    # ASSERT
    assert "multiPhotonFlag" in caplog.text
    assert "multiPhotonFlag" in data.branch_names


def test_validation_covers_the_tree_and_not_only_the_selection(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """Reading two branches of a broken tree is still reading a broken tree."""
    # ARRANGE
    incomplete = [name for name in NO_SYSTEM_BRANCHES if name != "edep"]
    path = hits_file(make_hits_root_file, incomplete)

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(HitsTreeValidationError):
        root_file.read(GateTree.HITS, ["eventID", "runID"])


def test_a_selection_of_a_valid_tree_reads_the_selected_branches(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Validation looks at the whole tree, extraction at what was asked for."""
    # ARRANGE
    # The A2 tree holds 46 branches; two of them are wanted.

    # ACT
    with RootFile(hits_variant_files["a2"]) as root_file:
        data = root_file.read(GateTree.HITS, ["eventID", "edep"])

    # ASSERT
    assert data.branch_names == ("eventID", "edep")


def test_other_trees_are_read_without_being_checked(
    make_gate_root_file: Callable[..., Path],
) -> None:
    """Only the structure of the hits tree is described, so only it is checked."""
    # ARRANGE
    path = make_gate_root_file(
        {"Singles": {"eventID": np.arange(3, dtype=np.int32), "energy": np.zeros(3)}}
    )

    # ACT
    with RootFile(path) as root_file:
        data = root_file.read(GateTree.SINGLES)

    # ASSERT
    assert data.branch_names == ("eventID", "energy")


def test_validation_looks_at_branches_it_could_not_load(
    make_hits_root_file: Callable[..., Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The check runs on branch types, so an unreadable branch is no obstacle.

    A branch of varying length cannot be loaded, but it can be looked at. It
    is reported as one the structure does not describe, and the branches that
    were asked for are read as usual.
    """
    # ARRANGE
    path = make_hits_root_file({"Hits": NO_SYSTEM_BRANCHES}, jagged=["hitTimes"])

    # ACT
    with caplog.at_level(logging.WARNING), RootFile(path) as root_file:
        data = root_file.read(GateTree.HITS, ["eventID", "edep"])

    # ASSERT
    assert "hitTimes" in caplog.text
    assert data.branch_names == ("eventID", "edep")


def test_an_unloadable_branch_is_still_refused_when_it_is_asked_for(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """Tolerating a branch in the tree is not the same as being able to read it."""
    # ARRANGE
    path = make_hits_root_file({"Hits": NO_SYSTEM_BRANCHES}, jagged=["hitTimes"])

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(UnsupportedBranchTypeError, match="hitTimes"):
        root_file.read(GateTree.HITS, ["eventID", "hitTimes"])


def test_hits_stored_under_another_name_are_found(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The GateToTree output calls its tree "tree", and still holds hits."""
    # ARRANGE
    # Nothing in the file is named "Hits".

    # ACT
    with RootFile(hits_variant_files["b1"]) as root_file:
        resolved = root_file.resolve_tree_name(GateTree.HITS)
        data = root_file.read(GateTree.HITS)

    # ASSERT
    assert resolved == "tree"
    assert len(data.branch_names) == 54


def test_a_named_tree_is_read_as_asked(hits_variant_files: Mapping[str, Path]) -> None:
    """Naming the tree is how one run or one detector is singled out."""
    # ARRANGE
    # The file holds one tree per run.

    # ACT
    with RootFile(hits_variant_files["multi-run"]) as root_file:
        data = root_file.read(GateTree.HITS, tree_name="Hits_run2")

    # ASSERT
    assert set(data["runID"].tolist()) == {2}


def test_naming_a_tree_that_is_absent_is_reported(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A name the file does not hold should say what the file does hold."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with RootFile(hits_variant_files["multi-run"]) as root_file:
        with pytest.raises(TreeNotFoundError, match="Hits_run7"):
            root_file.read(GateTree.HITS, tree_name="Hits_run7")


def test_naming_a_tree_that_holds_no_hits_is_reported(gate_hits_file: Path) -> None:
    """A named tree is still checked, so pointing at the wrong one is caught."""
    # ARRANGE
    # The file holds a "pet_data" tree of run metadata.

    # ACT / ASSERT
    with RootFile(gate_hits_file) as root_file:
        with pytest.raises(UnknownHitsVariantError, match="pet_data"):
            root_file.read(GateTree.HITS, tree_name="pet_data")


def test_other_trees_holding_hits_are_reported(
    hits_variant_files: Mapping[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reading one run of three without a word would hide two thirds of the data."""
    # ARRANGE
    # The file holds "Hits", "Hits_run1" and "Hits_run2".

    # ACT
    with caplog.at_level(logging.WARNING), RootFile(hits_variant_files["multi-run"]) as root_file:
        data = root_file.read(GateTree.HITS)

    # ASSERT
    assert data.entry_count == 500
    assert "Hits_run1" in caplog.text
    assert "Hits_run2" in caplog.text


def test_a_single_hits_tree_is_read_without_a_warning(
    hits_variant_files: Mapping[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The usual file holds one tree of hits and deserves no noise."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with caplog.at_level(logging.WARNING), RootFile(hits_variant_files["a1"]) as root_file:
        root_file.read(GateTree.HITS)

    # ASSERT
    assert caplog.text == ""


def test_several_unnamed_trees_of_hits_are_refused(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Choosing one detector over the other is not the package's call."""
    # ARRANGE
    # The file holds one tree per sensitive detector, neither named "Hits".

    # ACT
    with RootFile(hits_variant_files["multi-sd"]) as root_file:
        with pytest.raises(AmbiguousTreeError) as raised:
            root_file.read(GateTree.HITS)

    # ASSERT
    assert "Hits_DET_INNER" in str(raised.value)
    assert "Hits_DET_OUTER" in str(raised.value)


def test_trees_holding_hits_are_listed(hits_variant_files: Mapping[str, Path]) -> None:
    """The names are what the caller needs to pick one, or to read them all."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["multi-sd"]) as root_file:
        names = root_file.hits_tree_names()

    # ASSERT
    assert names == ("Hits_DET_INNER", "Hits_DET_OUTER")


def test_trees_that_hold_no_hits_are_left_out_of_the_listing(gate_hits_file: Path) -> None:
    """A GATE file holds trees of other things, and they are not hits."""
    # ARRANGE
    # The file holds "pet_data" and an empty "OpticalData" next to the hits.

    # ACT
    with RootFile(gate_hits_file) as root_file:
        names = root_file.hits_tree_names()

    # ASSERT
    assert names == ("Hits",)


def test_a_file_without_hits_reports_the_trees_it_holds(
    make_gate_root_file: Callable[..., Path],
) -> None:
    """Looking for hits that are not there must say what is there instead."""
    # ARRANGE
    path = make_gate_root_file({"pet_data": {"start_time_sec": np.zeros(1)}})

    # ACT
    with RootFile(path) as root_file:
        with pytest.raises(TreeNotFoundError, match="pet_data"):
            root_file.read(GateTree.HITS)


def test_other_trees_are_not_looked_for_by_their_structure(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """Only the structure of hits is described, so only hits can be found by it."""
    # ARRANGE
    path = hits_file(make_hits_root_file, tree_name="tree")

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(TreeNotFoundError):
        root_file.read(GateTree.SINGLES)


def test_branch_names_can_be_asked_for_by_tree_name(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Listing the branches of one run should not need the tree to be read."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["multi-sd"]) as root_file:
        names = root_file.branch_names(GateTree.HITS, "Hits_DET_OUTER")

    # ASSERT
    assert len(names) == 40
    assert "volumeID" in names


def test_the_structure_of_a_named_tree_is_recognised(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Recognition follows the same naming rules as reading does."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["multi-sd"]) as root_file:
        detection = root_file.detect_hits_tree("Hits_DET_INNER")

    # ASSERT
    assert detection.tree_name == "Hits_DET_INNER"
    assert detection.variant is HitsTreeVariant.NO_SYSTEM


def test_hits_split_per_run_are_read_as_one_dataset(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Three runs of a simulation are one measurement, not three."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["multi-run"]) as root_file:
        data = root_file.read_hits()

    # ASSERT
    assert data.entry_count == 1500
    assert sorted(set(data["runID"].tolist())) == [0, 1, 2]
    assert sorted(set(data[SOURCE_TREE_BRANCH])) == ["Hits", "Hits_run1", "Hits_run2"]


def test_hits_split_per_detector_keep_one_identity_per_event(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The same decay recorded in two detectors stays one event."""
    # ARRANGE
    # Both trees hold run 0, and their events overlap.

    # ACT
    with RootFile(hits_variant_files["multi-sd"]) as root_file:
        data = root_file.read_hits()

    # ASSERT
    key = np.stack([data["runID"], data["eventID"]], axis=1)
    assert data.entry_count == 1000
    assert len(np.unique(key, axis=0)) == len(np.unique(data["eventID"]))
    assert sorted(set(data[SOURCE_TREE_BRANCH])) == ["Hits_DET_INNER", "Hits_DET_OUTER"]


def test_reading_hits_can_be_limited_to_some_trees(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """Two runs out of three are a legitimate thing to ask for."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["multi-run"]) as root_file:
        data = root_file.read_hits(tree_names=["Hits", "Hits_run2"])

    # ASSERT
    assert data.entry_count == 1000
    assert sorted(set(data["runID"].tolist())) == [0, 2]


def test_reading_hits_of_a_file_holding_one_tree_works(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The usual file needs no special treatment to be read this way."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["a1"]) as root_file:
        data = root_file.read_hits()

    # ASSERT
    assert data.entry_count == 500
    assert set(data[SOURCE_TREE_BRANCH]) == {"Hits"}


def test_reading_hits_can_leave_out_the_source_column(
    hits_variant_files: Mapping[str, Path],
    hits_variant_layouts: Mapping[str, HitsVariantLayout],
) -> None:
    """A result meant to match the structure exactly carries no extra column."""
    # ARRANGE
    expected_branches_count = hits_variant_layouts["a1"].branch_count

    # ACT
    with RootFile(hits_variant_files["multi-run"]) as root_file:
        data = root_file.read_hits(add_source_branch=False)

    # ASSERT
    assert len(data.branch_names) == expected_branches_count
    assert SOURCE_TREE_BRANCH not in data.branch_names


def test_reading_hits_honours_a_branch_selection(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A selection applies to every tree that goes into the dataset."""
    # ARRANGE
    # No additional setup required.

    # ACT
    with RootFile(hits_variant_files["multi-run"]) as root_file:
        data = root_file.read_hits(["eventID", "edep"])

    # ASSERT
    assert data.branch_names == ("eventID", "edep", SOURCE_TREE_BRANCH)
    assert data.entry_count == 1500


def test_reading_hits_of_a_file_without_them_reports_the_trees_it_holds(
    make_gate_root_file: Callable[..., Path],
) -> None:
    """Asking for hits that are not there must say what is there instead."""
    # ARRANGE
    path = make_gate_root_file({"pet_data": {"start_time_sec": np.zeros(1)}})

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(TreeNotFoundError, match="pet_data"):
        root_file.read_hits()


def test_reading_hits_of_a_named_tree_that_is_absent_is_reported(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A tree named among the ones to read still has to exist."""
    # ARRANGE
    # No additional setup required.

    # ACT / ASSERT
    with RootFile(hits_variant_files["multi-run"]) as root_file:
        with pytest.raises(TreeNotFoundError, match="Hits_run7"):
            root_file.read_hits(tree_names=["Hits", "Hits_run7"])


def test_an_unrelated_tree_carrying_a_marker_branch_is_not_taken_for_hits(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """A file can hold other trees, and some of them name a branch like a marker."""
    # ARRANGE
    path = make_hits_root_file({"Hits": NO_SYSTEM_BRANCHES, "DoseByRegion": ["volumeID", "edep"]})

    # ACT
    with RootFile(path) as root_file:
        names = root_file.hits_tree_names()
        data = root_file.read_hits()

    # ASSERT
    assert names == ("Hits",)
    assert set(data[SOURCE_TREE_BRANCH]) == {"Hits"}


def test_reading_hits_of_an_unrecognised_tree_reports_the_structure(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """A file whose "Hits" tree is not a known structure says exactly that.

    The name resolution finds the tree, so the failure has to come from the
    structure rather than from there being nothing to merge.
    """
    # ARRANGE
    path = hits_file(make_hits_root_file, ["eventID", "edep"])

    # ACT / ASSERT
    with RootFile(path) as root_file, pytest.raises(UnknownHitsVariantError):
        root_file.read_hits()


def test_reading_hits_of_an_unrecognised_tree_is_possible_without_validation(
    make_hits_root_file: Callable[..., Path],
) -> None:
    """The escape hatch has to work while merging, as it does while reading."""
    # ARRANGE
    path = hits_file(make_hits_root_file, ["eventID", "edep"])

    # ACT
    with RootFile(path) as root_file:
        data = root_file.read_hits(validate=False)

    # ASSERT
    assert data.branch_names == ("eventID", "edep", SOURCE_TREE_BRANCH)
