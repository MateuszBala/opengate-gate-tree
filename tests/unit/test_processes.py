"""Unit tests for the names of the processes a "Hits" tree records."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from opengate_gate_tree.io.reader import read_tree
from opengate_gate_tree.tree.filters import is_process
from opengate_gate_tree.tree.gatetree import GateTree
from opengate_gate_tree.tree.hits.processes import Process


def test_every_process_the_reference_outputs_hold_is_named(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """The members are the names the files hold, and only those."""
    # ARRANGE
    written: set[str] = set()
    for key in ("a1", "a2", "a3", "a4", "a5", "b1"):
        frame = read_tree(hits_variant_files[key], GateTree.HITS, ["processName"]).to_dataframe()
        written |= set(frame["processName"])

    # ACT
    named = {str(member) for member in Process}

    # ASSERT
    assert named == written


@pytest.mark.parametrize(
    ("member", "name"),
    [
        (Process.COMPTON, "Compton"),
        (Process.PHOTOELECTRIC, "PhotoElectric"),
        (Process.RAYLEIGH, "RayleighScattering"),
    ],
)
def test_a_member_is_the_string_gate_writes(member: Process, name: str) -> None:
    """So it compares against a column of text with nothing in between."""
    # ARRANGE
    # No additional setup required.

    # ACT
    equal = member == name

    # ASSERT
    assert equal
    assert isinstance(member, str)


def test_the_filters_of_the_pandas_view_take_a_member(
    hits_variant_files: Mapping[str, Path],
) -> None:
    """A member and its name select the same rows."""
    # ARRANGE
    frame = read_tree(hits_variant_files["a1"], GateTree.HITS, ["processName"]).to_dataframe()

    # ACT
    by_member = is_process(frame["processName"], Process.COMPTON)
    by_name = is_process(frame["processName"], "Compton")

    # ASSERT
    assert by_member.equals(by_name)
    assert int(by_member.sum()) == 200
