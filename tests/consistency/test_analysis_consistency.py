"""Analysis consistency tests: the package against analyses made outside it.

One test per directory under ``cases/``; ``harness.py`` says what a case holds
and when two tables agree. A branch adding a feature an outside analysis
checks is not opened as a pull request until every case passes.
"""

from pathlib import Path

import pytest
from harness import case_directories, compare_tables, load_case

from opengate_gate_tree import GateTree, read_tree

CASES_DIR = Path(__file__).parent / "cases"

CASES = case_directories(CASES_DIR)


@pytest.mark.consistency
@pytest.mark.parametrize("directory", CASES, ids=[directory.name for directory in CASES])
def test_the_package_agrees_with_the_reference(directory: Path) -> None:
    """Same events, same columns, same values within the tolerance of the case."""
    # ARRANGE
    case = load_case(directory)
    data = read_tree(case.input_path, GateTree.HITS, validate=False)

    # ACT
    table = case.analysis.extract(data, case.variables)

    # ASSERT
    differences = compare_tables(table, case.reference(), case.rtol, case.atol)
    assert not differences, f"Case {case.name}:\n" + "\n".join(differences)
