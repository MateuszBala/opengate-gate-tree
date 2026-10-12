"""Analysis consistency tests: the package against analyses made outside it.

One test per directory under ``cases/``; ``harness.py`` says what a case holds
and when two tables agree. A branch adding a feature an outside analysis
checks is not opened as a pull request until every case passes.
"""

from pathlib import Path

import pytest
from harness import Case, compare_tables, discover_cases

from opengate_gate_tree import GateTree, read_tree

CASES_DIR = Path(__file__).parent / "cases"

CASES = discover_cases(CASES_DIR)


@pytest.mark.consistency
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_the_package_agrees_with_the_reference(case: Case) -> None:
    """Same events, same columns, same values within the tolerance of the case."""
    # ARRANGE
    data = read_tree(case.input_path, GateTree.HITS, validate=False)

    # ACT
    table = case.analysis.extract(data, case.variables)

    # ASSERT
    differences = compare_tables(table, case.reference(), case.rtol)
    assert not differences, f"Case {case.name}:\n" + "\n".join(differences)
