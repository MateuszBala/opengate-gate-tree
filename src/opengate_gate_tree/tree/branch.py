"""Module for branch validation in the gate tree."""

from opengate_gate_tree.tree.gatetree import GateTree

GATE_HITS_BRANCHES: list[str] = []
GATE_SINGLES_BRANCHES: list[str] = []
GATE_COINCIDENCES_BRANCHES: list[str] = []


def validate_branch_name(branches: list[str], gate_tree: GateTree) -> tuple[bool, list[str]]:
    """
    Validate branch names against the gate tree.

    Parameters
    ----------
    branches : list[str]
        List of branch names to validate.
    gate_tree : GateTree
        The gate tree structure to validate against.
    Returns
    -------
    tuple[bool, list[str]]
        A tuple containing a boolean indicating if all branches are valid,
        and a list of invalid branches.
    """
    invalid_branches: list[str] = []
    if gate_tree == GateTree.HITS:
        invalid_branches = [branch for branch in branches if branch not in GATE_HITS_BRANCHES]
    elif gate_tree == GateTree.SINGLES:
        invalid_branches = [branch for branch in branches if branch not in GATE_SINGLES_BRANCHES]
    elif gate_tree == GateTree.COINCIDENCES:
        invalid_branches = [
            branch for branch in branches if branch not in GATE_COINCIDENCES_BRANCHES
        ]
    return len(invalid_branches) == 0, invalid_branches
