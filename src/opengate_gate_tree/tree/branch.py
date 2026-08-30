"""Branch selection for GATE trees.

The module turns a requested list of branch names into the list that is
actually read. Which branches exist is decided by the file being read, not by
a fixed list built into the package, so the checks here cover the shape of the
request only. Whether a branch exists is checked against the opened file by
:mod:`opengate_gate_tree.io.validation`.

Public functions
----------------
validate_branch_selection(requested: Sequence[str]) -> None
    Check that a branch selection is well formed.
normalize_branch_selection(requested: Sequence[str], available: Sequence[str]) -> list[str]
    Turn a requested selection into the list of branches to read.
"""

from collections.abc import Sequence


def validate_branch_selection(requested: Sequence[str]) -> None:
    """Check that a branch selection is well formed.

    The check does not need the file, so it can run before one is opened.
    Repeated names are allowed; they are collapsed by
    :func:`normalize_branch_selection`.

    Parameters
    ----------
    requested : Sequence[str]
        Requested branch names.

    Raises
    ------
    ValueError
        If any requested name is empty or contains only whitespace.
    """
    if any(not name.strip() for name in requested):
        raise ValueError("Requested branch names must not be empty.")


def normalize_branch_selection(
    requested: Sequence[str],
    available: Sequence[str],
) -> list[str]:
    """Turn a requested selection into the list of branches to read.

    An empty selection means every branch of the tree. Repeated names are kept
    once, at the position of their first occurrence, because asking for the
    same branch twice is not a mistake worth refusing.

    Parameters
    ----------
    requested : Sequence[str]
        Requested branch names. Empty means every branch.
    available : Sequence[str]
        Branch names present in the tree, in file order.

    Returns
    -------
    list[str]
        Branch names to read, in the order they should appear.

    Raises
    ------
    ValueError
        If any requested name is empty or contains only whitespace.
    """
    validate_branch_selection(requested)
    if not requested:
        return list(available)
    return list(dict.fromkeys(requested))
