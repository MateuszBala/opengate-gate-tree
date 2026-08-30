"""GateTree enum and parser.

This module defines the GateTree enum and provides a function to parse
strings into GateTree enum members.
"""

from enum import Enum


class GateTree(Enum):
    HITS = "Hits"
    SINGLES = "Singles"
    COINCIDENCES = "Coincidences"


def parse_gate_tree(name: str) -> GateTree:
    """Parse a string into a GateTree enum member.

    Parameters
    ----------
    name : str
        Name of the gate tree member.

    Returns
    -------
    GateTree
        Corresponding GateTree enum member.

    Raises
    ------
    ValueError
        If the name does not correspond to any GateTree member.
    """
    try:
        return GateTree[name.upper()]
    except KeyError as err:
        raise ValueError(f"Unknown GateTree member: {name}") from err
