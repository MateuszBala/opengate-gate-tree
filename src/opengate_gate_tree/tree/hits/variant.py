"""Variants of the "Hits" tree and the naming of system identifier branches.

Which branches GATE writes into the "Hits" tree depends on the simulation: on
whether hits are attached to a system, on whether the Compton camera output is
enabled, on whether septal penetration is counted, and on which output module
wrote the file.

When a system is used, GATE adds one branch per level of the system hierarchy.
Their names depend on the type of the system, and two system types can share
one set of names, so what a set of names identifies is the naming scheme, not
the system itself. In the classic ROOT output GATE always writes six such
branches, whatever the depth of the system; the ``GateToTree`` output writes as
many as the system is deep, which makes its identifier block a prefix of the
scheme.

Public objects:

HitsTreeVariant
    Structure of a "Hits" tree.
GateSystemType
    Naming scheme of the system identifier branches.
SYSTEM_ID_BRANCHES
    Identifier branch names of every scheme, ordered from the top level down.
SYSTEM_ALIASES
    System types that share one naming scheme.
system_id_depth(system, branch_names) -> int
    Number of leading identifier branches of a scheme that are present.
find_system_type(branch_names) -> GateSystemType | None
    Naming scheme the branches follow, or ``None`` when it cannot be told.
"""

from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Final


class HitsTreeVariant(Enum):
    """Structure of a "Hits" tree."""

    NO_SYSTEM = "No system"
    SYSTEM = "System"
    SYSTEM_SEPTAL = "System with septal penetration"
    NO_SYSTEM_CC = "No system, Compton camera output"
    SYSTEM_CC = "System, Compton camera output"
    TREE_COMMON = "GateToTree common output"


class GateSystemType(Enum):
    """Naming scheme of the system identifier branches.

    A member stands for the names GATE gives the identifier branches, not for
    a single system type: the systems listed in :data:`SYSTEM_ALIASES` under
    one member cannot be told apart by branch names alone.
    """

    CYLINDRICAL_PET = "cylindricalPET"
    CPET = "CPET"
    SPECT_HEAD = "SPECThead"
    ECAT = "ecat"
    CT_SCANNER = "CTscanner"
    SCANNER = "scanner"


# Identifier branch names of every scheme, ordered from the top level down.
SYSTEM_ID_BRANCHES: Final[Mapping[GateSystemType, tuple[str, ...]]] = MappingProxyType(
    {
        GateSystemType.CYLINDRICAL_PET: (
            "gantryID",
            "rsectorID",
            "moduleID",
            "submoduleID",
            "crystalID",
            "layerID",
        ),
        GateSystemType.CPET: (
            "gantryID",
            "sectorID",
            "cassetteID",
            "moduleID",
            "crystalID",
            "layerID",
        ),
        GateSystemType.SPECT_HEAD: (
            "headID",
            "crystalID",
            "pixelID",
            "unused3ID",
            "unused4ID",
            "unused5ID",
        ),
        GateSystemType.ECAT: (
            "gantryID",
            "blockID",
            "crystalID",
            "unused3ID",
            "unused4ID",
            "unused5ID",
        ),
        GateSystemType.CT_SCANNER: (
            "gantryID",
            "moduleID",
            "clusterID",
            "pixelID",
            "unused4ID",
            "unused5ID",
        ),
        GateSystemType.SCANNER: (
            "baseID",
            "level1ID",
            "level2ID",
            "level3ID",
            "level4ID",
            "level5ID",
        ),
    }
)

# System types that write the same identifier branch names.
SYSTEM_ALIASES: Final[Mapping[GateSystemType, tuple[str, ...]]] = MappingProxyType(
    {
        GateSystemType.CYLINDRICAL_PET: ("cylindricalPET", "OPET"),
        GateSystemType.CPET: ("CPET",),
        GateSystemType.SPECT_HEAD: ("SPECThead", "OpticalSystem"),
        GateSystemType.ECAT: ("ecat", "ecatAccel"),
        GateSystemType.CT_SCANNER: ("CTscanner",),
        GateSystemType.SCANNER: ("scanner",),
    }
)


def system_id_depth(system: GateSystemType, branch_names: Sequence[str]) -> int:
    """Return how many leading identifier branches of a scheme are present.

    Counting stops at the first missing name, because GATE writes the levels
    of a system from the top down: a gap means the branches belong to another
    scheme rather than to a deeper level of this one.

    Parameters
    ----------
    system : GateSystemType
        Naming scheme to measure against.
    branch_names : Sequence[str]
        Branch names of the tree.

    Returns
    -------
    int
        Number of leading identifier branches present, from 0 to the depth of
        the scheme.
    """
    available = set(branch_names)
    depth = 0
    for name in SYSTEM_ID_BRANCHES[system]:
        if name not in available:
            break
        depth += 1
    return depth


def find_system_type(branch_names: Sequence[str]) -> GateSystemType | None:
    """Return the naming scheme the identifier branches follow.

    The scheme reaching furthest into its own names wins. Schemes overlap at
    their first level, so a tree carrying only a shared name such as
    ``gantryID`` matches several of them equally well; that is reported as
    "cannot be told" rather than resolved by picking one.

    Parameters
    ----------
    branch_names : Sequence[str]
        Branch names of the tree.

    Returns
    -------
    GateSystemType | None
        Matching scheme, or ``None`` when no identifier branch is present or
        several schemes match equally well.
    """
    depths = {system: system_id_depth(system, branch_names) for system in GateSystemType}
    deepest = max(depths.values())
    if deepest == 0:
        return None
    matching = [system for system, depth in depths.items() if depth == deepest]
    if len(matching) != 1:
        return None
    return matching[0]
