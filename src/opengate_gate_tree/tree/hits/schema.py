"""Branch schemas of the supported "Hits" tree variants.

A schema is the list of branches a variant of the tree holds, in the order
GATE writes them, each with the type it is written as. Variants that use a
system carry a block of identifier branches whose names depend on the system
(:mod:`opengate_gate_tree.tree.hits.variant`), so the schema of such a variant
is completed with the naming scheme in use.

The branch lists were measured on simulation output, one file per variant, and
the tests compare them against those files. Two of them are written out here,
and the rest are stated as the differences the reference material describes:
the identifier block, the septal penetration counter, and the Compton camera
columns.

Types follow the name of the branch, which is the same rule GATE follows: a
branch called ``edep`` is single precision wherever it appears. In the
``GateToTree`` layout ``volumeID`` is split into ten scalar branches named
``volumeID[0]`` to ``volumeID[9]``, which are integers like any other branch;
only the classic ``volumeID`` is a fixed-width array.

Public objects:

BranchKind
    Kind of value a branch holds.
BranchSpec
    Name and type of a single branch.
branch_spec(name) -> BranchSpec
    Specification of a branch of the given name.
expected_branches(variant, system, system_depth) -> tuple[BranchSpec, ...]
    Branches a variant holds, in file order.
supported_variants() -> tuple[HitsTreeVariant, ...]
    Variants the package supports.
uses_system(variant) -> bool
    Whether a variant carries system identifier branches.
variant_reference(variant) -> str
    Label the reference material gives a variant.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

from opengate_gate_tree.tree.hits.variant import (
    SYSTEM_ID_BRANCHES,
    GateSystemType,
    HitsTreeVariant,
)


class BranchKind(Enum):
    """Kind of value a branch holds."""

    INTEGER = "integer"
    FLOAT = "float"
    TEXT = "text"
    INTEGER_ARRAY = "integer array"


@dataclass(frozen=True)
class BranchSpec:
    """Name and type of a single branch.

    Attributes
    ----------
    name : str
        Branch name, as GATE writes it.
    kind : BranchKind
        Kind of value the branch holds.
    dtype : str
        Type name, such as ``"int32"``, ``"float64"``, ``"text"`` or
        ``"int32[10]"`` for a fixed-width array branch.
    """

    name: str
    kind: BranchKind
    dtype: str


# Branches GATE writes as double precision.
FLOAT64_BRANCHES: Final[frozenset[str]] = frozenset({"trackLocalTime", "time"})

# Branches GATE writes as single precision.
FLOAT32_BRANCHES: Final[frozenset[str]] = frozenset(
    {
        "edep",
        "stepLength",
        "trackLength",
        "posX",
        "posY",
        "posZ",
        "localPosX",
        "localPosY",
        "localPosZ",
        "sourcePosX",
        "sourcePosY",
        "sourcePosZ",
        "momDirX",
        "momDirY",
        "momDirZ",
        "axialPos",
        "rotationAngle",
        "sourceEnergy",
        "energyFinal",
        "energyIniT",
    }
)

# Branches GATE writes as text.
TEXT_BRANCHES: Final[frozenset[str]] = frozenset(
    {"processName", "comptVolName", "RayleighVolName", "postStepProcess"}
)

# Branches GATE writes as a fixed-width array, mapped to their width.
ARRAY_BRANCH_WIDTHS: Final[Mapping[str, int]] = MappingProxyType({"volumeID": 10})

# Stands in a branch list for the block of system identifier branches.
SYSTEM_ID_PLACEHOLDER: Final[str] = "<system identifiers>"

# Branches of the classic ROOT output without a system, in file order.
_NO_SYSTEM_BRANCHES: Final[tuple[str, ...]] = (
    "PDGEncoding",
    "trackID",
    "parentID",
    "trackLocalTime",
    "time",
    "edep",
    "stepLength",
    "trackLength",
    "posX",
    "posY",
    "posZ",
    "localPosX",
    "localPosY",
    "localPosZ",
    "sourcePosX",
    "sourcePosY",
    "sourcePosZ",
    "sourceID",
    "eventID",
    "runID",
    "volumeID",
    "processName",
    "momDirX",
    "momDirY",
    "momDirZ",
    "photonID",
    "nPhantomCompton",
    "nCrystalCompton",
    "nPhantomRayleigh",
    "nCrystalRayleigh",
    "nInteractions",
    "primaryID",
    "axialPos",
    "rotationAngle",
    "comptVolName",
    "RayleighVolName",
    "sourceType",
    "decayType",
    "gammaType",
    "decayIndex",
)

# Branches of the classic ROOT output with the Compton camera output enabled
# and without a system, in file order. From "sourceEnergy" on, this layout
# replaces the second half of the branches above rather than extending it.
_NO_SYSTEM_CC_BRANCHES: Final[tuple[str, ...]] = (
    "PDGEncoding",
    "trackID",
    "parentID",
    "trackLocalTime",
    "time",
    "edep",
    "stepLength",
    "trackLength",
    "posX",
    "posY",
    "posZ",
    "localPosX",
    "localPosY",
    "localPosZ",
    "sourcePosX",
    "sourcePosY",
    "sourcePosZ",
    "sourceID",
    "eventID",
    "runID",
    "volumeID",
    "processName",
    "sourceEnergy",
    "sourcePDG",
    "nCrystalConv",
    "nCrystalCompt",
    "nCrystalRayl",
    "energyFinal",
    "energyIniT",
    "postStepProcess",
)

# Branches of the common GateToTree output, in file order. The order differs
# from the classic output, volumeID is split into ten scalar branches, and
# nInteractions is not written at all.
_TREE_COMMON_BRANCHES: Final[tuple[str, ...]] = (
    "PDGEncoding",
    "trackID",
    "parentID",
    "trackLocalTime",
    "time",
    "runID",
    "eventID",
    "sourceID",
    "primaryID",
    "posX",
    "posY",
    "posZ",
    "localPosX",
    "localPosY",
    "localPosZ",
    "momDirX",
    "momDirY",
    "momDirZ",
    "edep",
    "stepLength",
    "trackLength",
    "rotationAngle",
    "axialPos",
    "processName",
    "comptVolName",
    "RayleighVolName",
    "volumeID[0]",
    "volumeID[1]",
    "volumeID[2]",
    "volumeID[3]",
    "volumeID[4]",
    "volumeID[5]",
    "volumeID[6]",
    "volumeID[7]",
    "volumeID[8]",
    "volumeID[9]",
    "sourcePosX",
    "sourcePosY",
    "sourcePosZ",
    "nPhantomCompton",
    "nCrystalCompton",
    "nPhantomRayleigh",
    "nCrystalRayleigh",
    SYSTEM_ID_PLACEHOLDER,
    "photonID",
    "sourceType",
    "decayType",
    "gammaType",
    "decayIndex",
)


def _insert_after(names: Sequence[str], anchor: str, inserted: Sequence[str]) -> tuple[str, ...]:
    """Return the names with further ones placed right after an anchor."""
    position = names.index(anchor) + 1
    return (*names[:position], *inserted, *names[position:])


def _insert_before(names: Sequence[str], anchor: str, inserted: Sequence[str]) -> tuple[str, ...]:
    """Return the names with further ones placed right before an anchor."""
    position = names.index(anchor)
    return (*names[:position], *inserted, *names[position:])


# A system adds its identifier branches right after "processName".
_SYSTEM_BRANCHES: Final[tuple[str, ...]] = _insert_after(
    _NO_SYSTEM_BRANCHES, "processName", (SYSTEM_ID_PLACEHOLDER,)
)

# Septal penetration counting adds one branch before the decay information.
_SYSTEM_SEPTAL_BRANCHES: Final[tuple[str, ...]] = _insert_before(
    _SYSTEM_BRANCHES, "sourceType", ("septalNb",)
)

_SYSTEM_CC_BRANCHES: Final[tuple[str, ...]] = _insert_after(
    _NO_SYSTEM_CC_BRANCHES, "processName", (SYSTEM_ID_PLACEHOLDER,)
)

# Branch list of every supported variant, in file order.
VARIANT_BRANCHES: Final[Mapping[HitsTreeVariant, tuple[str, ...]]] = MappingProxyType(
    {
        HitsTreeVariant.NO_SYSTEM: _NO_SYSTEM_BRANCHES,
        HitsTreeVariant.SYSTEM: _SYSTEM_BRANCHES,
        HitsTreeVariant.SYSTEM_SEPTAL: _SYSTEM_SEPTAL_BRANCHES,
        HitsTreeVariant.NO_SYSTEM_CC: _NO_SYSTEM_CC_BRANCHES,
        HitsTreeVariant.SYSTEM_CC: _SYSTEM_CC_BRANCHES,
        HitsTreeVariant.TREE_COMMON: _TREE_COMMON_BRANCHES,
    }
)

# Label the reference material gives each variant.
VARIANT_REFERENCES: Final[Mapping[HitsTreeVariant, str]] = MappingProxyType(
    {
        HitsTreeVariant.NO_SYSTEM: "A1",
        HitsTreeVariant.SYSTEM: "A2",
        HitsTreeVariant.SYSTEM_SEPTAL: "A3",
        HitsTreeVariant.NO_SYSTEM_CC: "A4",
        HitsTreeVariant.SYSTEM_CC: "A5",
        HitsTreeVariant.TREE_COMMON: "B1",
    }
)


def branch_spec(name: str) -> BranchSpec:
    """Return the specification of a branch of the given name.

    Parameters
    ----------
    name : str
        Branch name.

    Returns
    -------
    BranchSpec
        Name and type of the branch.
    """
    width = ARRAY_BRANCH_WIDTHS.get(name)
    if width is not None:
        return BranchSpec(name, BranchKind.INTEGER_ARRAY, f"int32[{width}]")
    if name in TEXT_BRANCHES:
        return BranchSpec(name, BranchKind.TEXT, "text")
    if name in FLOAT64_BRANCHES:
        return BranchSpec(name, BranchKind.FLOAT, "float64")
    if name in FLOAT32_BRANCHES:
        return BranchSpec(name, BranchKind.FLOAT, "float32")
    return BranchSpec(name, BranchKind.INTEGER, "int32")


def supported_variants() -> tuple[HitsTreeVariant, ...]:
    """Return the variants the package supports, in reference order."""
    return tuple(VARIANT_BRANCHES)


def variant_reference(variant: HitsTreeVariant) -> str:
    """Return the label the reference material gives a variant.

    Parameters
    ----------
    variant : HitsTreeVariant
        Variant to label.

    Returns
    -------
    str
        Label such as ``"A1"`` or ``"B1"``.
    """
    return VARIANT_REFERENCES[variant]


def uses_system(variant: HitsTreeVariant) -> bool:
    """Return whether a variant carries system identifier branches.

    Parameters
    ----------
    variant : HitsTreeVariant
        Variant to check.

    Returns
    -------
    bool
        ``True`` when the schema of the variant needs a naming scheme to be
        completed.
    """
    return SYSTEM_ID_PLACEHOLDER in VARIANT_BRANCHES[variant]


def expected_branches(
    variant: HitsTreeVariant,
    system: GateSystemType | None = None,
    system_depth: int | None = None,
) -> tuple[BranchSpec, ...]:
    """Return the branches a variant holds, in file order.

    Parameters
    ----------
    variant : HitsTreeVariant
        Variant to describe.
    system : GateSystemType | None
        Naming scheme of the identifier branches. Required for a variant that
        uses a system, rejected for one that does not.
    system_depth : int | None
        Number of identifier branches, for the ``GateToTree`` output only,
        where the block is as deep as the system. Defaults to the full depth
        of the scheme. The classic ROOT output always writes six identifier
        branches, so passing a depth for it is rejected.

    Returns
    -------
    tuple[BranchSpec, ...]
        Branches of the variant, in the order GATE writes them.

    Raises
    ------
    ValueError
        If the naming scheme is missing or given where it does not belong, or
        if the depth is out of range or given for a variant of fixed depth.
    """
    names = VARIANT_BRANCHES[variant]
    if not uses_system(variant):
        if system is not None:
            raise ValueError(
                f"Variant '{variant.value}' does not use a system, so no naming scheme applies."
            )
        if system_depth is not None:
            raise ValueError(
                f"Variant '{variant.value}' does not use a system, so no depth applies."
            )
        return tuple(branch_spec(name) for name in names)

    if system is None:
        raise ValueError(
            f"Variant '{variant.value}' uses a system, so its identifier naming scheme is "
            f"required to complete the schema."
        )

    identifiers = SYSTEM_ID_BRANCHES[system]
    if variant is HitsTreeVariant.TREE_COMMON:
        depth = len(identifiers) if system_depth is None else system_depth
        if not 1 <= depth <= len(identifiers):
            raise ValueError(
                f"System depth must be between 1 and {len(identifiers)}, got {system_depth}."
            )
    else:
        if system_depth is not None:
            raise ValueError(
                f"Variant '{variant.value}' always holds {len(identifiers)} identifier branches, "
                f"so a depth cannot be chosen."
            )
        depth = len(identifiers)

    completed = _insert_before(names, SYSTEM_ID_PLACEHOLDER, identifiers[:depth])
    return tuple(branch_spec(name) for name in completed if name != SYSTEM_ID_PLACEHOLDER)
