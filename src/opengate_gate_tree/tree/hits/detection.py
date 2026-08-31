"""Recognition of the structure a "Hits" tree has.

Which structure a tree has is decided by a handful of marker branches rather
than by comparing the whole branch list against a schema. Marker branches say
what the simulation did: whether hits were attached to a system, whether the
Compton camera output was enabled, whether septal penetration was counted, and
which output module wrote the file.

Splitting recognition from validation is deliberate. A file from a GATE build
that adds or drops a branch still recognises as the structure it is, so
:mod:`opengate_gate_tree.tree.hits.validation` can report what is missing from
it, instead of the package answering "unknown structure" to everything that is
not an exact match.

Two structures are recognised and refused: the per-collection ``GateToTree``
output with the Compton camera columns, and the output of the Compton camera
actor. Neither has a reference file that holds any data, so their schemas
cannot be confirmed; naming them in the error beats leaving the user with
"not recognised".

Public objects:

HitsTreeDetection
    Structure recognised in a tree.
detect_hits_variant(branch_names, tree_name) -> HitsTreeDetection
    Recognise the structure of a tree, or refuse it.
find_hits_variant(branch_names) -> HitsTreeDetection | None
    Recognise the structure of a tree, answering ``None`` when it is not one
    of the supported ones.
summarise_hits_tree(detection) -> str
    One line stating what a tree was recognised as.
describe_hits_tree(detection, dtypes, entry_count) -> str
    Human readable description of a recognised tree, branch by branch.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from opengate_gate_tree.errors import UnknownHitsVariantError
from opengate_gate_tree.tree.hits.schema import (
    expected_branches,
    supported_variants,
    takes_system_depth,
    uses_system,
    variant_reference,
)
from opengate_gate_tree.tree.hits.variant import (
    SYSTEM_ALIASES,
    GateSystemType,
    HitsTreeVariant,
    find_system_type,
    system_id_depth,
)

# Branch holding the detector hierarchy in the classic ROOT output.
VOLUME_ID_BRANCH: Final[str] = "volumeID"

# First of the ten branches the GateToTree output splits volumeID into.
SPLIT_VOLUME_ID_BRANCH: Final[str] = "volumeID[0]"

# Branch written only when hits carry the name of the layer they hit, which
# the Compton camera output does and no other structure does.
LAYER_NAME_BRANCH: Final[str] = "layerName"

# Branch written only when septal penetration is counted.
SEPTAL_BRANCH: Final[str] = "septalNb"

# Branches the Compton camera output of the classic writer adds.
COMPTON_CAMERA_BRANCHES: Final[frozenset[str]] = frozenset({"sourceEnergy", "postStepProcess"})

# Number of branch names an error message lists before summarising the rest.
MAX_REPORTED_BRANCHES: Final[int] = 12


@dataclass(frozen=True)
class HitsTreeDetection:
    """Structure recognised in a "Hits" tree.

    Attributes
    ----------
    variant : HitsTreeVariant
        Structure the tree has.
    system : GateSystemType | None
        Naming scheme of the system identifier branches, or ``None`` when the
        structure carries none.
    system_depth : int | None
        Number of identifier branches present, or ``None`` when the structure
        carries none.
    tree_name : str | None
        Name the tree is stored under, when it is known.
    branch_count : int
        Number of branches the tree holds.
    """

    variant: HitsTreeVariant
    system: GateSystemType | None
    system_depth: int | None
    tree_name: str | None
    branch_count: int


def detect_hits_variant(
    branch_names: Sequence[str],
    tree_name: str | None = None,
) -> HitsTreeDetection:
    """Recognise the structure of a "Hits" tree.

    Parameters
    ----------
    branch_names : Sequence[str]
        Branch names of the tree.
    tree_name : str | None
        Name the tree is stored under, reported back and named in errors.

    Returns
    -------
    HitsTreeDetection
        Structure of the tree.

    Raises
    ------
    UnknownHitsVariantError
        If the branches match no supported structure, or match one the
        package does not support.
    """
    available = set(branch_names)
    system = find_system_type(branch_names)
    depth = system_id_depth(system, branch_names) if system is not None else None
    # A tree carrying identifier branches came from a simulation using a
    # system, even when the names present do not settle which scheme they
    # belong to. Reading it as the system-less structure would answer a
    # question that stayed open.
    has_identifiers = any(
        system_id_depth(candidate, branch_names) > 0 for candidate in GateSystemType
    )

    if LAYER_NAME_BRANCH in available:
        raise UnknownHitsVariantError(
            _unsupported_message(
                "the Compton camera hits layout (GateComptonCameraActor or GateCCHitTree)",
                tree_name,
            )
        )

    if SPLIT_VOLUME_ID_BRANCH in available:
        if COMPTON_CAMERA_BRANCHES <= available:
            raise UnknownHitsVariantError(
                _unsupported_message(
                    "the per-collection GateToTree output with the Compton camera columns",
                    tree_name,
                )
            )
        return _detected(
            HitsTreeVariant.TREE_COMMON, system, depth, tree_name, branch_names, has_identifiers
        )

    if VOLUME_ID_BRANCH in available:
        if COMPTON_CAMERA_BRANCHES <= available:
            variant = (
                HitsTreeVariant.SYSTEM_CC if system is not None else HitsTreeVariant.NO_SYSTEM_CC
            )
        elif SEPTAL_BRANCH in available:
            variant = HitsTreeVariant.SYSTEM_SEPTAL
        elif system is not None:
            variant = HitsTreeVariant.SYSTEM
        else:
            variant = HitsTreeVariant.NO_SYSTEM
        return _detected(variant, system, depth, tree_name, branch_names, has_identifiers)

    raise UnknownHitsVariantError(
        f"{_tree_label(tree_name)} does not hold a branch that identifies a known Hits structure: "
        f"neither '{VOLUME_ID_BRANCH}' nor '{SPLIT_VOLUME_ID_BRANCH}' is present. "
        f"{_branch_summary(branch_names)}"
    )


def find_hits_variant(branch_names: Sequence[str]) -> HitsTreeDetection | None:
    """Recognise the structure of a tree without refusing unknown ones.

    Used where a tree is a candidate rather than a request, such as when the
    tree holding the hits is looked for by its structure.

    Parameters
    ----------
    branch_names : Sequence[str]
        Branch names of the tree.

    Returns
    -------
    HitsTreeDetection | None
        Structure of the tree, or ``None`` when it is not a supported one.
    """
    try:
        return detect_hits_variant(branch_names)
    except UnknownHitsVariantError:
        return None


def summarise_hits_tree(detection: HitsTreeDetection) -> str:
    """Return one line stating what a tree was recognised as.

    Used where the structure is worth reporting but the branches are not, such
    as the log of a run.

    Parameters
    ----------
    detection : HitsTreeDetection
        Structure recognised in the tree.

    Returns
    -------
    str
        Single line naming the variant, the tree and the identifier scheme.
    """
    parts = [
        f"Hits tree variant: {detection.variant.value} ({variant_reference(detection.variant)})"
    ]
    if detection.tree_name is not None:
        parts.append(f"stored as '{detection.tree_name}'")
    if detection.system is not None:
        parts.append(f"system {' / '.join(SYSTEM_ALIASES[detection.system])}")
    parts.append(f"{detection.branch_count} branches")
    return ", ".join(parts)


def describe_hits_tree(
    detection: HitsTreeDetection,
    dtypes: Mapping[str, str] | None = None,
    entry_count: int | None = None,
) -> str:
    """Describe a recognised tree in a form meant to be read.

    Parameters
    ----------
    detection : HitsTreeDetection
        Structure recognised in the tree.
    dtypes : Mapping[str, str] | None
        Branch names mapped to the type each of them was read with, in file
        order. When omitted, the branches and types of the schema are
        described instead, which needs no file to be open.
    entry_count : int | None
        Number of entries, when it is known.

    Returns
    -------
    str
        Description spanning several lines.
    """
    variant = detection.variant
    lines = [f"Detected Hits tree variant: {variant.value} ({variant_reference(variant)})"]
    if detection.tree_name is not None:
        lines.append(f"Tree name in the file: {detection.tree_name}")
    if detection.system is not None:
        systems = " / ".join(SYSTEM_ALIASES[detection.system])
        lines.append(f"System identifier scheme: {systems}")
    if entry_count is not None:
        lines.append(f"Entries: {entry_count}")

    branches = _described_branches(detection, dtypes)
    lines.append(f"Branches ({len(branches)}):")
    lines.extend(f"  - {name} / {dtype}" for name, dtype in branches)
    return "\n".join(lines)


def _detected(
    variant: HitsTreeVariant,
    system: GateSystemType | None,
    depth: int | None,
    tree_name: str | None,
    branch_names: Sequence[str],
    has_identifiers: bool,
) -> HitsTreeDetection:
    """Build the result, refusing a structure whose system stayed unknown."""
    if system is None and has_identifiers:
        raise UnknownHitsVariantError(
            f"{_tree_label(tree_name)} carries system identifier branches, but their naming "
            f"scheme could not be told apart: several schemes name their first levels alike. "
            f"{_branch_summary(branch_names)}"
        )
    if system is None and uses_system(variant):
        raise UnknownHitsVariantError(
            f"{_tree_label(tree_name)} looks like the '{variant.value}' structure, but it carries "
            f"no system identifier branches. {_branch_summary(branch_names)}"
        )
    return HitsTreeDetection(
        variant=variant,
        system=system if uses_system(variant) else None,
        system_depth=depth if uses_system(variant) else None,
        tree_name=tree_name,
        branch_count=len(branch_names),
    )


def _described_branches(
    detection: HitsTreeDetection,
    dtypes: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    """Return the branches to describe, with the type of each of them."""
    if dtypes is not None:
        return list(dtypes.items())
    depth = detection.system_depth if takes_system_depth(detection.variant) else None
    specs = expected_branches(detection.variant, detection.system, depth)
    return [(spec.name, spec.dtype) for spec in specs]


def _unsupported_message(layout: str, tree_name: str | None) -> str:
    """Return the message refusing a structure that is known but unsupported."""
    supported = ", ".join(
        f"{variant_reference(variant)} ({variant.value})" for variant in supported_variants()
    )
    return (
        f"{_tree_label(tree_name)} matches {layout}, which this version does not support. "
        f"Supported structures: {supported}."
    )


def _tree_label(tree_name: str | None) -> str:
    """Return how the tree is referred to in a message."""
    return "The tree" if tree_name is None else f"Tree '{tree_name}'"


def _branch_summary(branch_names: Sequence[str]) -> str:
    """Return a short listing of the branches, for use in a message."""
    if not branch_names:
        return "The tree holds no branches."
    listed = list(branch_names[:MAX_REPORTED_BRANCHES])
    remaining = len(branch_names) - len(listed)
    suffix = f" and {remaining} more" if remaining else ""
    return f"Branches present: {listed}{suffix}."
