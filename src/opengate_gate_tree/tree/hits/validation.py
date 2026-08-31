"""Checking a "Hits" tree against the structure it was recognised as.

Recognition names the structure from a few marker branches; validation asks
whether the tree really holds what that structure describes. The two are kept
apart so that a tree which almost matches gets a report of what is wrong with
it, instead of being turned away as unrecognised.

What counts as a failure is not symmetric:

- a branch the structure describes but the tree does not hold, or one whose
  type differs, is an error, because the data is not what it was taken for;
- a branch the tree holds beyond the structure is a warning. The reference
  files come from a GATE build carrying patches, and adding a branch is an
  ordinary thing for a GATE build to do. Refusing such a file would turn the
  package away from exactly the simulations it is written for.

Public functions:

validate_hits_tree(branch_names, dtypes, detection) -> HitsTreeDetection
    Check a tree against the structure it was recognised as.
"""

from collections.abc import Mapping, Sequence

from opengate_gate_tree.errors import HitsTreeValidationError
from opengate_gate_tree.logger import log
from opengate_gate_tree.tree.hits.detection import (
    HitsTreeDetection,
    detect_hits_variant,
    expected_branches_of,
)
from opengate_gate_tree.tree.hits.schema import variant_reference


def validate_hits_tree(
    branch_names: Sequence[str],
    dtypes: Mapping[str, str],
    detection: HitsTreeDetection | None = None,
) -> HitsTreeDetection:
    """Check a tree against the structure it was recognised as.

    Parameters
    ----------
    branch_names : Sequence[str]
        Branch names of the tree, in file order.
    dtypes : Mapping[str, str]
        Branch names mapped to the type each of them is stored with, in the
        vocabulary of :class:`~opengate_gate_tree.tree.hits.schema.BranchSpec`.
    detection : HitsTreeDetection | None
        Structure recognised in the tree. Recognised here when not given.

    Returns
    -------
    HitsTreeDetection
        Structure the tree was checked against.

    Raises
    ------
    UnknownHitsVariantError
        If the structure of the tree is not a supported one.
    HitsTreeValidationError
        If a branch the structure describes is missing or stored with another
        type.
    """
    if detection is None:
        detection = detect_hits_variant(branch_names)

    expected = {spec.name: spec.dtype for spec in expected_branches_of(detection)}

    present = set(branch_names)
    missing = [name for name in expected if name not in present]
    mismatched = {
        name: (expected[name], dtypes.get(name))
        for name in expected
        if name in present and dtypes.get(name) != expected[name]
    }

    if missing or mismatched:
        raise HitsTreeValidationError(_failure_message(detection, missing, mismatched))

    unexpected = [name for name in branch_names if name not in expected]
    if unexpected:
        log().warning(
            "%s holds %d branch(es) the '%s' structure does not describe: %s. "
            "They are read as they are.",
            _tree_label(detection),
            len(unexpected),
            detection.variant.value,
            unexpected,
        )

    return detection


def _failure_message(
    detection: HitsTreeDetection,
    missing: Sequence[str],
    mismatched: Mapping[str, tuple[str, str | None]],
) -> str:
    """Return the message describing how a tree departs from its structure."""
    label = variant_reference(detection.variant)
    parts = [
        f"{_tree_label(detection)} was recognised as the '{detection.variant.value}' structure "
        f"({label}), but does not match it."
    ]
    if missing:
        parts.append(f"Branches missing from the tree: {list(missing)}.")
    if mismatched:
        differences = ", ".join(
            f"'{name}' expected {expected}, stored as {actual}"
            for name, (expected, actual) in mismatched.items()
        )
        parts.append(f"Branches stored with another type: {differences}.")
    return " ".join(parts)


def _tree_label(detection: HitsTreeDetection) -> str:
    """Return how the tree is referred to in a message."""
    if detection.tree_name is None:
        return "The tree"
    return f"Tree '{detection.tree_name}'"
