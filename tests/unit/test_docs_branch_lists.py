"""Unit tests keeping the branch lists in the guide equal to the schemas.

The guide lists 253 branches across six structures. Such a listing drifts away
from the code at the first change of a schema, and the drift is invisible to a
reader of either side, so the two are compared here.

Format the guide has to keep: one section per structure, headed
``### <label> — <name> (<count> branches)``, followed by a fenced block holding
one branch per line as ``<name> / <type>``.
"""

import re
from pathlib import Path

import pytest

from opengate_gate_tree.tree.hits.schema import (
    expected_branches,
    supported_variants,
    uses_system,
    variant_reference,
)
from opengate_gate_tree.tree.hits.variant import GateSystemType, HitsTreeVariant

# Guide page holding the branch lists.
GUIDE_PAGE = Path(__file__).resolve().parents[2] / "docs" / "guide" / "hits.md"

# Section heading introducing the branches of one structure.
SECTION = re.compile(
    r"^### (?P<label>[AB]\d) — (?P<name>[^(]+) \((?P<count>\d+) branches\)$\n\n"
    r"```text\n(?P<branches>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)

# Naming scheme the guide shows the identifier branches with.
DOCUMENTED_SYSTEM = GateSystemType.CYLINDRICAL_PET


def documented_sections() -> dict[str, tuple[str, int, list[str]]]:
    """Return the branch lists of the guide, keyed by structure label."""
    page = GUIDE_PAGE.read_text(encoding="utf-8")
    return {
        match["label"]: (
            match["name"].strip(),
            int(match["count"]),
            match["branches"].splitlines(),
        )
        for match in SECTION.finditer(page)
    }


@pytest.mark.parametrize("variant", supported_variants(), ids=lambda variant: variant.name)
def test_the_guide_lists_the_branches_of_every_structure(variant: HitsTreeVariant) -> None:
    """A structure the package supports has to be documented branch by branch."""
    # ARRANGE
    sections = documented_sections()
    system = DOCUMENTED_SYSTEM if uses_system(variant) else None
    expected = [f"{spec.name} / {spec.dtype}" for spec in expected_branches(variant, system)]

    # ACT
    label = variant_reference(variant)
    documented = sections.get(label)

    # ASSERT
    assert documented is not None, f"the guide holds no section for {label}"
    name, count, branches = documented
    assert name == variant.value
    assert count == len(expected)
    assert branches == expected


def test_the_guide_documents_no_structure_the_package_does_not_support() -> None:
    """A section for a structure that cannot be read would be a promise."""
    # ARRANGE
    supported = {variant_reference(variant) for variant in supported_variants()}

    # ACT
    documented = set(documented_sections())

    # ASSERT
    assert documented == supported
