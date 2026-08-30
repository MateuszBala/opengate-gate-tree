"""Smoke tests for package version exposure and consistency."""

import re
from pathlib import Path

from opengate_gate_tree import __version__


def _extract_with_pattern(content: str, pattern: str, source_name: str) -> str:
    """Extract a single version value from content using a regex pattern."""
    match = re.search(pattern, content, re.MULTILINE)
    if match is None:
        raise AssertionError(f"Could not find version in {source_name}.")
    return match.group(1)


def test_package_exposes_version() -> None:
    """The package should expose a non-empty string version."""
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_version_is_consistent_across_all_version_files() -> None:
    """All files that store package version should use the same value."""
    repository_root = Path(__file__).resolve().parents[2]

    pyproject_content = (repository_root / "pyproject.toml").read_text()
    init_content = (repository_root / "src" / "opengate_gate_tree" / "__init__.py").read_text()
    readme_content = (repository_root / "README.md").read_text()

    pyproject_version = _extract_with_pattern(
        pyproject_content,
        r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"',
        "pyproject.toml",
    )
    init_version = _extract_with_pattern(
        init_content,
        r'^__version__\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"',
        "src/opengate_gate_tree/__init__.py",
    )
    readme_badge_version = _extract_with_pattern(
        readme_content,
        r"badge/version-([0-9]+\.[0-9]+\.[0-9]+)-informational",
        "README.md badge",
    )
    readme_stage_version = _extract_with_pattern(
        readme_content,
        r"^Current development stage:\s*version-([0-9]+\.[0-9]+\.[0-9]+)$",
        "README.md stage",
    )

    assert pyproject_version == __version__
    assert init_version == __version__
    assert readme_badge_version == __version__
    assert readme_stage_version == __version__
