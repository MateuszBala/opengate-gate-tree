#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash script/update_version_in_files.sh X.Y.Z

Examples:
  bash script/update_version_in_files.sh 1.2.3
  bash script/update_version_in_files.sh v1.2.3
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

VERSION="$1"
if [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  VERSION="${VERSION#v}"
fi

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: VERSION must match X.Y.Z (or vX.Y.Z)." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYPROJECT_FILE="${ROOT_DIR}/pyproject.toml"
README_FILE="${ROOT_DIR}/README.md"
INIT_FILE="${ROOT_DIR}/src/opengate_gate_tree/__init__.py"

if [[ ! -f "$PYPROJECT_FILE" ]]; then
  echo "Error: file not found: $PYPROJECT_FILE" >&2
  exit 1
fi

if [[ ! -f "$README_FILE" ]]; then
  echo "Error: file not found: $README_FILE" >&2
  exit 1
fi

if [[ ! -f "$INIT_FILE" ]]; then
  echo "Error: file not found: $INIT_FILE" >&2
  exit 1
fi

if ! grep -qE '^version[[:space:]]*=[[:space:]]*"[^"]+"' "$PYPROJECT_FILE"; then
  echo "Error: could not locate the version field in pyproject.toml" >&2
  exit 1
fi

if ! grep -qE '^__version__[[:space:]]*=[[:space:]]*"[^"]+"' "$INIT_FILE"; then
  echo "Error: could not locate the __version__ field in src/opengate_gate_tree/__init__.py" >&2
  exit 1
fi

if ! grep -qE 'badge/version-[0-9]+\.[0-9]+\.[0-9]+-informational' "$README_FILE"; then
  echo "Error: could not locate the version badge in README.md" >&2
  exit 1
fi

if ! grep -qE '^Current development stage:[[:space:]]*version-[0-9]+\.[0-9]+\.[0-9]+$' "$README_FILE"; then
  echo "Error: could not locate the current development stage in README.md" >&2
  exit 1
fi

sed -E -i "0,/^version[[:space:]]*=[[:space:]]*\"[^\"]+\"/s//version = \"${VERSION}\"/" "$PYPROJECT_FILE"
sed -E -i "0,/^__version__[[:space:]]*=[[:space:]]*\"[^\"]+\"/s//__version__ = \"${VERSION}\"/" "$INIT_FILE"
sed -E -i "0,/badge\/version-[0-9]+\.[0-9]+\.[0-9]+-informational/s//badge\/version-${VERSION}-informational/" "$README_FILE"
sed -E -i "0,/^Current development stage:[[:space:]]*version-[0-9]+\.[0-9]+\.[0-9]+$/s//Current development stage: version-${VERSION}/" "$README_FILE"

UPDATED_VERSION="$(sed -nE 's/^version[[:space:]]*=[[:space:]]*"([^"]+)"/\1/p' "$PYPROJECT_FILE" | head -n1)"
if [[ "$UPDATED_VERSION" != "$VERSION" ]]; then
  echo "Error: pyproject.toml version verification failed." >&2
  exit 1
fi

INIT_VERSION="$(sed -nE 's/^__version__[[:space:]]*=[[:space:]]*"([^"]+)"/\1/p' "$INIT_FILE" | head -n1)"
if [[ "$INIT_VERSION" != "$VERSION" ]]; then
  echo "Error: src/opengate_gate_tree/__init__.py version verification failed." >&2
  exit 1
fi

if ! grep -qE "badge/version-${VERSION//./\.}-informational" "$README_FILE"; then
  echo "Error: README.md badge verification failed." >&2
  exit 1
fi

if ! grep -qE "^Current development stage:[[:space:]]*version-${VERSION//./\.}$" "$README_FILE"; then
  echo "Error: README.md current development stage verification failed." >&2
  exit 1
fi

echo "Updated package version to ${VERSION}:"
echo "- ${PYPROJECT_FILE}"
echo "- ${INIT_FILE}"
echo "- ${README_FILE}"
