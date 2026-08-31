# Version Upgrade Conventions

This document defines the mandatory procedure for publishing a new package version.

## Critical Rule

Version publication is allowed only:

- on branch `main`
- by a repository admin

Any release executed outside this rule is invalid and must be reverted.

## Prerequisites

Required tools and access:

- `git`
- `python3`
- `gh` (GitHub CLI) authenticated for the target repository
- admin permission on the repository

Verify access and repository state:

```bash
git status --porcelain
git branch --show-current
gh auth status
gh repo view --json viewerPermission -q '.viewerPermission'
```

Expected conditions before release:

- working tree is clean
- current branch is `main`
- `viewerPermission` is `ADMIN`

## Release Variables

Set release variables once at the beginning:

```bash
export VERSION="X.Y.Z"
export TAG="v${VERSION}"
```

Optional semantic version guard:

```bash
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "VERSION must match X.Y.Z" >&2
  exit 1
}
```

## Release Procedure

### 1. Synchronize local `main`

```bash
git checkout main
git pull --ff-only origin main
```

### 2. Update version in `pyproject.toml` and README badge

Run the dedicated version-update script:

```bash
bash scripts/update_version_in_files.sh "$VERSION"
```

Verify updated values:

```bash
rg -n '^version\s*=\s*"' pyproject.toml
rg -n 'badge/version-' README.md
git diff -- pyproject.toml README.md
```

### 3. Run validation (mandatory)

```bash
make check
```

### 4. Commit release metadata

Commit message format is fixed:

```bash
git add pyproject.toml README.md
git commit -m "chore: ${TAG}"
```

Example:

- `chore: v1.4.0`

### 5. Create release tag

```bash
git tag -a "$TAG" -m "Release ${TAG}"
```

### 6. Push `main` and tag

```bash
git push origin main
git push origin "$TAG"
```

### 7. Publish GitHub Release via `gh`

```bash
gh release create "$TAG" \
  --verify-tag \
  --target main \
  --title "$TAG" \
  --generate-notes
```

### 8. Synchronize `develop` with `main`

```bash
git checkout develop
git pull --ff-only origin develop
git merge --no-ff main -m "chore: sync develop with main after ${TAG}"
git push origin develop
```

## Post-Release Verification

```bash
gh release view "$TAG"
git checkout main
git pull --ff-only origin main
git tag --list "$TAG"
```

## Recovery Notes

If a tag was created by mistake and release was not finalized yet:

```bash
git tag -d "$TAG"
git push --delete origin "$TAG"
```

If the release commit was pushed and must be reverted, use a regular revert commit (never force-push `main`):

```bash
git revert HEAD
git push origin main
```
