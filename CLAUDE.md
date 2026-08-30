# CLAUDE.md

This file provides project-level instructions for Claude Code.

## Scope and priorities

- Apply these rules to all work in this repository.
- Keep instructions specific, verifiable, and concise.
- Prefer project conventions over generic defaults.

## Project snapshot

- Package: opengate-gate-tree
- Source root: src/opengate_gate_tree
- Tests root: tests/unit
- Python support: 3.11-3.14
- Tooling: uv, ruff, mypy, pytest, pre-commit

## Required workflow

Use repository Make targets when available:

```bash
make init
make lint
make format
make typecheck
make test
make check
```

If CLI tools are needed, run them via uv in this repository:

```bash
uv run <tool>
```

## Non-negotiable checks before finishing changes

Run:

```bash
make check
```

If pre-commit is enabled, also run:

```bash
uv run pre-commit run --all-files
```

Do not consider work complete while these checks fail.

## Coding rules

- Keep code, comments, docs, and user-facing messages in English.
- Keep edits minimal and focused; avoid unrelated refactors.
- Follow strict type hints; this project uses strict mypy settings.
- Keep line length within Ruff limits (100).
- Preserve current package/module naming and folder layout.

Authoritative coding standard:
- docs/CODING_CONVENTIONS.md

## Testing rules

- Add or update tests when behavior changes.
- Keep unit tests under tests/unit.
- Target current modules from opengate_gate_macro_fold.
- Do not reintroduce legacy imports or names from other projects.

Authoritative testing standard:
- docs/TESTING_CONVENTIONS.md

## Versioning and release

- Version releases are allowed only on main and only by admins.
- For version updates in files, use:

```bash
bash script/update_version_in_files.sh "$VERSION"
```

Authoritative release process:
- docs/VERSION_UPGRADE_CONVENTIONS.md

## Commit and PR hygiene

- Follow docs/COMMIT_CONVENTIONS.md for commit message format.
- Use .github/PULL_REQUEST_TEMPLATE.md for PR descriptions.

## Repository-specific pitfalls to avoid

- Do not add obsolete console script aliases from older projects.
- Ensure tests and docs match the current package name and module paths.

## Agent configuration

Additional generalised settings for agents are in a directory `.agents`