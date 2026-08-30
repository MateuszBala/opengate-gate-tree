# Copilot Instructions

These instructions apply to all changes in this repository.

## Project Context

- Package name: `opengate-gate-tree`
- Source root: `src/opengate_gate_tree`
- Supported Python versions: 3.12-3.14
- Environment and dependency manager: `uv`

## Development Workflow

- Prefer Make targets for common tasks:
  - `make init`
  - `make lint`
  - `make format`
  - `make typecheck`
  - `make test`
  - `make check`
- Keep changes focused and minimal.
- Do not introduce unrelated refactors in the same change.

## Repository Standards

- Follow coding conventions from [../docs/CODING_CONVENTIONS.md](../docs/CODING_CONVENTIONS.md).
- Follow testing conventions from [../docs/TESTING_CONVENTIONS.md](../docs/TESTING_CONVENTIONS.md).

## Code Quality Requirements

- Keep code and user-facing messages in English.
- Follow type hints consistently; this repository uses strict mypy settings.
- Keep line length within Ruff limits (100 characters).
- Preserve existing module structure and naming conventions.
- Treat [../docs/CODING_CONVENTIONS.md](../docs/CODING_CONVENTIONS.md) as authoritative for style decisions.

## Testing Requirements

- Add or update tests when behavior changes.
- Place unit tests under `tests/unit`.
- Ensure tests target current modules from `opengate_gate_tree` (not legacy project names).
- Avoid adding test dependencies that are not declared in `pyproject.toml`.
- Treat [../docs/TESTING_CONVENTIONS.md](../docs/TESTING_CONVENTIONS.md) as authoritative for test structure.

## Validation Before Finishing

Run full validation before finalizing code changes:

```bash
make check
```

If pre-commit is enabled, ensure hooks pass:

```bash
uv run pre-commit run --all-files
```

## Versioning and Release Rules

- For version bumps, use:

```bash
bash script/update_version_in_files.sh "$VERSION"
```

- Follow the release procedure in `docs/VERSION_UPGRADE_CONVENTIONS.md`.
- Release publication is allowed only on `main` and only by repository admins.

## Commit and PR Hygiene

- Follow commit format rules from `docs/COMMIT_CONVENTIONS.md`.
- Keep commit messages explicit and scoped.
- Ensure PR descriptions use `.github/PULL_REQUEST_TEMPLATE.md`.

## Agent configuration

Additional generalised settings for agents are in a directory `.agents`