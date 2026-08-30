# Commit Conventions

This document is a portable template for commit-message rules. It is based on Conventional Commits, but you can adapt the format if your team uses a different convention.

## Purpose

A commit history should help readers understand what changed, why it changed, and how the change affects the repository.

## Language

Use a single language consistently across the repository. If the team prefers English, keep commit messages in English.

## Basic Format

```text
<type>: <subject>
```

## With Scope

```text
<type>(<scope>): <subject>
```

## Common Types

| Type | Purpose |
|------|---------|
| `feat` | New functionality |
| `fix` | Bug fix |
| `docs` | Documentation-only change |
| `test` | Test additions or updates |
| `refactor` | Code restructuring without behavior change |
| `style` | Formatting, linting, or naming cleanup |
| `perf` | Performance improvement |
| `ci` | Continuous integration change |
| `config` | Configuration update |
| `chore` | Maintenance task |
| `build` | Build system or dependency change |
| `revert` | Revert a previous commit |

## Subject Line Rules

- Use imperative mood.
- Keep the subject concise and specific.
- Avoid a trailing period.
- Prefer lowercase unless the repository convention says otherwise.
- Make the subject understandable without reading the diff.

## Scope Rules

- Use a scope when the change clearly belongs to one area.
- Skip the scope when the change is broad or the component is obvious.
- Keep scopes short and stable.

## Commit Body

Use the body when the change needs extra context:

- explain the reason for the change
- note any trade-offs or follow-up work
- mention related issues or pull requests
- describe behavior changes that are not obvious from the subject line

## Examples

```text
feat: add input validation
fix(api): handle missing response fields
docs: update setup instructions
test(auth): cover token refresh edge case
refactor: simplify error handling
```

## Team Notes

- If a repository already has a commit standard, document it here.
- If automation depends on the format, keep the rules strict.
- If the team uses a different style, replace this file with the local convention.
