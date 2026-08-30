# Contributing Guide

This document is a reusable contribution template. Copy it into a new repository and replace the placeholders with the project's real setup, branching model, and review process.

## What To Adapt

- Repository name and links
- Supported languages and tooling
- Branching rules and release flow
- Testing and documentation requirements
- Contact points for maintainers or reviewers

## Code Of Conduct

State the project's code-of-conduct expectations here or link to the canonical policy if the repository uses one.

## Getting Started

Before you begin, document the basics a contributor needs:

- required software and versions
- how to clone the repository
- how to install dependencies
- how to run the project locally
- how to run the tests

## Development Setup

Keep the setup steps short and verifiable.

1. Clone the repository.
2. Install the required tools.
3. Create a branch for your change.
4. Run the project or tests once to confirm the environment works.
5. Make the smallest useful change.

Replace the commands below with the ones used by the repository:

```bash
git clone <repository-url>
cd <repository-directory>
<install-dependencies>
<run-tests>
```

## Branching Strategy

Document the branch names the team expects.

- `main` or `master` for production-ready code, if applicable
- `develop` or another integration branch, if applicable
- short-lived feature branches for work in progress

If the project does not use branch rules, say so explicitly.

## Making Changes

- Keep each change focused on one concern.
- Add or update tests when behavior changes.
- Update docs when user-facing behavior changes.
- Rebase or merge from the integration branch as needed.
- Avoid unrelated formatting changes in the same pull request.

## Pull Request Process

Describe the local review flow used by the repository.

- open a pull request against the correct base branch
- include a short summary and testing notes
- link related issues when relevant
- wait for required checks and reviews before merging

## Code Review

- respond to review comments clearly
- keep review conversations scoped to the change
- prefer small follow-up commits over large rewrites

## Testing Requirements

Document what should be run before a change is merged.

- unit tests for behavior changes
- integration tests when workflows are affected
- linting or static analysis if the repository uses them
- manual verification for changes that cannot be covered automatically

## Documentation

- update README files, guides, and examples when behavior changes
- keep terminology consistent across the repository
- document assumptions that future contributors need to know

## Reporting Issues

Explain where contributors should report bugs, propose improvements, or ask questions.

## FAQ

Use this section for common setup questions, project conventions, or links to more detailed docs.
