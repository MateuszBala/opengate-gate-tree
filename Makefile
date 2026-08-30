# FILE: Makefile
# ABOUT: Project helper tasks: installation, validation, and worktree
#        management for feature branches created from develop.
# USAGE:
#   make init | install | install-user | test | lint | format | typecheck | check
#   make worktree BRANCH_NAME=feature/description
#   make switch-to-worktree BRANCH_NAME=feature/description

SHELL := /bin/bash

# Parent directory for worktrees and directory for lock files.
WORKTREE_ROOT ?= ./worktrees
WORKTREE_LOCKS_DIR ?= .tmp/worktree-locks

.PHONY: init install install-user test lint format typecheck check docs docs-check worktree switch-to-worktree

# Install dependencies and create the virtual environment (uv).
init:
	uv sync

# Install project dependencies and package into the local .venv.
install: init
	@echo "Installed in local virtual environment: .venv"
	@echo "Run via uv: uv run opengate-gate-tree --help"
	@echo "Or activate the environment: source .venv/bin/activate"

# Install the package for the current user (outside project .venv).
install-user:
	python3 -m pip install --user -e .
	@echo "Installed in user site-packages (--user)."
	@echo "If needed, add ~/.local/bin to PATH."

# Run the full test suite.
test:
	uv run pytest

# Check code style (lint).
lint:
	uv run ruff check

# Format code.
format:
	uv run ruff format

# Static type checking.
typecheck:
	uv run mypy src/

# Full local validation: lint, formatting check, types, tests.
check:
	uv run ruff check
	uv run ruff format --check
	uv run mypy src/
	uv run pytest

# Build the user documentation as HTML.
docs:
	uv run --extra docs sphinx-build -b html docs docs/_build/html
	@echo "Documentation built: docs/_build/html/index.html"

# Build the documentation with warnings treated as errors, as ReadTheDocs does.
docs-check:
	uv run --extra docs sphinx-build -b html -W --keep-going docs docs/_build/html

# Create a worktree for the given branch if it does not already exist.
worktree:
	@if [[ -z "$(BRANCH_NAME)" ]]; then \
		echo "Error: set BRANCH_NAME, e.g. make worktree BRANCH_NAME=worktree/branch/name." >&2; \
		exit 1; \
	fi
	@WORKTREE_DIR="$(WORKTREE_ROOT)/$$(echo "$(BRANCH_NAME)" | tr '/' '-')"; \
	mkdir -p "$(WORKTREE_ROOT)"; \
	if git worktree list --porcelain | grep -Fxq "branch refs/heads/$(BRANCH_NAME)"; then \
		echo "Worktree for branch $(BRANCH_NAME) already exists."; \
		exit 0; \
	fi; \
	if git show-ref --verify --quiet "refs/heads/$(BRANCH_NAME)"; then \
		git worktree add "$$WORKTREE_DIR" "$(BRANCH_NAME)"; \
	else \
		git worktree add -b "$(BRANCH_NAME)" "$$WORKTREE_DIR" develop; \
	fi; \
	echo "Created worktree: $$WORKTREE_DIR (branch: $(BRANCH_NAME))."

# Switch to an existing worktree if it is not locked by another instance.
switch-to-worktree:
	@if [[ -z "$(BRANCH_NAME)" ]]; then \
		echo "Error: set BRANCH_NAME, e.g. make switch-to-worktree BRANCH_NAME=worktree/branch/name." >&2; \
		exit 1; \
	fi
	@if ! git show-ref --verify --quiet "refs/heads/$(BRANCH_NAME)"; then \
		echo "Error: the specified branch does not exist: $(BRANCH_NAME)." >&2; \
		exit 1; \
	fi
	@WORKTREE_DIR="$$(git worktree list --porcelain | awk -v target="refs/heads/$(BRANCH_NAME)" '\
		$$1=="worktree" { path=$$2 } \
		$$1=="branch" && $$2==target { print path; exit }')"; \
	if [[ -z "$$WORKTREE_DIR" ]]; then \
		echo "Error: no worktree exists for branch $(BRANCH_NAME). First run make worktree BRANCH_NAME=$(BRANCH_NAME)." >&2; \
		exit 1; \
	fi; \
	LOCK_FILE="$(WORKTREE_LOCKS_DIR)/$$(echo "$(BRANCH_NAME)" | tr '/' '-').lock"; \
	mkdir -p "$(WORKTREE_LOCKS_DIR)"; \
	if [[ -f "$$LOCK_FILE" ]]; then \
		LOCK_PID="$$(cat "$$LOCK_FILE")"; \
		if [[ -n "$$LOCK_PID" ]] && kill -0 "$$LOCK_PID" 2>/dev/null; then \
			echo "Error: the worktree is already locked by another instance (PID: $$LOCK_PID)." >&2; \
			echo "How to release it: stop the process with PID $$LOCK_PID (for example, close the shell holding the lock)." >&2; \
			exit 1; \
		fi; \
		rm -f "$$LOCK_FILE"; \
	fi; \
	echo "$$BASHPID" > "$$LOCK_FILE"; \
	cleanup() { rm -f "$$LOCK_FILE"; }; \
	trap cleanup EXIT INT TERM; \
	echo "Switched to $$WORKTREE_DIR. To release the lock, close this shell."; \
	cd "$$WORKTREE_DIR"; \
	"$${SHELL:-/bin/bash}" -i