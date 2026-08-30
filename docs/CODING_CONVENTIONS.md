# Coding Conventions

This document is a reusable template for code-style rules. Copy it into a new repository and replace the placeholders, examples, and tool names with the standards used by that project.

## How To Adapt This File

- Set the target language or languages for the repository.
- Replace formatter, linter, type checker, and test runner names with the tools you actually use.
- Keep only the sections that apply to the codebase.
- Add project-specific rules where consistency matters.

## Table Of Contents

1. [Language Standards](#language-standards)
2. [Python Coding Style](#python-coding-style)
3. [Type Hints And Docstrings](#type-hints-and-docstrings)
4. [Naming Conventions](#naming-conventions)
5. [Code Organization](#code-organization)
6. [Shell Script Conventions](#shell-script-conventions)
7. [Tools And Analysis](#tools-and-analysis)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Language Standards

Document the language versions and tools used by the repository.

### Python

- Minimum supported version: `<python-version>`
- Maximum version tested: `<python-version>`
- Code style: PEP 8 or the repository's equivalent
- Formatting tool: `<formatter>`
- Linting tool: `<linter>`
- Type checker: `<type-checker>`

### Shell Scripts

- Shebang: `#!/usr/bin/env bash` or the repository's standard
- Shell version: `<shell-version>`
- Linting tool: `<shell-linter>`
- Formatting rule: `<indentation-and-style>`

---

## Python Coding Style

### Indentation

- Use the indentation style required by the language.
- Keep indentation consistent within a file.
- Avoid mixing tabs and spaces.

```python
# Good
class ExampleProcessor:
    def __init__(self, value: int) -> None:
        self.value = value

    def double(self) -> int:
        return self.value * 2

# Bad
class ExampleProcessor:
  def __init__(self, value: int) -> None:
      self.value = value
```

### Line Length

- Keep lines within the repository's configured limit.
- Break long expressions at logical boundaries.
- Prefer implicit continuation inside parentheses.

```python
# Good
result = (
    first_value
    + second_value
    + third_value
)

# Good
formatted_message = build_message(
    prefix=prefix,
    payload=payload,
    suffix=suffix,
)

# Bad
result = first_value + second_value + third_value + fourth_value + fifth_value
```

### Imports

- Group imports by origin.
- Keep the order consistent across the codebase.
- Remove unused imports promptly.

```python
# Standard library
from pathlib import Path
from typing import Final, Optional

# Third-party
import numpy as np

# Local
from .parser import parse_input
```

### Blank Lines

- Use two blank lines between top-level definitions.
- Use one blank line between methods inside a class.
- Use blank lines sparingly inside a function to separate logical steps.

```python
class ExampleProcessor:
    def __init__(self, value: int) -> None:
        self.value = value

    def double(self) -> int:
        return self.value * 2


def parse_value(raw_value: str) -> int:
    cleaned_value = raw_value.strip()
    return int(cleaned_value)
```

### Whitespace

```python
# Good
x = 1 + 2
items = [1, 2, 3]
function(arg1, arg2)

# Bad
x=1+2
items=[1,2,3]
function (arg1)
```

---

## Type Hints And Docstrings

### Type Hints

- Add type hints to public functions and methods.
- Add type hints to internal helpers when they improve clarity.
- Prefer explicit types over implied behavior.
- Keep annotations readable, especially for long signatures.

```python
from collections.abc import Sequence
from typing import Final

DEFAULT_LIMIT: Final[int] = 10


def normalize_values(values: Sequence[float]) -> list[float]:
    return [value / DEFAULT_LIMIT for value in values]
```

### Optional And Union Types

```python
from pathlib import Path
from typing import Optional


def load_config(path: Path, fallback: Optional[str] = None) -> str:
    if fallback is not None:
        return fallback
    return path.read_text(encoding="utf-8")
```

### Complex Types

```python
from collections.abc import Callable, Sequence

Processor = Callable[[Sequence[str]], list[str]]


def process_items(items: Sequence[str], processor: Processor) -> list[str]:
    return processor(items)
```

### Docstrings

- Use docstrings for public modules, classes, functions, and methods when they add value.
- Keep docstrings short and direct.
- Describe what the code does, not how it does it.
- Match the docstring style used by the repository.

```python
"""Utilities for formatting output values."""

from pathlib import Path


def format_output(path: Path, value: str) -> str:
    """Format a value for writing to disk."""
    return f"{path.name}: {value}"


class OutputWriter:
    """Write formatted output to a destination."""

    def __init__(self, destination: Path) -> None:
        """Store the destination path."""
        self.destination = destination
```

### Module Docstrings

Use module docstrings when a file exposes public behavior or needs a short summary.

```python
"""Parse and validate application input."""
```

### Comments

- Use comments to explain intent, constraints, or trade-offs.
- Avoid comments that simply repeat the code.
- Remove stale comments when code changes.

```python
# Good: explains the reason for the guard.
if retries > max_retries:
    raise RuntimeError("retry limit reached")

# Bad: repeats the code.
# Increase retries.
retries += 1
```

---

## Naming Conventions

### Module Names

- Use lowercase letters and underscores.
- Keep names descriptive and concise.
- Match the feature or responsibility of the module.

```python
# Good
input_parser.py
result_formatter.py
cache_store.py

# Bad
InputParser.py
utils.py
misc.py
```

### Class Names

- Use PascalCase for classes.
- Prefer noun-based names.
- Avoid suffixes like `Class`.

```python
class InputParser:
    pass


class ResultFormatter:
    pass
```

### Function And Method Names

- Use snake_case for functions and methods.
- Prefer action verbs for behavior.
- Keep names specific enough to understand without the implementation.

```python
def parse_input_file():
    pass


def format_result_value():
    pass
```

### Variable Names

- Use snake_case for variables.
- Choose names that reflect purpose.
- Avoid one-letter names except in small mathematical contexts.

```python
value_count = 3
input_path = Path("input.txt")
normalized_values = []
```

### Constants

- Use uppercase_with_underscores for module-level constants.
- Use `Final` where the language ecosystem supports it.

```python
from typing import Final

MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
```

### Private Members

- Prefix internal helpers and attributes with a single underscore.
- Use double underscores only when name collisions must be avoided.

```python
class CacheStore:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def _clear_expired_items(self) -> None:
        self._items.clear()
```

---

## Code Organization

### Module Structure

A module that exposes public APIs can follow a structure like this:

```python
"""Utilities for processing input data."""

from __future__ import annotations

from pathlib import Path

DEFAULT_SEPARATOR = ","


class InputProcessor:
    """Process input values from files or strings."""

    def __init__(self, separator: str = DEFAULT_SEPARATOR) -> None:
        """Store the separator used during parsing."""
        self.separator = separator

    def parse(self, raw_value: str) -> list[str]:
        """Split a raw value into tokens."""
        return [part.strip() for part in raw_value.split(self.separator)]


def load_text(path: Path) -> str:
    """Load text from a file path."""
    return path.read_text(encoding="utf-8")
```

### Class Organization

- Put class constants near the top of the class.
- Group public methods before private helpers when that improves readability.
- Keep object state minimal and explicit.

```python
class InputProcessor:
    DEFAULT_SEPARATOR = ","

    def __init__(self, separator: str = DEFAULT_SEPARATOR) -> None:
        """Store the separator used during parsing."""
        self.separator = separator

    def parse(self, raw_value: str) -> list[str]:
        """Split a raw value into tokens."""
        return self._split(raw_value)

    def _split(self, raw_value: str) -> list[str]:
        """Split the value using the configured separator."""
        return [part.strip() for part in raw_value.split(self.separator)]
```

### Public API Boundaries

- Keep public APIs small and stable.
- Validate inputs at the edges of the system.
- Convert external formats into internal data structures early.
- Keep internal helpers private unless they are meant to be reused.

---

## Shell Script Conventions

- Use shell scripts for automation, setup, or maintenance tasks.
- Keep scripts deterministic and easy to read.
- Quote variables and command substitutions.
- Prefer explicit error handling.

```bash
#!/usr/bin/env bash

set -euo pipefail

input_file="${1:?input file required}"
output_file="${2:?output file required}"

printf 'Processing %s -> %s\n' "$input_file" "$output_file"
```

### Shell Script Doc Comments

Use short comments to document script purpose and usage when it helps a future maintainer.

```bash
#!/usr/bin/env bash
# Usage: bash scripts/generate_report.sh <input> <output>
```

---

## Tools And Analysis

Document the repository's toolchain here.

- Formatter: `<formatter>`
- Linter: `<linter>`
- Type checker: `<type-checker>`
- Test runner: `<test-runner>`
- Coverage tool: `<coverage-tool>`

Use automated checks to catch style regressions early.

---

## Best Practices

- Prefer small, focused changes.
- Keep code and tests aligned.
- Make error messages actionable.
- Document assumptions that are easy to forget.
- Remove dead code rather than leaving it commented out.
- Update examples when the public API changes.

---

## Examples

### Example Module

```python
"""Format a path and value for display."""

from pathlib import Path


def format_entry(path: Path, value: str) -> str:
    """Create a human-readable entry."""
    return f"{path.name}: {value}"
```

### Example Class

```python
class EntryFormatter:
    """Format entries for output."""

    def __init__(self, prefix: str) -> None:
        """Store the prefix used for formatting."""
        self.prefix = prefix

    def format(self, value: str) -> str:
        """Format a value with the configured prefix."""
        return f"{self.prefix}{value}"
```

### Example Function With Validation

```python
from pathlib import Path


def read_non_empty_text(path: Path) -> str:
    """Read text from a file and reject empty content."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("file content must not be empty")
    return text
```
