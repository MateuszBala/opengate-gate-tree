# Testing Conventions

This document is a reusable template for test-writing rules. Copy it into a new repository and replace the placeholders, examples, and tool names with the standards used by that project.

## How To Adapt This File

- Replace `<test-framework>` with the real framework used by the repository.
- Replace `<test-runner>` with the command used to execute tests.
- Replace `<coverage-tool>` with the tool used for coverage reporting.
- Adjust the examples to match the language and directory layout.
- Remove sections that do not apply to the project.

## Table Of Contents

1. [General Principles](#general-principles)
2. [Test Organization](#test-organization)
3. [Test Structure](#test-structure)
4. [Running Tests](#running-tests)
5. [Pytest Or Equivalent Configuration](#pytest-or-equivalent-configuration)
6. [Fixtures And Setup](#fixtures-and-setup)
7. [Parameterized Tests](#parameterized-tests)
8. [Mocking](#mocking)
9. [Coverage](#coverage)
10. [Best Practices](#best-practices)
11. [Examples](#examples)

---

## General Principles

### Testing Philosophy

- Every important behavior should have automated tests.
- Tests should describe behavior, not implementation details.
- Tests should be independent and repeatable.
- Prefer fast unit tests, then add broader integration tests where needed.
- Keep test names clear enough that the intent is obvious from the file list.

### Test Goals

- Verify functionality works correctly.
- Catch regressions early.
- Document expected behavior.
- Enable safe refactoring.
- Maintain high code quality.

### Framework

- Testing framework: `<test-framework>`
- Test discovery: `<discovery-rules>`
- Virtual environment or package manager: `<environment-tool>`
- Coverage reporting: `<coverage-tool>`

---

## Test Organization

### Directory Structure

Use a layout that matches the repository, then keep it consistent.

```text
project/
├── src/
│   └── <package-name>/
│       ├── <module>.py
│       └── <module>.py
└── tests/
    ├── unit/
    │   ├── test_<module>.py
    │   └── test_<module>.py
    ├── integration/
    │   └── test_<workflow>.py
    └── conftest.py
```

### File Naming Convention

- Unit tests: `test_<module>.py`
- Integration tests: `test_<workflow>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<expected_behavior>`

```python
class TestParser:
    def test_parse_returns_expected_tokens(self):
        pass

    def test_parse_rejects_empty_input(self):
        pass


def test_normalize_values_returns_unit_length_values():
    pass


def test_normalize_values_raises_for_zero_vector():
    pass
```

---

## Test Structure

### Arrange-Act-Assert

Every test should follow the Arrange-Act-Assert pattern.

```python
def test_function_expected_behavior():
    # ARRANGE
    input_data = ...
    expected_result = ...

    # ACT
    actual_result = function_under_test(input_data)

    # ASSERT
    assert actual_result == expected_result
```

### Example Test

```python
import pytest

from <package>.module import function_under_test


def test_function_name_returns_something() -> None:
    """Example test for a representative happy path."""
    # ARRANGE
    input_value = "sample input"
    expected_result = "expected output"

    # ACT
    actual_result = function_under_test(input_value)

    # ASSERT
    assert actual_result == expected_result


@pytest.mark.parametrize(
    "input_value,expected_result",
    [
        ("input one", "output one"),
        ("input two", "output two"),
        ("input three", "output three"),
    ],
)
def test_function_name_handles_multiple_cases(
    input_value: str,
    expected_result: str,
) -> None:
    """Parameterized test for a generic function."""
    # ARRANGE
    # No additional setup required.

    # ACT
    actual_result = function_under_test(input_value)

    # ASSERT
    assert actual_result == expected_result
```

---

## Running Tests

Document the commands contributors should use, for example:

```bash
<test-runner>
<test-runner> path/to/specific-test
<test-runner> --watch
<test-runner> --coverage
```

If the repository has separate commands for unit, integration, or end-to-end tests, list them here.

### Example Commands

```bash
<environment-tool> sync
<environment-tool> run <test-runner>
<environment-tool> run <test-runner> --coverage
```

---

## Pytest Or Equivalent Configuration

### Example `pytest.ini` or `pyproject.toml`

```toml
[tool.pytest.ini_options]
minversion = "7.0"
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
testpaths = ["tests"]
addopts = [
  "--verbose",
  "--strict-markers",
  "--tb=short",
  "--cov=src",
  "--cov-report=term-missing",
]
markers = [
  "unit: Unit tests",
  "integration: Integration tests",
  "slow: Slow tests",
]
```

### Coverage Configuration Example

```ini
[run]
source = src

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:

precision = 2
```

---

## Fixtures And Setup

Fixtures provide reusable test setup.

### Basic Fixtures

```python
import pytest


@pytest.fixture
def sample_input() -> str:
    """Provide a sample input value for testing."""
    return "sample input"


@pytest.fixture
def sample_pair() -> tuple[str, str]:
    """Provide two related sample values."""
    return ("first value", "second value")


@pytest.fixture
def sample_config() -> dict[str, str]:
    """Provide sample configuration data."""
    return {
        "alpha": "one",
        "beta": "two",
        "gamma": "three",
    }
```

### Using Fixtures

```python
def test_function_with_sample_input(sample_input: str) -> None:
    """Test a function with a fixture."""
    result = function_under_test(sample_input)

    assert result == "expected output"


def test_function_with_sample_pair(sample_pair: tuple[str, str]) -> None:
    """Test a function with a pair fixture."""
    first_value, second_value = sample_pair

    result = combine_values(first_value, second_value)

    assert result == "first value-second value"
```

### Fixture Scopes

```python
import pytest


@pytest.fixture(scope="module")
def expensive_data_load() -> list[str]:
    """Load data once for the entire test module."""
    return ["one", "two", "three"]


@pytest.fixture(scope="class")
def test_helper() -> object:
    """Create a shared helper for one test class."""
    return object()


@pytest.fixture(scope="function")
def fresh_data() -> list[str]:
    """Create fresh data for each test."""
    return ["a", "b", "c"]
```

---

## Parameterized Tests

Use parameterized tests when several inputs should exercise the same behavior.

### Basic Parameterization

```python
import pytest


@pytest.mark.parametrize(
    "input_value,expected_result",
    [
        ("alpha", "ALPHA"),
        ("beta", "BETA"),
        ("gamma", "GAMMA"),
    ],
)
def test_function_multiple_cases(
    input_value: str,
    expected_result: str,
) -> None:
    """Test a function for several representative inputs."""
    actual_result = transform_value(input_value)

    assert actual_result == expected_result
```

### Indirect Parameterization

```python
import pytest


@pytest.fixture
def angle_data(request: pytest.FixtureRequest) -> dict[str, float]:
    """Fixture that uses parametrization."""
    return request.param


@pytest.mark.parametrize(
    "angle_data",
    [
        {"theta": 0.5, "phi": 1.0},
        {"theta": 1.0, "phi": 0.5},
        {"theta": 2.0, "phi": 3.0},
    ],
    indirect=True,
)
def test_with_parametrized_fixture(angle_data: dict[str, float]) -> None:
    """Test using a parametrized fixture."""
    assert angle_data["theta"] > 0
```

---

## Mocking

Mock external dependencies for isolated unit tests.

### Basic Mocking

```python
from unittest.mock import Mock, patch


def test_function_with_mocked_dependency() -> None:
    """Test a function while mocking an external dependency."""
    expected_data = {"status": "ok"}

    with patch("package.external_dependency") as mock_dependency:
        mock_dependency.return_value = expected_data

        result = function_under_test("input value")

        assert result == expected_data
        mock_dependency.assert_called_once_with("input value")


def test_function_with_mocked_response() -> None:
    """Test response handling with a mocked object."""
    with patch("package.get_response") as mock_get_response:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_get_response.return_value = mock_response

        result = process_response("resource-id")

        assert result["result"] == "success"
```

### When To Mock

Use mocks for:

- external APIs
- file I/O
- database operations
- time-dependent functions
- slow or flaky boundaries

### When Not To Mock

- when the dependency is simple and deterministic
- when a small fake is easier to understand
- when the real dependency is cheap and stable
- when mocking would hide the behavior under test

---

## Coverage

- Use coverage as a signal, not the only quality gate.
- Cover branches and error paths that matter to users.
- Do not chase coverage numbers at the expense of meaningful tests.
- Document any minimum threshold the repository enforces.

### Coverage Checklist

- new behavior is covered by tests
- existing behavior still passes
- failure paths are tested when relevant
- fixtures and helpers stay simple
- coverage configuration matches the repository's expectations

---

## Best Practices

- Keep tests deterministic.
- Use explicit assertions.
- Name failing cases clearly.
- Avoid test dependencies.
- Review tests with the same care as production code.
- Keep test code readable enough to serve as documentation.

---

## Examples

### Example Unit Test

```python
def test_read_non_empty_text_returns_content(tmp_path) -> None:
    """Read text from a file and reject empty content."""
    path = tmp_path / "input.txt"
    path.write_text("hello\n", encoding="utf-8")

    result = read_non_empty_text(path)

    assert result == "hello\n"
```

### Example Integration Test

```python
def test_full_pipeline_writes_output(tmp_path) -> None:
    """Run the pipeline end to end for a small input."""
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text("1,2,3\n", encoding="utf-8")

    run_pipeline(input_path, output_path)

    assert output_path.exists()
```

### Example Checklist For Review

Before merging, confirm that:

- new behavior is covered by tests
- existing behavior still passes
- test commands are documented
- any new fixtures or helpers are easy to understand
